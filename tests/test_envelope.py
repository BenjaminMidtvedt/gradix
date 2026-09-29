"""Envelopes (§6.4): derivation with headroom and buckets, presence, sources, in-envelope flags."""

import math

import pytest
import torch
from scenes import chain, emitters

import gradix as gx
from gradix._core.envelope import Entry, check_paths, in_envelope, widen
from gradix.schema.fields import FieldSpec


def _spec(quantity="length", constraint=None):
    return FieldSpec(
        kind="tensor",
        quantity=quantity,
        role="image",
        constraint=constraint,
        scale="log" if constraint == "positive" else "linear",
    )


def test_linear_buckets_are_strictly_outward():
    assert widen(0.1, 0.1, _spec()) == (0.0, 0.25)
    assert widen(0.0, 0.0, _spec()) == (-0.25, 0.25)
    lo, hi = widen(-1.0, 1.0, _spec())
    assert lo <= -1.25 and hi >= 1.25


def test_log_buckets_for_positive_sizes():
    lo, hi = widen(0.3, 0.3, _spec(constraint="positive"))
    # a single value keeps x1.25 of room on both sides, rounded outward to 10 % buckets
    assert lo <= 0.3 / 1.25 and hi >= 0.3 * 1.25
    steps = math.log(hi / lo) / math.log(1.1)
    assert steps == pytest.approx(round(steps)) and round(steps) <= 6
    lo, hi = widen(50.0, 50.0, _spec(constraint="positive"))  # a learnable magnification
    assert lo <= 40.0 and hi >= 62.5


def test_linear_zero_spans_get_multiplicative_room():
    lo, hi = widen(2.0, 2.0, _spec(quantity="dimensionless"))
    assert lo == pytest.approx(1.5) and hi == pytest.approx(2.5)


def test_constants_get_exact_entries_and_tensors_get_headroom():
    beads = emitters(2, 3, z=0.3)
    env = gx.envelope_of(
        {"beads": beads, "objective": gx.Objective(NA=0.7, focus=torch.tensor(0.1))}
    )
    assert env.range("objective.NA") == (0.7, 0.7)  # a Python number: a constant
    focus = env["objective.focus"]
    assert focus.lo < 0.1 < focus.hi and focus.source == "derived"
    entry = env["beads.position.z"]
    assert isinstance(beads.position, torch.Tensor)
    z = beads.position[..., 2]
    assert entry.lo < float(z.min()) and entry.hi > float(z.max())


def test_absent_slots_are_ignored():
    pos = torch.zeros(1, 3, 3)
    pos[0, 2, 2] = 100.0  # an absent slot far away
    beads = gx.Emitters(
        position=pos,
        photons=1.0,
        presence=torch.tensor([[1.0, 1.0, 0.0]]),
        emission=gx.Spectrum.line(0.6),
    )
    env = gx.envelope_of({"beads": beads})
    assert env["beads.position.z"].hi < 1.0


def test_tables_with_presence():
    pos, presence = gx.pad([torch.randn(n, 3) for n in (1, 4, 2)])
    env = gx.envelope_of({"beads.position": pos, "beads.presence": presence})
    assert {"beads.position.x", "beads.position.y", "beads.position.z"} <= set(env)


def test_sources_combine_entry_by_entry():
    derived = gx.envelope_of(chain(2, 3))
    combined = derived | {"beads.position.z": (-8.0, 8.0)}
    assert combined.range("beads.position.z") == (-8.0, 8.0)
    assert combined["beads.position.z"].source == "explicit"
    assert combined.range("objective.NA") == derived.range("objective.NA")
    assert ({"x": (0.0, 1.0)} | gx.Envelope({"x": (2.0, 3.0)})).range("x") == (2.0, 3.0)
    union = gx.Envelope({"x": (0.0, 1.0)}).union({"x": (-1.0, 0.5), "y": (2.0, 2.0)})
    assert union.range("x") == (-1.0, 1.0) and union.range("y") == (2.0, 2.0)


def test_bad_entries():
    with pytest.raises(gx.EnvelopeError):
        gx.Envelope({"x": (1.0, 0.0)})
    with pytest.raises(gx.EnvelopeError):
        gx.Envelope({"x": 1.0})  # ty: ignore[invalid-argument-type] - deliberately malformed


def test_json_roundtrip():
    env = gx.envelope_of(chain(2, 3)) | {"beads.position.z": (-1.0, 1.0)}
    back = gx.Envelope.from_json(env.to_json())
    assert back.key() == env.key() and back["beads.position.z"].source == "explicit"


def test_in_envelope_flags_per_image():
    c = chain(4, 3)
    pipe = gx.Pipeline(c, outputs=("expected",), envelope={"beads.position.z": (-0.5, 0.5)})
    beads = c.emitters["beads"]
    pos = beads.position.clone()
    pos[2, 1, 2] = 3.0  # image 2 leaves the z envelope
    moved = c.replace(emitters={"beads": beads.replace(position=pos)})
    flags = pipe(moved).meta["in_envelope"]
    assert flags.tolist() == [True, True, False, True]
    with pytest.raises(gx.EnvelopeError, match=r"images \[2\]"):
        pipe(moved, check="raise")


def test_changed_constant_flags_every_image():
    c = chain(2, 3)
    pipe = gx.Pipeline(c, outputs=("expected",))
    other = gx.tree.replace(c, {"imaging.objective.NA": 0.5})
    assert not pipe(other).meta["in_envelope"].any()


def test_entry_validation():
    assert Entry(0, 1).source == "explicit"


def test_presence_broader_than_values_is_supported():
    # blinking emitters with static positions: presence [B, T, N], positions [B, N, 3]
    beads = gx.Emitters(
        position=torch.zeros(2, 5, 3),
        photons=torch.full((2, 1, 5), 100.0),
        presence=torch.ones(2, 4, 5),
        emission=gx.Spectrum.line(0.6),
    )
    env = gx.envelope_of({"beads": beads})
    assert env.range("beads.position.z") is not None
    assert bool(in_envelope(env, {"beads": beads}, 2).all())


def test_non_finite_values_name_the_field():
    beads = emitters(1, 3, z=float("nan"))
    with pytest.raises(gx.EnvelopeError, match=r"beads\.position\.z"):
        gx.envelope_of({"beads": beads})


def test_tables_with_a_template_give_the_same_entries_as_nodes():
    beads = emitters(2, 3, z=0.3)
    nodes = gx.envelope_of({"beads": beads})
    tables = {"beads.position": beads.position, "beads.photons": beads.photons}
    like = {"beads": beads}
    from_tables = gx.envelope_of(tables, like=like)
    assert set(from_tables) == {"beads.position.x", "beads.position.y", "beads.position.z"}
    assert all(from_tables.range(p) == nodes.range(p) for p in from_tables)
    with pytest.raises(gx.EnvelopeError, match="did you mean"):
        gx.envelope_of({"beads.positon": beads.position}, like=like)


def test_declared_constants_are_widened():
    scene = {"beads": emitters(1, 3), "objective": gx.Objective(NA=0.7, focus=0.0)}
    exact = gx.envelope_of(scene)
    assert exact.range("objective.focus") == (0.0, 0.0)
    widened = gx.envelope_of(scene, inputs=("objective.focus",))
    bounds = widened.range("objective.focus")
    assert bounds is not None and bounds[0] < 0.0 < bounds[1]


def test_explicit_entries_are_checked_against_the_fields():
    scene = {"beads": emitters(1, 3), "objective": gx.Objective(NA=0.7)}
    check_paths({"beads.position.z": (-1, 1)}, scene)
    with pytest.raises(gx.EnvelopeError, match="did you mean"):
        check_paths({"beads.positon.z": (-1, 1)}, scene)
    with pytest.raises(gx.EnvelopeError, match=r"beads\.position\.x"):
        check_paths({"beads.position": (-1, 1)}, scene)
