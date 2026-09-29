"""Acquisitions (§4.2, §6.7, M1): frames and focus stacks on the acquisition axis A."""

import pytest
import torch
from scenes import emitters, optics

import gradix as gx


def _chain(beads, acquisition, **kw):
    objective, camera = optics(shape=(32, 32), **kw)
    return gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=acquisition,
    )


def test_frames_image_a_static_scene_over_time():
    c = _chain(emitters(2, 3, fov=4.0), gx.acq.Frames(4, interval=0.02))
    mu = c(outputs=("expected",))["expected"]
    assert mu.shape == (2, 4, 1, 32, 32)
    assert torch.equal(mu[:, 0], mu[:, 3])
    assert gx.Pipeline(c).frames == 4


def test_frames_take_per_frame_fields_and_check_their_count():
    base = emitters(2, 3, fov=4.0)
    moving = base.replace(position=base.position[:, None].repeat(1, 4, 1, 1) + 0.1)
    moving = moving.replace(
        position=moving.position * torch.linspace(1, 1.2, 4)[None, :, None, None]
    )
    mu = _chain(moving, gx.acq.Frames(4))(outputs=("expected",))["expected"]
    assert not torch.equal(mu[:, 0], mu[:, 3])
    three = base.replace(position=base.position[:, None].repeat(1, 3, 1, 1))
    with pytest.raises(gx.StructureError, match="acquisition declares 4"):
        _chain(three, gx.acq.Frames(4))(outputs=("expected",))


def test_a_focus_stack_equals_refocused_renders():
    beads = emitters(1, 3, fov=4.0)
    steps = torch.tensor([-0.3, 0.0, 0.4])
    stack = _chain(beads, gx.acq.FocusStack(focus=steps))(outputs=("expected",))["expected"]
    assert stack.shape == (1, 3, 32, 32)  # the focus axis goes to channels
    for a, step in enumerate(steps.tolist()):
        objective, camera = optics(shape=(32, 32))
        single = gx.Chain(
            emitters={"beads": beads},
            imaging=gx.imaging.Sprites(objective.replace(focus=step), camera),
            environment=gx.env.Homogeneous(1.33),
        )(outputs=("expected",))["expected"]
        assert torch.allclose(stack[:, a], single[:, 0], atol=1e-5)


def test_acquisition_must_be_an_acquisition():
    with pytest.raises(gx.StructureError, match=r"gx.acq"):
        _chain(emitters(1, 2), gx.Objective(NA=1.0))


def test_focus_stacks_enter_validity_and_psf_sizing():
    beads = emitters(1, 3, fov=4.0)
    deep = gx.acq.FocusStack(focus=torch.tensor([-1.5, 0.0, 1.5]))
    with pytest.warns(gx.GradixWarning, match="DOF"):
        gx.Pipeline(_chain(beads, deep), outputs=("expected",))
    objective, camera = optics(shape=(32, 32))
    shallow = gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.PointPSF(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    stacked = shallow.replace(acquisition=deep)
    roi_flat = gx.Pipeline(shallow).static("imaging")
    roi_stack = gx.Pipeline(stacked).static("imaging")
    assert isinstance(roi_flat, gx.imaging.PointPSFStatic)
    assert isinstance(roi_stack, gx.imaging.PointPSFStatic)
    assert roi_stack.roi > roi_flat.roi  # the ROI covers the stack's defocus


def test_plugin_acquisitions_bind_per_frame_settings():
    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Biplane(gx.acq.Acquisition):
        """Two focal planes separated by ``gap`` µm, as two channels."""

        gap: float = gx.knob(default=0.4)

        def index(self):
            return gx.AcqIndex((("plane", 2),))

        def bindings(self, chain):
            focus = torch.as_tensor(chain.objective.focus, dtype=torch.float32).reshape(-1, 1)
            return {"objective.focus": focus + torch.tensor([[-0.5, 0.5]]) * self.gap}

    beads = emitters(1, 3, fov=4.0)
    image = _chain(beads, Biplane())(outputs=("expected",))["expected"]
    assert image.shape == (1, 2, 32, 32)
    objective, camera = optics(shape=(32, 32))
    for c, focus in enumerate((-0.2, 0.2)):
        single = gx.Chain(
            emitters={"beads": beads},
            imaging=gx.imaging.Sprites(objective.replace(focus=focus), camera),
            environment=gx.env.Homogeneous(1.33),
        )(outputs=("expected",))["expected"]
        assert torch.allclose(image[:, c], single[:, 0], atol=1e-6)


def test_focus_drift_is_a_per_frame_setting():
    objective, camera = optics(shape=(32, 32))
    drift = torch.tensor([[0.0, 0.1, 0.2, 0.3]])  # [B=1, T=4]
    chain = gx.Chain(
        emitters={"beads": emitters(1, 3, fov=4.0)},
        imaging=gx.imaging.Sprites(objective.replace(focus=drift), camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.Frames(4),
    )
    mu = chain(outputs=("expected",))["expected"]
    assert mu.shape == (1, 4, 1, 32, 32) and not torch.equal(mu[:, 0], mu[:, 3])
