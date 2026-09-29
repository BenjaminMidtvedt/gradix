"""Registries and the plugin contract (§7.1, §11.13): complete schemas, duplicate names, lookups."""

import dataclasses

import pytest

import gradix as gx
from gradix._core.contract import Element


def test_every_registered_element_meets_the_contract():
    assert set(gx.registry.elements.names()) >= {
        "emit.gaussian",
        "detect.camera",
        "source.plane_waves",
    }
    for name in gx.registry.elements.names():
        cls = gx.registry.elements.get(name)
        assert issubclass(cls, Element) and cls.registry_name == name
        assert isinstance(cls.slot, gx.Slot) and isinstance(cls.caps, gx.Capabilities)
        for method in ("validity", "configure", "forward"):
            assert callable(getattr(cls, method))
        gx.schema_of(cls)  # a complete schema


def test_every_registered_data_object_has_a_schema():
    for registry in (gx.registry.object_sets, gx.registry.noise_models, gx.registry.labels):
        for name in registry.names():
            schema = gx.schema_of(registry.get(name))
            assert schema["name"] == name


def test_duplicate_names_need_override():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Probe(gx.DataObject):
        pass

    gx.register.object_set("probe_test")(Probe)
    try:
        with pytest.raises(gx.RegistryError, match="already registered"):

            @gx.register.object_set("probe_test")
            @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
            class Other(gx.DataObject):
                pass

        gx.register.object_set("probe_test", override=True)(Probe)
    finally:
        gx.registry.object_sets.unregister("probe_test")
    assert "probe_test" not in gx.registry.object_sets


def test_plugin_element_registration():
    @gx.register.element("wpm_test", slot="interact")
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class WPM(gx.Element):
        slot = gx.Slot.INTERACT
        caps = gx.Capabilities(produces=gx.Field, grad=frozenset({"checkpoint"}))
        dz: float | str = gx.knob(default="auto")

    try:
        assert gx.registry.elements.get("interact.wpm_test") is WPM
        assert gx.schema_of(WPM)["fields"][0]["name"] == "dz"
    finally:
        gx.registry.elements.unregister("interact.wpm_test")
    with pytest.raises(gx.RegistryError, match=r"not a gx\.Element"):
        gx.register.element("bad")(gx.Emitters)


def test_capabilities_json():
    caps = gx.imaging.Sprites.caps
    data = caps.to_json()
    assert data["accepts"] == ["EmitterSet"] and data["produces"] == ["Irradiance"]
    assert data["approximates"][0]["target"] == "emit.pupil_mft"
