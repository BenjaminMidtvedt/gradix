"""Carriers (§4.2, §4.4): fixed shapes that broadcast over B and A; grids; acquisition layout."""

import pytest
import torch

import gradix as gx
from gradix._core.axes import AcqIndex
from gradix.detect.camera import format_frames


def _emitter_set(b, a, n=3, s=1, bins=1):
    return gx.EmitterSet(
        position=torch.zeros(b, a, n, 3),
        photons=torch.ones(b, a, n),
        presence=torch.ones(b, a, n),
        species=torch.zeros(b, 1, n, dtype=torch.int64),
        wavelengths=torch.full((b, s, bins), 0.6),
        weights=torch.full((b, s, bins), 1.0 / bins),
    )


@pytest.mark.parametrize(("b", "a"), [(1, 1), (3, 1), (1, 4), (3, 4)])
def test_emitter_set_broadcasts_over_b_and_a(b, a):
    es = _emitter_set(b, a)
    assert es.slots == 3


def test_carrier_dimension_patterns_are_checked():
    with pytest.raises(gx.StructureError, match="dimension N disagrees"):
        gx.EmitterSet(
            position=torch.zeros(1, 1, 3, 3),
            photons=torch.ones(1, 1, 4),
            wavelengths=torch.full((1, 1, 1), 0.6),
            weights=torch.ones(1, 1, 1),
        )
    with pytest.raises(gx.StructureError, match="dimension B disagrees"):
        gx.EmitterSet(
            position=torch.zeros(2, 1, 3, 3),
            photons=torch.ones(5, 1, 3),
            wavelengths=torch.full((1, 1, 1), 0.6),
            weights=torch.ones(1, 1, 1),
        )
    with pytest.raises(gx.StructureError):
        gx.EmitterSet(
            position=torch.zeros(1, 3, 3),  # rank 3, needs 4
            photons=torch.ones(1, 1, 3),
            wavelengths=torch.full((1, 1, 1), 0.6),
            weights=torch.ones(1, 1, 1),
        )


@pytest.mark.parametrize(("b", "a", "bins"), [(1, 1, 1), (2, 3, 1), (2, 1, 4)])
def test_irradiance_broadcasts(b, a, bins):
    grid = gx.Grid2D((8, 6), 0.1)
    irr = gx.Irradiance(data=torch.ones(b, a, bins, 8, 6), grid=grid)
    assert (irr + irr).data.sum() == 2 * b * a * bins * 48
    with pytest.raises(gx.StructureError, match="grid"):
        gx.Irradiance(data=torch.ones(1, 1, 1, 8, 7), grid=grid)


def test_emitter_density_and_plane_waves_patterns():
    vol = gx.VolumeGrid(gx.Grid2D((4, 5), 0.1), z0=-1.0, dz=0.5, nz=3)
    dens = gx.EmitterDensity(
        data=torch.ones(2, 1, 1, 3, 4, 5),
        wavelengths=torch.full((1, 1, 1), 0.6),
        weights=torch.ones(1, 1, 1),
        grid=vol,
    )
    assert dens.grid.z().tolist() == [-1.0, -0.5, 0.0]
    waves = gx.light.PlaneWave(0.5, irradiance=torch.tensor([1.0, 4.0]))()
    assert waves.amplitude.shape == (2, 1, 1, 1, 1, 1)
    assert torch.allclose(waves.amplitude.abs().flatten() ** 2, torch.tensor([1.0, 4.0]))


def test_plane_wave_polarisation_modes():
    pol = gx.light.PlaneWave(0.5, irradiance=2.0, polarization=gx.Polarization.linear(0.3))()
    assert pol.amplitude.shape == (1, 1, 1, 1, 1, 2)
    assert torch.allclose((pol.amplitude.abs() ** 2).sum(), torch.tensor(2.0))
    unpol = gx.light.PlaneWave(0.5, irradiance=2.0, polarization=gx.Polarization.unpolarized())()
    assert unpol.amplitude.shape == (1, 1, 2, 1, 1, 2)  # two incoherent modes on M
    assert torch.allclose((unpol.amplitude.abs() ** 2).sum(), torch.tensor(2.0))
    circ = gx.Polarization.circular(-1).modes()
    assert torch.allclose(circ.abs() ** 2, torch.full((1, 1, 2), 0.5))


def test_acquisition_layout():
    data = torch.arange(2 * 6 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 6, 1, 3, 4)
    assert format_frames(data, AcqIndex((("time", 3), ("focus", 2)))).shape == (2, 3, 2, 3, 4)
    assert format_frames(data, AcqIndex((("focus", 6),))).shape == (2, 6, 3, 4)
    assert format_frames(data[:, :1], AcqIndex()).shape == (2, 1, 3, 4)
    with pytest.raises(ValueError):
        AcqIndex((("focus", 2), ("time", 3)))


def test_grids():
    assert gx.sampling.nice_size(97) == 98 and gx.sampling.is_smooth(2 * 3 * 5 * 7)
    grid = gx.Grid2D((4, 8), 0.25)
    assert grid.extent == (2.0, 1.0)
    f = grid.freq()
    assert torch.allclose(f.fx(), torch.fft.fftfreq(8, d=0.25))
    support = gx.sampling.detection_spacing(0.13, 0.6, 1.4)
    assert support[1] == 2 and support[0] <= 0.6 / (4 * 1.4)
    assert gx.sampling.slot_bucket(0) == 8 and gx.sampling.slot_bucket(9) == 16
    assert gx.sampling.batch_bucket(40) == 64 and gx.sampling.batch_bucket(1) == 1
