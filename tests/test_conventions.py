"""Conventions (§4.1, §11.1): Fourier signs, propagation, frame, Zernike, phases, matmuls."""

import math

import pytest
import torch

import gradix as gx
from gradix import conventions as cv
from gradix._core.precision import ieee_matmul, unit_phasor, wrap_phase
from gradix.ops.fourier import mft2
from gradix.special.zernike import zernike, zernike_ansi


def test_fourier_shift_matches_roll():
    x = torch.randn(16, 20, dtype=torch.float64)
    shifted = cv.fourier_shift(x, (3.0, -2.0))  # (dx along columns, dy along rows)
    rolled = torch.roll(x, shifts=(-2, 3), dims=(-2, -1))
    assert torch.allclose(shifted.real, rolled, atol=1e-12)
    assert shifted.imag.abs().max() < 1e-12


def test_propagation_composes_and_decays():
    fy, fx = cv.angular_frequencies((32, 32), 0.1)
    kz = cv.kz(1.33, 0.5, fx, fy)
    assert (kz.imag >= 0).all()
    evanescent = (fx**2 + fy**2) > (1.33 / 0.5) ** 2
    assert evanescent.any() and (kz.imag[evanescent] > 0).all() and (kz.real[evanescent] == 0).all()
    h_ab = cv.propagator(kz, 1.7) * cv.propagator(kz, 0.6)
    assert torch.allclose(h_ab, cv.propagator(kz, 2.3), atol=1e-12)
    assert (cv.propagator(kz, 1.0).abs() <= 1 + 1e-12).all()


def test_propagator_gradient_reaches_distance():
    fy, fx = cv.angular_frequencies((8, 8), 0.2)
    kz = cv.kz(1.0, 0.5, fx, fy)
    dz = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    cv.propagator(kz, dz).real.sum().backward()
    assert dz.grad is not None and torch.isfinite(dz.grad) and dz.grad != 0


def test_plane_wave_sign_and_fft_bin():
    # e^{+i k·r} on a periodic grid lands in the positive-frequency FFT bin with amplitude·N
    n, d = 32, 0.25
    grid = gx.Grid2D((n, n), d)
    f0 = 3 / (n * d)  # exactly on the grid
    wl = 0.5
    waves = gx.PlaneWaves(
        amplitude=torch.full((1, 1, 1, 1, 1, 1), 2.0, dtype=torch.complex128),
        u=torch.tensor([[[[[wl * f0, 0.0]]]]], dtype=torch.float64),
        wavelengths=torch.tensor([[wl]], dtype=torch.float64),
    )
    x, y = grid.x(dtype=torch.float64), grid.y(dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pts = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(1, 1, -1, 3)
    field = waves.at(pts, 1.0).reshape(n, n)
    direct = 2.0 * torch.exp(1j * 2 * math.pi * f0 * xx)
    assert torch.allclose(field, direct, atol=1e-12)
    spectrum = torch.fft.fft2(field, norm=cv.FFT_NORM)
    peak = spectrum.abs().flatten().argmax()
    assert int(peak) == 3  # row 0, column +3: the positive x frequency
    assert abs(abs(spectrum[0, 3]) - 2.0 * n * n) < 1e-9


def test_evanescent_plane_wave_decays_toward_the_objective():
    waves = gx.PlaneWaves(
        amplitude=torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.complex128),
        u=torch.tensor([[[[[1.5, 0.0]]]]], dtype=torch.float64),  # |u| > n = 1.33
        wavelengths=torch.tensor([[0.5]], dtype=torch.float64),
    )
    pts = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]]], dtype=torch.float64)
    mag = waves.at(pts, 1.33).abs().flatten()
    assert mag[0] > mag[1] > mag[2]


def test_defocus_and_z_helpers():
    assert cv.defocus(1.0, 0.25) == 0.75  # above the focal plane: toward the objective
    assert gx.coords.from_focus(0.5, 1.0) == 1.5
    assert gx.coords.depth(2.0) == -2.0
    obj = gx.coords.shift_focus(gx.Objective(NA=1.0, focus=0.5), 0.25)
    assert obj.focus == 0.75


def test_collection_efficiency_values():
    eta = float(cv.collection_efficiency(1.4, 1.518))
    assert abs(eta - 0.3067) < 5e-4
    assert float(cv.collection_efficiency(2.0, 1.0)) == 0.5  # clipped at the hemisphere


@pytest.mark.parametrize("j", range(15))
def test_ansi_indices_roundtrip(j):
    n, m = cv.ansi_to_nm(j)
    assert cv.nm_to_ansi(n, m) == j


def test_named_zernike_indices():
    assert cv.ansi_to_nm(4) == (2, 0)  # defocus
    assert cv.noll_to_nm(4) == (2, 0)
    assert cv.noll_to_nm(2) == (1, 1) and cv.noll_to_nm(3) == (1, -1)
    assert cv.noll_to_nm(5) == (2, -2) and cv.noll_to_nm(6) == (2, 2)
    assert cv.noll_to_ansi(11) == 12  # primary spherical


def test_zernike_orthonormal_rms():
    nr, nphi = 400, 256
    rho = (torch.arange(nr, dtype=torch.float64) + 0.5) / nr
    phi = torch.arange(nphi, dtype=torch.float64) * (2 * math.pi / nphi)
    r, p = torch.meshgrid(rho, phi, indexing="ij")
    weight = r * (1.0 / nr) * (2 * math.pi / nphi) / math.pi  # area element over the disc area
    zs = torch.stack([zernike_ansi(j, r, p) for j in range(15)])
    gram = torch.einsum("irp,jrp,rp->ij", zs, zs, weight)
    assert torch.allclose(gram, torch.eye(15, dtype=torch.float64), atol=2e-3)
    assert torch.allclose(
        zernike(2, 0, torch.tensor(1.0), torch.tensor(0.0)), torch.tensor(math.sqrt(3.0))
    )


def test_fp64_carrier_phase():
    # 2e4 waves of phase: float32 errs by ~1e-3; fp64 with the carrier removed by ~1e-7
    k = 2 * math.pi / 0.5
    z = torch.linspace(0, 1e4, 1001, dtype=torch.float64)
    exact = torch.polar(torch.ones_like(z), torch.remainder(k * z, 2 * math.pi))
    naive = torch.exp(1j * (k * z.float())).to(torch.complex128)
    careful = unit_phasor(k * z, dtype=torch.complex64).to(torch.complex128)
    assert (naive - exact).abs().max() > 1e-4
    assert (careful - exact).abs().max() < 1e-6
    assert wrap_phase(torch.tensor(3 * math.pi, dtype=torch.float64)).abs() == pytest.approx(
        math.pi
    )


def test_mft_equals_fft():
    x = torch.randn(12, 10, dtype=torch.complex128)
    coords = (torch.arange(12.0, dtype=torch.float64), torch.arange(10.0, dtype=torch.float64))
    freq = (torch.fft.fftfreq(12, dtype=torch.float64), torch.fft.fftfreq(10, dtype=torch.float64))
    assert torch.allclose(mft2(x, coords, freq), torch.fft.fft2(x), atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 needs a CUDA GPU")
def test_mft_runs_at_ieee_precision_even_when_tf32_is_on():
    matmul = torch.backends.cuda.matmul
    saved = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        ref = a.double() @ b.double()
        assert ((a @ b).double() - ref).abs().max() > 1e-3  # TF32 is really on
        x = torch.randn(64, 64, dtype=torch.complex64, device="cuda")
        coords = (torch.arange(64.0, device="cuda"), torch.arange(64.0, device="cuda"))
        freq = (
            torch.linspace(-0.3, 0.3, 50, device="cuda"),
            torch.linspace(-0.3, 0.3, 50, device="cuda"),
        )
        got = mft2(x, coords, freq)
        ref = mft2(
            x.to(torch.complex128),
            (coords[0].double(), coords[1].double()),
            (freq[0].double(), freq[1].double()),
        )
        rel = (got.to(torch.complex128) - ref).abs().max() / ref.abs().max()
        assert rel < 1e-5
        assert matmul.fp32_precision == "tf32"  # restored after the call
    finally:
        matmul.fp32_precision = saved


def test_ieee_matmul_is_reentrant():
    matmul = torch.backends.cuda.matmul
    before = matmul.fp32_precision
    with ieee_matmul():
        with ieee_matmul():
            assert matmul.fp32_precision == "ieee"
        assert matmul.fp32_precision == "ieee"
    assert matmul.fp32_precision == before


def test_pixel_centres():
    grid = gx.Grid2D((3, 4), 0.5, origin=(1.0, 2.0))
    assert torch.allclose(grid.x(), torch.tensor([1.25, 1.75, 2.25, 2.75]))
    assert torch.allclose(grid.y(), torch.tensor([2.25, 2.75, 3.25]))
    assert torch.allclose(cv.pixel_centers(3, 2.0), torch.tensor([1.0, 3.0, 5.0]))


def test_direction():
    u = cv.direction(math.pi / 6, math.pi / 2, n=1.5)
    assert torch.allclose(u, torch.tensor([0.0, 0.75], dtype=torch.float64), atol=1e-12)


def test_kz_gradient_is_finite_at_the_cutoff():
    n = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    f = torch.tensor([[1.0 / 0.5]], dtype=torch.float64)  # |f| = n/λ exactly
    k = gx.conventions.kz(n, 0.5, f, torch.zeros_like(f))
    (g,) = torch.autograd.grad(k.abs().sum(), n)
    assert torch.isfinite(g).all()


def test_angular_spectrum_factors_match_direct_summation():
    conv = gx.conventions
    d, size, w = 0.1, 64, 0.5
    x = (torch.arange(size, dtype=torch.float64) - size // 2) * d
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    field = torch.exp(-r2 / w**2).to(torch.complex128)
    # centre the Gaussian on sample 0 so the spectrum is real: A(k) = w²/(4π)·exp(−k²w²/4)
    spectrum = conv.to_angular_spectrum(torch.fft.ifftshift(field), d)
    fy, fx = conv.angular_frequencies((size, size), d)
    k2 = (2 * math.pi) ** 2 * (fx**2 + fy**2)
    analytic = w**2 / (4 * math.pi) * torch.exp(-k2 * w**2 / 4)
    assert torch.allclose(spectrum.real, analytic, atol=1e-12, rtol=1e-9)
    # direct summation Σ A(k) exp(i k·r) Δk² at a few points
    dk2 = conv.delta_k2((size, size), d)
    kx, ky = 2 * math.pi * fx, 2 * math.pi * fy
    for px, py in [(0.0, 0.0), (0.3, -0.2), (0.75, 0.4)]:
        direct = (spectrum * torch.exp(1j * (kx * px + ky * py))).sum() * dk2
        assert abs(complex(direct) - math.exp(-(px**2 + py**2) / w**2)) < 1e-9
    back = conv.from_angular_spectrum(spectrum, d)
    assert torch.allclose(back, torch.fft.ifftshift(field), atol=1e-12)


def test_zernike_radials_are_accurate_at_high_order_and_validate_indices():
    from fractions import Fraction

    from gradix.special.zernike import radial

    def exact(n, m, rho):
        rho = Fraction(rho)
        total = sum(
            (-1) ** k
            * math.comb(n - k, k)
            * math.comb(n - 2 * k, (n - m) // 2 - k)
            * rho ** (n - 2 * k)
            for k in range((n - m) // 2 + 1)
        )
        return float(total)

    rho = torch.tensor([0.0, 0.37, 0.8, 0.999, 1.0], dtype=torch.float32)
    for n, m in [(20, 0), (30, 4), (40, 2), (50, 10)]:
        got = radial(n, m, rho)
        want = torch.tensor([exact(n, m, float(r)) for r in rho], dtype=torch.float32)
        assert got.dtype == torch.float32 and torch.allclose(got, want, atol=1e-6)
    for bad in [(3, 0), (2, 4), (-1, 1)]:
        with pytest.raises(ValueError, match="Zernike index"):
            radial(*bad, torch.zeros(1))
