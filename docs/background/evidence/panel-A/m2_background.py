"""Does carrying the unscattered (zero-order) wave analytically matter?

Case A: off-grid tilted plane wave (Koehler source point) imaged through a soft pupil.
        Sampled total field on a periodic grid vs analytic background (exact: |P(u_s)|^2 |a|^2).
Case B: weak scatterer interference (iSCAT-like contrast ~1e-5..1e-3): complex64 sampled total
        field through FFT->pupil->iFFT->|.|^2 then I/I_bg - 1, vs explicit cross-term with
        analytic background; both vs complex128 reference.
"""
import math

import torch

dev = "cuda"
N, dx, lam, NA = 256, 0.1, 0.5, 0.8  # um
L = N * dx
f = torch.fft.fftfreq(N, d=dx, device=dev, dtype=torch.float64)
FY, FX = torch.meshgrid(f, f, indexing="ij")
fr = torch.sqrt(FX**2 + FY**2)
P = torch.sigmoid((NA / lam - fr) / (0.02 * NA / lam))  # soft pupil, real
y = (torch.arange(N, device=dev, dtype=torch.float64) - N / 2) * dx
Y, X = torch.meshgrid(y, y, indexing="ij")

print("Case A: off-grid tilted plane wave through pupil (background only)")
for ks in [0.3 / lam * 0.5, 0.61, 1.2]:  # cycles/um; 1/L = 0.039
    ramp = torch.exp(2j * math.pi * ks * X)
    img = torch.fft.ifft2(torch.fft.fft2(ramp) * P).abs() ** 2
    exact = torch.sigmoid(torch.tensor((NA / lam - ks) / (0.02 * NA / lam), dtype=torch.float64)) ** 2
    err = (img - exact).abs().max().item() / exact.item()
    rms = ((img - exact) ** 2).mean().sqrt().item() / exact.item()
    snapped = round(ks * L) / L
    ang_err_mrad = abs(math.asin(min(ks * lam, 0.999)) - math.asin(min(snapped * lam, 0.999))) * 1e3
    print(
        f"  k_s={ks:.3f}/um (k_s*L={ks*L:.2f}): sampled-background image max rel err {err:.2e}, "
        f"rms {rms:.2e}; snap-to-grid angle error {ang_err_mrad:.1f} mrad"
    )

print("Case B: weak scatterer contrast, background amplitude r=0.067 (glass/water)")
r = 0.067
g = torch.exp(-((X**2 + Y**2) / (2 * 0.15**2)))  # scatterer footprint before imaging
for c in [1e-5, 1e-4, 1e-3]:
    s128 = (c * r / 2) * g * torch.exp(1j * torch.tensor(0.7, dtype=torch.float64))  # contrast ~ 2|s|/r
    # reference in complex128: explicit cross term
    Es = torch.fft.ifft2(torch.fft.fft2(s128) * P)
    Eb = r * P[0, 0]
    contrast_ref = (2 * (Eb.conj() * Es).real + Es.abs() ** 2) / abs(Eb) ** 2
    # (i) complex64 sampled total field
    tot = (r + s128).to(torch.complex64)
    img = torch.fft.ifft2(torch.fft.fft2(tot) * P.float()).abs() ** 2
    bg = (abs(r * P[0, 0].float())) ** 2
    con_i = img / bg - 1
    # (ii) complex64 explicit cross term with analytic background
    Es32 = torch.fft.ifft2(torch.fft.fft2(s128.to(torch.complex64)) * P.float())
    Eb32 = torch.tensor(r, dtype=torch.complex64, device=dev) * P[0, 0].float()
    con_ii = (2 * (Eb32.conj() * Es32).real + Es32.abs() ** 2) / Eb32.abs() ** 2
    peak = contrast_ref.abs().max().item()
    e_i = (con_i.double() - contrast_ref).abs().max().item() / peak
    e_ii = (con_ii.double() - contrast_ref).abs().max().item() / peak
    print(f"  peak contrast {peak:.1e}: sampled-total c64 rel err {e_i:.2e} | analytic-bg c64 rel err {e_ii:.2e}")
