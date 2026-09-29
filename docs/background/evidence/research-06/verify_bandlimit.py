"""In-band spectral error of different voxelization filters for a sphere.

Reference: analytic FF(k). The DFT of the voxel grid (times h^3) should match FF(k) inside the
optically relevant band |k| < f * k_nyq. Report max relative error in band (normalised by FF(0))
and min voxel value (ringing/negativity) and max overshoot.
"""
import math

import torch

torch.set_default_dtype(torch.float64)
N, h = 128, 0.05
R = 0.3
x0 = 0.0123  # sub-voxel offset to avoid symmetric luck


def sphere_ff(k, R):
    kR = k * R
    small = kR < 1e-4
    kRs = torch.where(small, torch.ones_like(kR), kR)
    val = 4 * math.pi * R**3 * (torch.sin(kRs) - kRs * torch.cos(kRs)) / kRs**3
    return torch.where(small, 4 / 3 * math.pi * R**3 * torch.ones_like(kR), val)


def blurred_ball(r, R, sigma):
    s2 = math.sqrt(2.0) * sigma
    r = torch.clamp(r, min=1e-9)
    a = 0.5 * (torch.erf((R - r) / s2) + torch.erf((R + r) / s2))
    b = sigma / (r * math.sqrt(2 * math.pi)) * (
        torch.exp(-((R - r) ** 2) / (2 * sigma**2)) - torch.exp(-((R + r) ** 2) / (2 * sigma**2))
    )
    return a - b


ax = (torch.arange(N) - N // 2) * h
X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
Rr = torch.sqrt((X - x0) ** 2 + Y**2 + Z**2)
k1 = 2 * math.pi * torch.fft.fftfreq(N, d=h)
KX, KY, KZ = torch.meshgrid(k1, k1, k1, indexing="ij")
K = torch.sqrt(KX**2 + KY**2 + KZ**2)
knyq = math.pi / h
ref = sphere_ff(K, R) * torch.exp(-1j * KX * x0)
# grid centred at index N//2 -> account for that shift in DFT
shift = torch.exp(-1j * (KX + KY + KZ) * 0)  # we ifftshift before FFT so origin at index 0


def spectrum(vol):
    return torch.fft.fftn(torch.fft.ifftshift(vol)) * h**3


vols = {}
vols["point-sampled hard"] = (Rr <= R).double()
ss = 8
fa = (torch.arange(N * ss) + 0.5) * (h / ss) - (N // 2) * h - h / 2
FX, FY, FZ = torch.meshgrid(fa, fa, fa, indexing="ij")
vols["box (8x supersample)"] = torch.nn.functional.avg_pool3d(
    (((FX - x0) ** 2 + FY**2 + FZ**2) <= R**2).double()[None, None], ss
)[0, 0]
del FX, FY, FZ
vols["sdf ramp clamp(.5-d/h)"] = torch.clamp(0.5 - (Rr - R) / h, 0, 1)
vols["gauss blur 0.3h"] = blurred_ball(Rr, R, 0.3 * h)
vols["gauss blur 0.5h"] = blurred_ball(Rr, R, 0.5 * h)
# ideal: FF truncated to Nyquist (sinc band-limit) -> ifft
vols["k-space ideal (FF, no window)"] = torch.fft.fftshift(torch.fft.ifftn(ref).real) / h**3
# k-space with raised-cosine rolloff between 0.6 and 1.0 knyq (per-axis max norm)
kn = torch.maximum(torch.maximum(KX.abs(), KY.abs()), KZ.abs()) / knyq
W = torch.where(kn < 0.6, 1.0, torch.where(kn > 1.0, 0.0, 0.5 * (1 + torch.cos(math.pi * (kn - 0.6) / 0.4))))
vols["k-space FF*raised-cos(0.6-1.0)"] = torch.fft.fftshift(torch.fft.ifftn(ref * W).real) / h**3

ff0 = 4 / 3 * math.pi * R**3
for f in (0.3, 0.56):
    band = K < f * knyq
    print(f"--- band |k| < {f} k_nyq")
    for name, v in vols.items():
        S = spectrum(v)
        err = (S - ref).abs()[band].max().item() / ff0
        print(f"  {name:32s} max in-band err {err*100:7.3f}%   min {v.min().item():+.3f}  max {v.max().item():.3f}")
