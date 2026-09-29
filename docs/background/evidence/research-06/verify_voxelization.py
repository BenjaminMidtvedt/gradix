"""Quick numerical checks for the sample-representation research brief.

1. Closed-form Gaussian-blurred ball vs brute-force convolution.
2. Analytic sphere form factor vs FFT of a finely supersampled voxelization.
3. Gradients wrt radius / position under different voxelization schemes.
4. Timing of patch-based voxelization of many spheres.
"""
import math
import time

import torch

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev)


def blurred_ball(r, R, sigma):
    """Indicator of ball radius R convolved with isotropic 3D Gaussian sigma, at distance r."""
    s2 = math.sqrt(2.0) * sigma
    r = torch.clamp(r, min=1e-9)
    a = 0.5 * (torch.erf((R - r) / s2) + torch.erf((R + r) / s2))
    b = sigma / (r * math.sqrt(2 * math.pi)) * (
        torch.exp(-((R - r) ** 2) / (2 * sigma**2)) - torch.exp(-((R + r) ** 2) / (2 * sigma**2))
    )
    return a - b


def sphere_ff(k, R):
    """FT of ball indicator: 4 pi (sin kR - kR cos kR)/k^3, limit 4/3 pi R^3 at k=0."""
    kR = k * R
    small = kR < 1e-4
    kRs = torch.where(small, torch.ones_like(kR), kR)
    val = 4 * math.pi * R**3 * (torch.sin(kRs) - kRs * torch.cos(kRs)) / kRs**3
    return torch.where(small, 4 / 3 * math.pi * R**3 * torch.ones_like(kR), val)


# ---- 1. blurred ball closed form vs brute force ---------------------------------------
N, h = 96, 0.05  # grid of 96^3 at 50 nm = 4.8 um box
R, sigma = 0.6, 0.08
ax = (torch.arange(N) - N / 2) * h
X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
# fine supersampled indicator (8x) averaged -> near-exact box-filtered occupancy
ss = 8
fine_ax = (torch.arange(N * ss) - N * ss / 2 + 0.5) * (h / ss) - h / 2 + h / (2)  # centers
# simpler: build indicator at fine grid then avg-pool
fa = (torch.arange(N * ss) + 0.5) * (h / ss) - N / 2 * h - h / 2
FX, FY, FZ = torch.meshgrid(fa, fa, fa, indexing="ij")
ind_fine = ((FX**2 + FY**2 + FZ**2) <= R**2).double()
box = torch.nn.functional.avg_pool3d(ind_fine[None, None], ss)[0, 0]
del ind_fine, FX, FY, FZ
# Gaussian blur via FFT
k1 = 2 * math.pi * torch.fft.fftfreq(N, d=h)
KX, KY, KZ = torch.meshgrid(k1, k1, k1, indexing="ij")
K = torch.sqrt(KX**2 + KY**2 + KZ**2)
# convolve box-filtered indicator (approx) with gaussian -> compare with closed form (which convolves exact indicator)
G = torch.exp(-0.5 * (K * sigma) ** 2)
# use analytic FT of indicator for a reference that has no voxel filter: sample FF on grid, multiply gaussian, ifft
phase = torch.ones_like(K)  # centered at origin -> index 0 at grid center requires shift
ref = torch.fft.ifft(torch.fft.ifft(torch.fft.ifft(sphere_ff(K, R) * G, dim=0), dim=1), dim=2).real / h**3
ref = torch.fft.fftshift(ref)
closed = blurred_ball(torch.sqrt(X**2 + Y**2 + Z**2), R, sigma)
print("1) max |closed-form blurred ball - FFT(FF*gauss)| =", (closed - ref).abs().max().item())

# ---- 2. sphere FF vs FFT of box-filtered voxelization ------------------------------------
Fvox = torch.fft.fftn(torch.fft.ifftshift(box)) * h**3
ff = sphere_ff(K, R)
mask = K < (math.pi / h) * 0.5  # compare below half Nyquist
rel = ((Fvox.real - ff).abs()[mask].max() / ff.abs().max()).item()
print("2) max rel err sphere FF vs FFT(8x supersampled voxels), |k|<k_nyq/2:", rel)

# ---- 3. gradients wrt radius and position --------------------------------------------
Rt = torch.tensor(0.6, requires_grad=True)
x0 = torch.tensor(0.013, requires_grad=True)  # sub-voxel offset


def mass_and_centroid(occ):
    m = occ.sum() * h**3
    cx = (occ * X).sum() * h**3 / m
    return m, cx


schemes = {}
# hard threshold
schemes["hard"] = lambda: (((X - x0) ** 2 + Y**2 + Z**2) <= Rt**2).double()
# linear SDF ramp: clamp(0.5 - sdf/h)
schemes["sdf_ramp"] = lambda: torch.clamp(
    0.5 - (torch.sqrt((X - x0) ** 2 + Y**2 + Z**2) - Rt) / h, 0, 1
)
# sigmoid SDF
schemes["sdf_sigmoid(tau=h/4)"] = lambda: torch.sigmoid(
    -(torch.sqrt((X - x0) ** 2 + Y**2 + Z**2) - Rt) / (h / 4)
)
# blurred ball sigma=0.5h
schemes["gauss_blur(0.5h)"] = lambda: blurred_ball(
    torch.sqrt((X - x0) ** 2 + Y**2 + Z**2), Rt, 0.5 * h
)


# k-space: FF * phase ramp * gaussian(0.5h) -> ifft
def kspace():
    F = sphere_ff(K, Rt) * torch.exp(-1j * KX * x0) * torch.exp(-0.5 * (K * 0.5 * h) ** 2)
    return torch.fft.fftshift(torch.fft.ifftn(F).real) / h**3


schemes["kspace_FF(0.5h)"] = kspace

true_dm_dR = 4 * math.pi * 0.6**2
print(f"3) true dV/dR = {true_dm_dR:.4f}, true dcx/dx0 = 1")
for name, fn in schemes.items():
    occ = fn()
    m, cx = mass_and_centroid(occ)
    if not m.requires_grad:
        gR = gx = None
    else:
        gR = torch.autograd.grad(m, Rt, retain_graph=True, allow_unused=True)[0]
        gx = torch.autograd.grad(cx, x0, allow_unused=True)[0]
    gR = 0.0 if gR is None else gR.item()
    gx = 0.0 if gx is None else gx.item()
    print(
        f"   {name:22s} V={m.item():.5f} (true {4/3*math.pi*0.6**3:.5f})  dV/dR={gR:.4f}  dcx/dx0={gx:.4f}"
    )

# ---- 4. timing: patch-based voxelization of many spheres ------------------------------
torch.set_default_dtype(torch.float32)
for Nobj in (100, 1000, 10000):
    Ng = (256, 256, 64)
    hh = 0.05
    centers = torch.rand(Nobj, 3, device=dev) * torch.tensor(
        [Ng[0] * hh, Ng[1] * hh, Ng[2] * hh], device=dev
    )
    radii = (0.2 + 0.3 * torch.rand(Nobj, device=dev)).requires_grad_(True)
    centers.requires_grad_(True)
    P = 24  # patch side in voxels (covers R<=0.5um + blur margin at 50nm)
    off = torch.arange(P, device=dev) - P // 2
    OX, OY, OZ = torch.meshgrid(off, off, off, indexing="ij")
    offs = torch.stack([OX, OY, OZ], -1).reshape(-1, 3)  # [P^3,3]

    def voxelize():
        base = torch.round(centers.detach() / hh).long()  # [N,3] integer anchor
        idx = base[:, None, :] + offs[None]  # [N,P^3,3]
        pos = idx.float() * hh
        d = torch.linalg.norm(pos - centers[:, None, :], dim=-1)
        occ = blurred_ball_f(d, radii[:, None], 0.5 * hh)
        valid = (
            (idx[..., 0] >= 0) & (idx[..., 0] < Ng[0]) & (idx[..., 1] >= 0) & (idx[..., 1] < Ng[1])
            & (idx[..., 2] >= 0) & (idx[..., 2] < Ng[2])
        )
        lin = (idx[..., 0] * Ng[1] + idx[..., 1]) * Ng[2] + idx[..., 2]
        vol = torch.zeros(Ng[0] * Ng[1] * Ng[2], device=dev)
        vol = vol.index_add(0, lin[valid], (occ * 0.05)[valid])
        return vol.view(Ng)

    def blurred_ball_f(r, R, sigma):
        s2 = math.sqrt(2.0) * sigma
        r = torch.clamp(r, min=1e-6)
        a = 0.5 * (torch.erf((R - r) / s2) + torch.erf((R + r) / s2))
        b = sigma / (r * math.sqrt(2 * math.pi)) * (
            torch.exp(-((R - r) ** 2) / (2 * sigma**2)) - torch.exp(-((R + r) ** 2) / (2 * sigma**2))
        )
        return a - b

    for _ in range(2):
        v = voxelize()
        loss = (v**2).sum()
        loss.backward()
    if dev == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    reps = 5
    for _ in range(reps):
        v = voxelize()
        loss = (v**2).sum()
        loss.backward()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
    print(f"4) {Nobj:6d} spheres, patch {P}^3, grid {Ng}: fwd+bwd {dt*1e3:.1f} ms, peak {peak:.0f} MB")
