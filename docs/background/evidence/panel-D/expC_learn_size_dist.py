"""Exp C: learn a particle-size *distribution* (LogNormal mu, sigma) from unlabeled images.

Forward model (fully batched, analytic, band-limited):
  sphere projection phase via kz=0 slice of ball form factor -> Fourier phase ramp for position
  -> exp(i*phi) thin-object transmission -> pupil with soft NA edge + exact defocus -> |E|^2
  -> photons -> Poisson (scaled straight-through).
"Real" data: same model with true (mu, sigma) and independent seeds; positions/z/photons random.
Loss: MMD (multi-bandwidth RBF) between sets of position-invariant features:
  32 intensity quantiles + 16 log radial power-spectrum bins.
"""
import math
import time

import torch

dev = "cuda"
N, dx, lam, n_m, NA, dn = 64, 0.1, 0.5, 1.33, 0.8, 0.04
k0 = 2 * math.pi / lam
f = torch.fft.fftfreq(N, d=dx, device=dev)
FY, FX = torch.meshgrid(f, f, indexing="ij")
FR = torch.sqrt(FX**2 + FY**2)
q = 2 * math.pi * FR
pupil = torch.sigmoid((NA / lam - FR) / (0.5 / (N * dx)))
kz = 2 * math.pi * (torch.sqrt(torch.clamp((n_m / lam) ** 2 - FR**2, min=0)) - n_m / lam)
rbin = torch.clamp((FR / FR.max() * 16).long(), max=15).flatten()


def ball_ft(qq, R):
    x = qq * R
    small = x < 1e-2
    xs = torch.where(small, torch.ones_like(x), x)
    big = 4 * math.pi * (torch.sin(xs) - xs * torch.cos(xs)) / (qq.clamp_min(1e-12) ** 3)
    ser = 4 / 3 * math.pi * R**3 * (1 - x**2 / 10)
    return torch.where(small, ser, big)


def render(R, pos, z, photons, g):
    F = ball_ft(q[None], R[:, None, None])  # projected-thickness FT [B,N,N] (um^3)
    ramp = torch.exp(-2j * math.pi * (FX * pos[:, 0, None, None] + FY * pos[:, 1, None, None]))
    phi = torch.fft.ifft2(k0 * dn * F * ramp / dx**2).real  # phase map
    E = torch.fft.ifft2(torch.fft.fft2(torch.exp(1j * phi)) * pupil * torch.exp(1j * kz * z[:, None, None]))
    I = E.real**2 + E.imag**2
    lamb = photons[:, None, None] * I
    n = torch.poisson(lamb.detach(), generator=g)
    s = lamb.clamp_min(1e-6).sqrt()
    return lamb + s * ((n - lamb) / s).detach()  # scaled straight-through Poisson


def sample_batch(mu, log_sigma, B, g):
    eps = torch.randn(B, device=dev, generator=g)
    R = torch.exp(mu + torch.exp(log_sigma) * eps)  # reparameterized LogNormal
    pos = (torch.rand(B, 2, device=dev, generator=g) - 0.5) * 2.0 + N * dx / 2
    z = (torch.rand(B, device=dev, generator=g) - 0.5) * 1.0
    photons = 500 + 500 * torch.rand(B, device=dev, generator=g)
    return render(R, pos, z, photons, g) / photons[:, None, None]


def features(img):
    B = img.shape[0]
    qs = torch.quantile(img.flatten(1), torch.linspace(0.01, 0.99, 32, device=dev), dim=1).T
    ps = torch.fft.fft2(img - img.mean((-2, -1), keepdim=True)).abs().square().flatten(1)
    rad = torch.zeros(B, 16, device=dev).index_add(1, rbin, ps) / torch.bincount(rbin, minlength=16)
    return torch.cat([qs, torch.log(rad + 1e-6)], 1)


def mmd(x, y):
    xy = torch.cat([x, y])
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-6)
    d2 = torch.cdist(xy, xy).square()
    K = sum(torch.exp(-d2 / (2 * s * s)) for s in (1.0, 2.0, 4.0, 8.0))
    n = x.shape[0]
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()


true_mu, true_sig = math.log(0.45), 0.25
g_real = torch.Generator(device=dev).manual_seed(123)
g_sim = torch.Generator(device=dev).manual_seed(7)
with torch.no_grad():
    real_pool = torch.cat([sample_batch(torch.tensor(true_mu, device=dev),
                                        torch.tensor(math.log(true_sig), device=dev), 512, g_real)
                           for _ in range(8)])  # 4096 "experimental" images
    real_feat = features(real_pool)

mu = torch.tensor(math.log(0.25), device=dev, requires_grad=True)
log_sig = torch.tensor(math.log(0.08), device=dev, requires_grad=True)
opt = torch.optim.Adam([mu, log_sig], lr=0.03)
B = 256
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(401):
    idx = torch.randint(0, real_feat.shape[0], (B,), device=dev, generator=g_real)
    loss = mmd(features(sample_batch(mu, log_sig, B, g_sim)), real_feat[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d} loss {loss.item():.4f} median r {math.exp(mu.item()):.3f} um "
              f"(true 0.450)  sigma {math.exp(log_sig.item()):.3f} (true {true_sig})")
torch.cuda.synchronize()
print(f"time/step (B=256 sim + MMD, 64^2): {(time.perf_counter() - t0) / 401 * 1e3:.1f} ms")
