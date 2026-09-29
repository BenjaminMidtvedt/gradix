# Test 5 (GPU): "Camera integration in v1 is exact s x s sum-pooling" vs true pixel (box) integration.
import math, torch
dev="cuda"
def fine_psf(lam, NA, n, pitch, over, N, off=(0.0,0.0), dz=0.0):
    d = pitch/over; M = N*over
    f = torch.fft.fftfreq(M, d, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
    u = lam*torch.sqrt(FX**2+FY**2); inside = u <= NA
    kz = 2*math.pi/lam*torch.sqrt(torch.clamp(n*n-u*u, min=1e-12))
    # emitter at x = off + (N/2) pitch ; fine samples at (j+0.5) d  -> shift PSF by -(d/2) so sample j is at (j+0.5)d
    x0 = off[0] + (N//2)*pitch - d/2; y0 = off[1] + (N//2)*pitch - d/2
    pupil = torch.where(inside, (n*n-u*u).clamp(min=1e-12)**-0.25, 0)*torch.exp(1j*kz*dz)*torch.exp(-2j*math.pi*(FX*x0+FY*y0))
    return torch.fft.ifft2(pupil).abs()**2, M
def run(lam, NA, n, pitch, N=48, over=15, off=(0.0,0.0), dz=0.0):
    I, M = fine_psf(lam, NA, n, pitch, over, N, off, dz)
    box = I.reshape(N, over, N, over).mean(dim=(1, 3)); ref = box/box.sum()
    res = {}
    for s in (1, 3, 5):
        idx = [int(round((i+0.5)*over/s - 0.5)) for i in range(s)]
        acc = sum(I[iy::over, ix::over] for iy in idx for ix in idx)/(s*s); v = acc/acc.sum()
        res[s] = ((torch.linalg.norm(v-ref)/torch.linalg.norm(ref)).item(), v.max().item()/ref.max().item())
    return res
for (lam, NA, n, pitch, name) in [(0.68, 1.45, 1.518, 0.110, "SMLM ex.(b) 11um/100x NA1.45 680nm"),
                                  (0.52, 1.4, 1.518, 0.52/(4*1.4), "pitch exactly lambda/(4NA)")]:
    print(f"{name}: lambda/4NA = {lam/(4*NA)*1e3:.0f} nm, pitch {pitch*1e3:.0f} nm -> plan picks s=1")
    for off in [(0.0, 0.0), (0.5*pitch, 0.3*pitch)]:
        for dz in [0.0, 0.5]:
            r = run(lam, NA, n, pitch, off=off, dz=dz)
            print(f"  offset ({off[0]/pitch:.1f},{off[1]/pitch:.1f}) px dz={dz}: " +
                  " | ".join(f"s={s}: relL2 {e:.4f} peak x{p:.3f}" for s, (e, p) in r.items()))
