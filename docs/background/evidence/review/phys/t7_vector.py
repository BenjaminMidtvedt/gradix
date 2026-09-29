# Test 7 (GPU): scalar vs vectorial PSF of an isotropic (freely rotating) dipole, index-matched, aplanatic collection.
import math, torch
dev = "cuda"
def psfs(NAoverN, n=1.518, lam=0.6, N=256, pad=4, dz=0.0):
    NA = NAoverN*n; u = torch.linspace(-NA, NA, N, device=dev, dtype=torch.float64)
    UY, UX = torch.meshgrid(u, u, indexing="ij"); ur = torch.sqrt(UX**2+UY**2); inside = (ur <= NA)
    st = (ur/n).clamp(max=1); ct = torch.sqrt(1-st**2); phi = torch.atan2(UY, UX)
    apo = torch.where(inside, ct**-0.5, 0)*torch.exp(1j*2*math.pi/lam*n*ct*dz)
    cp, sp = torch.cos(phi), torch.sin(phi)
    # far field of dipole mu: E_theta = mu.e_theta, E_phi = mu.e_phi ; after aplanatic lens: radial <- E_theta
    eth = torch.stack([ct*cp, ct*sp, -st]); eph = torch.stack([-sp, cp, torch.zeros_like(sp)])
    def img(p):
        P = torch.zeros(N*pad, N*pad, dtype=torch.complex128, device=dev); P[:N, :N] = p
        I = torch.fft.fftshift(torch.fft.ifft2(P).abs()**2); c = N*pad//2; return I[c-64:c+64, c-64:c+64]
    Iv = 0
    for m in range(3):
        Eth, Eph = eth[m], eph[m]
        Ex = (Eth*cp - Eph*sp)*apo; Ey = (Eth*sp + Eph*cp)*apo
        Iv = Iv + img(Ex) + img(Ey)
    Is = img(apo)
    Iv, Is = Iv/Iv.sum(), Is/Is.sum()
    return (torch.linalg.norm(Iv-Is)/torch.linalg.norm(Iv)).item(), Is.max().item()/Iv.max().item()
for r in [0.5, 0.6, 0.7, 0.8, 0.92]:
    e0, p0 = psfs(r); e1, p1 = psfs(r, dz=0.5)
    print(f"NA/n={r:.2f}: in focus rel-L2 {e0:.3f} (peak scalar/vector {p0:.3f}); dz=0.5um rel-L2 {e1:.3f}")
print("low-NA sanity:")
for r in [0.05, 0.1, 0.2, 0.3, 0.4]:
    e0, p0 = psfs(r, N=128, pad=16)
    print(f"  NA/n={r:.2f}: rel-L2 {e0:.4f}, peak ratio {p0:.4f}, expected ~ (NA/n)^2/4 scale: {r*r/4:.4f}")
