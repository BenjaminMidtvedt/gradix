import math, torch
dev="cuda"
def pupil_psf(lam, NA, n, d, M, dz):
    f = torch.fft.fftfreq(M, d, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
    u = lam*torch.sqrt(FX**2+FY**2); kz = 2*math.pi/lam*torch.sqrt(torch.clamp(n*n-u*u, min=1e-12))
    P = torch.where(u <= NA, (n*n-u*u).clamp(min=1e-12)**-0.25, 0)*torch.exp(1j*kz*dz)
    I = torch.fft.fftshift(torch.fft.ifft2(P).abs()**2); return I/I.sum()
for NA, n in [(0.5, 1.0), (0.8, 1.0), (0.8, 1.33)]:
    lam = 0.6; d = lam/(4*NA)/8; M = 1024
    x = (torch.arange(M, device=dev, dtype=torch.float64) - M//2)*d; Y, X = torch.meshgrid(x, x, indexing="ij")
    for dz in [0.0, 0.2]:
        I = pupil_psf(lam, NA, n, d, M, dz)
        best = None
        for s in torch.linspace(0.15, 0.35, 201).tolist():
            sig = s*lam/NA; G = torch.exp(-(X**2+Y**2)/(2*sig**2)); G = G/G.sum()
            e = (torch.linalg.norm(G-I)/torch.linalg.norm(I)).item()
            if abs(s-0.21) < 5e-4: e21 = e
            if best is None or e < best[0]: best = (e, s)
        print(f"NA={NA} n={n} dz={dz}: rel-L2 Gaussian(0.21 lam/NA) {e21:.3f}; best sigma {best[1]:.3f} lam/NA gives {best[0]:.3f}")
print("pixel-integrated at lambda/(4NA) (8x8 fine samples per pixel), emitter at a pixel centre and at a corner:")
for NA, n in [(0.5, 1.0), (0.8, 1.33)]:
    lam = 0.6; d = lam/(4*NA)/8; M = 1024
    for dz in [0.0, 0.2]:
        I = pupil_psf(lam, NA, n, d, M, dz)
        x = (torch.arange(M, device=dev, dtype=torch.float64) - M//2)*d; Y, X = torch.meshgrid(x, x, indexing="ij")
        for sh in [4, 0]:
            Is = torch.roll(I, (sh, sh), (0, 1)).reshape(128, 8, 128, 8).sum((1, 3))
            res = []
            for s in [0.21, 0.22, 0.23]:
                sig = s*lam/NA; G = torch.exp(-(X**2+Y**2)/(2*sig**2)); G = torch.roll(G/G.sum(), (sh, sh), (0, 1)).reshape(128, 8, 128, 8).sum((1, 3))
                res.append(f"sigma {s}: {(torch.linalg.norm(G-Is)/torch.linalg.norm(Is)).item():.3f}")
            print(f"  NA={NA} n={n} dz={dz} {'centre' if sh==4 else 'corner'}: " + ", ".join(res))
