"""Fair version: band-limited weak cell-like phase object, ground truth = scattered+analytic march on a 4x larger
grid (1024^2, same dx, so no wrap in the region of interest). Compare on the 256^2 grid:
 (1) plan: scattered-field + analytic background; (2) demodulated total-field march (H at q+k_b); (3) naive sampled tilt.
Off-grid tilt sin=0.2037 (Koehler-type mode)."""
import math, torch
torch.set_default_dtype(torch.float64)
dev = "cuda"
dx, lam, nm, dz, S = 0.1, 0.532, 1.333, 0.2, 16
def grids(N):
    f = torch.fft.fftfreq(N, dx, device=dev); FY, FX = torch.meshgrid(f, f, indexing="ij")
    xs = (torch.arange(N, device=dev) - N//2)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
    return FX, FY, X, Y
def make_obj(N):
    FX, FY, X, Y = grids(N)
    g = torch.Generator(device=dev).manual_seed(3)
    base = torch.randn(S, 64, 64, device=dev, generator=g)
    big = torch.zeros(S, N, N, device=dev); c = N//2
    big[:, c-32:c+32, c-32:c+32] = base
    F = torch.exp(-(FX**2+FY**2)/(2*0.8**2))                     # band-limit texture to ~0.8 cyc/um
    tex = torch.fft.ifft2(torch.fft.fft2(big)*F).real
    cell = torch.sigmoid((5.0 - torch.sqrt(X**2+Y**2))/0.15)       # soft 5 um cell
    ph = 2*math.pi/lam*0.03*dz*cell + 0.05*tex/tex.abs().max()*cell
    return ph
def march(N, ph, sinx, mode, cd=torch.complex128):
    FX, FY, X, Y = grids(N)
    t = torch.polar(torch.ones_like(ph), ph).to(cd)
    fbx = nm*sinx/lam; kzb = 2*math.pi*math.sqrt((nm/lam)**2 - fbx**2)
    lat = torch.polar(torch.ones_like(X), 2*math.pi*fbx*X).to(cd)
    def H(fx, fy):
        a = (nm/lam)**2 - fx**2 - fy**2
        kz = 2*math.pi*torch.sqrt(a.clamp(min=0))
        return torch.where(a > 0, torch.polar(torch.ones_like(kz), (kz-kzb)*dz), 0*kz).to(cd)
    if mode == "scat":
        Hl = H(FX, FY); Es = torch.zeros(N, N, dtype=cd, device=dev)
        for s in range(S): Es = torch.fft.ifft2(torch.fft.fft2((t[s]-1)*lat + t[s]*Es)*Hl)
        return lat + Es
    if mode == "demod":
        Hd = H(FX + fbx, FY); u = torch.ones(N, N, dtype=cd, device=dev)
        for s in range(S): u = torch.fft.ifft2(torch.fft.fft2(t[s]*u)*Hd)
        return lat*u
    Hl = H(FX, FY); v = lat.clone()
    for s in range(S): v = torch.fft.ifft2(torch.fft.fft2(t[s]*v)*Hl)
    return v
sinx = 0.2037
ph_big = make_obj(1024); c = 512
ph = ph_big[:, c-128:c+128, c-128:c+128].contiguous()
truth = march(1024, ph_big, sinx, "scat")[c-128:c+128, c-128:c+128]
FX, FY, X, Y = grids(256); inner = (X**2+Y**2) < 8.0**2
rel = lambda a: ((a-truth)[inner].abs().norm()/(truth[inner]-truth[inner].mean()).abs().norm()).item()
for mode in ("scat", "demod", "naive"):
    for cd, nmn in ((torch.complex128, "c128"), (torch.complex64, "c64")):
        r = march(256, ph, sinx, mode, cd).to(torch.complex128)
        print(f"{mode:6s} {nmn}: error in ROI rel. to scattered-field energy {rel(r):.2e}")
