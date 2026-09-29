"""Tight-grid check (accurate tier: grid Nyquist ~1.05 n_m/lambda) with an off-grid Koehler-type tilt NA_ill 0.38.
Truth: plan's scattered-field + analytic-background march on a 4x larger grid (same dx).
Compare on the small grid: (1) plan method; (2) full demodulation (H at q+k_b); (3) residual demodulation
(k_b = k_grid + dk: on-grid part carried exactly as a periodic carrier, only the sub-bin residual dk demodulated,
H at q+dk); (4) naive sampled tilt. Also: residual-demod cost per slice equals the plain march (no extra ops)."""
import math, torch
torch.set_default_dtype(torch.float64); dev = "cuda"
lam, nm, S = 0.532, 1.333, 16
fmax = nm/lam; dx = 1/(2*1.05*fmax); dz = 0.2
def grids(N):
    f = torch.fft.fftfreq(N, dx, device=dev); FY, FX = torch.meshgrid(f, f, indexing="ij")
    xs = (torch.arange(N, device=dev) - N//2)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
    return FX, FY, X, Y
def make_obj(N):
    FX, FY, X, Y = grids(N); g = torch.Generator(device=dev).manual_seed(3)
    base = torch.zeros(S, N, N, device=dev); c = N//2
    base[:, c-24:c+24, c-24:c+24] = torch.randn(S, 48, 48, device=dev, generator=g)
    tex = torch.fft.ifft2(torch.fft.fft2(base)*torch.exp(-(FX**2+FY**2)/(2*1.5**2))).real
    cell = torch.sigmoid((4.0 - torch.sqrt(X**2+Y**2))/0.15)
    return 2*math.pi/lam*0.03*dz*cell + 0.1*tex/tex.abs().max()*cell
def H_at(fx, fy, kzb):
    a = (nm/lam)**2 - fx**2 - fy**2; kz = 2*math.pi*torch.sqrt(a.clamp(min=0))
    return torch.where(a > 0, torch.polar(torch.ones_like(kz), (kz-kzb)*dz), 0*kz).to(torch.complex128)
def march(N, ph, fbx, mode):
    FX, FY, X, Y = grids(N); t = torch.polar(torch.ones_like(ph), ph).to(torch.complex128)
    kzb = 2*math.pi*math.sqrt((nm/lam)**2 - fbx**2); L = N*dx
    lat = torch.polar(torch.ones_like(X), 2*math.pi*fbx*X).to(torch.complex128)
    if mode == "plan":
        H = H_at(FX, FY, kzb); Es = torch.zeros_like(lat)
        for s in range(S): Es = torch.fft.ifft2(torch.fft.fft2((t[s]-1)*lat + t[s]*Es)*H)
        return lat + Es
    if mode == "full_demod":
        H = H_at(FX + fbx, FY, kzb); u = torch.ones_like(lat)
        for s in range(S): u = torch.fft.ifft2(torch.fft.fft2(t[s]*u)*H)
        return lat*u
    if mode == "resid_demod":
        fg = round(fbx*L)/L; dk = fbx - fg                           # on-grid part + sub-bin residual
        carrier = torch.polar(torch.ones_like(X), 2*math.pi*fg*X).to(torch.complex128)   # periodic, exact
        H = H_at(FX + dk, FY, kzb); v = carrier.clone()
        for s in range(S): v = torch.fft.ifft2(torch.fft.fft2(t[s]*v)*H)
        return torch.polar(torch.ones_like(X), 2*math.pi*dk*X).to(torch.complex128)*v
    H = H_at(FX, FY, kzb); v = lat.clone()
    for s in range(S): v = torch.fft.ifft2(torch.fft.fft2(t[s]*v)*H)
    return v
fbx = 0.38/lam*0.9973                                             # off-grid
Nb, Ns = 1024, 256; c = Nb//2
ph_big = make_obj(Nb); ph = ph_big[:, c-Ns//2:c+Ns//2, c-Ns//2:c+Ns//2].contiguous()
truth = march(Nb, ph_big, fbx, "plan")[c-Ns//2:c+Ns//2, c-Ns//2:c+Ns//2]
FX, FY, X, Y = grids(Ns); inner = (X**2+Y**2) < 6.0**2
bg = torch.polar(torch.ones_like(X), 2*math.pi*fbx*X)
den = (truth - bg)[inner].abs().norm()
for mode in ("plan", "resid_demod", "full_demod", "naive"):
    r = march(Ns, ph, fbx, mode)
    print(f"{mode:12s}: error / scattered-field norm in ROI = {((r-truth)[inner].abs().norm()/den).item():.2e}")
