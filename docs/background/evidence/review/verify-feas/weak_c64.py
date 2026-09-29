"""FEAS-5 caveat: complex64 precision of the residual-demodulated total-field march (scattered part recovered as
v - carrier) vs the plan's scattered-field march, for dense objects of decreasing phase contrast.
Reference: same method in complex128 on the same grid. Error is relative to the scattered-field norm."""
import math, torch
dev = "cuda"; lam, nm, S, N = 0.532, 1.333, 16, 256
fmax = nm/lam; dx = 1/(2*1.05*fmax); dz = 0.2
f = torch.fft.fftfreq(N, dx, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
xs = (torch.arange(N, device=dev, dtype=torch.float64) - N//2)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
g = torch.Generator(device=dev).manual_seed(3)
base = torch.randn(S, N, N, device=dev, dtype=torch.float64, generator=g)
tex = torch.fft.ifft2(torch.fft.fft2(base)*torch.exp(-(FX**2+FY**2)/(2*1.5**2))).real
cell = torch.sigmoid((4.0 - torch.sqrt(X**2+Y**2))/0.15)
shape = (tex/tex.abs().max()*0.5 + 0.5)*cell                       # 0..1 texture inside a disc
fbx = 0.38/lam*0.9973; L = N*dx; fg = round(fbx*L)/L; dk = fbx - fg
kzb = 2*math.pi*math.sqrt((nm/lam)**2 - fbx**2)
def H_at(fx, fy):
    a = (nm/lam)**2 - fx**2 - fy**2; kz = 2*math.pi*torch.sqrt(a.clamp(min=0))
    return torch.where(a > 0, torch.polar(torch.ones_like(kz), (kz-kzb)*dz), 0*kz)
def run(ph, cd, mode):
    t = torch.polar(torch.ones_like(ph), ph).to(cd)
    if mode == "plan":
        lat = torch.polar(torch.ones_like(X), 2*math.pi*fbx*X).to(cd); H = H_at(FX, FY).to(cd)
        Es = torch.zeros_like(lat)
        for s in range(S): Es = torch.fft.ifft2(torch.fft.fft2((t[s]-1)*lat + t[s]*Es)*H)
        return Es                                                   # scattered part (lab frame)
    carrier = torch.polar(torch.ones_like(X), 2*math.pi*fg*X).to(cd); H = H_at(FX + dk, FY).to(cd)
    v = carrier.clone()
    for s in range(S): v = torch.fft.ifft2(torch.fft.fft2(t[s]*v)*H)
    demod = torch.polar(torch.ones_like(X), 2*math.pi*dk*X).to(cd)
    return demod*(v - carrier)                                      # scattered part = total - analytic background
for phi in (0.5, 1e-2, 1e-3, 1e-4, 1e-5):
    ph = phi*shape/S                  # total phase ~ phi across the stack
    for mode in ("plan", "resid"):
        ref = run(ph, torch.complex128, mode); r32 = run(ph.float(), torch.complex64, mode).to(torch.complex128)
        print(f"phase {phi:7.0e}  {mode:5s}: c64 rel. error of scattered field {((r32-ref).abs().norm()/ref.abs().norm()).item():.1e}")
