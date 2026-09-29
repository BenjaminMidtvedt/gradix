# Test 9 (GPU): linear 2-beam SIM with a DENSE label density: product rho * I_ex formed on the detection grid
# (lambda_em/(4NA), the only lateral rule in the plan's SamplingPlan) vs formed on a fine grid.
import math, torch
dev = "cuda"; torch.manual_seed(0)
lam_ex, lam_em, NA = 0.488, 0.52, 1.4
h = lam_em/(4*NA); N = 256; over = 8; M = N*over; d = h/over
fc_em = 2*NA/lam_em; p = 0.9*2*NA/lam_ex
print(f"detection Nyquist {1/(2*h):.2f}/um, OTF cutoff {fc_em:.2f}/um, SIM pattern {p:.2f}/um (needed rho band {p+fc_em:.2f}/um)")
def freqs(M, d):
    f = torch.fft.fftfreq(M, d, device=dev, dtype=torch.float64); return torch.meshgrid(f, f, indexing="ij")
def otf(FY, FX):
    s = torch.clamp(torch.sqrt(FX**2+FY**2)/fc_em, max=1); return (2/math.pi)*(torch.acos(s)-s*torch.sqrt(1-s*s))
# dense sub-resolution structure: random filaments + speckle-like texture, band-limited to 12/um on the fine grid
FYf, FXf = freqs(M, d)
rho = torch.fft.ifft2(torch.fft.fft2(torch.rand(M, M, device=dev, dtype=torch.float64))*(torch.sqrt(FXf**2+FYf**2) < 12)).real
rho = torch.clamp(rho - rho.mean(), min=0)
x = (torch.arange(M, device=dev, dtype=torch.float64))*d; Y, X = torch.meshgrid(x, x, indexing="ij")
xc = (torch.arange(N, device=dev, dtype=torch.float64))*h; Yc, Xc = torch.meshgrid(xc, xc, indexing="ij")
FYc, FXc = freqs(N, h)
# best possible coarse representation of rho: ideal low-pass to the coarse Nyquist, then sample
rho_c = torch.fft.ifft2(torch.fft.fft2(rho)*((FXf.abs() < 1/(2*h)) & (FYf.abs() < 1/(2*h)))).real[::over, ::over]
errs = []; mods_ref = []; mods_c = []
for ph in [0, 2*math.pi/3, 4*math.pi/3]:
    Iex = 1 + torch.cos(2*math.pi*p*X + ph); Iexc = 1 + torch.cos(2*math.pi*p*Xc + ph)
    ref = torch.fft.ifft2(torch.fft.fft2(rho*Iex)*otf(FYf, FXf)).real[::over, ::over]
    img = torch.fft.ifft2(torch.fft.fft2(rho_c*Iexc)*otf(FYc, FXc)).real
    errs.append((torch.linalg.norm(img-ref)/torch.linalg.norm(ref)).item()); mods_ref.append(ref); mods_c.append(img)
m_ref = mods_ref[0] - mods_ref[1]; m_c = mods_c[0] - mods_c[1]
print("raw SIM frame rel-L2 error per phase:", [round(e, 3) for e in errs])
print("modulated (phase-difference) component rel-L2 error:", round((torch.linalg.norm(m_c-m_ref)/torch.linalg.norm(m_ref)).item(), 3))
# with a 2x finer product grid (rule: Nyquist >= p + fc_em) and decimation after the OTF
h2 = h/2; over2 = over//2; FY2, FX2 = freqs(2*N, h2)
rho_2 = torch.fft.ifft2(torch.fft.fft2(rho)*((FXf.abs() < 1/(2*h2)) & (FYf.abs() < 1/(2*h2)))).real[::over2, ::over2]
x2 = torch.arange(2*N, device=dev, dtype=torch.float64)*h2; Y2, X2 = torch.meshgrid(x2, x2, indexing="ij")
e2 = []
for ph in [0, 2*math.pi/3, 4*math.pi/3]:
    Iex = 1 + torch.cos(2*math.pi*p*X + ph); ref = torch.fft.ifft2(torch.fft.fft2(rho*Iex)*otf(FYf, FXf)).real[::over, ::over]
    img2 = torch.fft.ifft2(torch.fft.fft2(rho_2*(1+torch.cos(2*math.pi*p*X2+ph)))*otf(FY2, FX2)).real[::2, ::2]
    e2.append((torch.linalg.norm(img2-ref)/torch.linalg.norm(ref)).item())
print("with product grid Nyquist >= p + 2NA/lam_em: rel-L2 per phase:", [f"{e:.1e}" for e in e2])
