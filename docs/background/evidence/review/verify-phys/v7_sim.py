# Verify PHYS-7: linear SIM with a dense label density; product rho*I_exc formed on the detection grid vs a fine grid.
import math, torch
dev="cuda"; torch.manual_seed(0)
lex, lem, NA = 0.488, 0.520, 1.4
fmax_ex = 2*NA/lex; p = 0.9*fmax_ex            # pattern frequency (cycles/um)
d_det = lem/(4*NA)                              # plan's detection spacing
s = 4; d_f = d_det/s                            # fine grid
Nd = 256; Nf = Nd*s
def otf(N, d):
    f = torch.fft.fftfreq(N, d, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
    fr = torch.sqrt(FX**2+FY**2)/(2*NA/lem); x = fr.clamp(max=1)
    return torch.where(fr < 1, (2/math.pi)*(torch.acos(x) - x*torch.sqrt(1-x*x)), 0.0), FX, FY
# dense label density: white noise band-limited to 0.9 of the FINE grid Nyquist (sub-resolution structure)
f = torch.fft.fftfreq(Nf, d_f, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
lp = (torch.sqrt(FX**2+FY**2) < 0.9/(2*d_f)).double()
rho_f = torch.fft.ifft2(torch.fft.fft2(torch.rand(Nf, Nf, device=dev, dtype=torch.float64))*lp).real
rho_f = rho_f - rho_f.min()
# band-limited version on the detection grid (ideal decimation after anti-alias filter)
lpd = (FX.abs() < 1/(2*d_det)) & (FY.abs() < 1/(2*d_det))
rho_bl = torch.fft.ifft2(torch.fft.fft2(rho_f)*lpd).real
rho_d = rho_bl[::s, ::s]
Hf, _, _ = otf(Nf, d_f); Hd, _, _ = otf(Nd, d_det)
xf = (torch.arange(Nf, device=dev, dtype=torch.float64))*d_f; Xf = xf[None, :].expand(Nf, Nf)
xd = (torch.arange(Nd, device=dev, dtype=torch.float64))*d_det; Xd = xd[None, :].expand(Nd, Nd)
# choose p commensurate with both periodic grids to avoid a seam
p = round(p*Nd*d_det)/(Nd*d_det)
frames_ref, frames_det = [], []
for ph in (0, 2*math.pi/3, 4*math.pi/3):
    If = 1 + torch.cos(2*math.pi*p*Xf + ph); Id = 1 + torch.cos(2*math.pi*p*Xd + ph)
    ref = torch.fft.ifft2(torch.fft.fft2(rho_f*If)*Hf).real[::s, ::s]
    det = torch.fft.ifft2(torch.fft.fft2(rho_d*Id)*Hd).real
    frames_ref.append(ref); frames_det.append(det)
rl2 = lambda a, b: (torch.linalg.norm(a-b)/torch.linalg.norm(b)).item()
print(f"p = {p:.3f}/um, detection Nyquist {1/(2*d_det):.3f}/um, needed rho band p+2NA/lem = {p+2*NA/lem:.3f}/um")
print("per-frame rel-L2 (product on detection grid):", [round(rl2(a, b), 3) for a, b in zip(frames_det, frames_ref)])
print("modulated component (frame0-frame1) rel-L2:", round(rl2(frames_det[0]-frames_det[1], frames_ref[0]-frames_ref[1]), 3))
# fix: product on a grid with Nyquist >= p + 2NA/lem (s2 = 2 here), then OTF, then decimate
s2 = 2; rho2 = rho_f[::s//s2, ::s//s2]  # rho_f is band-limited to 0.9*fine Nyquist -> resample with anti-alias
lp2 = (FX.abs() < 1/(2*d_det/s2)) & (FY.abs() < 1/(2*d_det/s2))
rho2 = torch.fft.ifft2(torch.fft.fft2(rho_f)*lp2).real[::s//s2, ::s//s2]
H2, _, _ = otf(Nd*s2, d_det/s2); x2 = torch.arange(Nd*s2, device=dev, dtype=torch.float64)*d_det/s2; X2 = x2[None, :].expand(Nd*s2, Nd*s2)
fix = [torch.fft.ifft2(torch.fft.fft2(rho2*(1+torch.cos(2*math.pi*p*X2+ph)))*H2).real[::s2, ::s2] for ph in (0, 2*math.pi/3, 4*math.pi/3)]
print("fixed grid Nyquist", 1/(2*d_det/s2), "per-frame:", [round(rl2(a, b), 4) for a, b in zip(fix, frames_ref)],
      "modulated:", round(rl2(fix[0]-fix[1], frames_ref[0]-frames_ref[1]), 4))
