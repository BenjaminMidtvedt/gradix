import math, torch
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
N, dx, lam = 256, 0.1, 0.55          # um
n_m = 1.335
NA, NAc = 0.4, 0.3
# thin phase object: a few soft discs with ~1 rad phase
y, x = torch.meshgrid(torch.arange(N, device=dev) * dx, torch.arange(N, device=dev) * dx, indexing="ij")
phi = torch.zeros(N, N, device=dev)
for cx, cy, r in [(8, 8, 3.0), (17, 12, 2.0), (12, 19, 2.5)]:
    d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    phi += 1.0 * torch.sigmoid((r - d) / 0.1)
t = torch.exp(1j * phi)
f = torch.fft.fftfreq(N, dx, device=dev)
fy, fx = torch.meshgrid(f, f, indexing="ij")
fr = torch.sqrt(fx ** 2 + fy ** 2)
P = (fr <= NA / lam).to(torch.complex64)
T = torch.fft.fft2(t)
df = 1.0 / (N * dx)

def image_for(src_idx):  # src_idx: list of (iy, ix) integer frequency shifts (on-grid)
    out = torch.zeros(N, N, device=dev)
    for iy, ix in src_idx:
        Ts = torch.roll(T, shifts=(iy, ix), dims=(0, 1))   # spectrum of t * exp(i2pi k_s r)
        out += torch.fft.ifft2(Ts * P).abs() ** 2
    return out / len(src_idx)

# on-grid source points inside condenser disc (exact Abbe reference)
ks = int(NAc / lam / df)
pts = [(iy, ix) for iy in range(-ks, ks + 1) for ix in range(-ks, ks + 1) if (iy * df) ** 2 + (ix * df) ** 2 <= (NAc / lam) ** 2]
I_ref = image_for(pts)
print("exact Abbe points:", len(pts))
for K in [2, 4, 8, 32]:
    errs = []
    for trial in range(20):
        sel = [pts[i] for i in torch.randint(len(pts), (K,)).tolist()]
        I_k = image_for(sel)
        errs.append(((I_k - I_ref).pow(2).mean().sqrt() / I_ref.mean()).item())
    print(f"stochastic K={K:3d}: rel RMS deviation from exact Abbe = {sum(errs)/len(errs):.3f}")
for ph in [1e3, 1e4]:
    print(f"shot-noise rel RMS at {ph:.0e} photons/px = {1/math.sqrt(ph):.3f}")

# annular source (phase contrast style): single on-axis mode vs single on-annulus mode vs full annulus
NAi, NAo = 0.25, 0.28
ann = [(iy, ix) for iy in range(-ks, ks + 1) for ix in range(-ks, ks + 1)
       if (NAi / lam) ** 2 <= (iy * df) ** 2 + (ix * df) ** 2 <= (NAo / lam) ** 2]
ring = ((fr >= 0.24 / lam) & (fr <= 0.29 / lam))
Pr = P * torch.where(ring, 0.25 * torch.exp(torch.tensor(1j * math.pi / 2, device=dev)), torch.tensor(1.0 + 0j, device=dev))
def pc_image(src):
    out = torch.zeros(N, N, device=dev)
    for iy, ix in src:
        Ts = torch.roll(T, shifts=(iy, ix), dims=(0, 1))
        out += torch.fft.ifft2(Ts * Pr).abs() ** 2
    return out / len(src)
I_pc = pc_image(ann)
I_axis = pc_image([(0, 0)])
I_one = pc_image([ann[0]])
def contrast(I):  # phase-object contrast: (inside - outside)/outside at disc 1 centre
    return ((I[80, 80] - I[5, 5]) / I[5, 5]).item()
print("annulus points:", len(ann))
print(f"phase-contrast object contrast: full annulus {contrast(I_pc):+.3f}, single on-annulus {contrast(I_one):+.3f}, single on-axis {contrast(I_axis):+.3f}")
