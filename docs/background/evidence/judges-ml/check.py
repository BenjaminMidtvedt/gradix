import torch, math
torch.manual_seed(0)
dev = "cuda"
# ---- 1) float32 subnormal behaviour on CUDA (B's "flushed" claim) ----
a = torch.tensor(1e-7, device=dev, dtype=torch.float32)
print("a^6 cuda f32:", (a**6).item(), " a*a*a*a*a*a:", (a*a*a*a*a*a).item())
alpha = 4*math.pi*(1e-8)**3*0.3
t = torch.tensor(alpha, device=dev, dtype=torch.float32)
print("alpha(10nm, metres)^2 f32:", (t*t).item())

# ---- 2) stochastic Abbe K=2 vs deterministic quadrature: per-image residual vs shot noise ----
N, dx, lam, NA, NAc = 256, 0.1, 0.55, 0.4, 0.2   # um
f = torch.fft.fftfreq(N, dx, device=dev)
FY, FX = torch.meshgrid(f, f, indexing="ij")
FR = torch.sqrt(FX**2 + FY**2)
P = (FR <= NA/lam).to(torch.complex64)
# thin phase object: a few soft discs (cells) with phase ~1 rad + texture
y = (torch.arange(N, device=dev) - N/2) * dx
Y, X = torch.meshgrid(y, y, indexing="ij")
phi = torch.zeros(N, N, device=dev)
for (cx, cy, r) in [(-4, -3, 3.0), (5, 4, 2.5), (2, -6, 2.0), (-6, 6, 1.5)]:
    phi += 1.0 * torch.sigmoid((r - torch.sqrt((X-cx)**2 + (Y-cy)**2)) / 0.15)
phi += 0.1 * torch.randn(N, N, device=dev).mul(1.0)
t_obj = torch.exp(1j * phi.to(torch.complex64))
T = torch.fft.fft2(t_obj)
df = 1.0 / (N * dx)
# on-grid source points inside condenser disc
src = [(i, j) for i in range(-40, 41) for j in range(-40, 41) if math.hypot(i*df, j*df) <= NAc/lam]
def img_for(points):
    acc = torch.zeros(N, N, device=dev)
    for (i, j) in points:
        Ts = torch.roll(T, shifts=(j, i), dims=(0, 1))   # tilt = spectrum shift (thin sample)
        e = torch.fft.ifft2(Ts * P)
        acc += e.real**2 + e.imag**2
    return acc / len(points)
I_ref = img_for(src)
print("n source points (exact on-grid Abbe):", len(src))
g = torch.Generator().manual_seed(1)
res = []
for rep in range(8):
    idx = torch.randint(len(src), (2,), generator=g).tolist()
    I2 = img_for([src[k] for k in idx])
    res.append(((I2 - I_ref).pow(2).mean().sqrt() / I_ref.mean()).item())
print("stochastic K=2: per-image rel RMS residual vs exact Abbe: mean %.3f" % (sum(res)/len(res)))
for K in (9, 25):
    # deterministic quadrature: evenly spread subset (every k-th point)
    step = max(1, len(src)//K)
    Iq = img_for(src[::step][:K])
    print("deterministic subset K=%d rel RMS: %.4f" % (K, ((Iq - I_ref).pow(2).mean().sqrt() / I_ref.mean()).item()))
for photons in (1e3, 1e4):
    print("relative shot noise at %d photons/px: %.3f" % (photons, 1/math.sqrt(photons)))
