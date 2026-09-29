# Does Mie-in-pupil holography/brightfield hit A's ">=2k frames/s with 7 Koehler modes, 5 particles, 256^2"?
# Reuses proposal C's W3 kernel structure; adds M tilted modes (theta measured from each k_in) and an
# optional pupil-support-only evaluation (B's idea). Forward only, inference mode (data-generation case).
import math, time, sys, torch
sys.path.insert(0, "propC")
from bench_v1_tiers import mie_coeffs
dev = "cuda"; torch.backends.cuda.matmul.fp32_precision = "ieee"
wl, NA, n_m = 0.6, 1.3, 1.33       # um
def make(B=64, Pn=5, N=256, dx=0.1, Nth=512, M=1, support=False, NAc=0.3):
    k = 2*math.pi*n_m/wl; P = B*Pn
    g = torch.Generator(device=dev).manual_seed(0)
    radius = torch.rand(P, device=dev, dtype=torch.float64, generator=g)*0.4+0.2
    nre = torch.full((P,), 1.58, device=dev, dtype=torch.float64)
    pos = torch.rand(B, Pn, 3, device=dev, generator=g)*torch.tensor([N*dx, N*dx, 10.0], device=dev)
    f = torch.fft.fftfreq(N, dx, device=dev); fy, fx = torch.meshgrid(f, f, indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    fr = torch.sqrt(fx**2+fy**2)
    idx = torch.nonzero(fr <= NA/wl*1.02).squeeze(1) if support else torch.arange(N*N, device=dev)
    fxs, fys = fx[idx], fy[idx]
    kz = 2*math.pi*torch.sqrt(torch.clamp((n_m/wl)**2 - fxs**2 - fys**2, min=0))
    inNA = torch.sigmoid((NA/wl - torch.sqrt(fxs**2+fys**2))*wl*50)
    # M source directions on a ring inside condenser NA (plus centre)
    ang = torch.arange(M, device=dev)*2*math.pi/max(M-1,1)
    rad = torch.where(torch.arange(M, device=dev)==0, 0.0, NAc/wl*0.7)
    kin = torch.stack([rad*torch.cos(ang), rad*torch.sin(ang)], -1)          # [M,2] cycles/um
    theta_max = math.asin(min(1.0,(NA+NAc)/n_m))
    theta_grid = torch.linspace(0, theta_max, Nth, device=dev, dtype=torch.float64)
    def run():
        with torch.inference_mode():
            x = k*radius; nmax = int(x.max().item() + 4*x.max().item()**(1/3) + 2)
            a, b = mie_coeffs(x, torch.complex(nre/n_m, torch.zeros_like(nre)), nmax)
            mu = torch.cos(theta_grid); pi_prev = torch.zeros_like(mu); pi_cur = torch.ones_like(mu)
            S1 = torch.zeros(P, Nth, dtype=torch.complex128, device=dev); S2 = torch.zeros_like(S1)
            for n in range(1, nmax+1):
                tau = n*mu*pi_cur - (n+1)*pi_prev; c = (2*n+1)/(n*(n+1))
                S1 = S1 + c*(a[:, n-1, None]*pi_cur + b[:, n-1, None]*tau)
                S2 = S2 + c*(a[:, n-1, None]*tau + b[:, n-1, None]*pi_cur)
                pi_prev, pi_cur = pi_cur, ((2*n+1)*mu*pi_cur - (n+1)*pi_prev)/n
            S1 = S1.to(torch.complex64); S2 = S2.to(torch.complex64)
            I = torch.zeros(B, N*N, device=dev)
            for m in range(M):
                # direction cosines of k and k_in -> scattering angle
                sx, sy = fxs*wl/n_m, fys*wl/n_m; sz = torch.sqrt(torch.clamp(1-sx**2-sy**2, min=0))
                ix, iy = kin[m,0]*wl/n_m, kin[m,1]*wl/n_m; iz = torch.sqrt(1-ix**2-iy**2)
                cth = (sx*ix+sy*iy+sz*iz).clamp(-1,1); th = torch.acos(cth)
                tq = (th/theta_max).clamp(max=1)*(Nth-1); i0 = tq.floor().long().clamp(max=Nth-2); w = (tq-i0).float()
                phi = torch.atan2(fys-kin[m,1], fxs-kin[m,0])
                s1 = S1[:, i0]*(1-w) + S1[:, i0+1]*w; s2 = S2[:, i0]*(1-w) + S2[:, i0+1]*w
                Es = (s2*torch.cos(phi)**2 + s1*torch.sin(phi)**2).view(B, Pn, -1)
                ph = (-2*math.pi*(fxs*pos[...,0,None] + fys*pos[...,1,None]) + kz*pos[...,2,None])
                Es = (Es*torch.polar(torch.ones_like(ph), ph)/sz.clamp(min=0.1)).sum(1)*inNA*1e-3
                full = torch.zeros(B, N*N, dtype=torch.complex64, device=dev)
                full[:, idx] = Es
                E = torch.fft.ifft2(full.view(B, N, N)).reshape(B, -1)
                I += (1 + 2*E.real + E.real**2 + E.imag**2)/M
            return I
    return run
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1e3
for M in [1, 7]:
    for sup in [False, True]:
        t = bench(make(M=M, support=sup))
        print(f"M={M} support_only={sup}: {t:.1f} ms / 64 images -> {64/t*1e3:.0f} img/s")
