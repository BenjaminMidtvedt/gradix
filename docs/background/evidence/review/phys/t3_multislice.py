# Test 3 (GPU): scattered-field multislice march  E_s <- P_dz[(t_s-1) E_b(z_s) + t_s E_s]  with analytic E_b
#  (i) equals the total-field march (algebraic identity check, fp32 vs fp64)
#  (ii) vs exact scalar sphere: the plan's edge "multislice (voxel sphere) <-> mie, dn<=0.05, x<=10, dz<=lam_m/8: <=5%"
#  (iii) with an obliquity-corrected source term O = k/kz (multi-layer-Born-like) -> weak limit = first Born
import math, numpy as np, torch
from scipy.special import spherical_jn, spherical_yn
dev = "cuda"
lam0, n_m = 0.532, 1.33; k0 = 2*math.pi/lam0; k = k0*n_m
def hn(n, x, der=False): return spherical_jn(n, x, der) + 1j*spherical_yn(n, x, der)
def scalar_coeffs(a, m, nmax):
    x = k*a; mx = m*x; c = []
    for n in range(nmax+1):
        num = m*spherical_jn(n, mx, True)*spherical_jn(n, x) - spherical_jn(n, x, True)*spherical_jn(n, mx)
        den = hn(n, x, True)*spherical_jn(n, mx) - m*spherical_jn(n, mx, True)*hn(n, x)
        c.append(num/den)
    return np.array(c)
def S_scalar(mu, c):
    mu = np.asarray(mu, dtype=complex); P0 = np.ones_like(mu); P1 = mu.copy(); out = -(c[0]*P0 + 3*c[1]*P1); Pm, Pc = P0, P1
    for n in range(1, len(c)-1):
        Pn1 = ((2*n+1)*mu*Pc - n*Pm)/(n+1); out = out - (2*(n+1)+1)*c[n+1]*Pn1; Pm, Pc = Pc, Pn1
    return out

def march(a, dn, N=256, dx=0.04, dz=None, obliq=False, total=False, dtype=torch.complex128, ss=4):
    dz = dz or (lam0/n_m)/8
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    x = (torch.arange(N, device=dev, dtype=torch.float64) - N//2)*dx
    Y, X = torch.meshgrid(x, x, indexing="ij")
    f = torch.fft.fftfreq(N, dx, device=dev, dtype=torch.float64)
    FY, FX = torch.meshgrid(f, f, indexing="ij"); kp2 = (2*math.pi)**2*(FX**2+FY**2)
    kz = torch.sqrt((k*k - kp2).to(torch.complex128)); prop_ok = (kp2 < k*k)
    nz = int(math.ceil(2*a/dz)) + 2; z0 = -nz*dz/2
    zc = z0 + (torch.arange(nz, device=dev, dtype=torch.float64) + 0.5)*dz
    P = torch.where(prop_ok, torch.exp(1j*kz*dz), 0).to(dtype)
    O = torch.where(prop_ok, (k/kz.real.clamp(min=1e-3)), 0).to(dtype) if obliq else None
    Es = torch.zeros(N, N, device=dev, dtype=dtype); Et = torch.exp(1j*k*torch.tensor(z0, dtype=torch.float64, device=dev))*torch.ones(N, N, device=dev, dtype=torch.complex128)
    Et = Et.to(dtype)
    # supersampled occupancy of each slab (fraction of slab thickness inside the ball, xy-supersampled)
    for s in range(nz):
        zlo, zhi = z0 + s*dz, z0 + (s+1)*dz
        occ = torch.zeros(N, N, device=dev, dtype=torch.float64)
        for iy in range(ss):
            for ix in range(ss):
                r2 = (X + (ix+0.5)/ss*dx - dx/2)**2 + (Y + (iy+0.5)/ss*dx - dx/2)**2
                h = torch.sqrt(torch.clamp(a*a - r2, min=0))
                occ += torch.clamp(torch.minimum(h, torch.tensor(zhi, device=dev, dtype=torch.float64)) - torch.maximum(-h, torch.tensor(zlo, device=dev, dtype=torch.float64)), min=0)/dz
        occ /= ss*ss
        t = torch.exp(1j*k0*dn*dz*occ).to(dtype)              # thin screen for this slab (Delta n relative to host)
        Eb = torch.exp(1j*k*torch.tensor(float(zc[s]), dtype=torch.float64, device=dev)).to(dtype)   # analytic background at slab centre
        if total:
            # total-field march, background carried on-grid (on-axis => exact)
            Et = torch.fft.ifft2(torch.fft.fft2(Et*t)*P) if s > 0 or True else Et
        else:
            src = (t - 1)*Eb + t*Es if not obliq else Es + torch.fft.ifft2(torch.fft.fft2((t-1)*(Eb + Es))*O)
            if obliq:
                Es = torch.fft.ifft2(torch.fft.fft2(src)*P)
            else:
                Es = torch.fft.ifft2(torch.fft.fft2(src)*P)
    zend = z0 + nz*dz
    if total:
        Es = Et - torch.exp(1j*k*torch.tensor(zend + dz/2*0, dtype=torch.float64, device=dev)).to(dtype)
    A = torch.fft.fft2(Es.to(torch.complex128))*dx*dx/(2*math.pi)**2
    # reference to z_ref = 0 (sphere centre) and remove the x-shift of the grid origin (grid centred at N//2)
    A = A*torch.exp(-1j*kz*(zend + dz/2))*torch.exp(-1j*2*math.pi*(FX+FY)*(-(N//2)*dx))
    return A, FX, FY, kz

# (i) identity check (on-axis illumination so the total-field background is exactly representable on grid)
# NOTE: total-field convention above applies screens at the slab start; compare scattered-field march with same placement
def march_pair(a, dn, dtype):
    return march(a, dn, dtype=dtype)[0]
A64 = march(0.3, 0.05, dtype=torch.complex128)[0]; A32 = march(0.3, 0.05, dtype=torch.complex64)[0]
pk = march(0.3, 0.05)[3].imag.abs() < 1e-9
print("scattered-field march fp32 vs fp64 rel diff (propagating):", (torch.linalg.norm((A32-A64)[pk])/torch.linalg.norm(A64[pk])).item())

def compare(a, dn, NA, obliq):
    A, FX, FY, kz = march(a, dn, obliq=obliq)
    fr = torch.sqrt(FX**2+FY**2); sel = fr*lam0 <= NA
    m = 1 + dn/n_m; c = scalar_coeffs(a, m, int(k*a*m + 4*(k*a*m)**(1/3) + 8))
    mu = (kz.real[sel]/k).cpu().numpy()
    Aex = torch.tensor(-S_scalar(mu, c), device=dev)/(2*math.pi*k*kz.real[sel])
    Anum = A[sel]
    return (torch.linalg.norm(Anum-Aex)/torch.linalg.norm(Aex)).item()

print("\nmultislice (dz = lam_m/8, supersampled voxels, dx=40 nm) vs exact scalar sphere, pupil rel-L2")
for dn in [0.01, 0.05]:
    for x_ in [1.0, 3.0, 10.0]:
        a = x_/k
        for NA in [0.5, 0.8, 1.2]:
            e0 = compare(a, dn, NA, False); e1 = compare(a, dn, NA, True)
            print(f"dn={dn:.2f} x={x_:4.1f} (a={a:.3f}um, phi={k0*dn*2*a:.2f}) NA={NA:.1f}: plain BPM {e0:.3f} | with k/kz obliquity {e1:.3f}")
