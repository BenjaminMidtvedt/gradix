import numpy as np
from scipy.special import spherical_jn, spherical_yn
def hn(n, x, der=False): return spherical_jn(n, x, der) + 1j*spherical_yn(n, x, der)
def bh_ab(x, m, nmax):
    mx = m*x; a = []; b = []
    for n in range(1, nmax+1):
        psi = x*spherical_jn(n, x); dpsi = spherical_jn(n, x) + x*spherical_jn(n, x, True)
        xi = x*hn(n, x); dxi = hn(n, x) + x*hn(n, x, True)
        psim = mx*spherical_jn(n, mx); dpsim = spherical_jn(n, mx) + mx*spherical_jn(n, mx, True)
        a.append((m*psim*dpsi - psi*dpsim)/(m*psim*dxi - xi*dpsim)); b.append((psim*dpsi - m*psi*dpsim)/(psim*dxi - m*xi*dpsim))
    return np.array(a), np.array(b)
def S12(a, b, mu):
    pp = np.zeros_like(mu); pc = np.ones_like(mu); S1 = 0j*mu; S2 = 0j*mu
    for n in range(1, len(a)+1):
        tau = n*mu*pc - (n+1)*pp; c = (2*n+1)/(n*(n+1)); S1 += c*(a[n-1]*pc + b[n-1]*tau); S2 += c*(a[n-1]*tau + b[n-1]*pc)
        pp, pc = pc, ((2*n+1)*mu*pc - (n+1)*pp)/n
    return S1, S2
k = 2*np.pi*1.33/0.532
for NA in [0.8, 1.2]:
    th = np.linspace(1e-4, np.arcsin(NA/1.33), 400); ph = np.linspace(0, 2*np.pi, 360, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij"); w = np.sin(TH)*np.cos(TH)
    for r in [0.05, 0.25, 0.5]:
        x = k*r; a, b = bh_ab(x, 1.59/1.33, int(x+4*x**(1/3)+4)); S1, S2 = S12(a, b, np.cos(th))
        S1, S2 = S1[:, None], S2[:, None]
        obj = S2*np.cos(TH)*np.cos(PH)**2 + S1*np.sin(PH)**2        # object-space Cartesian x component
        lens = S2*np.cos(PH)**2 + S1*np.sin(PH)**2                    # x component after aplanatic lens rotation
        e = np.sqrt((np.abs(obj-lens)**2*w).sum()/(np.abs(lens)**2*w).sum())
        print(f"NA={NA} polystyrene r={r} um: rel-L2 between the two 'x-pol projection' definitions over the pupil = {e:.3f}")
