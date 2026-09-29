# bare quasi-static alpha0 = 4 pi a^3 (m^2-1)/(m^2+2)  vs  a1 (exact electric dipole), and with the radiative correction only
import numpy as np
from scipy.special import spherical_jn, spherical_yn
def hn(n, x, der=False): return spherical_jn(n, x, der) + 1j*spherical_yn(n, x, der)
def a1(x, m):
    mx = m*x; n = 1
    psi = x*spherical_jn(n, x); dpsi = spherical_jn(n, x) + x*spherical_jn(n, x, True)
    xi = x*hn(n, x); dxi = hn(n, x) + x*hn(n, x, True)
    psim = mx*spherical_jn(n, mx); dpsim = spherical_jn(n, mx) + mx*spherical_jn(n, mx, True)
    return (m*psim*dpsi - psi*dpsim)/(m*psim*dxi - xi*dpsim)
for name, m in [("polystyrene/water", 1.59/1.33), ("TiO2/water", 2.6/1.33), ("gold/water 532", (0.54+2.14j)/1.33), ("silver/water 450", (0.04+2.46j)/1.33)]:
    for x in [0.1, 0.2, 0.3]:
        al0 = 4*np.pi*(m*m-1)/(m*m+2)            # alpha0 / a^3 with k a = x -> in units a^3; S ~ k^3 alpha
        a1_bare = -1j*x**3*al0/(6*np.pi)           # a1 = -i k^3 alpha/(6pi) with alpha = al0 a^3
        a1_rc = -1j*x**3*(al0/(1 - 1j*x**3*al0/(6*np.pi)))/(6*np.pi)
        ex = a1(x, m)
        print(f"{name:17s} x={x:.1f}: |bare/exact-1|={abs(a1_bare/ex-1):.3f}, phase err {np.angle(a1_bare/ex):+.3f} rad | radiative-corrected: {abs(a1_rc/ex-1):.3f}")
