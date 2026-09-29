# Paraxial in-focus Airy (scalar, uniform pupil) vs unit-sum Gaussian, fine sampling (no pixel integration), 2-D rel-L2
import numpy as np
from scipy.special import j1
lam, NA = 1.0, 1.0
d = lam/(4*NA)/8; N = 1024
x = (np.arange(N)-N/2)*d; X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
v = 2*np.pi*NA/lam*r; v[v==0] = 1e-12
A = (2*j1(v)/v)**2; A /= A.sum()
for c in [0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
    s = c*lam/NA; G = np.exp(-r**2/(2*s*s)); G /= G.sum()
    print(f"sigma={c:.2f} lam/NA: rel-L2 {np.linalg.norm(G-A)/np.linalg.norm(A):.3f}")
