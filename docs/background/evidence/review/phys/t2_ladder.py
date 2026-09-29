# Test 2: validity thresholds of the interaction rungs against exact solutions.
#  (a) born_ff validity "Delta n * d < 0.35 lambda0 (phi <~ 2.2 rad)" vs exact scalar sphere
#  (b) projection (thin screen at centre plane) vs exact, as a function of thickness/DOF and NA (obliquity)
#  (c) Mie-dressed dipole (alpha = i 6 pi a1 / k^3) vs full vector Mie (BH) at x = 0.3, 1: is it "Mie truncated at n=1"?
import numpy as np
from scipy.special import spherical_jn, spherical_yn, j0
from scipy.integrate import quad
lam0, n_m = 0.532, 1.33
k0 = 2*np.pi/lam0; k = k0*n_m
def hn(n, x, der=False): return spherical_jn(n, x, der) + 1j*spherical_yn(n, x, der)
def scalar_coeffs(a, m, nmax):
    x = k*a; mx = m*x; c = []
    for n in range(nmax+1):
        num = m*spherical_jn(n, mx, True)*spherical_jn(n, x) - spherical_jn(n, x, True)*spherical_jn(n, mx)
        den = hn(n, x, True)*spherical_jn(n, mx) - m*spherical_jn(n, mx, True)*hn(n, x)
        c.append(num/den)
    return np.array(c)
def S_scalar(mu, c):
    mu = np.asarray(mu, dtype=complex); P0 = np.ones_like(mu); P1 = mu.copy(); out = -(c[0]*P0 + 3*c[1]*P1)
    Pm, Pc = P0, P1
    for n in range(1, len(c)-1):
        Pn1 = ((2*n+1)*mu*Pc - n*Pm)/(n+1); out = out - (2*(n+1)+1)*c[n+1]*Pn1; Pm, Pc = Pc, Pn1
    return out
def S_born(theta, a, m):
    q = 2*k*np.sin(theta/2); qa = np.maximum(q*a, 1e-9)
    F = 4*np.pi*a**3*spherical_jn(1, qa)/qa
    A = 1j*k0**2*(n_m**2*m**2 - n_m**2)*F/(8*np.pi**2*k*np.cos(theta))
    return -2*np.pi*k*(k*np.cos(theta))*A
def S_proj(theta, a, m, obliquity=False):
    # thin screen at centre plane: A(kperp) = (1/2pi) int_0^a (t-1) J0(kperp rho) rho drho ; S = -2 pi k kz A
    dn = n_m*(m-1); kp = k*np.sin(theta); out = []
    for kpi, th in zip(kp, theta):
        f = lambda r: (np.exp(1j*k0*dn*2*np.sqrt(max(a*a-r*r, 0))) - 1)*j0(kpi*r)*r
        A = (quad(lambda r: f(r).real, 0, a, limit=200)[0] + 1j*quad(lambda r: f(r).imag, 0, a, limit=200)[0])/(2*np.pi)
        if obliquity: A = A/np.cos(th)
        out.append(-2*np.pi*k*(k*np.cos(th))*A)
    return np.array(out)
def rel_l2_pupil(Sa, Sb, theta):  # rel-L2 of the angular spectra A ~ S/kz over the pupil, measure d^2k ~ sin cos dth
    Aa, Ab = Sa/np.cos(theta), Sb/np.cos(theta); w = np.sin(theta)*np.cos(theta)
    return np.sqrt(np.sum(np.abs(Aa-Ab)**2*w)/np.sum(np.abs(Ab)**2*w))

print("=== (a) born_ff vs exact scalar sphere, NA 0.8 pupil (water) ===")
th = np.linspace(1e-3, np.arcsin(0.8/n_m), 200)
for a in [0.25, 1.0]:
    for phi in [0.1, 0.3, 0.5, 1.0, 2.2]:
        dn = phi/(k0*2*a); m = 1 + dn/n_m
        c = scalar_coeffs(a, m, int(k*a*m + 4*(k*a*m)**(1/3) + 8))
        Se = S_scalar(np.cos(th), c); Sb = S_born(th, a, m)
        fwd_phase = np.angle(Sb[0]/Se[0])
        print(f"a={a:4.2f} um  phi=k0*dn*d={phi:4.1f} rad  (dn*d/lam0={dn*2*a/lam0:.3f}) : pupil rel-L2 {rel_l2_pupil(Sb, Se, th):6.3f}, "
              f"forward |S| ratio {abs(Sb[0])/abs(Se[0]):.3f}, forward phase err {fwd_phase:+.3f} rad")

print("\n=== (b) projection (centre-plane thin screen) vs exact scalar sphere; small phase (phi=0.2) ===")
for NA in [0.5, 0.8, 1.0]:
    th = np.linspace(1e-3, np.arcsin(NA/n_m), 80)
    DOF = n_m*lam0/NA**2
    for frac in [0.1, 0.25, 1.0]:
        a = frac*DOF/2; phi = 0.2; dn = phi/(k0*2*a); m = 1 + dn/n_m
        c = scalar_coeffs(a, m, int(k*a*m + 4*(k*a*m)**(1/3) + 8))
        Se = S_scalar(np.cos(th), c)
        e0 = rel_l2_pupil(S_proj(th, a, m), Se, th); e1 = rel_l2_pupil(S_proj(th, a, m, True), Se, th)
        edge = abs(S_proj(th[-1:], a, m)[0])/abs(Se[-1])
        print(f"NA={NA:.2f} (NA/n={NA/n_m:.2f}) d={2*a:5.3f} um = {frac:4.2f} DOF : rel-L2 plain {e0:.3f} | with 1/cos(theta_out) {e1:.3f} | |A_proj/A_exact| at pupil edge {edge:.3f} (cos={np.cos(th[-1]):.3f})")

print("\n=== (c) dipole (electric, alpha from a1) vs full vector Mie (BH) ===")
def bh_ab(x, m, nmax):
    mx = m*x; a = []; b = []
    for n in range(1, nmax+1):
        psi = x*spherical_jn(n, x); dpsi = spherical_jn(n, x) + x*spherical_jn(n, x, True)
        xi = x*hn(n, x); dxi = hn(n, x) + x*hn(n, x, True)
        psim = mx*spherical_jn(n, mx); dpsim = spherical_jn(n, mx) + mx*spherical_jn(n, mx, True)
        a.append((m*psim*dpsi - psi*dpsim)/(m*psim*dxi - xi*dpsim))
        b.append((psim*dpsi - m*psi*dpsim)/(psim*dxi - m*xi*dpsim))
    return np.array(a), np.array(b)
def S12(ab, mu, upto=None, only_a1=False):
    a, b = ab; N = len(a) if upto is None else upto
    pi_prev = np.zeros_like(mu); pi_cur = np.ones_like(mu); S1 = 0j*mu; S2 = 0j*mu
    for n in range(1, N+1):
        tau = n*mu*pi_cur - (n+1)*pi_prev; cn = (2*n+1)/(n*(n+1))
        an = a[n-1]; bn = 0 if only_a1 else b[n-1]
        S1 = S1 + cn*(an*pi_cur + bn*tau); S2 = S2 + cn*(an*tau + bn*pi_cur)
        pi_prev, pi_cur = pi_cur, ((2*n+1)*mu*pi_cur - (n+1)*pi_prev)/n
    return S1, S2
mu_full = np.cos(np.linspace(0, np.pi, 721))   # all angles (backscatter matters for iSCAT)
mu_pup = np.cos(np.linspace(0, np.arcsin(0.8/n_m), 200))
for name, mrel in [("polystyrene/water", 1.59/1.33), ("silica/water", 1.46/1.33), ("TiO2/water", 2.6/1.33),
                   ("gold/water 532nm", (0.54+2.14j)/1.33)]:
    for x in [0.3, 0.6, 1.0]:
        ab = bh_ab(x, mrel, int(x + 4*x**(1/3) + 6))
        def err(mu, **kw):
            S1f, S2f = S12(ab, mu); S1d, S2d = S12(ab, mu, **kw)
            num = np.abs(S1f-S1d)**2 + np.abs(S2f-S2d)**2; den = np.abs(S1f)**2 + np.abs(S2f)**2
            return np.sqrt(num.sum()/den.sum())
        print(f"{name:18s} x={x:3.1f}: a1-only dipole rel-L2 all angles {err(mu_full, upto=1, only_a1=True):.3f}, "
              f"pupil NA0.8 {err(mu_pup, upto=1, only_a1=True):.3f} | 'Mie truncated at n=1' (a1+b1) all angles {err(mu_full, upto=1):.3f} "
              f"| |b1/a1|={abs(ab[1][0]/ab[0][0]):.3f}")
