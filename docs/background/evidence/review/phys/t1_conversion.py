# Test 1: far-field -> angular-spectrum conversion A = -S/(2 pi k k_z) e^{-i(k-k_in).r_p}
# against a direct near-field partial-wave sum of an exact scalar sphere solution.
# Also: Born form factor A = i k0^2 (n^2-n_m^2) F(k-k_in)/(8 pi^2 k_z) vs the same exact S in the weak limit,
# and the optical theorem C_ext = (4pi/k^2) Re S(0) through the cross term.
import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre, j0, j1
from scipy.integrate import quad

lam0, n_m = 0.532, 1.33
k0 = 2*np.pi/lam0; k = k0*n_m

def hn(n, x, der=False):
    return spherical_jn(n, x, der) + 1j*spherical_yn(n, x, der)

def scalar_coeffs(a, m, nmax):
    x = k*a; mx = m*x
    c = []
    for n in range(nmax+1):
        num = m*spherical_jn(n, mx, True)*spherical_jn(n, x) - spherical_jn(n, x, True)*spherical_jn(n, mx)
        den = hn(n, x, True)*spherical_jn(n, mx) - m*spherical_jn(n, mx, True)*hn(n, x)
        c.append(num/den)
    return np.array(c)

def S_of(mu, c):  # S(theta) = -sum (2n+1) c_n P_n(cos theta); mu may be complex
    mu = np.asarray(mu, dtype=complex)
    # Legendre recurrence valid for complex argument
    P0 = np.ones_like(mu); P1 = mu.copy(); out = -(1*c[0]*P0 + 3*c[1]*P1)
    Pm, Pc = P0, P1
    for n in range(1, len(c)-1):
        Pn1 = ((2*n+1)*mu*Pc - n*Pm)/(n+1)
        out = out - (2*(n+1)+1)*c[n+1]*Pn1
        Pm, Pc = Pc, Pn1
    return out

def direct_field(rho, z, c):
    r = np.hypot(rho, z); mu = z/r
    return sum((1j**n)*(2*n+1)*c[n]*hn(n, k*r)*eval_legendre(n, mu) for n in range(len(c)))

def spectrum_field(rho, z, c):
    # E = int A e^{i(k.r)} d^2k,  A = -S/(2 pi k kz); polar integral with kz substitution (see notes)
    fr = lambda kz: (-(1/k)*S_of(kz/k, c)*np.exp(1j*kz*z)*j0(rho*np.sqrt(max(k*k-kz*kz, 0.0))))
    re = quad(lambda t: fr(t).real, 0, k, limit=400)[0]; im = quad(lambda t: fr(t).imag, 0, k, limit=400)[0]
    fe = lambda kap: ((1j/k)*S_of(1j*kap/k, c)*np.exp(-kap*z)*j0(rho*np.sqrt(k*k+kap*kap)))
    re2 = quad(lambda t: fe(t).real, 0, 60/z, limit=400)[0]; im2 = quad(lambda t: fe(t).imag, 0, 60/z, limit=400)[0]
    return re + 1j*im + re2 + 1j*im2

a, m = 0.6, 1.08
x = k*a; nmax = int(x + 4*x**(1/3) + 6)
c = scalar_coeffs(a, m, nmax)
print(f"x={x:.2f}, m={m}, nmax={nmax}")
print(" rho     z     |direct|        rel.err(spectrum formula)")
for rho, z in [(0.0, 1.0), (0.5, 1.0), (1.5, 1.2), (0.0, 3.0), (2.0, 2.0), (0.3, 0.75)]:
    d = direct_field(rho, z, c); s = spectrum_field(rho, z, c)
    print(f"{rho:5.2f} {z:5.2f}  {abs(d):.4e}   {abs(s-d)/abs(d):.2e}")

# sign check of the opposite convention (+S): error should be ~2
d = direct_field(0.5, 1.0, c); s = spectrum_field(0.5, 1.0, -c)
print("flipped-sign formula rel.err:", abs(s-d)/abs(d))

# optical theorem: extinction via 2 Re int E_b^* E_s d^2r = 2 Re (2pi)^2 A(0) (on axis, z_ref=0)
Cext_ot = 4*np.pi/k**2*S_of(1.0, c).real
# direct scalar extinction = scattered power (lossless): Csca = (2pi/k^2) int |S|^2 sin th dth
Csca = 2*np.pi/k**2*quad(lambda t: abs(S_of(np.cos(t), c))**2*np.sin(t), 0, np.pi, limit=400)[0]
A0 = -S_of(1.0, c)/(2*np.pi*k*k)
Cext_cross = -2*((2*np.pi)**2*A0).real
print(f"optical theorem: (4pi/k^2)ReS(0)={Cext_ot:.6e}  cross-term={Cext_cross:.6e}  Csca={Csca:.6e}")

# Born form factor vs exact scalar S in the weak limit (Rayleigh-Gans):
def S_born(theta, a, m):
    q = 2*k*np.sin(theta/2); qa = np.maximum(q*a, 1e-9)
    F = 4*np.pi*a**3*spherical_jn(1, qa)/qa           # int e^{-iq.r} d^3r over the ball
    A = 1j*k0**2*(n_m**2*m**2 - n_m**2)*F/(8*np.pi**2*k*np.cos(theta))  # plan's formula (kz = k cos th)
    return -2*np.pi*k*(k*np.cos(theta))*A              # back to S through the plan's conversion
for a_, m_ in [(0.2, 1.002), (0.6, 1.002), (0.6, 1.01)]:
    cc = scalar_coeffs(a_, m_, int(k*a_+4*(k*a_)**(1/3)+6))
    th = np.linspace(0.01, 1.0, 50)
    e = np.abs(S_born(th, a_, m_) - S_of(np.cos(th), cc))/np.abs(S_of(np.cos(th), cc)).max()
    print(f"Born (plan formula) vs exact scalar, a={a_}, m={m_}: max rel err {e.max():.3e}")
