import numpy as np, re
src = open("../phys/t2_ladder.py").read()
start = src.index("def bh_ab"); end = src.index("mu_full =")
ns = {"np": np}
exec("from scipy.special import spherical_jn, spherical_yn\n" + src[src.index("def hn"):src.index("def scalar_coeffs")] + src[start:end], ns)
bh_ab, S12 = ns["bh_ab"], ns["S12"]
mu = np.cos(np.linspace(0, np.pi, 721))
for name, m in [("gold/water 532", (0.54+2.14j)/1.33), ("silver/water 450", (0.04+2.46j)/1.33), ("silver/water 400", (0.05+2.1j)/1.33), ("PS/water", 1.59/1.33)]:
    row = []
    for x in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ab = bh_ab(x, m, int(x + 4*x**(1/3) + 6))
        S1f, S2f = S12(ab, mu)
        def e(**kw):
            S1d, S2d = S12(ab, mu, **kw)
            return np.sqrt(((np.abs(S1f-S1d)**2+np.abs(S2f-S2d)**2).sum())/((np.abs(S1f)**2+np.abs(S2f)**2).sum()))
        row.append(f"x={x}: a1 {e(upto=1, only_a1=True):.3f} a1+b1 {e(upto=1):.3f}")
    print(name, " | ".join(row))
