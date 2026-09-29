import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
def bucket(n):
    b = 8
    while b < n: b *= 2
    return b
for lam in [5, 10, 14, 20, 30, 45, 60]:
    maxes = rng.poisson(lam, (4000, 256)).max(1)
    nm = np.array([bucket(m) for m in maxes])
    p_img = np.mean(stats.poisson.sf(nm, lam))
    print(f"Poisson({lam}): probed max median {np.median(maxes):.0f}, bucket values {sorted(set(nm))}, per-image P(N>bucket) {p_img:.2e}, per-batch64 {1-(1-p_img)**64:.2e}")
# heavy-tailed count
for s in [0.5, 0.8, 1.0]:
    draws = lambda size: np.floor(rng.lognormal(3, s, size)).astype(int)
    maxes = draws((2000, 256)).max(1); nm = np.array([bucket(m) for m in maxes])
    new = draws((2000, 64))
    p_img = np.mean(new > nm[:, None]); p_b = np.mean((new > nm[:, None]).any(1))
    print(f"floor(lognormal(3,{s})): per-image {p_img:.2e}, per-batch {p_b:.3f}")
