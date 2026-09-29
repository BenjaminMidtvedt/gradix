"""§10.2 L2: BatchedDataset 'probes the DT DAG (256 CPU resolutions) to estimate bounds and N_max for opaque lambdas';
'Guards mark images whose values leave the probed bounds (in_bounds mask)'; §6.6 default: warn + re-plan before the NEXT call.
How often does a later image exceed the probed N_max / z-range, for typical DT lambdas?"""
import numpy as np
rng = np.random.default_rng(0)
R, B, trials = 256, 64, 20000
# (a) particle count: particle ^ (lambda: np.random.poisson(30))  (DT multi-particle idiom)
p_img, p_batch = [], []
for _ in range(trials):
    nmax = rng.poisson(30, R).max()
    new = rng.poisson(30, B)
    p_img.append((new > nmax).mean()); p_batch.append((new > nmax).any())
print(f"(a) N ~ Poisson(30): per-image P(N > probed N_max) = {np.mean(p_img):.4f}; "
      f"P(batch of 64 has >=1 such image) = {np.mean(p_batch):.3f}")
# (b) defocus per particle: z = lambda: np.random.normal(0, 5) (um), 30 particles per image
p_img = []
for _ in range(2000):
    probe = rng.normal(0, 5, (R, 30)); lo, hi = probe.min(), probe.max()
    new = rng.normal(0, 5, (B, 30))
    p_img.append(((new < lo) | (new > hi)).any(1).mean())
print(f"(b) z ~ N(0,5) per particle, 30/image: per-image P(any z outside probed range) = {np.mean(p_img):.4f}; "
      f"P(batch has >=1) = {1-(1-np.mean(p_img))**B:.3f}")
