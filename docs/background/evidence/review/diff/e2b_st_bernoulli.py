"""E2b: identity straight-through Bernoulli (b = hard + p - stopgrad(p)) on the same quadratic count losses."""
import math, torch
dev = "cuda"
gen = torch.Generator(device=dev).manual_seed(2)
for K, p0, target in ((64, 0.25, 20.0), (8, 0.6, 3.0), (8, 0.6, 6.0), (8, 0.05, 1.0), (8, 0.95, 6.0), (64, 0.05, 8.0)):
    reps = 200_000
    w = 0.5 + torch.rand(reps, K, device=dev, generator=gen)
    sw, sw2 = w.sum(-1), (w ** 2).sum(-1)
    true = (2 * (p0 * sw - target) * sw + (1 - 2 * p0) * sw2).mean().item()
    u = torch.rand(reps, K, device=dev, generator=gen)
    b = (u < p0).float()
    C = (b * w).sum(-1)
    st = (2 * (C - target) * sw)          # dL/dp with db_i/dp = 1
    # 'midpoint' ST: evaluate dL/db_i at b_i = 1/2 (exact for quadratic L): L(b_i=1)-L(b_i=0) = dL/db_i|_{b_i=1/2}
    mid = (2 * ((C[:, None] - b * w + 0.5 * w) - target) * w).sum(-1)
    print(f"K={K:>2} p={p0:4.2f} t={target:4.1f} true={true:8.2f} | ST-Bernoulli {st.mean().item():8.2f} (se {st.std().item()/math.sqrt(reps):.2f})"
          f" | midpoint-ST {mid.mean().item():8.2f} (se {mid.std().item()/math.sqrt(reps):.2f})")
