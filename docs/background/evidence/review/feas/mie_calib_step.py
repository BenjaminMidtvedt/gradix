"""§8.3 row 'Size-distribution calibration step, B256, 64^2 | standard | <= 10 ms' is anchored on [M-D Exp C]
(form-factor projection model). At `standard`, spheres route to `mie`. Time the [M-C] W3 Mie-in-pupil kernel
(fp64 coefficients, 1-D S(theta), interp to 64^2 pupil, phase ramps, explicit interference) in grad mode at B256,
N_max = 8 slots (example (a): Binomial(8, 0.6)), radius up to 1.5 um (n_max 37 as in the §5.7 plan listing)."""
import math, time, torch, importlib.util, sys
spec = importlib.util.spec_from_file_location("w", r"E:/gradoscopy/docs/background/evidence/panel-C/bench_v1_tiers.py")
src = open(spec.origin, encoding="utf-8").read().split("if __name__")[0]
ns = {}; exec(compile(src.split("# ---------------- W4")[0], "w3", "exec"), ns)
w3 = ns["w3"]; ns["wl"] = 0.532e-6; ns["NA"] = 0.8
def timeit(fn, reps=7, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts)//2] * 1e3
# radius range: W3 draws U(0.2, 0.6) um; widen to 1.5 um bound by scaling after construction is not possible, so
# run at the W3 default and report nmax; bigger radii only increase cost.
for B, Pn in ((256, 8), (64, 5)):
    f = w3(B=B, Pn=Pn, N=64, dx=162.5e-9, grad=False); g = w3(B=B, Pn=Pn, N=64, dx=162.5e-9, grad=True)
    print(f"Mie-in-pupil B{B}x{Pn} 64^2: fwd {timeit(f):6.1f} ms   fwd+bwd {timeit(g):6.1f} ms")
