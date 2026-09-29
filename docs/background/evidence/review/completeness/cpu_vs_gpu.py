# Order-of-magnitude CPU vs GPU for two representative kernels (no claims beyond ratios).
import torch, time, math, os
torch.set_num_threads(os.cpu_count())
def bench(fn, dev, reps=5):
    fn();
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps): fn()
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps
print("threads", torch.get_num_threads(), "torch", torch.__version__)
for dev in ("cpu", "cuda"):
    # (1) multislice-like step: 64 images x 256^2, fft2 -> multiply -> ifft2, 32 slices
    x = torch.randn(64, 256, 256, dtype=torch.complex64, device=dev)
    H = torch.randn(256, 256, dtype=torch.complex64, device=dev)
    def march():
        y = x
        for _ in range(32):
            y = torch.fft.ifft2(torch.fft.fft2(y) * H)
        return y
    t1 = bench(march, dev, reps=2 if dev == "cpu" else 5)
    # (2) sparse emitters: B=64 x 50 emitters, 64^2 pupil -> 32^2 ROI via two matmuls (MFT)
    B, N, P, R = 64, 50, 64, 32
    pupil = torch.randn(B, N, P, P, dtype=torch.complex64, device=dev)
    Ay = torch.randn(R, P, dtype=torch.complex64, device=dev); Ax = torch.randn(P, R, dtype=torch.complex64, device=dev)
    def mft():
        f = Ay @ pupil @ Ax
        return (f.real**2 + f.imag**2).sum(1)
    t2 = bench(mft, dev)
    print(f"{dev}: 32-slice march 64x256^2: {t1*1e3:.1f} ms ({64/t1:.0f} img/s); sparse MFT 64x50 emitters: {t2*1e3:.2f} ms ({64/t2:.0f} img/s)")
