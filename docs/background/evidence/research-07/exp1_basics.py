import torch, time, math, warnings
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
dev = "cuda"

# 1. complex gradient convention
z = torch.tensor(1 + 1j, requires_grad=True)
(z.abs() ** 2).backward()
print("grad of |z|^2 at 1+1j:", z.grad)

# 2. poisson gradient
lam = torch.tensor([10.0, 100.0], requires_grad=True)
y = torch.poisson(lam)
y.sum().backward()
print("poisson grad:", lam.grad)
from torch.distributions import Poisson, Gamma, Beta
print("Poisson.has_rsample", Poisson.has_rsample, "Gamma", Gamma.has_rsample, "Beta", Beta.has_rsample)

# 3. normal(mean,std) gradient
m = torch.tensor(1.0, requires_grad=True)
s = torch.tensor(2.0, requires_grad=True)
torch.normal(m.expand(3), s.expand(3)).sum().backward()
print("torch.normal grads:", m.grad, s.grad)

# 4. grid_sample complex
x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
g = torch.zeros(1, 4, 4, 2)
try:
    torch.nn.functional.grid_sample(x, g, align_corners=False)
    print("grid_sample complex: OK")
except Exception as e:
    print("grid_sample complex error:", str(e)[:120])

# 5. special functions complex / grad
for name in ["bessel_j0", "bessel_j1", "spherical_bessel_j0"]:
    f = getattr(torch.special, name)
    xr = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        f(xr).sum().backward()
        print(name, "real grad", xr.grad)
    except Exception as e:
        print(name, "grad error", str(e)[:100])
    try:
        f(torch.tensor([1.0 + 0.5j]))
        print(name, "complex OK")
    except Exception as e:
        print(name, "complex error", str(e)[:100])

# 6. chalf coverage
a = torch.randn(256, 256, dtype=torch.complex32, device=dev) if False else torch.randn(256, 256, device=dev, dtype=torch.complex64).to(torch.complex32)
for opname, fn in [("fft2", lambda t: torch.fft.fft2(t)), ("mul", lambda t: t * t), ("exp", lambda t: torch.exp(t)), ("abs", lambda t: t.abs()), ("sum", lambda t: t.sum()), ("add", lambda t: t + t), ("conj_mul", lambda t: t * t.conj())]:
    try:
        r = fn(a); torch.cuda.synchronize()
        print("chalf", opname, "OK", r.dtype)
    except Exception as e:
        print("chalf", opname, "ERR", str(e)[:90])
try:
    torch.fft.fft2(torch.randn(300, 300, device=dev).half())
    print("half fft non-pow2 OK")
except Exception as e:
    print("half fft non-pow2 ERR", str(e)[:100])

# 7. cufft plan cache
print("cufft plan cache max", torch.backends.cuda.cufft_plan_cache.max_size)
print("fp32 precision matmul", torch.backends.cuda.matmul.fp32_precision, "cudnn conv", torch.backends.cudnn.conv.fp32_precision if hasattr(torch.backends.cudnn, "conv") else "n/a")
