import torch, time
from torch.autograd import gradcheck, gradgradcheck
torch.manual_seed(0)

# gradcheck on complex function (float64/complex128 required)
H = torch.exp(1j * torch.rand(16, 16, dtype=torch.float64))
def prop(u, phi):
    return torch.fft.ifft2(torch.fft.fft2(u * torch.exp(1j * phi)) * H)
u = torch.randn(2, 16, 16, dtype=torch.complex128, requires_grad=True)
phi = torch.rand(2, 16, 16, dtype=torch.float64, requires_grad=True)
t = time.perf_counter(); ok = gradcheck(prop, (u, phi), fast_mode=False); print("gradcheck full", ok, f"{(time.perf_counter()-t):.2f}s")
t = time.perf_counter(); ok = gradcheck(prop, (u, phi), fast_mode=True); print("gradcheck fast", ok, f"{(time.perf_counter()-t):.2f}s")

# Fourier sub-pixel shift: gradient wrt position
N = 64
fx = torch.fft.fftfreq(N, dtype=torch.float64)
def render(pos, amp):
    # gaussian spot template in Fourier domain, shifted by pos (pixels)
    KX, KY = torch.meshgrid(fx, fx, indexing="ij")
    tmpl = torch.exp(-(KX**2 + KY**2) * (2 * torch.pi * 2.0) ** 2 / 2)  # sigma=2px
    ph = torch.exp(-2j * torch.pi * (KX[None] * pos[:, 0, None, None] + KY[None] * pos[:, 1, None, None]))
    F = (amp[:, None, None] * tmpl[None] * ph).sum(0)
    return torch.fft.ifft2(F).real
pos = torch.tensor([[20.3, 30.7], [40.1, 10.2]], dtype=torch.float64, requires_grad=True)
amp = torch.tensor([1.0, 0.5], dtype=torch.float64, requires_grad=True)
print("gradcheck fourier-shift render", gradcheck(render, (pos, amp)))

# complex grid_sample workaround: stack re/im as channels
import torch.nn.functional as F
z = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
grid = torch.rand(1, 5, 5, 2) * 2 - 1
zs = torch.cat([z.real, z.imag], 1)
out = F.grid_sample(zs, grid, align_corners=False)
zc = torch.complex(out[:, :1], out[:, 1:])
print("complex grid_sample via channels ok", zc.shape, zc.dtype)

# vmap randomness
from torch.func import vmap
def noisy(x):
    return x + torch.randn(())
try:
    vmap(noisy)(torch.zeros(3))
except Exception as e:
    print("vmap default randomness:", type(e).__name__, str(e)[:80])
print("vmap different:", vmap(noisy, randomness="different")(torch.zeros(3)))
print("vmap same:", vmap(noisy, randomness="same")(torch.zeros(3)))

# torch.poisson on MPS? (skip) ; Poisson ST estimators
lam = torch.tensor([5.0, 50.0, 500.0], requires_grad=True)
n = torch.poisson(lam.detach())
st_plain = lam + (n - lam).detach()
eps = ((n - lam) / lam.sqrt()).detach()
st_scaled = lam + lam.sqrt() * eps
g1 = torch.autograd.grad(st_plain.sum(), lam)[0]
g2 = torch.autograd.grad(st_scaled.sum(), lam)[0]
print("ST plain grad", g1, "ST scaled (Gaussian-reparam-consistent) grad", g2, "values equal:", torch.allclose(st_plain, st_scaled))
