"""E13: custom autograd.Functions (recompute-in-backward rasteriser, custom S(theta) backward, Bessel with custom
backward) under forward-mode AD (torch.func.jacfwd / jvp), the cheap route to Fisher/CRLB for <= 10 parameters,
and under double backward (Gauss-Newton / Hessian-vector products for L-BFGS diagnostics)."""
import torch
from torch.func import jacfwd, jacrev
dev = "cuda"
class Raster(torch.autograd.Function):
    @staticmethod
    def forward(r, x):
        return torch.sigmoid((r - x.abs()) / 0.1)
    @staticmethod
    def setup_context(ctx, inputs, output):
        r, x = inputs; ctx.save_for_backward(r, x)
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, go):
        r, x = ctx.saved_tensors
        s = torch.sigmoid((r - x.abs()) / 0.1)
        return (go * s * (1 - s) / 0.1).sum().reshape(r.shape), None
x = torch.linspace(-1, 1, 64, device=dev)
def mu(params):  # expected image as a function of 2 parameters
    return params[1] * Raster.apply(params[0], x) + 1.0
p = torch.tensor([0.4, 100.0], device=dev)
for name, fn in (("jacrev (64 outputs -> 64 backward passes)", lambda: jacrev(mu)(p)),
                 ("jacfwd (2 params -> 2 JVPs)", lambda: jacfwd(mu)(p))):
    try:
        J = fn(); print(f"{name:<45} OK, J shape {tuple(J.shape)}")
    except Exception as e:
        print(f"{name:<45} {type(e).__name__}: {str(e).splitlines()[0][:120]}")
pp = p.clone().requires_grad_()
try:
    (g,) = torch.autograd.grad(mu(pp).square().sum(), pp, create_graph=True)
    g.sum().backward(); print("double backward: OK")
except Exception as e:
    print("double backward:", type(e).__name__, str(e).splitlines()[0][:120])
