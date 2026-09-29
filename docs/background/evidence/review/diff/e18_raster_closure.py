"""E18: 'recompute-in-backward' rasteriser as a custom autograd.Function that receives geometry through a callable
(sdf(v, x), FieldComponent.from_module(nn.Module)). Parameters captured by the callable are not Function inputs."""
import torch
from torch.utils.checkpoint import checkpoint
dev = "cuda"
field = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(dev)   # neural field
radius = torch.tensor(0.5, device=dev, requires_grad=True)
x = torch.rand(4096, 3, device=dev) * 2 - 1
def sdf(r, pts):                     # user callable: analytic sphere + learned residual
    return pts.norm(dim=-1) - r + 0.05 * field(pts).squeeze(-1)
class RasterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pts, fn):
        ctx.fn = fn; ctx.save_for_backward(r, pts)
        with torch.no_grad():
            return torch.sigmoid(-fn(r, pts) / 0.05)
    @staticmethod
    def backward(ctx, go):
        r, pts = ctx.saved_tensors
        with torch.enable_grad():
            r_ = r.detach().requires_grad_()
            occ = torch.sigmoid(-ctx.fn(r_, pts) / 0.05)
            (gr,) = torch.autograd.grad(occ, r_, go)
        return gr, None, None
occ = RasterFn.apply(radius, x, sdf)
params = [radius] + list(field.parameters())
g = torch.autograd.grad(occ.sum(), params, allow_unused=True)
print("custom Function: d/dradius =", f"{g[0].item():.3f}", "| neural-field grads:", [None if t is None else float(t.abs().sum()) for t in g[1:]])
occ2 = checkpoint(lambda r, p: torch.sigmoid(-sdf(r, p) / 0.05), radius, x, use_reentrant=False)
g2 = torch.autograd.grad(occ2.sum(), params, allow_unused=True)
print("non-reentrant checkpoint: d/dradius =", f"{g2[0].item():.3f}", "| neural-field grads:", [round(float(t.abs().sum()), 3) for t in g2[1:]])
