"""E5: torch.utils.checkpoint(use_reentrant=False, preserve_rng_state=True) with draws from an EXPLICIT
torch.Generator inside the checkpointed region (plan: numerics/detector streams are explicit generators;
chunked reductions and procedural slices are checkpointed with preserve_rng_state=True)."""
import torch
from torch.utils.checkpoint import checkpoint
dev = "cuda"
theta = torch.tensor(2.0, device=dev, requires_grad=True)

def chunk(th, gen):
    # e.g. stochastic Abbe nodes (numerics stream) or scaled-ST Poisson (detector stream) inside a mode chunk
    eps = torch.randn(1000, device=dev, generator=gen)
    lam = th * (1 + 0.1 * eps).abs()
    n = torch.poisson(lam.detach(), generator=gen)
    s = lam.sqrt()
    y = lam + s * ((n - lam) / s).detach()
    return (y - 2.0).square().mean()

for label, use_ckpt in (("no checkpoint", False), ("checkpoint, explicit gen", True)):
    gen = torch.Generator(device=dev).manual_seed(0)
    if use_ckpt:
        loss = checkpoint(chunk, theta, gen, use_reentrant=False, preserve_rng_state=True)
    else:
        loss = chunk(theta, gen)
    state_after_fwd = gen.get_state().clone()
    (gr,) = torch.autograd.grad(loss, theta)
    moved = not torch.equal(state_after_fwd, gen.get_state())
    print(f"{label:<26} loss={loss.item():.6f} grad={gr.item():.6f}  generator advanced by backward: {moved}")

# counter-keyed draws are pure functions of (key, index) -> recompute-safe
def chunk_counter(th, key):
    idx = torch.arange(1000, device=dev, dtype=torch.int64)
    h = (idx * 0x9E3779B97F4A7C15 + key) & 0xFFFFFFFFFFFF
    u = (h.double() + 0.5) / 2.0**48
    eps = torch.special.ndtri(u).float()
    lam = th * (1 + 0.1 * eps).abs()
    return (lam - 2.0).square().mean()
key = torch.tensor(42, device=dev)
l1 = chunk_counter(theta, key); (g1,) = torch.autograd.grad(l1, theta)
l2 = checkpoint(chunk_counter, theta, key, use_reentrant=False); (g2,) = torch.autograd.grad(l2, theta)
print(f"counter-keyed draws: grad no-ckpt {g1.item():.6f} vs ckpt {g2.item():.6f}")
