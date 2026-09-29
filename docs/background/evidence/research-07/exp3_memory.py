import torch, time, math
from torch.utils.checkpoint import checkpoint
dev = "cuda"
torch.manual_seed(0)
B, N, S = 16, 512, 32
lam, dx, dz = 0.5, 0.1, 0.2
fx = torch.fft.fftfreq(N, d=dx, device=dev)
KX, KY = torch.meshgrid(fx, fx, indexing="ij")
kz = torch.sqrt(torch.clamp(1 / lam**2 - KX**2 - KY**2, min=0))
H = torch.exp(2j * torch.pi * (kz - 1 / lam) * dz).to(torch.complex64)  # carrier removed
field_mb = B * N * N * 8 / 2**20

def step(u, phi):
    return torch.fft.ifft2(torch.fft.fft2(u * torch.exp(1j * phi)) * H)

def naive(u, phis):
    for k in range(phis.shape[1]):
        u = step(u, phis[:, k])
    return u

def ckpt(u, phis, seg):
    def run(u, p):
        for k in range(p.shape[1]):
            u = step(u, p[:, k])
        return u
    for s0 in range(0, phis.shape[1], seg):
        u = checkpoint(run, u, phis[:, s0:s0 + seg], use_reentrant=False)
    return u

class MultisliceAdjoint(torch.autograd.Function):
    """Phase-only multislice, O(1) activation memory: u_k recomputed by
    back-propagating the output field (valid when |t|=1 and H is unitary on its support)."""
    @staticmethod
    def forward(u0, phis):
        u = u0
        for k in range(phis.shape[1]):
            u = torch.fft.ifft2(torch.fft.fft2(u * torch.exp(1j * phis[:, k])) * H)
        return u
    @staticmethod
    def setup_context(ctx, inputs, output):
        u0, phis = inputs
        ctx.save_for_backward(phis, output)
    @staticmethod
    def backward(ctx, g):
        phis, u = ctx.saved_tensors
        Hc = H.conj()
        gphi = torch.empty_like(phis)
        lam_ = g  # adjoint field (PyTorch grad convention)
        for k in reversed(range(phis.shape[1])):
            t = torch.exp(1j * phis[:, k])
            # undo propagation: v_k = t_k * u_k  = P^H u_{k+1}
            v = torch.fft.ifft2(torch.fft.fft2(u) * Hc)
            lv = torch.fft.ifft2(torch.fft.fft2(lam_) * Hc)   # adjoint through P
            # v = u_k * t ; phi real: dL/dphi = Re(conj(lv) * d v/dphi) = Re(conj(lv) * 1j * v)
            gphi[:, k] = (lv.conj() * 1j * v).real
            lam_ = lv * t.conj()
            u = v * t.conj()
        return lam_, gphi

def measure(name, fn, u0, phis):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    out = fn(u0, phis)
    loss = (out.abs().pow(2) * W).mean()
    loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1e3
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    g = phis.grad.clone(); phis.grad = None
    print(f"{name:28s} peak {peak:8.0f} MB  ({peak/field_mb:6.1f} fields, {peak/field_mb/S:5.2f}/slice)  {dt:7.1f} ms")
    return g

u0 = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
W = torch.rand(N, N, device=dev)
phis = (torch.rand(B, S, N, N, device=dev) * 0.3).requires_grad_()
print(f"B={B} N={N} S={S}; one field = {field_mb:.0f} MB; phis = {phis.numel()*4/2**20:.0f} MB")
for _ in range(2):
    g0 = measure("naive autograd", naive, u0, phis)
g1 = measure("checkpoint seg=sqrt(S)", lambda u, p: ckpt(u, p, int(math.sqrt(S))), u0, phis)
g1 = measure("checkpoint seg=sqrt(S)", lambda u, p: ckpt(u, p, int(math.sqrt(S))), u0, phis)
g2 = measure("custom adjoint (reversible)", lambda u, p: MultisliceAdjoint.apply(u, p), u0, phis)
print("rel diff ckpt", ((g1 - g0).norm() / g0.norm()).item(), "adjoint", ((g2 - g0).norm() / g0.norm()).item())
del g0, g1, g2

# torch.compile attempt
import warnings
warnings.simplefilter("ignore")
try:
    import torch._dynamo
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(naive)(u0, phis.detach())
    print("dynamo explain: graphs", explanation.graph_count, "graph breaks", explanation.graph_break_count)
    cnaive = torch.compile(naive, backend="aot_eager")
    for i in range(3):
        measure(f"compiled naive (iter {i})", cnaive, u0, phis)
except Exception as e:
    print("compile ERR", type(e).__name__, str(e)[:300])

# real-view formulation: split re/im, elementwise fused by inductor
def step_realview(ur, ui, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    vr = ur * c - ui * s
    vi = ur * s + ui * c
    V = torch.fft.fft2(torch.complex(vr, vi)) * H
    v = torch.fft.ifft2(V)
    return v.real, v.imag
def naive_rv(u, phis):
    ur, ui = u.real.contiguous(), u.imag.contiguous()
    for k in range(phis.shape[1]):
        ur, ui = step_realview(ur, ui, phis[:, k])
    return torch.complex(ur, ui)
try:
    torch._dynamo.reset()
    crv = torch.compile(naive_rv, backend="aot_eager")
    for i in range(3):
        measure(f"compiled real-view (iter {i})", crv, u0, phis)
    measure("eager real-view", naive_rv, u0, phis)
except Exception as e:
    print("compile rv ERR", type(e).__name__, str(e)[:300])
