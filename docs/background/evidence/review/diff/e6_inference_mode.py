"""E6: interplay of torch.inference_mode (plan sec 6.3: data generation runs under inference_mode automatically)
with (a) a network trained on the generated batch (example (a) stage 2); (b) caches hoisted during an
inference-mode call and reused in a later grad-mode call (P7 hoisting; alternating calibration/generation);
(c) custom autograd.Function saving an inference tensor; (d) CUDA-graph static outputs + prefetch aliasing."""
import torch
dev = "cuda"
net = torch.nn.Conv2d(1, 2, 3, padding=1).to(dev)

def attempt(label, fn):
    try:
        fn(); print(f"{label:<70} OK")
    except Exception as e:
        print(f"{label:<70} {type(e).__name__}: {str(e).splitlines()[0][:110]}")

with torch.inference_mode():
    x = torch.rand(4, 1, 32, 32, device=dev)          # simulator output
    pupil_cache = torch.rand(32, 32, device=dev)       # hoisted invariant
attempt("(a) net(x_inference).sum().backward()", lambda: net(x).sum().backward())
attempt("(a') net(x_inference.clone()) outside inference_mode", lambda: net(x.clone()).sum().backward())
attempt("(a'') x_inference.requires_grad_() (input-gradient use)", lambda: x.requires_grad_())
w = torch.nn.Parameter(torch.ones(32, 32, device=dev))
attempt("(b) grad-mode step reusing a cache created under inference_mode", lambda: (w * pupil_cache).sum().backward())

class Raster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, grid):
        ctx.save_for_backward(r, grid); return torch.sigmoid((r - grid) / 0.1)
    @staticmethod
    def backward(ctx, go):
        r, grid = ctx.saved_tensors; s = torch.sigmoid((r - grid) / 0.1)
        return (go * s * (1 - s) / 0.1).sum().reshape(r.shape), None
r = torch.tensor(0.5, device=dev, requires_grad=True)
attempt("(c) custom Function saving an inference-mode grid", lambda: Raster.apply(r, pupil_cache).sum().backward())

# (d) CUDA graph: output buffer is overwritten by the next replay (prefetch of batch k+1 while training on k)
static_in = torch.zeros(4, device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    out = static_in * 2
torch.cuda.current_stream().wait_stream(s)
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph):
    out = static_in * 2
static_in.fill_(1.0); gph.replay(); batch_k = out            # handed to the training loop without a copy
static_in.fill_(5.0); gph.replay()                            # prefetch of batch k+1
torch.cuda.synchronize()
print("(d) batch_k after next replay:", batch_k.tolist(), "(was [2,2,2,2])")
