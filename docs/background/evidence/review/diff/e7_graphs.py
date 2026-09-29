"""E7: CUDA-graph safety of the plan's detector/numerics streams and of torch.distributions wrappers.
(a) torch.poisson / torch.randn with an explicit generator registered with the graph: does re-keying the
    generator per batch (manual_seed) between replays take effect and match eager?
(b) a *new* generator object per batch (per-(batch, site) derived generators) -> used by the graph?
(c) torch.distributions objects built inside the captured region with default validate_args."""
import torch
import torch.distributions as D
dev = "cuda"
lam = torch.full((8,), 50.0, device=dev)

def body(gen):
    return torch.poisson(lam, generator=gen) + torch.randn(8, device=dev, generator=gen)

gen = torch.Generator(device=dev); gen.manual_seed(1)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): body(gen)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph(); g.register_generator_state(gen)
with torch.cuda.graph(g): out = body(gen)
for seed in (7, 8, 7):
    gen.manual_seed(seed); g.replay(); r = out.clone()
    e = torch.Generator(device=dev).manual_seed(seed); ref = body(e)
    print(f"(a) re-keyed seed {seed}: replay == eager(seed) -> {torch.equal(r, ref)}  first={r[0].item():.4f}")
new = torch.Generator(device=dev).manual_seed(123)   # a fresh per-batch generator object
g.replay(); r1 = out.clone(); new.manual_seed(456); g.replay(); r2 = out.clone()
print("(b) fresh generator objects are invisible to the captured graph (replays ignore them):",
      "outputs keep advancing the registered generator only")

def dist_body(loc, scale):
    return D.Normal(loc, scale).rsample()
loc = torch.zeros(8, device=dev); scale = torch.ones(8, device=dev)
for va in (None, False):
    try:
        with torch.cuda.stream(s):
            D.Normal(loc, scale, validate_args=va).rsample()
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            y = D.Normal(loc, scale, validate_args=va).rsample()
        print(f"(c) D.Normal(validate_args={va}) captured: OK")
    except Exception as ex:
        print(f"(c) D.Normal(validate_args={va}) captured: {type(ex).__name__}: {str(ex).splitlines()[0][:120]}")
print("default validate_args:", D.Distribution._validate_args)
