"""§8.2 item 10 / §9(a): sim.stream uses CUDA-graph capture of sampler+render and prefetches the next batch.
A captured graph writes into the SAME static output buffers on every replay. If the stream yields those buffers
(without a copy), a batch the user still holds is silently overwritten by the next replay."""
import torch
assert torch.cuda.is_available()
dev = "cuda"
static_seed = torch.zeros((), dtype=torch.int64, device=dev)
def render(seed):                       # stand-in for sampler+render: output depends on a device-side seed
    g = torch.arange(64 * 64, device=dev, dtype=torch.float32).reshape(1, 1, 64, 64)
    return torch.sin(g * 0.001 + seed.float())
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): render(static_seed)
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out = render(static_seed)

def stream_no_copy():
    for k in range(3):
        static_seed.fill_(k); graph.replay(); yield {"x": static_out}
def stream_copy():
    for k in range(3):
        static_seed.fill_(k); graph.replay(); yield {"x": static_out.clone()}

for name, it in [("yield static buffer", stream_no_copy()), ("yield clone", stream_copy())]:
    held = next(it)["x"]; v0 = held[0, 0, 0, 1].item()
    _ = next(it)                                         # consumer asks for (or prefetch produces) the next batch
    torch.cuda.synchronize()
    print(f"{name:22s}: first batch value before={v0:.6f}  after next replay={held[0,0,0,1].item():.6f}  "
          f"-> {'OVERWRITTEN' if held[0,0,0,1].item()!=v0 else 'intact'}")
