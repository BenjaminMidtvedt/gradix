import torch
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
out = {}
def f(): out["x"] = torch.randn(4, device=dev, generator=g)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): f()
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr): f()
gr.replay(); a = out["x"].clone(); gr.replay(); b = out["x"].clone()
print("replays differ (fresh numbers from custom generator):", not torch.equal(a, b))
g2 = torch.Generator(device=dev).manual_seed(0)
for _ in range(2): torch.randn(4, device=dev, generator=g2)
ref = [torch.randn(4, device=dev, generator=g2) for _ in range(3)]
print("replay stream reproducible vs eager draws:", any(torch.equal(b, r) for r in ref), any(torch.equal(a, r) for r in ref))
