import sys, torch
import torch.distributions as D
dev = "cuda"
va = {"none": None, "false": False}[sys.argv[1]]
loc = torch.zeros(8, device=dev); scale = torch.ones(8, device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    D.Normal(loc, scale, validate_args=va).rsample()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
try:
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        y = D.Normal(loc, scale, validate_args=va).rsample()
    print(f"D.Normal(validate_args={va}) inside capture: OK")
except Exception as ex:
    print(f"D.Normal(validate_args={va}) inside capture: {type(ex).__name__}: {str(ex).splitlines()[0][:160]}")
