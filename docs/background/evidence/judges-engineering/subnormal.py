import torch
for dev in ["cpu","cuda"]:
    a = torch.tensor(1e-7, dtype=torch.float32, device=dev)
    a6 = a**6
    a6b = a*a*a*a*a*a
    print(dev, "a**6 =", a6.item(), " chained mult =", a6b.item(), " (fp32 min normal 1.18e-38; min subnormal 1.4e-45)")
    x = torch.tensor(1e-24, dtype=torch.float32, device=dev)
    print(dev, "(1e-24)^2 =", (x*x).item())
