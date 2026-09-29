import torch
for dev in ("cpu", "cuda"):
    for dt in (torch.float32, torch.float64):
        z = torch.tensor([-13.72, -6.28, -5.0, 6.2, 9.0], device=dev, dtype=dt)
        A = torch.special.ndtr(z)
        p = torch.tensor([1e-10, 1e-20, 1e-30, 1e-38, 1e-44], device=dev, dtype=dt)
        print(dev, dt, "ndtr:", A.tolist(), " ndtri:", torch.special.ndtri(p).tolist(),
              " log_ndtr(-13.72):", torch.special.log_ndtr(z[:1]).tolist())
