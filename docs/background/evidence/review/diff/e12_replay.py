"""E12: 'gs.replay(trace) re-renders bit-exactly ... including detector noise' with the default
(non-deterministic) atomic scatter-add of emitter ROIs. Render the same expected image twice with index_add_
(ROIs overlapping), then draw Poisson noise with the same generator key."""
import torch
dev = "cuda"
torch.manual_seed(0)
B, N, R, H = 64, 200, 32, 128
roi = torch.rand(B, N, R, R, device=dev) * 50.0              # per-emitter ROI photons
oy = torch.randint(0, H - R, (B, N), device=dev); ox = torch.randint(0, H - R, (B, N), device=dev)
ar = torch.arange(R, device=dev)
idx = (torch.arange(B, device=dev)[:, None, None, None] * H * H
       + (oy[:, :, None, None] + ar[None, None, :, None]) * H + (ox[:, :, None, None] + ar[None, None, None, :])).flatten()
def render():
    img = torch.zeros(B * H * H, device=dev)
    img.index_add_(0, idx, roi.flatten())
    return img.view(B, H, H) + 3.0
diff_lam, diff_n = 0, 0
for trial in range(20):
    a, b = render(), render()
    ga = torch.Generator(device=dev).manual_seed(7); gb = torch.Generator(device=dev).manual_seed(7)
    na, nb = torch.poisson(a, generator=ga), torch.poisson(b, generator=gb)
    diff_lam += (a != b).sum().item(); diff_n += (na != nb).sum().item()
print(f"over 20 re-renders of {B}x{H}x{H}: expected-image pixels differing (ulp) = {diff_lam}, "
      f"Poisson counts differing with identical key = {diff_n}")
torch.use_deterministic_algorithms(True)
a, b = render(), render()
print("deterministic mode: pixels differing =", (a != b).sum().item())
