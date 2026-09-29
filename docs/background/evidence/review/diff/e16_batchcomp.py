"""E16: detector noise from an explicit generator keyed per batch: is image i's noise independent of batch composition?"""
import torch
dev = "cuda"
H = 64
lam_img = {i: torch.full((H, H), 20.0 + i, device=dev) for i in range(8)}   # expected images of dataset items 0..7
def render_batch(indices, batch_key):
    g = torch.Generator(device=dev).manual_seed(batch_key)
    lam = torch.stack([lam_img[i] for i in indices])
    return torch.poisson(lam, generator=g)
a = render_batch([0, 1, 2, 3], batch_key=11)   # image 2 at position 2
b = render_batch([2, 5, 6, 7], batch_key=11)   # image 2 at position 0, same batch key
c = render_batch([4, 5, 2, 3], batch_key=11)   # image 2 at position 2, other neighbours
print("image 2 noise identical across compositions: pos2 vs pos0:", torch.equal(a[2], b[0]),
      "| pos2 vs pos2 (other neighbours, same key):", torch.equal(a[2], c[2]))
