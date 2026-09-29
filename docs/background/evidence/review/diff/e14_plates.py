"""E14: plate shapes as tabulated in sec 6.2 ('image' -> [B], '<population>' -> [B, N_max], 'frame' -> [B, T, N_max])
combined with ordinary torch broadcasting, e.g. the hierarchical prior of sec 6.2:
   radius ~ LogNormal(loc = Normal(...) [image plate], scale) inside a population [B, N_max].
Example (a): Binomial(8, .) -> N_max = 8; a batch of B = 8 makes the error silent."""
import torch
B, N = 8, 8
g = torch.Generator().manual_seed(0)
loc_img = torch.arange(B, dtype=torch.float32)            # per-image mean (image b has mean b)
eps = torch.zeros(B, N)                                   # zero scatter to expose the indexing
x = loc_img + 0.1 * eps                                   # right-aligned broadcasting [B] vs [B, N]
print("image 0 objects' means:", x[0].tolist(), "(should all be 0.0)")
print("image 5 objects' means:", x[5].tolist(), "(should all be 5.0)")
try:
    torch.zeros(B) + torch.zeros(B, 6)
except RuntimeError as e:
    print("with N_max = 6 it fails loudly:", str(e).splitlines()[0][:90])
# per-object [B, N] vs per-frame-per-object [B, T, N]
T = 8
v_obj = torch.arange(B * N, dtype=torch.float32).reshape(B, N)
v = v_obj + torch.zeros(B, T, N)
print("[B,N] + [B,T,N] with B == T:", "value for (b=1, t=0, n=0) =", v[1, 0, 0].item(), "(should be", v_obj[1, 0].item(), ")")
