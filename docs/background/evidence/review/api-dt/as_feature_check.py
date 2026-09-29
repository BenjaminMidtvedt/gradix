"""§10.2 item 4: sim.as_feature() 'serves per-sample slices from an internally batched render'.
DT training idiom (DTDV431/DTEx252): per sample -> loss -> backward -> optimizer.step().
Check what happens when slices of ONE batched render (sharing one autograd graph) are consumed that way."""
import torch
theta = torch.nn.Parameter(torch.tensor(1.0))            # e.g. a learnable radius / Zernike coefficient
opt = torch.optim.Adam([theta], lr=0.1)
class AsFeature:                                         # minimal model of the described behaviour
    def __init__(self, B): self.B, self.buf, self.k = B, None, 0
    def __call__(self):
        if self.buf is None or self.k == self.B:
            base = torch.linspace(0, 1, 16)
            self.buf = torch.exp(-(base[None] - 0.5) ** 2 / theta ** 2).repeat(self.B, 1) * 1.0  # one graph for B
            self.buf = self.buf * torch.arange(1, self.B + 1)[:, None]
            self.k, self.theta_at_render = 0, theta.item()
        out = self.buf[self.k]; self.k += 1
        return out
feat = AsFeature(B=8)
for step in range(3):
    img = feat()
    loss = (img - 0.3).pow(2).mean()
    try:
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"step {step}: OK  (image rendered with theta={feat.theta_at_render:.3f}, current theta={theta.item():.3f})")
    except RuntimeError as e:
        print(f"step {step}: RuntimeError: {str(e).splitlines()[0][:110]}")
