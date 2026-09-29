import torch, runpy, sys, io
ns = runpy.run_path("splitmix.py", run_name="x") if False else None
exec(open("splitmix.py").read().split("def bench")[0])
with torch.no_grad():
    u = uniforms()
    c = torch.corrcoef(u)                       # [B,B] image-image correlations over 1020 elements
    off = c[~torch.eye(B, dtype=bool, device=c.device)]
    print(f"image-pair corr: mean {off.mean().item():.1e}  std {off.std().item():.3f}  (iid expect std {1/U**0.5:.3f})")
    ct = torch.corrcoef(u.T); offt = ct[~torch.eye(U, dtype=bool, device=ct.device)]
    print(f"element-pair corr across images: mean {offt.mean().item():.1e} std {offt.std().item():.3f} (iid {1/B**0.5:.3f})")
