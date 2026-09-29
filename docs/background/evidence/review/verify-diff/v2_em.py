import torch
dev="cuda"; torch.manual_seed(0)
G=100.0; M=4_000_000
for lam0 in (0.2, 1.0, 3.0):
    res={}
    for mode in ("double_where","double_where+ST0"):
        lam = torch.full((M,), lam0, device=dev, requires_grad=True)
        g = torch.Generator(device=dev).manual_seed(1)
        n = torch.poisson(lam.detach(), generator=g)
        lam_d = lam.detach()
        gf = torch.where(lam_d>0, (n+lam_d)/(2*lam_d.clamp_min(1e-30)), torch.ones_like(lam_d))
        ne = n + (lam - lam_d)*gf               # scaled-ST, integer-exact forward
        pos = n>0
        n_safe = torch.where(pos, ne, torch.ones_like(ne))
        em = torch.where(pos, G*torch._standard_gamma(n_safe, generator=g), torch.zeros_like(ne))
        if mode.endswith("ST0"):
            em = em + torch.where(pos, torch.zeros_like(ne), G*(ne - ne.detach()))
        em.sum().backward()
        res[mode]=(lam.grad.mean().item()/G, torch.isfinite(lam.grad).all().item())
    print(f"lam={lam0}: true dE[y]/dlam /G = 1 ; " + " ; ".join(f"{k}: {v[0]:.4f} finite={v[1]}" for k,v in res.items()), f"; predicted double-where 1-exp(-lam)/2 = {1-torch.exp(torch.tensor(-lam0)).item()/2:.4f}")
