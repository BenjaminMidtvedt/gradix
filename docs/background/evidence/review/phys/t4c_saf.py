import math, torch
dev="cuda"; lam0=0.68; k0=2*math.pi/lam0; ns, ng, NA = 1.33, 1.518, 1.45; kw=k0*ns
Np=4096; u=torch.linspace(-NA,NA,Np,device=dev,dtype=torch.float64); dk2=(k0*(u[1]-u[0]))**2
UY,UX=torch.meshgrid(u,u,indexing="ij"); ur=torch.sqrt(UX**2+UY**2); inside=ur<=NA
kz=lambda n: k0*torch.sqrt((n*n-ur*ur).to(torch.complex128)); kzw,kzg=kz(ns),kz(ng); ts=2*kzw/(kzw+kzg)
for h in [0.0, 0.05, 0.1, 0.2, 0.5]:
    A=-1/(2*math.pi*kw*kzw)*torch.exp(1j*kzw*h); F=(2*math.pi)**2*(kzg.real/k0)*(ts*A).abs()**2*dk2/(ns*4*math.pi/kw**2)
    tot=F[inside].sum().item(); sup=F[inside&(ur>ns)].sum().item()
    print(f"h={h:.2f} um: collected {tot:.3f}, supercritical share {sup/tot:.2f}")
