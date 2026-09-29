import math, torch
dev="cuda"; lam0=0.68; k0=2*math.pi/lam0; ns, ng, NA = 1.33, 1.518, 1.45; kw=k0*ns
Np=512; u=torch.linspace(-NA,NA,Np,device=dev,dtype=torch.float64); du=(u[1]-u[0]).item()
UY,UX=torch.meshgrid(u,u,indexing="ij"); ur=torch.sqrt(UX**2+UY**2); inside=ur<=NA
kz=lambda n: k0*torch.sqrt((n*n-ur*ur).to(torch.complex128))
kzw,kzg=kz(ns),kz(ng)
print("max |exp(i kz_w Delta)| inside NA: ", {d: round(torch.exp(1j*kzw*d)[inside].abs().max().item(),1) for d in (-0.5,-1.0,-2.0)})
def psf(pupil, pad=4):
    P=torch.zeros(Np*pad,Np*pad,dtype=torch.complex128,device=dev); P[:Np,:Np]=torch.where(inside,pupil,0)
    I=torch.fft.fftshift(torch.fft.ifft2(P).abs()**2); c=Np*pad//2; I=I[c-48:c+48,c-48:c+48]; return I/I.sum()
for h in [0.0, 0.1, 0.5]:
    A=-1/(2*math.pi*kw*kzw)*torch.exp(1j*kzw*h)
    pa=torch.sqrt(kzw/kw)*A                        # literal plan rule
    pc=torch.sqrt(kzg/(k0*ng))*(2*kzw/(kzw+kzg))*A # immersion apodisation + scalar Fresnel t
    Ia,Ic=psf(pa),psf(pc)
    print(f"h={h:.1f} um: PSF rel-L2 literal vs t-corrected = {(torch.linalg.norm(Ia-Ic)/torch.linalg.norm(Ic)).item():.3f}; "
          f"peak ratio {Ia.max().item()/Ic.max().item():.3f}; pupil |p| max/median literal {(pa.abs()[inside].max()/pa.abs()[inside].median()).item():.1f} vs {(pc.abs()[inside].max()/pc.abs()[inside].median()).item():.1f}")
