# Test 6 (GPU): band-limited soft voxelisation (Gaussian prefilter sigma = 0.3 h) - effect on images.
# 2-D slice of a labelled / refractive object: ellipse + small discs.  Compare images from the soft raster on grid h
# against the image of the exact continuous object (fine grid, area-exact indicator), sampled on the same grid.
import math, torch
dev = "cuda"
def indicator_fine(Mf, df):
    x = (torch.arange(Mf, device=dev, dtype=torch.float64) + 0.5)*df; Y, X = torch.meshgrid(x, x, indexing="ij")
    L = Mf*df; cx = cy = L/2
    ind = (((X-cx)/2.0)**2 + ((Y-cy)/0.8)**2 <= 1).double()
    for (px, py, r) in [(L/2+1.2, L/2+0.2, 0.15), (L/2-1.0, L/2-0.3, 0.25), (L/2+3.2, L/2+2.2, 0.35)]:
        ind = torch.clamp(ind + (((X-px)**2+(Y-py)**2) <= r*r).double(), max=1)
    return ind
def gauss_blur(img, sigma, d):
    M = img.shape[0]; f = torch.fft.fftfreq(M, d, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
    return torch.fft.ifft2(torch.fft.fft2(img)*torch.exp(-2*math.pi**2*sigma**2*(FX**2+FY**2))).real
def transfer(M, d, lam, NA, kind):
    f = torch.fft.fftfreq(M, d, device=dev, dtype=torch.float64); FY, FX = torch.meshgrid(f, f, indexing="ij")
    fr = torch.sqrt(FX**2+FY**2); fc = NA/lam
    if kind == "otf":   # in-focus incoherent OTF (paraxial)
        s = torch.clamp(fr/(2*fc), max=1); return (2/math.pi)*(torch.acos(s) - s*torch.sqrt(1-s*s))
    return (fr <= fc).double()   # coherent: pupil
def image(obj, d, lam, NA, kind):
    H = transfer(obj.shape[0], d, lam, NA, kind); X = torch.fft.fft2(obj)*H
    return torch.fft.ifft2(X).real if kind == "otf" else torch.fft.ifft2(X)
def case(label, lam, NA, h, kind, sigmas=(0.3, 0.5), N=128, sub=16):
    df = h/sub; Mf = N*sub
    ind = indicator_fine(Mf, df)
    ref_f = image(ind, df, lam, NA, kind)
    ref = ref_f[sub//2::sub, sub//2::sub]                         # exact continuous image at pixel centres
    out = []
    for sfac in sigmas:
        soft = gauss_blur(ind, sfac*h, df)[sub//2::sub, sub//2::sub]    # blurred-object sampled on grid h (s=1)
        img = image(soft, h, lam, NA, kind)
        out.append(f"sigma={sfac}h: rel-L2 {(torch.linalg.norm(img-ref)/torch.linalg.norm(ref - (ref.mean() if kind=='otf' else 0))).item():.3f}")
    # s = 4 supersampled raster then box-decimate (plan's reference for the edge)
    h4 = h/4; soft4 = gauss_blur(ind, 0.3*h4, df)[sub//8::sub//4, sub//8::sub//4]
    img4 = image(soft4, h4, lam, NA, kind)[1::4, 1::4] if False else image(soft4, h4, lam, NA, kind)
    img4 = img4.reshape(N, 4, N, 4)[:, 1:3, :, 1:3].mean(dim=(1, 3)) if False else img4[2::4, 2::4]
    out.append(f"s=4 raster: rel-L2 {(torch.linalg.norm(img4-ref)/torch.linalg.norm(ref-(ref.mean() if kind=='otf' else 0))).item():.3f}")
    print(f"{label} (h={h*1e3:.0f} nm): " + " | ".join(out))
lam, NA = 0.52, 1.4
case("fluorescence, grid = lambda/(4NA), contrast-normalised", lam, NA, lam/(4*NA), "otf")
lam, NA, NAi, n = 0.55, 0.95, 0.38, 1.33
case("coherent scattered field, standard band (NA+NAill)*1.1", lam, NA, lam/(2*1.1*(NA+NAi)), "ctf")
case("coherent scattered field, accurate band n_m", lam, NA, lam/(2*n), "ctf")
