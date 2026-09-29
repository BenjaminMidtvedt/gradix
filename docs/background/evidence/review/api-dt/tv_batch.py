"""Does torchvision.transforms.v2 draw random params per call (whole batch) or per sample?
And do tv_tensors.KeyPoints on a padded [B, N, 2] table co-transform per sample?"""
import torch
try:
    import torchvision
    from torchvision.transforms import v2
    from torchvision import tv_tensors
except Exception as e:
    print("torchvision not available:", e); raise SystemExit
print("torchvision", torchvision.__version__)
torch.manual_seed(0)
B = 64
x = torch.arange(8.).repeat(B, 1, 8, 1)             # [B,1,8,8], each row 0..7 -> flip detectable
t = v2.RandomHorizontalFlip(p=0.5)
flipped = []
for trial in range(20):
    y = t(x)
    per = (y[:, 0, 0, 0] == 7).float()          # 1 if that sample was flipped
    flipped.append(per.mean().item())
print("fraction of batch flipped, per call (20 calls):", sorted(set(round(f, 3) for f in flipped)))
if hasattr(tv_tensors, "KeyPoints"):
    kp = tv_tensors.KeyPoints(torch.tensor([[[1., 2.], [3., 4.]]]).repeat(B, 1, 1), canvas_size=(8, 8))
    img = tv_tensors.Image(x)
    y, k = v2.RandomHorizontalFlip(p=1.0)(img, kp)
    print("KeyPoints [B,N,2] accepted; x of sample0 kp0 before/after:", kp[0,0,0].item(), k[0,0,0].item())
