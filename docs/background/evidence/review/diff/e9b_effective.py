"""E9b: which precision does a complex64 matmul actually use inside ieee_matmul() when user code set a legacy flag?"""
import sys, torch
dev = "cuda"
torch.manual_seed(0)
A = torch.randn(256, 512, dtype=torch.complex64, device=dev); B = torch.randn(512, 256, dtype=torch.complex64, device=dev)
ref = (A.to(torch.complex128) @ B.to(torch.complex128))
def err(): return ((A @ B).to(torch.complex128) - ref).abs().pow(2).sum().sqrt().item() / ref.abs().pow(2).sum().sqrt().item()
mode = sys.argv[1]
if mode == "legacy_true": torch.backends.cuda.matmul.allow_tf32 = True
if mode == "setprec_high": torch.set_float32_matmul_precision("high")
if mode == "new_tf32": torch.backends.cuda.matmul.fp32_precision = "tf32"
e_out = err()
old = torch.backends.cuda.matmul.fp32_precision
torch.backends.cuda.matmul.fp32_precision = "ieee"
e_in = err()
torch.backends.cuda.matmul.fp32_precision = old
print(f"user setting {mode:<14}: rel err outside ctx {e_out:.2e}, inside ieee_matmul() {e_in:.2e}, after {err():.2e}")
