"""API-consistency checks for synthesis.md (API/DT reviewer).
1) Fidelity as declared (@dataclass(frozen=True, kw_only=True)) vs the positional calls used in §5.6/§9.
2) Learnable conditioning: synthesis 'value = T(scale*u)' vs proposal C 'value = T(u)*scale':
   Adam per-step physical move for several constraints, compared with the claimed 'about lr x scale'.
3) ANSI index list range(4,22) used in example (b): which Zernike modes it contains.
"""
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
import math, torch

# ---------- 1) Fidelity kw_only ----------
@dataclass(frozen=True, kw_only=True)
class Fidelity:
    preset: Literal["draft", "standard", "accurate", "reference", "legacy_dt"] = "standard"
    psf: str | None = None
    per_population: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

for call in ['Fidelity("standard", psf="vectorial")', 'Fidelity(preset="standard", psf="vectorial")']:
    try:
        eval(call); print(f"[1] {call:45s} -> OK")
    except TypeError as e:
        print(f"[1] {call:45s} -> TypeError: {e}")

# ---------- 2) Learnable conditioning ----------
def adam_move(value_fn, u0, lr=1e-2, steps=20):
    """Physical move per Adam step for a loss with constant sign of gradient (d loss/d value = 1)."""
    u = torch.tensor(float(u0), requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    v0 = value_fn(u).item()
    for _ in range(steps):
        opt.zero_grad(); value_fn(u).backward(); opt.step()
    return abs(value_fn(u).item() - v0) / steps, v0

sig = torch.sigmoid
cases = {
  # name: (scale, synthesis T(scale*u), proposal-C T(u)*scale, u0_syn, u0_C)
  "Zernike coeff, unconstrained, scale=0.01um":
     (0.01, lambda u: 0.01*u, lambda u: u*0.01, 0.0, 0.0),
  "Zernike coeff, bounds=(-0.1,0.1)um, scale=0.01um":
     (0.01, lambda u: -0.1 + 0.2*sig(0.01*u), lambda u: 0.01*(-10 + 20*sig(u)), 0.0, 0.0),
  "radius 0.30um, positive, scale=1um":
     (1.0, lambda u: torch.exp(1.0*u), lambda u: torch.exp(u)*1.0, math.log(0.30), math.log(0.30)),
  "photons 2000, positive(log), scale=1 (unit)":
     (1.0, lambda u: torch.exp(1.0*u), lambda u: torch.exp(u)*1.0, math.log(2000), math.log(2000)),
  "pixel-pupil phase, scale=0.1 rad, unconstrained":
     (0.1, lambda u: 0.1*u, lambda u: 0.1*u, 0.0, 0.0),
}
print("\n[2] Adam lr=1e-2: mean physical move per step (claim in §6.1: 'about lr x scale')")
for name, (scale, syn, C, u0s, u0c) in cases.items():
    ms, v0 = adam_move(syn, u0s); mc, _ = adam_move(C, u0c)
    print(f"    {name:52s} init={v0:9.4g}  claim={1e-2*scale:8.2e}  synthesis={ms:8.2e}  proposalC={mc:8.2e}")

# ---------- 3) ANSI range(4,22) ----------
def ansi_nm(j):
    n = int(math.ceil((-3 + math.sqrt(9 + 8*j)) / 2)); m = 2*j - n*(n+2); return n, m
names = {(0,0):"piston",(1,-1):"tilt y",(1,1):"tilt x",(2,-2):"oblique astig",(2,0):"DEFOCUS",(2,2):"vertical astig",
         (3,-1):"vert coma",(3,1):"horiz coma",(4,0):"primary spherical",(6,0):"secondary spherical"}
sel = list(range(4, 22))
print(f"\n[3] ansi=range(4,22): {len(sel)} terms")
print("    contains:", [(j, ansi_nm(j), names.get(ansi_nm(j), "")) for j in sel if ansi_nm(j) in names])
print("    j=3 (oblique astig) included?", 3 in sel, "| j=4 (defocus) included?", 4 in sel,
      "| j=24 (secondary spherical) included?", 24 in sel, "| last term:", ansi_nm(21))
