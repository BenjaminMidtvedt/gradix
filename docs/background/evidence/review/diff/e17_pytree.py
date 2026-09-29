import dataclasses, typing, torch, inspect
print("public torch.utils.pytree:", hasattr(torch.utils, "pytree"))
try:
    import torch.utils.pytree as P
    print("  has register_dataclass:", hasattr(P, "register_dataclass"), "| register_pytree_node:", hasattr(P, "register_pytree_node"))
except Exception as e:
    print("  import torch.utils.pytree failed:", e)
import torch.utils._pytree as _P
print("_pytree.register_dataclass signature:", inspect.signature(_P.register_dataclass))
@dataclasses.dataclass(frozen=True)
class PlaneWaves:
    amplitude: torch.Tensor
    u: torch.Tensor
    travel: int = dataclasses.field(default=1, metadata={"static": True})
    z0: typing.Union[float, torch.Tensor] = 0.0
try:
    _P.register_dataclass(PlaneWaves)
    pw = PlaneWaves(torch.ones(2), torch.zeros(2, 2), 1, 0.5)
    leaves, spec = _P.tree_flatten(pw)
    print("flatten leaves:", [type(l).__name__ for l in leaves])
    pw2 = _P.tree_unflatten([l * 2 if isinstance(l, torch.Tensor) else l for l in leaves], spec)
    print("unflatten ok, frozen preserved:", dataclasses.is_dataclass(pw2), pw2.z0)
except Exception as e:
    print("register_dataclass failed:", type(e).__name__, str(e)[:200])
