"""Architecture (§3.2, §11.13): the layering contract and the source bans on randomness.

Layers may import only what the table allows. Element packages may use each other's data types
(the imaging elements hold a ``gx.Objective`` and a ``gx.Camera``), as labels and coordinates do.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "gradix"

GROUPS = {
    "L0": (
        "gradix._version",
        "gradix.units",
        "gradix.conventions",
        "gradix._core",
        "gradix.special",
        "gradix.schema",
        "gradix.tree",
        "gradix.register",
        "gradix.registry",
        "gradix.sampling",
    ),
    "L1": ("gradix.ops",),
    "objects": (
        "gradix.objects",
        "gradix.devices",
        "gradix.env",
        "gradix.dipole",
        "gradix.acq",
        "gradix.materials",
    ),
    "lower": ("gradix.lower",),
    "elements": (
        "gradix.light",
        "gradix.interact",
        "gradix.excite",
        "gradix.imaging",
        "gradix.detect",
        "gradix.optics",
        "gradix.noise",
        "gradix.pupil",
    ),
    "labels": ("gradix.labels", "gradix.coords"),
    "L3": ("gradix.compose", "gradix.out"),
    "L4": ("gradix.planner", "gradix.containers", "gradix.presets", "gradix.api"),
    "testing": ("gradix.testing", "gradix.accel", "gradix.experimental"),
}
BELOW = ["L0", "L1", "objects", "lower", "elements", "labels", "L3", "L4"]
ALLOWED = {group: set(BELOW[: i + 1]) for i, group in enumerate(BELOW)}
ALLOWED["testing"] = {*GROUPS, "root"}  # harnesses may resolve the public namespace
ALLOWED["root"] = set(GROUPS)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _group(module: str) -> str | None:
    if module == "gradix":
        return "root"
    best = None
    for group, prefixes in GROUPS.items():
        for prefix in prefixes:
            if (module == prefix or module.startswith(prefix + ".")) and (
                best is None or len(prefix) > len(best[1])
            ):
                best = (group, prefix)
    return None if best is None else best[0]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    known = {_module_name(p) for p in SRC.rglob("*.py")}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                sub = f"{node.module}.{alias.name}"
                yield node.lineno, sub if sub in known else node.module


FILES = sorted(SRC.rglob("*.py"))


def test_every_module_belongs_to_a_layer():
    missing = [_module_name(p) for p in FILES if _group(_module_name(p)) is None]
    assert not missing, missing


@pytest.mark.parametrize("path", FILES, ids=lambda p: _module_name(p))
def test_layering(path):
    here = _group(_module_name(path))
    assert here is not None
    for line, target in _imports(path):
        if not target.startswith("gradix"):
            continue
        there = _group(target)
        assert there in ALLOWED[here], (
            f"{_module_name(path)}:{line} ({here}) imports {target} ({there})"
        )


BANNED = {
    "torch.manual_seed",
    "torch.seed",
    "torch.rand",
    "torch.randn",
    "torch.randint",
    "torch.randperm",
    "torch.rand_like",
    "torch.randn_like",
    "torch.randint_like",
    "torch.normal",
    "torch.poisson",
    "torch.bernoulli",
    "torch.binomial",
    "torch.multinomial",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
    "torch.cuda.seed",
    "torch.cuda.seed_all",
    "torch.random.manual_seed",
    "torch.random.seed",
    "torch.random.fork_rng",
    "torch._standard_gamma",
    "torch._sample_dirichlet",
}
"""Calls that use or reset global random state (§6.3: gradix owns no random state)."""
BANNED_PREFIXES = (
    "torch.distributions",
    "torch.nn.functional.dropout",
    "torch.nn.functional.alpha_dropout",
    "torch.nn.functional.feature_alpha_dropout",
    "torch.nn.Dropout",
    "torch.nn.AlphaDropout",
    "numpy.random",
    "random.",
)
BANNED_METHODS = {
    "normal_",
    "bernoulli_",
    "exponential_",
    "geometric_",
    "cauchy_",
    "log_normal_",
    "uniform_",
    "random_",
    "rsample",
}
SAMPLERS = BANNED - {n for n in BANNED if "seed" in n or n.endswith("fork_rng")}
"""Sampling calls that exempt modules may make, with an explicit generator."""
EXEMPT = {"gradix._core.keys", "gradix.ops.detect"}


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to the dotted names they stand for (``T`` → ``torch``, …)."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    names[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    names[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
    return names


def _dotted(node: ast.expr, aliases: dict[str, str]) -> str | None:
    chain: list[str] = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    root = aliases.get(node.id, node.id)
    return ".".join([root, *reversed(chain)])


def _banned(name: str) -> bool:
    return name in BANNED or name.startswith(BANNED_PREFIXES) or name == "random"


def _violations(path: Path, *, exempt: bool = False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            yield node.lineno, "relative import (use absolute gradix.* imports)"
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not exempt:
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [f"{node.module}.{a.name}" for a in node.names if node.module]
            )
            for name in modules:
                if name == "random" or name.startswith(("numpy.random", "torch.distributions")):
                    yield node.lineno, f"imports {name}"
        if isinstance(node, ast.Call):
            name = _dotted(node.func, aliases)
            generator = next((k for k in node.keywords if k.arg == "generator"), None)
            explicit = generator is not None and not (
                isinstance(generator.value, ast.Constant) and generator.value.value is None
            )
            if name is not None and _banned(name):
                if not (exempt and name in SAMPLERS and explicit):
                    yield node.lineno, name
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_METHODS:
                if not (exempt and explicit):
                    yield node.lineno, f".{node.func.attr}("


@pytest.mark.parametrize("path", FILES, ids=lambda p: _module_name(p))
def test_source_bans(path):
    found = list(_violations(path, exempt=_module_name(path) in EXEMPT))
    assert not found, f"{_module_name(path)}: {found}"


def test_the_ban_sees_through_aliases(tmp_path):
    probe = tmp_path / "probe.py"
    source = [
        "import torch as T",
        "import torch.nn.functional as F",
        "from torch import rand",
        "import numpy as np",
        "from . import sibling",
        "a = T.randn(3)",
        "b = rand(2)",
        "c = T.rand_like(a)",
        "T.cuda.manual_seed(0)",
        "T.random.manual_seed(0)",
        "d = F.dropout(a)",
        "e = np.random.default_rng()",
        "f = T.poisson(a, generator=None)",
    ]
    probe.write_text(chr(10).join(source) + chr(10), encoding="utf-8")
    found = {what for _line, what in _violations(probe)}
    assert {
        "torch.randn",
        "torch.rand",
        "torch.rand_like",
        "torch.cuda.manual_seed",
        "torch.random.manual_seed",
        "torch.nn.functional.dropout",
        "numpy.random.default_rng",
        "torch.poisson",
        "relative import (use absolute gradix.* imports)",
    } <= found
    exempt = {what for _line, what in _violations(probe, exempt=True)}
    assert "torch.poisson" in exempt  # generator=None is not an explicit generator


def test_package_ships_a_type_marker():
    assert (SRC / "py.typed").exists()
