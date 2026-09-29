"""The docstring gate: every public module, class, function and method passes numpydoc validation.

The checks and exclusions come from ``[tool.numpydoc_validation]`` in pyproject.toml. Validation
imports the package and inspects real signatures (``numpydoc.validate``), so the constructors
that ``@dataclass`` generates are checked against the class docstring's Parameters section.
"""

import importlib
import inspect
import pkgutil
import re
import sys
from pathlib import Path

from numpydoc.docscrape import get_doc_object
from numpydoc.validate import Validator, validate

import gradix

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: the gate must still run
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


class _Validator(Validator):
    """numpydoc's validator; tolerates modules without any def or class (numpydoc 1.11 raises)."""

    def __init__(self, obj_name):
        super().__init__(get_doc_object(Validator._load_obj(obj_name)))

    @property
    def source_file_def_line(self):
        try:
            return super().source_file_def_line
        except StopIteration:
            return 1


def _config():
    with open(ROOT / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)["tool"]["numpydoc_validation"]
    checks = set(config["checks"])
    if "all" in checks:
        from numpydoc.validate import ERROR_MSGS

        checks = set(ERROR_MSGS) - (checks - {"all"})
    return checks, [re.compile(p) for p in config.get("exclude", [])]


def _public_objects():
    """Yield ``(module, local_name)`` for every public module, class, function, method, property."""
    yield "gradix", ""
    for info in pkgutil.walk_packages(gradix.__path__, prefix="gradix."):
        module = importlib.import_module(info.name)
        yield info.name, ""
        for attr, obj in vars(module).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != info.name:
                continue
            if inspect.isclass(obj):
                yield info.name, attr
                for member, value in vars(obj).items():
                    if member.startswith("_") and member != "__call__":
                        continue
                    wrapped = (property, classmethod, staticmethod)
                    if inspect.isfunction(value) or isinstance(value, wrapped):
                        yield info.name, f"{attr}.{member}"
            elif inspect.isfunction(obj):
                yield info.name, attr


def test_numpydoc():
    checks, excludes = _config()
    problems = []
    for module, local in _public_objects():
        # The exclude patterns describe names inside a module (private helpers, magic methods).
        if local and any(p.search(f".{local}") for p in excludes):
            continue
        name = f"{module}.{local}" if local else module
        result = validate(name, validator_cls=_Validator)
        for code, message in result["errors"]:
            if code in checks:
                problems.append(f"{name}: {code} {message}")
    assert not problems, chr(10).join(problems)
