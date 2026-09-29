"""Tutorials and example notebooks run (§12.7): their code cells execute in order.

Examples run in smoke mode (``GRADIX_SMOKE=1``): small shapes and few iterations.
"""

import json
import warnings
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "docs").rglob("*.ipynb"))


@pytest.fixture
def headless():
    """Draw figures off-screen: an interactive backend would block the suite on plt.show()."""
    try:
        import matplotlib
    except ImportError:  # notebooks that never plot need no backend
        yield
        return
    matplotlib.use("Agg", force=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
        yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path, capsys, headless, monkeypatch):
    monkeypatch.setenv("GRADIX_SMOKE", "1")  # examples shrink to small shapes and few steps
    notebook = json.loads(path.read_text(encoding="utf-8"))
    scope: dict = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name}[{i}]", "exec"), scope)
    capsys.readouterr()
