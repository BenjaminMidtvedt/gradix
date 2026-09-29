"""The documented examples (§0.3, §9) parse, and every gx.* name they use exists or is planned."""

from pathlib import Path

import pytest

from gradix.testing.examples import check_document, python_blocks

PLAN = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"


def test_plan_examples_use_only_real_or_planned_names():
    problems = check_document(PLAN)
    assert not problems, "\n".join(problems)


def test_plan_has_examples():
    assert len(python_blocks(PLAN.read_text(encoding="utf-8"))) >= 20


def test_readme_example_runs(capsys):
    readme = (PLAN.parents[1] / "README.md").read_text(encoding="utf-8")
    for block in python_blocks(readme):
        exec(compile(block, "README.md", "exec"), {})
    assert "Plan" in capsys.readouterr().out


def test_the_checker_rejects_typos_in_planned_families_and_wrong_keywords():
    from gradix.testing import examples

    for typo in ("interact.Meeeie", "pupil.Zernikee"):
        with pytest.raises(KeyError):
            examples.resolve(typo)
    assert examples.resolve("pupil.Zernike") == "implemented"
    assert examples.resolve("pupil.PhaseRing") == "M4"
    bad = examples.keyword_problems("gx.Camera(pixel_size=6.5, shape=(8, 8), gian=2.0)")
    assert bad == ["gx.Camera(gian=…) has no parameter 'gian'"]
    assert examples.milestone_problems() == []
