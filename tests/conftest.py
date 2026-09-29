"""Test-suite fixtures: every test starts from the same global random state.

gradix itself never touches global random state (the architecture gate enforces it); tests that
draw example inputs with ``torch.rand`` get reproducible values from this seed.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seeded_global_rng():
    state = torch.get_rng_state()
    torch.manual_seed(20260929)
    yield
    torch.set_rng_state(state)
