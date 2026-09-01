from __future__ import annotations

import pytest

from nrem.methods import build_method


@pytest.mark.parametrize(
    "name", ("gdn", "mtgflow", "normfaae", "carla", "ocsvm", "lof")
)
def test_external15_trained_methods_are_available_to_the_common_evaluator(name):
    cfg = {
        "method": {"name": name},
        "data": {"channels": 6},
        "model": {"channels": 6},
        "nrem": {"window_size": 32, "stride": 16},
        "baseline": {"hidden_dim": 8, "latent_dim": 4, "topk": 2},
        "classical": {"n_neighbors": 3, "n_jobs": 1},
        "runtime": {"seed": 2025},
    }

    method = build_method(cfg)

    assert method.variant == name
