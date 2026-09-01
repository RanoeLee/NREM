from pathlib import Path

import pytest
import torch

from nrem.config import load_config
from nrem.methods.factory import build_method


@pytest.mark.parametrize("name", ("Benchmark", "SRE", "CTM", "NREM"))
def test_release_config_builds_and_scores(name):
    path = Path("configs/release") / f"{name}.yaml"
    cfg = load_config(path)
    assert cfg["experiment"]["name"] == name
    assert cfg["method"]["name"] == "nrem"

    model = build_method(cfg).eval()
    sample = torch.zeros(1, 6, 2048)
    with torch.inference_mode():
        output = model.forward({"x": sample}, training=False)
    assert output.sample_score.shape == (1,)
    assert torch.isfinite(output.sample_score).all()


def test_final_nrem_configuration_matches_reported_method():
    cfg = load_config(Path("configs/release/NREM.yaml"))
    assert cfg["nrem"]["backbone"] == "mlp"
    assert cfg["nrem"]["mask_ratio"] == pytest.approx(0.30)
    assert cfg["nrem"]["resa_energy"] == ["u1", "u2"]
    assert cfg["nrem"]["resa_stream_weights"] == pytest.approx([0.20, 0.80])

    model = build_method(cfg)
    assert model.model.resa_stream_weights == pytest.approx([0.20, 0.80])
