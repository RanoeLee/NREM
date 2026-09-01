import torch

from nrem.methods.reconstruction_baselines import ReconstructionBaselineMethod


def _config(method: str) -> dict:
    return {
        "method": {"name": method},
        "data": {"channels": 6},
        "model": {"channels": 6},
        "nrem": {"window_size": 128, "stride": 64},
        "baseline": {"hidden_dim": 16, "mlp_hidden_dim": 32, "latent_dim": 8},
    }


def test_mlp_ae_accepts_multichannel_windows() -> None:
    method = ReconstructionBaselineMethod(_config("mlp_ae"))
    batch = {"x": torch.randn(2, 6, 256)}

    output = method(batch, training=True)

    assert output.loss is not None
    assert output.sample_score.shape == (2,)
    assert output.window_energy.shape == (2, 3)
    assert torch.isfinite(output.loss)


def test_transformer_ae_shape_contract() -> None:
    cfg = _config("transformer_ae")
    cfg["baseline"].update({"heads": 4, "layers": 1})
    method = ReconstructionBaselineMethod(cfg)
    batch = {"x": torch.randn(2, 6, 256)}

    output = method(batch, training=False)

    assert output.sample_score.shape == (2,)
    assert output.window_energy.shape == (2, 3)
