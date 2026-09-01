from pathlib import Path

import pytest

from nrem.utils.complexity import profile_many


PAPER_SPECS = (
    ("NREM", "configs/release/NREM.yaml"),
    ("DeepSVDD", "configs/external/DeepSVDD_official.yaml"),
    ("TCN-AE", "configs/external/TCN_AE_canonical.yaml"),
    ("Transformer-AE", "configs/external/Transformer_AE.yaml"),
    ("TranAD", "configs/external/TranAD_official.yaml"),
    ("Anomaly Transformer", "configs/external/AnomalyTransformer_official.yaml"),
)


@pytest.mark.parametrize(
    ("label", "config", "expected_params", "expected_flops_g"),
    (
        ("NREM", PAPER_SPECS[0][1], 13_489_088, 8.52),
        ("DeepSVDD", PAPER_SPECS[1][1], 852_608, 0.64),
        ("TCN-AE", PAPER_SPECS[2][1], 24_486, 4.66),
        ("Transformer-AE", PAPER_SPECS[3][1], 166_342, 68.25),
        ("TranAD", PAPER_SPECS[4][1], 68_108, 27.34),
        ("Anomaly Transformer", PAPER_SPECS[5][1], 15_306, 3.24),
    ),
)
def test_paper_complexity_values(label, config, expected_params, expected_flops_g):
    assert Path(config).is_file()
    result = profile_many(((label, config),))[0]
    assert result.parameters == expected_params
    assert round(result.flops_giga_per_sample, 2) == expected_flops_g
    assert result.input_shape == "1x6x48000"
