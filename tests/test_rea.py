import pytest
import torch

from nrem.modules.rea import ResidualEvidenceAggregator


def _inputs():
    u1 = torch.tensor([[0.2, 0.8, 1.4]], dtype=torch.float32)
    u2 = torch.tensor([[0.1, 1.2, 2.0]], dtype=torch.float32)
    ref1 = torch.tensor([[0.1, 0.5, 1.0], [0.2, 0.7, 1.3]], dtype=torch.float32)
    ref2 = torch.tensor([[0.1, 0.9, 1.6], [0.3, 1.1, 1.8]], dtype=torch.float32)
    return [u1, u2], [ref1, ref2]


def test_equal_weights_reproduce_unweighted_two_stream_rea():
    energies, references = _inputs()
    rea = ResidualEvidenceAggregator(cauchy_clip=2.0)
    unweighted = rea(energies, references)
    weighted = rea(energies, references, stream_weights=[0.5, 0.5])
    torch.testing.assert_close(weighted, unweighted)


def test_weights_are_normalized_internally():
    energies, references = _inputs()
    rea = ResidualEvidenceAggregator(cauchy_clip=2.0)
    torch.testing.assert_close(
        rea(energies, references, stream_weights=[2.0, 8.0]),
        rea(energies, references, stream_weights=[0.2, 0.8]),
    )


@pytest.mark.parametrize("weights", [[1.0], [0.0, 0.0], [-1.0, 2.0]])
def test_invalid_stream_weights_are_rejected(weights):
    energies, references = _inputs()
    rea = ResidualEvidenceAggregator()
    with pytest.raises(ValueError, match="weight"):
        rea(energies, references, stream_weights=weights)
