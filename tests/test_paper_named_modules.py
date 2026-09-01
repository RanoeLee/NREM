from nrem.models.ctm import ContextualTrajectoryModel
from nrem.models.sre import SampleResidualEvidence
from nrem.modules.rea import ResidualEvidenceAggregator


def test_paper_named_modules_are_publicly_importable():
    assert SampleResidualEvidence.__name__ == "SampleResidualEvidence"
    assert ContextualTrajectoryModel.__name__ == "ContextualTrajectoryModel"
    assert ResidualEvidenceAggregator.__name__ == "ResidualEvidenceAggregator"
