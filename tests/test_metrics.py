from __future__ import annotations

import numpy as np
import pytest

from nrem.utils.metrics import average_precision, roc_auc


def test_binary_ranking_metrics_treat_equal_scores_as_ties():
    labels = np.array([1, 0])
    scores = np.array([0.0, 0.0])

    assert roc_auc(labels, scores) == pytest.approx(0.5)
    assert average_precision(labels, scores) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [roc_auc, average_precision])
def test_binary_ranking_metrics_reject_nonfinite_scores(metric):
    labels = np.array([0, 1])
    scores = np.array([0.0, np.nan])

    with pytest.raises(ValueError, match="finite"):
        metric(labels, scores)
