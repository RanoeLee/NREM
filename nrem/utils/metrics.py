from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class BinaryMetrics:
    thr: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    acc: float
    fpr: float
    tpr: float

    def to_dict(self):
        return asdict(self)


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> BinaryMetrics:
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores).astype(np.float64)
    pred = (s >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    tpr = recall
    return BinaryMetrics(float(thr), tp, fp, tn, fn, precision, recall, f1, acc, fpr, tpr)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores).astype(np.float64)
    if s.size == 0:
        return 0.0, threshold_metrics(y, s, 0.0)
    uniq = np.unique(s)
    thrs = np.quantile(s, np.linspace(0.01, 0.99, 199)) if uniq.shape[0] > 512 else uniq
    best_thr = float(thrs[0])
    best = threshold_metrics(y, s, best_thr)
    for thr in thrs[1:]:
        m = threshold_metrics(y, s, float(thr))
        if m.f1 > best.f1:
            best_thr = float(thr)
            best = m
    return best_thr, best


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind='mergesort')
    sorted_values = a[order]
    ranks = np.empty_like(order, dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores).astype(np.float64)
    if not np.isfinite(s).all():
        raise ValueError('scores must be finite')
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _rankdata(s) + 1.0
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores).astype(np.float64)
    if not np.isfinite(s).all():
        raise ValueError('scores must be finite')
    if y.sum() == 0:
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    group_end = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    end_indices = np.flatnonzero(group_end)
    group_positives = np.diff(np.r_[0, tp[end_indices]])
    precision = tp[end_indices] / (end_indices + 1)
    recall_delta = group_positives / int(y.sum())
    return float((precision * recall_delta).sum())


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, fixed_thr: Optional[float] = None) -> Dict:
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores).astype(np.float64)
    out = {
        'n': int(y.shape[0]),
        'positive': int((y == 1).sum()),
        'negative': int((y == 0).sum()),
        'positive_ratio': float((y == 1).mean()) if y.size else float('nan'),
        'auroc': roc_auc(y, s),
        'auprc': average_precision(y, s),
    }
    if fixed_thr is None:
        thr, best = best_f1_threshold(y, s)
        out['thr'] = float(thr)
        out['oracle_best_f1'] = float(best.f1)
        out.update({f'oracle_{k}': v for k, v in best.to_dict().items()})
    else:
        m = threshold_metrics(y, s, float(fixed_thr))
        out['thr'] = float(fixed_thr)
        out['f1_at_val_thr'] = float(m.f1)
        out.update(m.to_dict())
    return out
