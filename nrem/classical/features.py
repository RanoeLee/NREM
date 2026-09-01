"""Fixed label-free product descriptors for classical baselines.

The deep baselines consume windowed waveforms. Kernel, neighbourhood, and tree
baselines become prohibitively large if a 6 x 48,000 record is flattened. This
module therefore maps each complete product record to a deterministic set of
time- and frequency-domain summaries. No labels or test-set statistics enter
the representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


FEATURES_PER_CHANNEL = (
    "mean",
    "std",
    "median",
    "mad",
    "q05",
    "q25",
    "q75",
    "q95",
    "rms",
    "abs_mean",
    "peak_to_peak",
    "crest_factor",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "diff_rms",
    "lag1_autocorrelation",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_band_1",
    "spectral_band_2",
    "spectral_band_3",
    "spectral_band_4",
)


def _safe_float(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _channel_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(len(FEATURES_PER_CHANNEL), dtype=np.float32)

    mean = float(np.mean(x))
    std = float(np.std(x))
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    q05, q25, q75, q95 = np.quantile(x, [0.05, 0.25, 0.75, 0.95])
    rms = float(np.sqrt(np.mean(np.square(x))))
    abs_mean = float(np.mean(np.abs(x)))
    peak_to_peak = float(np.ptp(x))
    crest = float(np.max(np.abs(x)) / max(rms, 1e-12))
    skewness = _safe_float(stats.skew(x, bias=False)) if std > 1e-12 else 0.0
    kurtosis = _safe_float(stats.kurtosis(x, fisher=True, bias=False)) if std > 1e-12 else 0.0
    zero_cross = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if x.size > 1 else 0.0
    diff_rms = float(np.sqrt(np.mean(np.square(np.diff(x))))) if x.size > 1 else 0.0
    if x.size > 1 and std > 1e-12:
        lag1 = _safe_float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        lag1 = 0.0

    centered = x - mean
    power = np.square(np.abs(np.fft.rfft(centered)))
    total_power = float(power.sum())
    if power.size and total_power > 1e-20:
        freq = np.linspace(0.0, 1.0, power.size, dtype=np.float64)
        prob = power / total_power
        centroid = float(np.sum(freq * prob))
        entropy = float(-np.sum(prob * np.log(prob + 1e-20)) / np.log(max(prob.size, 2)))
        bands = []
        edges = np.linspace(0, power.size, 5, dtype=int)
        for left, right in zip(edges[:-1], edges[1:]):
            bands.append(float(power[left:right].sum() / total_power))
    else:
        centroid = entropy = 0.0
        bands = [0.0] * 4

    values = [
        mean, std, median, mad, q05, q25, q75, q95, rms, abs_mean,
        peak_to_peak, crest, skewness, kurtosis, zero_cross, diff_rms,
        lag1, centroid, entropy, *bands,
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), copy=False)


@dataclass(frozen=True)
class ProductFeatureExtractor:
    """Convert one complete `[channels, time]` record to a feature vector."""

    channels: int = 6

    @property
    def feature_names(self) -> list[str]:
        return [f"ch{channel}_{name}" for channel in range(self.channels)
                for name in FEATURES_PER_CHANNEL]

    @property
    def output_dim(self) -> int:
        return self.channels * len(FEATURES_PER_CHANNEL)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Expected a two-dimensional record, got {x.shape}")
        if x.shape[0] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[0]}")
        return np.concatenate([_channel_features(x[channel]) for channel in range(self.channels)])

    def transform(self, records: list[np.ndarray]) -> np.ndarray:
        if not records:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.stack([self.transform_one(record) for record in records]).astype(np.float32)
