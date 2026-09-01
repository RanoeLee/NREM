from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_UNKNOWN = -1
LABEL_OK = 0
LABEL_NG = 1

def infer_label_from_path(path: Path) -> int:
    """Infer OK/NG label from directory names or filename tokens.

    Supports both directory-level labels, e.g. ``val/ok/*.npy`` and
    filename-level labels, e.g. ``ok_OK_000001_xxx.npy`` or
    ``ng_NG_000001_xxx.npy``.
    """
    ok_tokens = {'ok', 'normal', 'good', 'clean', 'healthy', 'pass'}
    ng_tokens = {'ng', 'bad', 'anomaly', 'abnormal', 'defect', 'fault', 'fail'}

    parts = {p.lower() for p in Path(path).parts}
    if parts & ng_tokens:
        return LABEL_NG
    if parts & ok_tokens:
        return LABEL_OK

    stem = Path(path).stem.lower()
    name = Path(path).name.lower()
    normalized = stem.replace('-', '_').replace('.', '_').replace(' ', '_')
    tokens = {t for t in normalized.split('_') if t}

    if tokens & ng_tokens:
        return LABEL_NG
    if tokens & ok_tokens:
        return LABEL_OK

    if name.startswith('ng_') or '_ng_' in name or '-ng-' in name:
        return LABEL_NG
    if name.startswith('ok_') or '_ok_' in name or '-ok-' in name:
        return LABEL_OK

    if any(k in name for k in ['abnormal', 'anomaly', 'defect', 'fault', 'fail']):
        return LABEL_NG
    if any(k in name for k in ['normal', 'healthy', 'good', 'pass']):
        return LABEL_OK

    return LABEL_UNKNOWN


def _load_npy(path: Path, mmap_mode=None) -> np.ndarray:
    arr = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f'Expected 1D/2D npy array, got shape {arr.shape} from {path}')
    return arr


def standardize_array_shape(arr: np.ndarray, transpose_if_tall: bool = True) -> np.ndarray:
    # Return [C, T]. If data is [T, C] and T>C, transpose.
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D array, got {arr.shape}')
    if transpose_if_tall and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return np.asarray(arr, dtype=np.float32)


def pad_or_crop(x: np.ndarray, max_len: int | None = None, pad_value: float = 0.0) -> np.ndarray:
    if max_len is None or max_len <= 0:
        return x
    c, t = x.shape
    if t == max_len:
        return x
    if t > max_len:
        return x[:, :max_len]
    out = np.full((c, max_len), pad_value, dtype=np.float32)
    out[:, :t] = x
    return out


class FolderTimeSeriesDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        exts: Sequence[str] = ('.npy',),
        recursive: bool = True,
        transpose_if_tall: bool = True,
        max_len: int | None = None,
        strict_checks: bool = True,
        expected_channels: int | None = None,
        npy_mmap_mode=None,
        normalize: str = 'none',
        norm_stats: Dict[str, np.ndarray] | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f'Dataset root does not exist: {self.root}')
        self.exts = tuple(exts)
        self.recursive = bool(recursive)
        self.transpose_if_tall = bool(transpose_if_tall)
        self.max_len = max_len
        self.strict_checks = bool(strict_checks)
        self.expected_channels = int(expected_channels) if expected_channels is not None else None
        self.npy_mmap_mode = npy_mmap_mode
        self.normalize = str(normalize).lower()
        self.norm_stats = norm_stats

        files: List[Path] = []
        iterator = self.root.rglob('*') if self.recursive else self.root.glob('*')
        for p in iterator:
            if p.is_file() and p.suffix.lower() in self.exts:
                files.append(p)
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError(f'No files with extensions {self.exts} found under {self.root}')

        self.labels = [infer_label_from_path(p) for p in self.files]

        if self.strict_checks and self.expected_channels is not None:
            bad = []
            for p in self.files:
                arr = standardize_array_shape(_load_npy(p, mmap_mode=self.npy_mmap_mode), self.transpose_if_tall)
                if arr.shape[0] != self.expected_channels:
                    bad.append((str(p), tuple(arr.shape)))
                    if len(bad) >= 10:
                        break
            if bad:
                msg = '\n'.join([f'  {path}: shape={shape}' for path, shape in bad])
                raise ValueError(
                    f'Dataset strict check failed: expected {self.expected_channels} channels, '
                    f'but found incompatible files under {self.root}:\n{msg}'
                )

    def __len__(self):
        return len(self.files)

    def _read_raw_unpadded(self, idx: int) -> np.ndarray:
        path = self.files[idx]
        arr = _load_npy(path, mmap_mode=self.npy_mmap_mode)
        x = standardize_array_shape(arr, transpose_if_tall=self.transpose_if_tall)
        if self.strict_checks and self.expected_channels is not None and x.shape[0] != self.expected_channels:
            raise ValueError(f'Expected {self.expected_channels} channels but got {x.shape[0]} from {path}')
        return x.astype(np.float32, copy=False)

    def _read(self, idx: int) -> np.ndarray:
        x = self._read_raw_unpadded(idx)
        # Normalization is applied before padding, so padding zeros do not affect
        # fitted statistics. Padded zeros stay zeros in normalized space.
        if self.normalize != 'none' and self.norm_stats is not None:
            if self.normalize == 'zscore':
                mean = self.norm_stats['mean'].reshape(-1, 1)
                std = self.norm_stats['std'].reshape(-1, 1)
                x = (x - mean) / np.maximum(std, 1e-6)
            elif self.normalize == 'minmax':
                mn = self.norm_stats['min'].reshape(-1, 1)
                mx = self.norm_stats['max'].reshape(-1, 1)
                x = (x - mn) / np.maximum(mx - mn, 1e-6)
            elif self.normalize in {'robust_mad', 'mad', 'median_mad'}:
                # Contamination-robust channel-wise normalization:
                #   x' = (x - median) / (1.4826 * MAD)
                # Statistics are estimated from the current training set only,
                # without labels, clean references, or contamination-rate estimates.
                median = self.norm_stats['median'].reshape(-1, 1)
                scale = self.norm_stats['robust_scale'].reshape(-1, 1)
                x = (x - median) / np.maximum(scale, 1e-6)
            else:
                raise ValueError(f'Unknown normalize: {self.normalize}')
        x = pad_or_crop(x, self.max_len)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        x = self._read(idx)
        return {
            'x': torch.from_numpy(x),
            'label': int(self.labels[idx]),
            'path': str(self.files[idx]),
        }


def pad_collate_fn(batch):
    xs = [b['x'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    paths = [b['path'] for b in batch]
    c = max(int(x.shape[0]) for x in xs)
    t = max(int(x.shape[1]) for x in xs)
    out = torch.zeros(len(xs), c, t, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, :x.shape[0], :x.shape[1]] = x
    return {'x': out, 'label': labels, 'path': paths}



def estimate_norm_stats(
    dataset: FolderTimeSeriesDataset,
    max_samples: int | None = None,
    robust_max_points_per_channel: int = 500_000,
    robust_seed: int = 2025,
) -> Dict[str, np.ndarray]:
    """Estimate normalization statistics on raw, unpadded signals.

    Padding zeros are deliberately excluded. Otherwise short sequences would
    bias min/max and mean/std toward zero and create artificial all-zero windows.

    In addition to mean/std/min/max, this function also estimates robust
    channel-wise median/MAD statistics for contaminated training:

        median_c = median(x_c)
        robust_scale_c = 1.4826 * median(|x_c - median_c|)

    Exact median/MAD over very long signals can be memory-heavy, so the robust
    statistics are estimated from a deterministic per-channel time-point sample.
    This uses the current training set only; it does not require clean data,
    labels, contamination ratios, or manually selected low-risk samples.
    """
    n = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    sums = None
    sums2 = None
    mn = None
    mx = None
    count = 0

    rng = np.random.default_rng(int(robust_seed))
    robust_samples: list[list[np.ndarray]] | None = None
    # Number of sampled time points per file. This keeps total robust samples
    # around robust_max_points_per_channel per channel.
    robust_max_points_per_channel = int(robust_max_points_per_channel or 0)
    if robust_max_points_per_channel > 0:
        per_file_points = max(1, int(np.ceil(robust_max_points_per_channel / max(n, 1))))
    else:
        per_file_points = 0

    for idx in range(n):
        x = dataset._read_raw_unpadded(idx)
        if sums is None:
            c = x.shape[0]
            sums = np.zeros(c, dtype=np.float64)
            sums2 = np.zeros(c, dtype=np.float64)
            mn = np.full(c, np.inf, dtype=np.float64)
            mx = np.full(c, -np.inf, dtype=np.float64)
            robust_samples = [[] for _ in range(c)]
        sums += x.sum(axis=1)
        sums2 += (x ** 2).sum(axis=1)
        mn = np.minimum(mn, x.min(axis=1))
        mx = np.maximum(mx, x.max(axis=1))
        count += x.shape[1]

        if robust_samples is not None and per_file_points > 0:
            t = x.shape[1]
            if t <= per_file_points:
                sel = np.arange(t)
            else:
                sel = rng.choice(t, size=per_file_points, replace=False)
            sampled = x[:, sel]
            for ch in range(sampled.shape[0]):
                robust_samples[ch].append(sampled[ch].astype(np.float32, copy=False))

    mean = sums / max(count, 1)
    var = np.maximum(sums2 / max(count, 1) - mean ** 2, 1e-12)
    std = np.sqrt(var)

    # Robust median/MAD. Fall back to mean/std when sampling is disabled.
    if robust_samples is not None and per_file_points > 0:
        med = np.zeros_like(mean, dtype=np.float64)
        mad = np.zeros_like(mean, dtype=np.float64)
        for ch, chunks in enumerate(robust_samples):
            vals = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
            med[ch] = np.median(vals)
            mad[ch] = np.median(np.abs(vals - med[ch]))
    else:
        med = mean.copy()
        mad = std / 1.4826

    robust_scale = 1.4826 * mad
    # Degenerate channels can have near-zero MAD; fall back to std for those.
    fallback = robust_scale < 1e-6
    robust_scale[fallback] = np.maximum(std[fallback], 1e-6)

    return {
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
        'min': mn.astype(np.float32),
        'max': mx.astype(np.float32),
        'median': med.astype(np.float32),
        'mad': mad.astype(np.float32),
        'robust_scale': robust_scale.astype(np.float32),
        'robust_sample_points_per_channel': np.asarray([per_file_points * n], dtype=np.int64),
        'robust_seed': np.asarray([int(robust_seed)], dtype=np.int64),
    }
