from __future__ import annotations

from pathlib import Path
from typing import List
import os
import random
import re
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> str:
    device = str(device).lower()
    if device == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def configure_threads(cfg: dict, device: str) -> None:
    rcfg = cfg.get('runtime', {}) or {}
    num_threads = rcfg.get('num_threads', None)
    num_interop = rcfg.get('num_interop_threads', None)
    try:
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
        elif device == 'cpu':
            torch.set_num_threads(min(8, os.cpu_count() or 8))
    except Exception:
        pass
    try:
        if num_interop is not None:
            torch.set_num_interop_threads(int(num_interop))
        elif device == 'cpu':
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    if device == 'cuda':
        try:
            torch.backends.cudnn.benchmark = bool(rcfg.get('cudnn_benchmark', True))
            allow_tf32 = bool(rcfg.get('allow_tf32', True))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except Exception:
            pass


def discover_contam_dirs(root: str, prefix: str = 'contam_') -> List[Path]:
    rp = Path(root).expanduser().resolve()
    if not rp.exists():
        raise FileNotFoundError(f'train_root does not exist: {rp}')
    dirs = [p for p in rp.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    def key(p: Path):
        m = re.search(r'(\d+)', p.name)
        return int(m.group(1)) if m else 10**9
    return sorted(dirs, key=key)


def parse_contam_value(name: str) -> float:
    m = re.search(r'(\d+(?:\.\d+)?)', str(name))
    return float(m.group(1)) if m else float('nan')


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')
