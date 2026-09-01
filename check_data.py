#!/usr/bin/env python3
"""Validate that a directory satisfies the NREM NumPy data contract."""

from __future__ import annotations

import argparse

from nrem.data import FolderTimeSeriesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one sample and report its normalized NREM tensor shape."
    )
    parser.add_argument("--root", required=True, help="Directory containing .npy samples.")
    parser.add_argument("--max-len", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FolderTimeSeriesDataset(
        args.root,
        max_len=args.max_len,
        expected_channels=args.channels,
    )
    item = dataset[0]
    print(f"samples={len(dataset)}")
    print(f"shape={tuple(item['x'].shape)}")
    print(f"label={item['label']}")
    print(f"path={item['path']}")


if __name__ == "__main__":
    main()
