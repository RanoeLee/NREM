#!/usr/bin/env python3
"""Reproduce the parameter/FLOP columns used in the NREM manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nrem.utils.complexity import profile_many


DEFAULT_SPECS = (
    ("NREM", "configs/release/NREM.yaml"),
    ("DeepSVDD", "configs/external/DeepSVDD_official.yaml"),
    ("TCN-AE", "configs/external/TCN_AE_canonical.yaml"),
    ("Transformer-AE", "configs/external/Transformer_AE.yaml"),
    ("TranAD", "configs/external/TranAD_official.yaml"),
    ("Anomaly Transformer", "configs/external/AnomalyTransformer_official.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile model parameters and inference FLOPs for one complete sample."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Profile a custom config; repeat as needed. Defaults to the six paper methods.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/complexity_profile.csv"))
    parser.add_argument("--json", type=Path, default=Path("results/complexity_profile.json"))
    return parser.parse_args()


def parse_specs(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_SPECS)
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        specs.append((label.strip(), path.strip()))
    return specs


def main() -> None:
    args = parse_args()
    results = profile_many(parse_specs(args.config), device=args.device, max_len=args.max_len)
    rows = [result.to_dict() for result in results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"{row['method']:<20} "
            f"params={row['parameters_millions']:.3f}M "
            f"FLOPs/sample={row['flops_giga_per_sample']:.2f}G"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
