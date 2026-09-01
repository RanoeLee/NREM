#!/usr/bin/env python3
"""Verify hashes, task scope, and metrics in an NREM-Final artifact bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from tools.build_final_release import EXPECTED_COUNTS, sha256


def verify_release(
    root: Path,
    expected_counts: Mapping[tuple[str, str], int] | None = EXPECTED_COUNTS,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    root = root.resolve(strict=True)
    manifest = root / "MANIFEST.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("release manifest is empty")

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    task_identity: dict[str, tuple[str, str]] = {}
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden: {relative}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"size mismatch: {relative}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        by_task[row["task_key"]].append(row)
        identity = (row["phase"], row["group"])
        previous = task_identity.setdefault(row["task_key"], identity)
        if previous != identity:
            raise ValueError(f"inconsistent task identity: {row['task_key']}")

    if expected_counts is not None:
        observed = Counter(task_identity.values())
        if observed != Counter(expected_counts):
            raise ValueError(f"accepted task counts do not match: {dict(observed)}")

    recomputed = 0
    for task_key, task_rows in by_task.items():
        files = {PurePosixPath(row["relative_path"]).name: row for row in task_rows}
        for required in ("metrics.json", "scores.npz", "provenance.json"):
            if required not in files:
                raise ValueError(f"task {task_key} lacks {required}")
        metrics_path = root.joinpath(*PurePosixPath(files["metrics.json"]["relative_path"]).parts)
        scores_path = root.joinpath(*PurePosixPath(files["scores.npz"]["relative_path"]).parts)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(scores_path, allow_pickle=False) as archive:
            scores = np.asarray(archive["scores"], dtype=np.float64)
            labels = np.asarray(archive["labels"], dtype=np.int64)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
            raise ValueError(f"invalid score vector shape for {task_key}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite scores for {task_key}")
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        if abs(auroc - float(metrics["auroc"])) > tolerance or abs(auprc - float(metrics["auprc"])) > tolerance:
            raise ValueError(f"metric mismatch for {task_key}")
        recomputed += 1

    return {"files": len(rows), "tasks": len(by_task), "metrics_recomputed": recomputed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
