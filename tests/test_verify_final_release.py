import csv
import hashlib
import json

import numpy as np
import pytest

from tools.verify_final_release import verify_release


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(tmp_path):
    root = tmp_path / "EXP_RESULTS"
    leaf = root / "NREM" / "REA" / "contam_20" / "seed_2034"
    leaf.mkdir(parents=True)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    np.savez(leaf / "scores.npz", scores=scores, labels=labels)
    (leaf / "metrics.json").write_text('{"auroc": 1.0, "auprc": 1.0}', encoding="utf-8")
    (leaf / "provenance.json").write_text(
        json.dumps({"task_key": "abc", "auroc": 1.0, "auprc": 1.0}), encoding="utf-8"
    )
    fields = ["task_key", "phase", "group", "method_or_variant", "contamination_ratio", "seed", "relative_path", "bytes", "sha256"]
    with (root / "MANIFEST.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for path in sorted(leaf.iterdir()):
            writer.writerow(
                {
                    "task_key": "abc", "phase": "reporting", "group": "main_chain",
                    "method_or_variant": "NREM", "contamination_ratio": "20", "seed": "2034",
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size, "sha256": digest(path),
                }
            )
    return root


def test_verify_release_checks_hashes_and_recomputes_metrics(tmp_path):
    root = make_bundle(tmp_path)
    report = verify_release(root, expected_counts=None)
    assert report == {"files": 3, "tasks": 1, "metrics_recomputed": 1}


def test_verify_release_rejects_tampered_file(tmp_path):
    root = make_bundle(tmp_path)
    (root / "NREM" / "REA" / "contam_20" / "seed_2034" / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify_release(root, expected_counts=None)


def test_verify_release_rejects_metric_mismatch(tmp_path):
    root = make_bundle(tmp_path)
    metrics = root / "NREM" / "REA" / "contam_20" / "seed_2034" / "metrics.json"
    metrics.write_text('{"auroc": 0.5, "auprc": 1.0}', encoding="utf-8")
    manifest = root / "MANIFEST.tsv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
    for row in rows:
        if row["relative_path"].endswith("metrics.json"):
            row["bytes"] = str(metrics.stat().st_size)
            row["sha256"] = digest(metrics)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="metric mismatch"):
        verify_release(root, expected_counts=None)
