from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from evaluate import (
    attach_protocol_provenance,
    load_split_protocol,
    validate_eval_population,
)
from nrem.engine.runner import finalize_evaluation_metrics


FIELDS = (
    "split",
    "label",
    "source_relative_path",
    "destination_relative_path",
    "size_bytes",
    "sha256",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_protocol(tmp_path: Path) -> tuple[Path, Path]:
    split_root = tmp_path / "evaluation_split"
    rows = []
    for split, labels in {
        "val": {"ok": ("ok_a.npy", "ok-b"), "ng": ("ng_a.npy", "ng-b")},
        "test": {
            "ok": ("ok_b.npy", "ok-test-1"),
            "ng": ("ng_b.npy", "ng-test-1"),
        },
    }.items():
        for label, (name, content) in labels.items():
            path = split_root / split / label / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "source_relative_path": f"source/{label}/{name}",
                    "destination_relative_path": f"{split}/{label}/{name}",
                    "size_bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )

    manifest = split_root / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema": "nrem-evaluation-split-v1",
        "seed": 2025,
        "counts": {"val": {"ok": 1, "ng": 1}, "test": {"ok": 1, "ng": 1}},
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
    }
    (split_root / "split_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return split_root, manifest


def _load_and_validate(split_root: Path, manifest: Path, split: str = "val"):
    protocol = load_split_protocol(manifest, split)
    return validate_eval_population(
        split_root / split,
        protocol,
        expected_ok=1,
        expected_ng=1,
    )


def test_valid_population_returns_stable_protocol_provenance(tmp_path):
    split_root, manifest = _write_protocol(tmp_path)

    first = _load_and_validate(split_root, manifest)
    second = _load_and_validate(split_root, manifest)

    assert first == second
    assert first["eval_split"] == "val"
    assert first["eval_ok"] == 1
    assert first["eval_ng"] == 1
    assert first["eval_manifest_sha256"] == _sha256(manifest)
    assert len(first["eval_population_sha256"]) == 64


def test_load_rejects_manifest_that_no_longer_matches_metadata(tmp_path):
    _, manifest = _write_protocol(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest SHA-256"):
        load_split_protocol(manifest, "val")


def test_load_rejects_unknown_split(tmp_path):
    _, manifest = _write_protocol(tmp_path)

    with pytest.raises(ValueError, match="split.*holdout"):
        load_split_protocol(manifest, "holdout")


def test_validation_rejects_wrong_expected_class_counts(tmp_path):
    split_root, manifest = _write_protocol(tmp_path)
    protocol = load_split_protocol(manifest, "val")

    with pytest.raises(RuntimeError, match="expected OK=2, NG=1"):
        validate_eval_population(
            split_root / "val", protocol, expected_ok=2, expected_ng=1
        )


def test_validation_rejects_extra_file(tmp_path):
    split_root, manifest = _write_protocol(tmp_path)
    (split_root / "val" / "ok" / "extra.npy").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="unexpected evaluation files"):
        _load_and_validate(split_root, manifest)


def test_validation_rejects_missing_file(tmp_path):
    split_root, manifest = _write_protocol(tmp_path)
    (split_root / "val" / "ng" / "ng_a.npy").unlink()

    with pytest.raises(RuntimeError, match="(?i)missing evaluation files"):
        _load_and_validate(split_root, manifest)


def test_validation_rejects_changed_file_content(tmp_path):
    split_root, manifest = _write_protocol(tmp_path)
    (split_root / "val" / "ok" / "ok_a.npy").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="content mismatch"):
        _load_and_validate(split_root, manifest)


def test_load_rejects_duplicate_destination_membership(tmp_path):
    _, manifest = _write_protocol(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = manifest.with_name("split_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = _sha256(manifest)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate destination"):
        load_split_protocol(manifest, "val")


def test_metrics_receive_provenance_only_when_label_counts_match():
    provenance = {
        "eval_split": "test",
        "eval_manifest_sha256": "a" * 64,
        "eval_population_sha256": "b" * 64,
        "eval_ok": 3,
        "eval_ng": 3,
    }

    enriched = attach_protocol_provenance(
        {"n": 6, "positive": 3, "negative": 3, "auroc": 0.9}, provenance
    )

    assert enriched["eval_split"] == "test"
    assert enriched["eval_manifest_sha256"] == "a" * 64
    assert enriched["eval_population_sha256"] == "b" * 64
    assert enriched["eval_ok"] == 3
    assert enriched["eval_ng"] == 3

    with pytest.raises(RuntimeError, match="metric label-count mismatch"):
        attach_protocol_provenance(
            {"n": 6, "positive": 2, "negative": 4}, provenance
        )


def test_runner_finalizes_metrics_before_the_artifact_is_written():
    provenance = {
        "eval_split": "test",
        "eval_manifest_sha256": "a" * 64,
        "eval_population_sha256": "b" * 64,
        "eval_ok": 3,
        "eval_ng": 3,
    }

    finalized = finalize_evaluation_metrics(
        {"n": 6, "positive": 3, "negative": 3},
        protocol_provenance=provenance,
    )

    assert finalized["eval_split"] == "test"
    assert finalized["eval_population_sha256"] == "b" * 64

    with pytest.raises(RuntimeError, match="metric label-count mismatch"):
        finalize_evaluation_metrics(
            {"n": 6, "positive": 4, "negative": 2},
            protocol_provenance=provenance,
        )
