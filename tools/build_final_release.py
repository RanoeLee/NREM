#!/usr/bin/env python3
"""Build the path-free NREM-Final verification bundle from accepted artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import yaml


EXPECTED_COUNTS = {
    ("development", "mask_ratio"): 210,
    ("development", "rea_weight"): 210,
    ("reporting", "main_chain"): 280,
    ("reporting", "mechanism"): 390,
    ("reporting", "stochastic_external"): 840,
    ("reporting", "deterministic_external"): 14,
}

MECHANISM_MODULE = {
    "masked_input": "SRE",
    "symmetric_view": "SRE",
    "no_covariance": "SRE",
    "dyn_mask_005": "CTM",
    "dyn_mask_030": "CTM",
    "eval_groups_2": "CTM",
    "eval_groups_8": "CTM",
    "u1_only": "REA",
    "u2_only": "REA",
    "legacy_sum": "REA",
    "three_streams": "REA",
    "no_clipping": "REA",
    "equal_fusion": "REA",
}

PUBLIC_RUN_FIELDS = [
    "phase",
    "task_key",
    "group",
    "method_or_variant",
    "seed",
    "contamination_ratio",
    "split",
    "auroc",
    "auprc",
    "manifest_sha256",
    "population_sha256",
]


def map_release_leaf(row: Mapping[str, str]) -> PurePosixPath:
    """Map one accepted task to its public method/ratio/seed directory."""
    group = row["group"]
    method = row["method_or_variant"]
    ratio = f"contam_{row['contamination_ratio']}"
    seed = f"seed_{row['seed']}"
    if group == "main_chain":
        return PurePosixPath("NREM", "REA" if method == "NREM" else method, ratio, seed)
    if group == "mechanism":
        if method not in MECHANISM_MODULE:
            raise ValueError(f"unknown mechanism variant: {method}")
        return PurePosixPath("NREM", "Ablations", MECHANISM_MODULE[method], method, ratio, seed)
    if group == "mask_ratio":
        return PurePosixPath("NREM", "Selection", "SRE_mask_ratio", method, ratio, seed)
    if group == "rea_weight":
        return PurePosixPath("NREM", "Selection", "REA_weight", method, ratio, seed)
    if group in {"stochastic_external", "deterministic_external"}:
        return PurePosixPath("Comparison", method, ratio, seed)
    raise ValueError(f"unknown experiment group: {group}")


def validate_accepted_counts(
    rows: Iterable[Mapping[str, str]], expected: Mapping[tuple[str, str], int] = EXPECTED_COUNTS
) -> None:
    observed = Counter((row["phase"], row["group"]) for row in rows)
    if observed != Counter(expected):
        raise ValueError(f"accepted task counts do not match: expected={dict(expected)} observed={dict(observed)}")


def export_public_results(accepted_csv: Path, summary_csv: Path, output: Path) -> None:
    """Export accepted tables while removing server-only artifact paths."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with accepted_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with (output / "paper_400_600_run_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_RUN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in PUBLIC_RUN_FIELDS} for row in rows)
    shutil.copy2(summary_csv, output / "paper_400_600_summary.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file():
        raise FileNotFoundError(f"required artifact is not a file: {resolved}")
    return resolved


def _resolve_checkpoint(run: Path, checkpoint: str, root: Path) -> Path:
    for candidate in (run / checkpoint, run / "checkpoints" / checkpoint):
        if candidate.exists():
            return _inside(candidate, root)
    raise FileNotFoundError(f"checkpoint not found for {run}: {checkpoint}")


def _copy(source: Path, destination: Path) -> dict[str, str | int]:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    if destination.is_symlink():
        raise ValueError(f"release artifact must be a real file: {destination}")
    return {"file": destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def build_release(
    *,
    accepted_csv: Path,
    task_metadata: Mapping[str, Mapping[str, str]],
    source_root: Path,
    output_root: Path,
    expected_counts: Mapping[tuple[str, str], int] | None = EXPECTED_COUNTS,
) -> Path:
    """Copy accepted artifacts and return the generated manifest path."""
    if output_root.exists():
        raise FileExistsError(output_root)
    with accepted_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if expected_counts is not None:
        validate_accepted_counts(rows, expected_counts)
    if len({row["task_key"] for row in rows}) != len(rows):
        raise ValueError("accepted task keys are not unique")

    root = source_root.resolve(strict=True)
    planned = [(row, map_release_leaf(row)) for row in rows]
    if len({leaf for _, leaf in planned}) != len(planned):
        raise ValueError("release destination leaves are not unique")

    output_root.mkdir(parents=True)
    manifest_rows: list[dict[str, str | int]] = []
    for row, relative_leaf in sorted(planned, key=lambda item: item[1].as_posix()):
        metadata = task_metadata.get(row["task_key"])
        if metadata is None:
            raise KeyError(f"missing source metadata for task {row['task_key']}")
        source_run = Path(metadata["source_run"])
        run = source_run if source_run.is_absolute() else root / source_run
        run = run.resolve(strict=True)
        run.relative_to(root)
        checkpoint = _resolve_checkpoint(run, metadata["checkpoint"], root)
        result = _inside(Path(row["corrected_result"]), root).parent if Path(row["corrected_result"]).is_file() else Path(row["corrected_result"]).resolve(strict=True)
        result.relative_to(root)
        metrics = _inside(result / "metrics.json", root)
        scores = _inside(result / "scores.npz", root)
        config = _inside(run / "config.yaml", root)

        leaf = output_root.joinpath(*relative_leaf.parts)
        leaf.mkdir(parents=True)
        artifacts = [
            _copy(checkpoint, leaf / "checkpoint.pt"),
            _copy(config, leaf / "config.yaml"),
            _copy(metrics, leaf / "metrics.json"),
            _copy(scores, leaf / "scores.npz"),
        ]
        for name in ("normalizer.npz", "resa_reference.npz"):
            optional = run / name
            if optional.is_file():
                artifacts.append(_copy(_inside(optional, root), leaf / name))

        provenance = {
            "task_key": row["task_key"],
            "phase": row["phase"],
            "group": row["group"],
            "method_or_variant": row["method_or_variant"],
            "seed": row["seed"],
            "contamination_ratio": row["contamination_ratio"],
            "split": row["split"],
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "protocol_manifest_sha256": row["manifest_sha256"],
            "population_sha256": row["population_sha256"],
            "artifacts": artifacts,
        }
        provenance_path = leaf / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append({"file": provenance_path.name, "bytes": provenance_path.stat().st_size, "sha256": sha256(provenance_path)})

        for artifact in artifacts:
            manifest_rows.append(
                {
                    "task_key": row["task_key"],
                    "phase": row["phase"],
                    "group": row["group"],
                    "method_or_variant": row["method_or_variant"],
                    "contamination_ratio": row["contamination_ratio"],
                    "seed": row["seed"],
                    "relative_path": (relative_leaf / str(artifact["file"])).as_posix(),
                    "bytes": int(artifact["bytes"]),
                    "sha256": str(artifact["sha256"]),
                }
            )

    manifest = output_root / "MANIFEST.tsv"
    fields = ["task_key", "phase", "group", "method_or_variant", "contamination_ratio", "seed", "relative_path", "bytes", "sha256"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest


def load_task_metadata(completion_manifest: Path, reporting_tsv: Path) -> dict[str, dict[str, str]]:
    document = yaml.safe_load(completion_manifest.read_text(encoding="utf-8"))
    metadata = {
        row["task_key"]: {"source_run": row["source_run"], "checkpoint": row["checkpoint"]}
        for row in document["experiments"]
    }
    with reporting_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            metadata[row["task_key"]] = {"source_run": row["source_run"], "checkpoint": row["checkpoint"]}
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-csv", type=Path, required=True)
    parser.add_argument("--reporting-tsv", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_task_metadata(args.completion_manifest, args.reporting_tsv)
    manifest = build_release(
        accepted_csv=args.accepted_csv,
        task_metadata=metadata,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(manifest)


if __name__ == "__main__":
    main()
