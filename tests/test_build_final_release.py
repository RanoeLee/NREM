import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.build_final_release import (
    EXPECTED_COUNTS,
    build_release,
    export_public_results,
    map_release_leaf,
    validate_accepted_counts,
)


def row(**updates):
    value = {
        "phase": "reporting",
        "group": "main_chain",
        "method_or_variant": "NREM",
        "seed": "2034",
        "contamination_ratio": "20",
        "task_key": "abc",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "NREM/REA/contam_20/seed_2034"),
        ({"method_or_variant": "SRE"}, "NREM/SRE/contam_20/seed_2034"),
        (
            {"group": "mechanism", "method_or_variant": "no_covariance"},
            "NREM/Ablations/SRE/no_covariance/contam_20/seed_2034",
        ),
        (
            {
                "phase": "development",
                "group": "mask_ratio",
                "method_or_variant": "NREM_mask030",
            },
            "NREM/Selection/SRE_mask_ratio/NREM_mask030/contam_20/seed_2034",
        ),
        (
            {
                "phase": "development",
                "group": "rea_weight",
                "method_or_variant": "weight_020",
            },
            "NREM/Selection/REA_weight/weight_020/contam_20/seed_2034",
        ),
        (
            {"group": "stochastic_external", "method_or_variant": "DeepSVDD"},
            "Comparison/DeepSVDD/contam_20/seed_2034",
        ),
        (
            {
                "group": "deterministic_external",
                "method_or_variant": "OCSVM",
                "seed": "fixed",
            },
            "Comparison/OCSVM/contam_20/seed_fixed",
        ),
    ],
)
def test_map_release_leaf(updates, expected):
    assert map_release_leaf(row(**updates)).as_posix() == expected


def test_validate_accepted_counts_rejects_incomplete_scope():
    with pytest.raises(ValueError, match="accepted task counts"):
        validate_accepted_counts([row()])


def test_build_release_copies_real_files_and_writes_path_free_manifest(tmp_path):
    source_root = tmp_path / "source"
    run = source_root / "runs" / "main" / "seed_2034" / "contam_20"
    result = run / "eval_release"
    checkpoint = run / "checkpoints" / "ema.pt"
    result.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    (run / "config.yaml").write_text("seed: 2034\n", encoding="utf-8")
    (run / "normalizer.npz").write_bytes(b"normalizer")
    (result / "metrics.json").write_text('{"auroc": 0.9, "auprc": 0.8}', encoding="utf-8")
    (result / "scores.npz").write_bytes(b"scores")

    accepted = tmp_path / "accepted.csv"
    fields = [
        "phase", "task_key", "group", "method_or_variant", "seed",
        "contamination_ratio", "split", "auroc", "auprc", "source_result",
        "corrected_result", "manifest_sha256", "population_sha256",
    ]
    values = row()
    values.update(
        split="test", auroc="0.9", auprc="0.8",
        source_result=str(result), corrected_result=str(result),
        manifest_sha256="m", population_sha256="p",
    )
    with accepted.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(values)

    metadata = {
        "abc": {"source_run": "runs/main/seed_2034/contam_20", "checkpoint": "ema.pt"}
    }
    output = tmp_path / "EXP_RESULTS"
    manifest = build_release(
        accepted_csv=accepted,
        task_metadata=metadata,
        source_root=source_root,
        output_root=output,
        expected_counts=None,
    )

    leaf = output / "NREM" / "REA" / "contam_20" / "seed_2034"
    assert not (leaf / "checkpoint.pt").is_symlink()
    assert (leaf / "checkpoint.pt").read_bytes() == b"checkpoint"
    assert (leaf / "config.yaml").read_text(encoding="utf-8") == "seed: 2034\n"
    assert (leaf / "normalizer.npz").read_bytes() == b"normalizer"
    assert (leaf / "metrics.json").is_file()
    assert (leaf / "scores.npz").is_file()
    provenance = json.loads((leaf / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_key"] == "abc"
    assert "source" not in json.dumps(provenance).lower()
    text = manifest.read_text(encoding="utf-8")
    assert str(source_root) not in text
    assert hashlib.sha256((leaf / "checkpoint.pt").read_bytes()).hexdigest() in text


def test_build_release_refuses_existing_output(tmp_path):
    output = tmp_path / "EXP_RESULTS"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_release(
            accepted_csv=tmp_path / "missing.csv",
            task_metadata={},
            source_root=tmp_path,
            output_root=output,
            expected_counts=EXPECTED_COUNTS,
        )


def test_export_public_results_removes_server_paths(tmp_path):
    accepted = tmp_path / "accepted.csv"
    accepted.write_text(
        "phase,task_key,group,method_or_variant,seed,contamination_ratio,split,auroc,auprc,source_result,corrected_result,manifest_sha256,population_sha256\n"
        "reporting,abc,main_chain,NREM,2034,20,test,0.9,0.8,/private/source,/private/result,m,p\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "phase,group,method_or_variant,contamination_ratio,n,auroc_mean,auroc_std,auprc_mean,auprc_std\n"
        "reporting,main_chain,NREM,20,10,0.9,0.1,0.8,0.1\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"
    export_public_results(accepted, summary, output)
    run_text = (output / "paper_400_600_run_level.csv").read_text(encoding="utf-8")
    assert "/shared_data" not in run_text
    assert "source_result" not in run_text
    assert "corrected_result" not in run_text
    assert (output / "paper_400_600_summary.csv").read_text(encoding="utf-8") == summary.read_text(encoding="utf-8")
