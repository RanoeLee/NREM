import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _seven_places(value: str) -> str:
    return str(Decimal(value).quantize(Decimal("0.0000001"), rounding=ROUND_HALF_UP))


def test_aggregate_metrics_match_public_run_summaries() -> None:
    aggregate = {row["method"]: row for row in _rows(ROOT / "results" / "aggregate_efficiency.csv")}
    summaries = _rows(ROOT / "results" / "paper_400_600_summary.csv")
    display = {"Anomaly Transformer": "Anomaly-Transformer"}
    for method in ("NREM", "DeepSVDD", "TCN-AE", "Transformer-AE", "TranAD", "Anomaly Transformer"):
        shown = aggregate[method]
        source_name = display.get(method, method)
        rows = [
            row for row in summaries
            if row["phase"] == "reporting" and row["method_or_variant"] == source_name
        ]
        assert len(rows) == 7
        mean_auroc = sum(Decimal(row["auroc_mean"]) for row in rows) / 7
        mean_auprc = sum(Decimal(row["auprc_mean"]) for row in rows) / 7
        assert shown["mean_auroc"] == _seven_places(str(mean_auroc))
        assert shown["mean_auprc"] == _seven_places(str(mean_auprc))


def test_aggregate_complexity_matches_profile() -> None:
    aggregate = {row["method"]: row for row in _rows(ROOT / "results" / "aggregate_efficiency.csv")}
    profile = {row["method"]: row for row in _rows(ROOT / "results" / "complexity_profile.csv")}

    for method, shown in aggregate.items():
        exact = profile[method]
        assert Decimal(shown["parameters_millions"]) == Decimal(exact["parameters_millions"])
        assert Decimal(shown["flops_giga_per_sample"]) == Decimal(exact["flops_giga_per_sample"])
