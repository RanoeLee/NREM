"""Collect scattered validation results into a single CSV.

Usage
-----
  python tools/collect_results.py --runs_root runs/
  python tools/collect_results.py --runs_root runs/ --out results/summary.csv
  python tools/collect_results.py --runs_root runs/ --group main

The script walks every directory under runs_root that contains a
metrics.json file inside an  eval_*/  sub-directory and assembles
one row per evaluation into a flat CSV.

Expected run-directory structure (created by runner.py / train.py):

  runs/
    {group}/
      seed_{seed}/
        {experiment}/
          contam_{pct}/
            eval_{eval_name}_{ckpt}_{score_mode}/
              metrics.json

Columns produced
----------------
  group         — sub-folder name directly under runs_root (e.g. main)
  experiment    — config/experiment name (e.g. H3_global)
  seed          — integer seed
  contam_pct    — contamination percentage as a float (0, 1, 5, …)
  eval_name     — name of the evaluation split (e.g. val)
  ckpt          — checkpoint used (ema, best, last, …)
  score_mode    — scoring mode (mr_resa, rec_mean, …)
  + all numeric/string fields present in metrics.json
  run_path      — path of the run directory relative to runs_root
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import csv


# ── Path parsing ─────────────────────────────────────────────────────────────

def _parse_seed(token: str) -> int | None:
    """'seed_2025' → 2025, '2025' → 2025, else None."""
    m = re.fullmatch(r"(?:seed_)?(\d+)", token, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_contam(token: str) -> float | None:
    """'contam_00' → 0.0, 'contam_20' → 20.0, 'contam_0.05' → 0.05."""
    m = re.search(r"(\d+(?:\.\d+)?)", token)
    return float(m.group(1)) if m else None


def _parse_eval_dir(name: str) -> dict:
    """'eval_val_ema_mr_resa' → {eval_name, ckpt, score_mode}.

    Format: eval_{eval_name}_{ckpt}_{score_mode}
    eval_name and score_mode may themselves contain underscores, so we
    anchor on common checkpoint names as the split point.
    """
    # strip leading 'eval_'
    body = re.sub(r"^eval_", "", name)

    # Known checkpoint stems
    ckpt_names = ("ema", "best", "last", "final")
    ckpt_pat   = "|".join(ckpt_names)

    m = re.match(
        rf"^(?P<eval_name>.+?)_(?P<ckpt>{ckpt_pat})_(?P<score_mode>.+)$",
        body,
    )
    if m:
        return {
            "eval_name":  m.group("eval_name"),
            "ckpt":       m.group("ckpt"),
            "score_mode": m.group("score_mode"),
        }

    # Fallback: split on last two underscores
    parts = body.rsplit("_", 2)
    if len(parts) == 3:
        return {"eval_name": parts[0], "ckpt": parts[1], "score_mode": parts[2]}
    return {"eval_name": body, "ckpt": "", "score_mode": ""}


def _infer_metadata_from_path(
    metrics_path: Path, runs_root: Path
) -> dict:
    """Parse as much structured metadata as possible from the path.

    The path is expected to look like (levels relative to runs_root):
      {group} / seed_{seed} / {experiment} / contam_{pct} / eval_*/ metrics.json

    Parts that cannot be parsed are left as empty string / -1.
    """
    rel   = metrics_path.relative_to(runs_root)
    parts = rel.parts          # e.g. ('main','seed_2025','H3_global','contam_20','eval_val_ema_mr_resa','metrics.json')

    meta: dict = {
        "group":      "",
        "experiment": "",
        "seed":       -1,
        "contam_pct": -1.0,
        "eval_name":  "",
        "ckpt":       "",
        "score_mode": "",
        "run_path":   str(metrics_path.parent.parent.relative_to(runs_root)),
    }

    # eval dir is always the direct parent of metrics.json
    if len(parts) >= 2:
        eval_meta = _parse_eval_dir(parts[-2])
        meta.update(eval_meta)

    # Walk up the remaining levels and match by content
    above = list(parts[:-2])   # everything above the eval_* dir

    # Assign by typical depth
    depth_map = {0: "group", 1: "seed_token", 2: "experiment", 3: "contam_token"}
    tokens: dict = {}
    for i, tok in enumerate(above):
        key = depth_map.get(i, f"extra_{i}")
        tokens[key] = tok

    if "group" in tokens:
        meta["group"] = tokens["group"]

    if "seed_token" in tokens:
        s = _parse_seed(tokens["seed_token"])
        meta["seed"] = s if s is not None else tokens["seed_token"]

    if "experiment" in tokens:
        meta["experiment"] = tokens["experiment"]

    if "contam_token" in tokens:
        c = _parse_contam(tokens["contam_token"])
        if c is not None:
            meta["contam_pct"] = c

    # If depth differs (e.g. no group sub-folder), try heuristic re-scan
    for tok in above:
        if meta["seed"] == -1:
            s = _parse_seed(tok)
            if s is not None:
                meta["seed"] = s
        if meta["contam_pct"] == -1.0 and re.search(r"contam", tok, re.IGNORECASE):
            c = _parse_contam(tok)
            if c is not None:
                meta["contam_pct"] = c

    return meta


# ── Collection ───────────────────────────────────────────────────────────────

def collect_rows(runs_root: Path, group_filter: str | None = None) -> list[dict]:
    """Find all metrics.json files and build one row per evaluation."""
    rows: list[dict] = []
    seen: set[str] = set()

    # metrics.json is always inside an eval_* directory
    for metrics_path in sorted(runs_root.rglob("eval_*/metrics.json")):
        key = str(metrics_path)
        if key in seen:
            continue
        seen.add(key)

        # Load the JSON
        try:
            with open(metrics_path) as fh:
                data: dict = json.load(fh)
        except Exception as e:
            print(f"[WARN] Could not read {metrics_path}: {e}", file=sys.stderr)
            continue

        meta = _infer_metadata_from_path(metrics_path, runs_root)

        # Apply group filter
        if group_filter and meta["group"] != group_filter:
            continue

        # Merge: metadata first, then metrics (metrics override if key clash
        # except for the reserved names we set above)
        row = {}
        row.update(meta)
        for k, v in data.items():
            if k not in row:     # don't overwrite path-derived keys
                row[k] = v

        rows.append(row)

    return rows


# ── CSV writing ───────────────────────────────────────────────────────────────

# Column priority order — these appear first in the CSV; all other keys follow
_PRIORITY_COLS = [
    "group", "experiment", "seed", "contam_pct",
    "eval_name", "ckpt", "score_mode",
    "auroc", "ap", "f1", "threshold",
    "resa_ref_samples", "resa_ref_windows",
    "run_path",
]


def write_csv(rows: list[dict], out_path: Path) -> list[str]:
    """Write rows to CSV; return the column order used."""
    if not rows:
        return []

    # Build column order: priority first, then alphabetic remainder
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())

    ordered  = [c for c in _PRIORITY_COLS if c in all_keys]
    ordered += sorted(all_keys - set(ordered))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ordered})

    return ordered


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("No results found.")
        return

    groups      = sorted({r.get("group", "")      for r in rows})
    experiments = sorted({r.get("experiment", "")  for r in rows})
    seeds       = sorted({r.get("seed", -1)        for r in rows})
    contams     = sorted({r.get("contam_pct", -1)  for r in rows})

    print(f"\nTotal rows   : {len(rows)}")
    print(f"Groups       : {groups}")
    print(f"Experiments  : {experiments}")
    print(f"Seeds        : {seeds}")
    print(f"Contam levels: {contams}")

    # AUROC summary by experiment × contam (mean over seeds)
    if "auroc" in cols:
        from collections import defaultdict
        bucket: dict = defaultdict(list)
        for r in rows:
            k = (r.get("experiment",""), r.get("contam_pct", -1))
            v = r.get("auroc")
            if v not in ("", None):
                try:
                    bucket[k].append(float(v))
                except (ValueError, TypeError):
                    pass

        if bucket:
            print("\nAUROC summary (mean ± std over seeds):")
            header = f"  {'Experiment':<40} {'Contam':>8}  {'AUROC mean':>12}  {'std':>8}  {'n':>4}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for (exp, contam), vals in sorted(bucket.items()):
                mean = sum(vals) / len(vals)
                std  = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                print(f"  {exp:<40} {str(contam):>8}  {mean:>12.4f}  {std:>8.4f}  {len(vals):>4}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--runs_root", required=True,
        help="Root directory containing run sub-directories.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output CSV path. Defaults to <runs_root>/results_summary.csv",
    )
    ap.add_argument(
        "--group", default=None,
        help="Only include runs from this group (e.g. main, mechanism).",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        print(f"ERROR: --runs_root does not exist: {runs_root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else runs_root / "results_summary.csv"

    print(f"Scanning : {runs_root}")
    if args.group:
        print(f"Filter   : group = {args.group}")

    rows = collect_rows(runs_root, group_filter=args.group)

    if not rows:
        print("No metrics.json files found. Have evaluations been run yet?")
        sys.exit(0)

    cols = write_csv(rows, out_path)
    print(f"\nWrote {len(rows)} rows → {out_path}")
    print_summary(rows, cols)


if __name__ == "__main__":
    main()
