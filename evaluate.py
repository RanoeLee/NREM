#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from nrem.config import load_config
from nrem.engine.runner import evaluate_run, finalize_evaluation_metrics
from nrem.data.dataset import infer_label_from_path, LABEL_OK, LABEL_NG
from nrem.utils.misc import parse_contam_value, pick_device


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained NREM runs.')
    p.add_argument('--root', type=str, required=True, help='Session root containing seed_*/exp/contam_* runs, or a single run with config.yaml.')
    p.add_argument('--eval_root', type=str, required=True, help='Directory containing labeled OK/NG .npy samples.')
    p.add_argument('--mode', type=str, default='fixed')
    p.add_argument('--ckpt_name', type=str, default='ema.pt', help='Checkpoint to evaluate. Default uses EMA for more stable self-supervised training; falls back to best/last if unavailable.')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--score_mode', type=str, default=None, help="Override cfg nrem.score_mode. Recommended formal score: resa.")
    p.add_argument('--resa_energy', type=str, default=None,
                   help="Override nrem.resa_energy. Use a single key or a comma-separated stream list, e.g. u1 or u1,u2,rec.")
    p.add_argument('--resa_cauchy_clip', type=float, default=None,
                   help="Override nrem.resa_cauchy_clip for post-hoc REA scoring.")
    p.add_argument('--resa_stream_weights', type=str, default=None,
                   help="Comma-separated non-negative stream weights aligned with --resa_energy.")
    p.add_argument('--eval_dyn_groups', type=int, default=None,
                   help="Override nrem.eval_dyn_groups. This may require a matching alternate reference.")
    p.add_argument('--reference_path', type=str, default=None,
                   help="Reference file to load or build. Defaults to RUN_DIR/resa_reference.npz.")
    p.add_argument('--result_tag', type=str, default=None,
                   help="Safe suffix for an independent result directory; required with post-hoc overrides.")
    p.add_argument('--eval_batch_size', type=int, default=None,
                   help='Override only the evaluation DataLoader batch size; does not change training or scoring.')
    p.add_argument('--split_manifest', type=str, required=True,
                   help='CSV manifest defining the immutable evaluation population.')
    p.add_argument('--split_name', type=str, required=True,
                   help='Manifest split to evaluate, normally val or test.')
    p.add_argument('--expected_ok', type=int, required=True,
                   help='Exact number of OK products required by the protocol.')
    p.add_argument('--expected_ng', type=int, required=True,
                   help='Exact number of NG products required by the protocol.')
    p.add_argument('--skip_failed', action='store_true')
    return p.parse_args()


@dataclass(frozen=True)
class SplitProtocol:
    manifest_path: Path
    manifest_sha256: str
    split: str
    rows: tuple[dict[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_protocol(manifest_path: Path, split: str) -> SplitProtocol:
    manifest_path = Path(manifest_path).expanduser().resolve()
    metadata_path = manifest_path.with_name('split_metadata.json')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Split manifest not found: {manifest_path}')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Split metadata not found: {metadata_path}')

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    actual_manifest_sha256 = sha256_file(manifest_path)
    recorded_manifest_sha256 = str(metadata.get('manifest_sha256', ''))
    if actual_manifest_sha256 != recorded_manifest_sha256:
        raise RuntimeError(
            'Split manifest SHA-256 does not match split_metadata.json: '
            f'actual={actual_manifest_sha256}, recorded={recorded_manifest_sha256}'
        )
    if metadata.get('manifest') not in (None, '', manifest_path.name):
        raise RuntimeError(
            f'Split metadata names a different manifest: {metadata.get("manifest")}'
        )

    with manifest_path.open(newline='', encoding='utf-8') as handle:
        all_rows = list(csv.DictReader(handle))
    available_splits = sorted({str(row.get('split', '')) for row in all_rows})
    if split not in available_splits:
        raise ValueError(
            f'Unknown split {split!r}; manifest contains {available_splits}'
        )

    destinations: set[str] = set()
    sources: set[str] = set()
    for row in all_rows:
        destination = str(row.get('destination_relative_path', ''))
        source = str(row.get('source_relative_path', ''))
        if destination in destinations:
            raise RuntimeError(f'Split manifest has duplicate destination: {destination}')
        if source in sources:
            raise RuntimeError(f'Split manifest has duplicate source membership: {source}')
        destinations.add(destination)
        sources.add(source)

    rows = tuple(row for row in all_rows if row.get('split') == split)
    return SplitProtocol(
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha256,
        split=split,
        rows=rows,
    )


def validate_eval_population(
    eval_root: Path,
    protocol: SplitProtocol,
    *,
    expected_ok: int,
    expected_ng: int,
) -> dict[str, object]:
    eval_root = Path(eval_root).expanduser().resolve()
    if expected_ok < 1 or expected_ng < 1:
        raise ValueError('Expected OK and NG counts must both be positive')
    if eval_root.name != protocol.split:
        raise RuntimeError(
            f'Evaluation root basename must match split {protocol.split!r}: {eval_root}'
        )

    expected: dict[str, dict[str, str]] = {}
    counts = {'ok': 0, 'ng': 0}
    canonical_rows: list[dict[str, object]] = []
    for row in protocol.rows:
        destination = Path(str(row['destination_relative_path']))
        if not destination.parts or destination.parts[0] != protocol.split:
            raise RuntimeError(
                f'Manifest destination is outside split {protocol.split}: {destination}'
            )
        relative = Path(*destination.parts[1:]).as_posix()
        label = str(row.get('label', '')).lower()
        if label not in counts:
            raise RuntimeError(f'Unknown manifest label {label!r}: {destination}')
        if relative in expected:
            raise RuntimeError(f'Split manifest has duplicate destination: {relative}')
        expected[relative] = row
        counts[label] += 1
        canonical_rows.append(
            {
                'relative_path': relative,
                'label': label,
                'size_bytes': int(row['size_bytes']),
                'sha256': str(row['sha256']),
            }
        )

    if counts['ok'] != expected_ok or counts['ng'] != expected_ng:
        raise RuntimeError(
            f'Protocol class-count mismatch: expected OK={expected_ok}, NG={expected_ng}; '
            f'manifest has OK={counts["ok"]}, NG={counts["ng"]}'
        )

    actual_paths = {
        path.relative_to(eval_root).as_posix(): path
        for path in eval_root.rglob('*.npy')
        if path.is_file()
    }
    missing = sorted(set(expected) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected))
    if missing:
        raise RuntimeError(f'Missing evaluation files: {missing[:5]}')
    if unexpected:
        raise RuntimeError(f'Found unexpected evaluation files: {unexpected[:5]}')

    for relative, row in expected.items():
        path = actual_paths[relative]
        recorded_size = int(row['size_bytes'])
        recorded_sha256 = str(row['sha256'])
        if path.stat().st_size != recorded_size or sha256_file(path) != recorded_sha256:
            raise RuntimeError(f'Evaluation file content mismatch: {path}')

    canonical = json.dumps(
        sorted(canonical_rows, key=lambda row: str(row['relative_path'])),
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return {
        'eval_split': protocol.split,
        'eval_manifest_sha256': protocol.manifest_sha256,
        'eval_population_sha256': hashlib.sha256(canonical).hexdigest(),
        'eval_ok': counts['ok'],
        'eval_ng': counts['ng'],
    }


def attach_protocol_provenance(
    metrics: dict, provenance: dict[str, object]
) -> dict:
    return finalize_evaluation_metrics(
        metrics, protocol_provenance=provenance
    )


def parse_resa_energy(value: str | None):
    if value is None:
        return None
    streams = [item.strip().lower() for item in value.split(',') if item.strip()]
    if not streams:
        raise ValueError('--resa_energy must contain at least one stream name')
    allowed = {'rec', 'u1', 'u2', 'u12'}
    unknown = sorted(set(streams) - allowed)
    if unknown:
        raise ValueError(f'Unknown --resa_energy stream(s): {unknown}; allowed={sorted(allowed)}')
    return streams[0] if len(streams) == 1 else streams


def parse_resa_stream_weights(value: str | None):
    if value is None:
        return None
    weights = [float(item.strip()) for item in value.split(',') if item.strip()]
    if (
        not weights
        or any((not math.isfinite(weight)) or weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError(
            '--resa_stream_weights must contain finite, non-negative values with a positive sum'
        )
    total = sum(weights)
    return [weight / total for weight in weights]


def validate_posthoc_args(args) -> None:
    has_override = any(
        value is not None
        for value in (
            args.resa_energy, args.resa_cauchy_clip, args.resa_stream_weights,
            args.eval_dyn_groups, args.reference_path,
        )
    )
    if has_override and not args.result_tag:
        raise ValueError('--result_tag is required when using post-hoc evaluation overrides')
    if args.result_tag and not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', args.result_tag):
        raise ValueError('--result_tag must match [a-z0-9][a-z0-9_-]*')
    if args.resa_cauchy_clip is not None and args.resa_cauchy_clip < 0:
        raise ValueError('--resa_cauchy_clip must be non-negative')
    if args.eval_dyn_groups is not None and args.eval_dyn_groups < 1:
        raise ValueError('--eval_dyn_groups must be at least 1')
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError('--eval_batch_size must be at least 1')
    if args.eval_dyn_groups is not None and not args.reference_path:
        raise ValueError(
            '--reference_path is required with --eval_dyn_groups so the matching '
            'training reference is rebuilt outside the original run artifacts'
        )


def find_runs(root: Path):
    if (root / 'config.yaml').exists():
        return [root]
    runs = []
    for p in root.rglob('config.yaml'):
        runs.append(p.parent)
    return sorted(runs)


def scan_eval_labels(eval_root: Path) -> dict:
    files = sorted(eval_root.rglob('*.npy'))
    ok = ng = unknown = 0
    for p in files:
        y = infer_label_from_path(p)
        if y == LABEL_OK:
            ok += 1
        elif y == LABEL_NG:
            ng += 1
        else:
            unknown += 1
    return {'num_files': len(files), 'ok': ok, 'ng': ng, 'unknown': unknown}


def extract_seed(run: Path) -> str:
    for part in run.parts:
        if part.startswith('seed_'):
            return part.replace('seed_', '')
    return ''


def assert_valid_metrics(metrics: dict, eval_root: Path, run: Path) -> None:
    pos = int(metrics.get('positive', 0) or 0)
    neg = int(metrics.get('negative', 0) or 0)
    if pos == 0 and neg == 0:
        raise RuntimeError(
            f'No valid OK/NG labels were found during evaluation. run={run}, eval_root={eval_root}. '
            f'Check filename/folder labels and infer_label_from_path(). Metrics={metrics}'
        )
    if pos == 0 or neg == 0:
        raise RuntimeError(
            f'Evaluation set must contain both OK and NG samples, got positive={pos}, negative={neg}, '
            f'run={run}, eval_root={eval_root}.'
        )


def main():
    args = parse_args()
    validate_posthoc_args(args)
    resa_energy_override = parse_resa_energy(args.resa_energy)
    resa_stream_weights_override = parse_resa_stream_weights(args.resa_stream_weights)
    root = Path(args.root).expanduser().resolve()
    eval_root = Path(args.eval_root).expanduser().resolve()
    split_manifest = Path(args.split_manifest).expanduser().resolve()
    device = pick_device(args.device)

    if not root.exists():
        raise FileNotFoundError(f'root does not exist: {root}')
    if not eval_root.exists():
        raise FileNotFoundError(f'eval_root does not exist: {eval_root}')

    protocol = load_split_protocol(split_manifest, args.split_name)
    protocol_provenance = validate_eval_population(
        eval_root,
        protocol,
        expected_ok=args.expected_ok,
        expected_ng=args.expected_ng,
    )

    scan = scan_eval_labels(eval_root)
    print('========================================')
    print('Evaluation label scan')
    print(f'eval_root : {eval_root}')
    print(f"num_files : {scan['num_files']}")
    print(f"OK        : {scan['ok']}")
    print(f"NG        : {scan['ng']}")
    print(f"UNKNOWN   : {scan['unknown']}")
    print('========================================')
    if scan['num_files'] == 0 or scan['ok'] == 0 or scan['ng'] == 0:
        raise RuntimeError(f'Invalid eval labels: {scan}')

    runs = find_runs(root)
    rows = []
    print(f'Evaluating {len(runs)} runs on {eval_root} | score_mode={args.score_mode or "cfg-default"}')
    for run in tqdm(runs, desc='evaluate'):
        try:
            cfg = load_config(run / 'config.yaml')
            if args.eval_batch_size is not None:
                cfg.setdefault('train', {})['batch_size'] = args.eval_batch_size
            if resa_energy_override is not None:
                cfg.setdefault('nrem', {})['resa_energy'] = resa_energy_override
            if args.resa_cauchy_clip is not None:
                cfg.setdefault('nrem', {})['resa_cauchy_clip'] = args.resa_cauchy_clip
            if resa_stream_weights_override is not None:
                configured = resa_energy_override
                if configured is None:
                    configured = cfg.setdefault('nrem', {}).get('resa_energy', [])
                streams = configured if isinstance(configured, list) else [configured]
                if len(resa_stream_weights_override) != len(streams):
                    raise ValueError(
                        'resa_stream_weights must align one-to-one with resa_energy streams'
                    )
                cfg.setdefault('nrem', {})['resa_stream_weights'] = resa_stream_weights_override
            if args.eval_dyn_groups is not None:
                cfg.setdefault('nrem', {})['eval_dyn_groups'] = args.eval_dyn_groups
            metrics = evaluate_run(
                run,
                eval_root,
                cfg=cfg,
                ckpt_name=args.ckpt_name,
                device=device,
                score_mode=args.score_mode,
                reference_path=args.reference_path,
                result_tag=args.result_tag,
                protocol_provenance=protocol_provenance,
            )
            assert_valid_metrics(metrics, eval_root, run)
            seed = extract_seed(run)
            exp_name = run.parent.name if run.name.startswith('contam_') else run.name
            train_name = run.name
            row = {
                'seed': seed,
                'exp_name': exp_name,
                'train_name': train_name,
                'train_contam': parse_contam_value(train_name),
                'eval_name': eval_root.name,
                'eval_tag': args.mode,
                'ckpt_name': args.ckpt_name,
                'score_mode': metrics.get('score_mode', args.score_mode or ''),
                'run_dir': str(run),
                'eval_root': str(eval_root),
                'source_exp_name': exp_name,
                'result_tag': args.result_tag or '',
                'resa_energy_override': args.resa_energy or '',
                'resa_cauchy_clip_override': '' if args.resa_cauchy_clip is None else args.resa_cauchy_clip,
                'resa_stream_weights_override': args.resa_stream_weights or '',
                'eval_dyn_groups_override': '' if args.eval_dyn_groups is None else args.eval_dyn_groups,
                'reference_path_override': args.reference_path or '',
            }
            row.update(metrics)
            rows.append(row)

        except Exception as e:
            print(f'[WARN] failed {run}: {e}')
            if args.skip_failed:
                continue
            raise
    if rows:
        header = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    header.append(k)
                    seen.add(k)
        # When --score_mode is not specified, each run uses its own YAML-defined
        # score_mode. Use a neutral filename to avoid implying that all rows share
        # the first row's score mode.
        mode_name = str(args.score_mode).lower() if args.score_mode else 'cfg'
        tag_suffix = f'_{args.result_tag}' if args.result_tag else ''
        out_csv = root / f'evaluation_index_{Path(args.ckpt_name).stem}_{mode_name}{tag_suffix}.csv'
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
