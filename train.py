#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

from nrem.config import discover_config_paths, load_config
from nrem.engine.runner import train_one
from nrem.utils.misc import configure_threads, discover_contam_dirs, parse_contam_value, pick_device, set_seed, timestamp


def parse_args():
    p = argparse.ArgumentParser(description='Train NREM experiments.')
    p.add_argument('--config', action='append', default=[], help='Config yaml. Can be repeated.')
    p.add_argument('--config_glob', action='append', default=[])
    p.add_argument('--train_root', type=str, required=True)
    p.add_argument('--sweep_contam', type=str, default='true')
    p.add_argument('--contam_prefix', type=str, default='contam_')
    p.add_argument('--out', type=str, default='runs')
    p.add_argument('--session_name', type=str, default='')
    p.add_argument('--seeds', type=str, default='2025')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def str2bool(x: str) -> bool:
    return str(x).lower() in {'1', 'true', 'yes', 'y'}


def main():
    args = parse_args()
    device = pick_device(args.device)
    
    if not args.config and not args.config_glob:
        args.config_glob = ['configs/release/*.yaml']
    cfg_paths = discover_config_paths(args.config, args.config_glob)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    session = args.session_name or f'NREM_{timestamp()}'
    session_root = Path(args.out).expanduser().resolve() / session
    session_root.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.train_root).expanduser().resolve()
    if str2bool(args.sweep_contam):
        train_dirs = discover_contam_dirs(str(train_root), prefix=args.contam_prefix)
    else:
        train_dirs = [train_root]

    jobs = []
    for seed in seeds:
        for cfg_path in cfg_paths:
            cfg = load_config(cfg_path)
            # Evaluation-only configurations reuse a completed training run.
            if bool((cfg.get('train') or {}).get('eval_only', False)):
                print(f'[INFO] skip eval-only config during training: {cfg_path}')
                continue
            exp_name = str((cfg.get('experiment') or {}).get('name', cfg_path.stem))
            configure_threads(cfg, device)
            for tr in train_dirs:
                jobs.append((seed, cfg_path, cfg, exp_name, tr))

    print(f'Using device: {device}')
    print(f'Jobs: {len(jobs)} | session={session_root}')
    for seed, cfg_path, cfg, exp_name, tr in tqdm(jobs, desc='NREM train'):
        set_seed(seed)
        contam_name = tr.name
        run_dir = session_root / f'seed_{seed}' / exp_name / contam_name
        print('=' * 80)
        print(f'NREM train | seed={seed} | exp={exp_name} | train={tr}')
        print('=' * 80)
        train_one(cfg_path, cfg, tr, run_dir, seed, device)


if __name__ == '__main__':
    main()
