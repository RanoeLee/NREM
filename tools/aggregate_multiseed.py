#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('csv')
p.add_argument('--metrics', default='auroc,auprc,oracle_best_f1')
p.add_argument('--out', default='')
args = p.parse_args()

df = pd.read_csv(args.csv)
metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
group_cols = ['exp_name', 'train_contam']
rows = []
for keys, g in df.groupby(group_cols):
    row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
    for m in metrics:
        if m in g:
            row[f'{m}_mean'] = g[m].mean()
            row[f'{m}_std'] = g[m].std(ddof=1)
    rows.append(row)
out = pd.DataFrame(rows).sort_values(group_cols)
print(out.to_string(index=False))
if args.out:
    out.to_csv(args.out, index=False)
