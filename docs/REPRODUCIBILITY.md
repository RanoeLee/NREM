# Reproducing the paper experiments

## Accepted scope

The frozen protocol contains 420 development tasks on 400-val and 1,524
reporting tasks on 600-test. Ratios are `00, 01, 05, 08, 12, 16, 20`; stochastic
methods use seeds 2025--2034. `experiments/paper_400_600_scope.yaml` defines the
complete method and ablation matrix.

## Verify the published results

Download and extract `EXP_RESULTS` at the repository root, then run:

```bash
python -m tools.verify_final_release EXP_RESULTS
```

The verifier is the release acceptance check. It checks every real-file
SHA-256 and size, rejects links and unsafe paths, requires exactly 420
development tasks and 1,524 reporting tasks, rejects non-finite scores, and
recomputes every AUROC/AUPRC pair from `scores.npz` with scikit-learn. A
successful invocation prints JSON whose `tasks` and `metrics_recomputed`
fields are both `1944`; any missing, altered, incomplete, or inconsistent
artifact causes a non-zero exit instead.

This procedure reproduces and audits the reported metrics from the accepted
score vectors and metadata. It does not promise bit-for-bit identical model
weights from fresh GPU training, because hardware and kernel nondeterminism can
change newly trained weights while leaving the documented protocol unchanged.

```text
EXP_RESULTS/NREM/{Benchmark,SRE,CTM,REA}/contam_XX/seed_YYYY/
EXP_RESULTS/NREM/Ablations/{SRE,CTM,REA}/<variant>/contam_XX/seed_YYYY/
EXP_RESULTS/NREM/Selection/{SRE_mask_ratio,REA_weight}/<variant>/contam_XX/seed_YYYY/
EXP_RESULTS/Comparison/<method>/contam_XX/seed_YYYY/
```

Each leaf contains `checkpoint.pt`, `config.yaml`, `metrics.json`, `scores.npz`,
`provenance.json`, and normalization/reference state when needed.

## Retrain from scratch

Download `IPML-MotorEOL.zip` from the [dataset download page](DATA.md#download)
and extract it to `data/IPML-MotorEOL/`, preserving the supplied train/val/test
folders. Use the split manifest and metadata in this repository's `protocol/`
directory.

Install the project and run its tests:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Submit one traceable experiment as one one-GPU Slurm job:

```bash
sbatch scripts/slurm/run_single_config.sbatch \
  --config configs/release/NREM.yaml \
  --train-base data/IPML-MotorEOL/train \
  --contam contam_20 \
  --seed 2025 \
  --out runs \
  --session-name reproduce-nrem
```

Use the config stored in the matching `EXP_RESULTS` leaf to retrain any
comparison or ablation. Always use a new output root.

Evaluate against the fixed reporting split:

```bash
python evaluate.py \
  --root runs/reproduce-nrem/seed_2025/NREM/contam_20 \
  --eval_root data/IPML-MotorEOL/test \
  --split_manifest protocol/split_manifest.csv \
  --split_name test --expected_ok 300 --expected_ng 300 --device cuda
```

Development selection uses `--split_name val --expected_ok 200 --expected_ng
200`. Final reporting never selects hyperparameters on test data.
