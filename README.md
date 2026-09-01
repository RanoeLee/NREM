# NREM

Official research code for **Nested Residual Evidence Modeling (NREM)**, a
sample-level motor end-of-line anomaly detector for contaminated training.

NREM models reconstruction residuals at three nested scopes: Sample Residual
Evidence (SRE), Contextual Trajectory Modeling (CTM), and Residual Evidence
Aggregation (REA). The frozen release uses an MLP backbone, SRE mask ratio 0.30,
and SRE/CTM weights 0.20/0.80.

## Release contents

This final 400/600-protocol project contains the model and comparison
implementations, paper configurations, one-GPU Slurm entry points, fixed split
membership, tests, and path-free results for all 1,944 accepted tasks. The
separately downloadable `EXP_RESULTS` bundle adds the exact checkpoint, run
configuration, accepted score vector, metric file, and provenance record for
every task.

## Installation

Python 3.10 or later is required. Install a CUDA-compatible PyTorch build for
GPU training, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

## Data and verification

The 32 GB data archive is distributed separately. Put it under
`data/IPML-MotorEOL/` and follow [docs/DATA.md](docs/DATA.md). After downloading the separate results asset, place its `EXP_RESULTS` directory
at the repository root. The command below verifies every original core artifact
and independently recomputes all 1,944 core AUROC/AUPRC pairs:

```bash
python -m tools.verify_final_release EXP_RESULTS
```

### Reproduce the reported results from `EXP_RESULTS`

The public code and the separately distributed `EXP_RESULTS` bundle form one
reproducible release. A user does not need access to the training server or to
internal experiment logs: download this repository, extract `EXP_RESULTS/` at
the repository root, install the dependencies, and run the verification command
above from that root. A successful run returns a JSON summary with 1,944 tasks
and 1,944 recomputed metric pairs. It also confirms every artifact hash and
size, the required development/reporting task scope, required metadata files,
finite score vectors, and agreement between saved and independently recomputed
AUROC/AUPRC values.

This verifies the published experiment results without claiming bit-for-bit
identical weights from a fresh GPU retraining. Fresh retraining instructions,
fixed split checks, and the exact accepted artifact layout are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Full commands are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), and
the accepted results are in [docs/RESULTS.md](docs/RESULTS.md).

## Project structure

```text
NREM-Final/
|-- nrem/                 # Models, methods, data loading, training engine
|-- configs/              # Release, ablation, sensitivity, comparison configs
|-- scripts/              # Batch and one-task/one-GPU Slurm entry points
|-- tools/                # Evaluation, aggregation, release verification
|-- tests/                # Lightweight regression and protocol tests
|-- experiments/          # Complete scientific task scope
|-- protocol/             # Fixed split membership and acceptance hashes
|-- results/              # Accepted path-free 1,944-run tables
|-- EXP_RESULTS/          # Supplied by the separate results asset
|-- train.py
`-- evaluate.py
```

## Official NREM configurations

| Configuration | Role |
|---|---|
| `configs/release/Benchmark.yaml` | Reconstruction-only benchmark |
| `configs/release/SRE.yaml` | Sample Residual Evidence |
| `configs/release/CTM.yaml` | Contextual Trajectory Modeling |
| `configs/release/NREM.yaml` | Full NREM with REA |

Exact configs for every comparison and ablation are also stored beside their
accepted checkpoint in `EXP_RESULTS`.

## Results

NREM achieves aggregate AUROC 0.956244 and AUPRC 0.953457 across seven
contamination ratios. It is best overall, but is not claimed to dominate every
method in every condition. Machine-readable sources are:

- `results/paper_400_600_run_level.csv`
- `results/paper_400_600_summary.csv`
- `results/complexity_profile.csv`
- `results/complexity_profile.json`
- `protocol/acceptance.json`

## Known limitation

The released legacy CTM scale branch saturates at its upper clamp in the
accepted checkpoints and must not be interpreted as learned heteroscedastic
uncertainty. The released artifacts are retained unchanged so the reported
legacy results remain reproducible; internal diagnostic notes and experiment
ledgers are not part of the public code package.

## Citation and license

Publication metadata will be added to `CITATION.cff` after acceptance or a
public preprint release. No public software license has been selected yet; until
the authors add one, the repository permits verification but grants no general
reuse rights.
