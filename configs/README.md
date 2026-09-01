# Configuration layout

The public configuration tree contains only settings used in the paper.

- `release/`: final Benchmark, SRE, CTM, and NREM configurations. The reported
  NREM uses the MLP backbone, a 0.30 SRE training split, and 0.20/0.80 REA
  stream weights for `u1`/`u2`.
- `ablations/sre/`: SRE mechanism ablations.
- `ablations/ctm/`: CTM masking and inference-group ablations.
- `ablations/rea/`: REA stream, clipping, and weight-sensitivity ablations.
- `mask_ratio_sensitivity/`: the 0.30/0.40/0.50 SRE training-split study.
- `external/`: external methods reported in the comparison and efficiency
  tables.

All ablation files inherit from `configs/release/NREM.yaml`. They therefore
express only the factor changed by that experiment. Historical H0/H1/H2/H3,
WS/WC/G, CWRU, SMD, and exploratory configuration trees are intentionally not
part of the release surface.
