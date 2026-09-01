# Accepted 400/600 results

Central acceptance validated 1,944/1,944 tasks: 420 development tasks on the
balanced 400-product validation split and 1,524 reporting tasks on the balanced
600-product test split. Every metric was independently recomputed from its
saved score vector; all values matched scikit-learn within `1e-12`.

Across seven contamination ratios, NREM averages **0.956244 AUROC** and
**0.953457 AUPRC**. The strongest external aggregate is OCSVM at 0.927301
AUROC and 0.937601 AUPRC; DeepSVDD gives 0.926743 and 0.937196.

| Contamination | NREM AUROC | NREM AUPRC |
|---:|---:|---:|
| 0% | 0.982393 | 0.972547 |
| 1% | 0.971297 | 0.961500 |
| 5% | 0.958130 | 0.952907 |
| 8% | 0.957984 | 0.953541 |
| 12% | 0.939612 | 0.942597 |
| 16% | 0.944962 | 0.946140 |
| 20% | 0.939327 | 0.944968 |

NREM is best in AUROC at 16% and 20%, and best in AUPRC at 12%. It is not
claimed to dominate every method at every contamination ratio.

Authoritative path-free tables:

- `results/paper_400_600_run_level.csv` (1,944 rows)
- `results/paper_400_600_summary.csv` (207 grouped rows)
- `protocol/acceptance.json`

Accepted source revision:
`0cd79f5bfe0a6583cfa67f08c82a7d2961df3190`.
