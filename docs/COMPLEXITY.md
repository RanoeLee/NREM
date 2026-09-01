# Complexity profiling

`tools/profile_complexity.py` is the source of the parameter-count and FLOP
columns used in the manuscript. It builds every method through the same
configuration and factory used by training and runs a real inference forward
pass on one complete sample with shape `1 x 6 x 48000`.

## Reproduce

From the repository root:

```bash
python tools/profile_complexity.py
```

The command writes exact counts and normalized values to:

- `results/complexity_profile.csv`
- `results/complexity_profile.json`

Custom configurations can be profiled with repeatable `LABEL=PATH` arguments:

```bash
python tools/profile_complexity.py \
  --config "NREM=configs/release/NREM.yaml" \
  --config "DeepSVDD=configs/external/DeepSVDD_official.yaml"
```

## FLOP convention

- One multiply-add is two FLOPs.
- Linear, Conv1d, and ConvTranspose1d operations are counted from runtime input
  and output tensor shapes.
- Multi-head attention includes Q, K, V, and output projections plus the
  `QK^T` and attention-value matrix multiplications.
- The explicit association matrices in the local Anomaly Transformer adapter
  are counted in the same way.
- Transformer encoder normalization, residual, activation, and other
  pointwise work uses a fixed 16-FLOP-per-output-element convention.
- Data loading, normalization, file I/O, threshold selection, reference-set
  construction, and metric computation are outside the model-forward count.

These rules make the values deterministic across supported PyTorch versions
and avoid hidden changes in third-party profiler operator registries. FLOPs are
an architecture-level proxy; they do not establish wall-clock latency,
throughput, energy use, or peak GPU memory.

## Paper configurations

| Method | Configuration |
|---|---|
| NREM | `configs/release/NREM.yaml` |
| DeepSVDD | `configs/external/DeepSVDD_official.yaml` |
| TCN-AE | `configs/external/TCN_AE_canonical.yaml` |
| Transformer-AE | `configs/external/Transformer_AE.yaml` |
| TranAD | `configs/external/TranAD_official.yaml` |
| Anomaly Transformer | `configs/external/AnomalyTransformer_official.yaml` |
