# Contributing

Contributions should be small, reproducible, and scoped to a documented
scientific or engineering question.

1. Create an isolated Python environment and install `.[dev]`.
2. Add or update a focused test for behavior changes.
3. Run `python -m pytest` and `python -m compileall -q nrem tools tests`.
4. Do not commit datasets, checkpoints, run directories, logs, credentials, or
   machine-specific paths.
5. Keep paper-facing configuration names aligned with Benchmark, SRE, CTM,
   REA, and NREM terminology.

Bug reports should include the configuration path, Python and PyTorch versions,
input shape, command, and the complete traceback. Do not attach proprietary
signal data unless its release is authorized.
