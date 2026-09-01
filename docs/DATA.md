# IPML-MotorEOL data and fixed 400/600 split

NREM expects one two-dimensional NumPy array per motor product in either
`[channels, time]` or `[time, channels]` layout. Download the separate
`IPML-MotorEOL` asset and place it as:

```text
data/IPML-MotorEOL/
|-- train/
|   |-- contam_00/
|   |-- contam_01/
|   |-- contam_05/
|   |-- contam_08/
|   |-- contam_12/
|   |-- contam_16/
|   `-- contam_20/
|-- val/ok, val/ng         # 200 OK + 200 NG
|-- test/ok, test/ng       # 300 OK + 300 NG
|-- split_manifest.csv
`-- split_metadata.json
```

Training never uses OK/NG labels. The 400-product validation split is used only
to freeze SRE mask ratio 0.30 and SRE/CTM weight 0.20/0.80. All paper-facing
comparisons use the disjoint 600-product test split. Do not resplit the public
asset: it is the exact seed-2025 membership used by the paper.

The canonical `split_manifest.csv` has SHA-256
`cc5a0066979718664efc04969dad9af764a6610a074c38a6eda0d354c7f06304`.
The derived population hashes are:

- 400-val: `77338fe0a19ee4201f698132c9cc39767f8f8e7aa4bc67cd3ee9baeefa893171`
- 600-test: `835578521bcf50c94ae8aea0ae7efd093bda5bd83a9bba2e72f343efd0f24f54`

`evaluate.py` refuses a manifest hash mismatch, wrong split name, wrong class
counts, duplicate products, or population mismatch.
