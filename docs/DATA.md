# IPML-MotorEOL data and fixed 400/600 split

## Download

**IPML-MotorEOL: A Product-Level Vibro-Acoustic Dataset for Motor End-of-Line
Quality Inspection** is distributed as `IPML-MotorEOL.zip`:

- **Baidu Netdisk / 百度网盘:** [Download IPML-MotorEOL.zip](https://pan.baidu.com/s/1zyJVdGbBN1FD6UIPlrq5pQ?pwd=EAAI)
- **Extraction code / 提取码:** `EAAI`

## Signal format and directory layout

Each `.npy` file stores one complete motor product as a `(6, 48000)` array
in channel-by-time order with dtype `float32`. The two-second signal is sampled
at 24 kHz and contains three acoustic channels and three acceleration channels.
OK and NG labels describe overall product quality, not defect type or location.

Extract the archive so that the following structure is under the repository's
`data/` directory. If the ZIP already contains an `IPML-MotorEOL/` folder,
extract it into `data/` to avoid nesting that folder twice.

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
|-- val/
|   |-- ok/               # 200 OK
|   `-- ng/               # 200 NG
|-- test/
|   |-- ok/               # 300 OK
|   `-- ng/               # 300 NG
`-- README.md
```

The release contains 29,000 `.npy` files, totaling 33,411,712,000 bytes
(31.12 GiB) of uncompressed signal data. These represent 5,800 unique motors;
the training conditions reuse products from a common pool of 4,000 OK and
800 NG motors and should not be combined into one training set.

| Training folder | NG contamination | OK | NG | Total |
| --- | ---: | ---: | ---: | ---: |
| `contam_00` | 0% | 4,000 | 0 | 4,000 |
| `contam_01` | 1% | 3,960 | 40 | 4,000 |
| `contam_05` | 5% | 3,800 | 200 | 4,000 |
| `contam_08` | 8% | 3,680 | 320 | 4,000 |
| `contam_12` | 12% | 3,520 | 480 | 4,000 |
| `contam_16` | 16% | 3,360 | 640 | 4,000 |
| `contam_20` | 20% | 3,200 | 800 | 4,000 |

## Fixed evaluation protocol

The archive contains only the three data folders and its README. Use the
versioned [`protocol/split_manifest.csv`](../protocol/split_manifest.csv) and
[`protocol/split_metadata.json`](../protocol/split_metadata.json) from this
code repository; these files do not need to be copied into the dataset folder.
For the evaluation command, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

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
