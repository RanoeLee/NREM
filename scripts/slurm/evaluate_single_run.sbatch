#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=NREM_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  sbatch scripts/slurm/evaluate_single_run.sbatch \
    --run-dir RUN_DIR \
    --eval-root EVAL_ROOT \
    --score-mode SCORE_MODE \
    --split-manifest SPLIT_MANIFEST \
    --split-name SPLIT_NAME \
    --expected-ok INTEGER \
    --expected-ng INTEGER

Optional:
    --project-dir PROJECT_DIR
    --conda-sh CONDA_SH
    --conda-env CONDA_ENV
    --ckpt-name CKPT_NAME
    --device DEVICE
    --mode MODE
    --resa-energy STREAM_OR_COMMA_LIST
    --resa-cauchy-clip FLOAT
    --resa-stream-weights COMMA_LIST
    --eval-dyn-groups INTEGER
    --reference-path PATH
    --result-tag SAFE_TAG
    --eval-batch-size INTEGER

This script evaluates exactly one completed NREM run on one Slurm-allocated GPU.
It refuses to overwrite an existing metrics.json file.
USAGE
}

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_SH=""
CONDA_ENV=""
CKPT_NAME="ema.pt"
DEVICE="cuda"
MODE="fixed"

RUN_DIR=""
EVAL_ROOT=""
SCORE_MODE=""
SPLIT_MANIFEST=""
SPLIT_NAME=""
EXPECTED_OK=""
EXPECTED_NG=""
RESA_ENERGY=""
RESA_CAUCHY_CLIP=""
RESA_STREAM_WEIGHTS=""
EVAL_DYN_GROUPS=""
REFERENCE_PATH=""
RESULT_TAG=""
EVAL_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)      RUN_DIR="$2"; shift 2 ;;
        --eval-root)    EVAL_ROOT="$2"; shift 2 ;;
        --score-mode)   SCORE_MODE="$2"; shift 2 ;;
        --split-manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
        --split-name)   SPLIT_NAME="$2"; shift 2 ;;
        --expected-ok)  EXPECTED_OK="$2"; shift 2 ;;
        --expected-ng)  EXPECTED_NG="$2"; shift 2 ;;
        --project-dir)  PROJECT_DIR="$2"; shift 2 ;;
        --conda-sh)     CONDA_SH="$2"; shift 2 ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --ckpt-name)    CKPT_NAME="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --resa-energy)  RESA_ENERGY="$2"; shift 2 ;;
        --resa-cauchy-clip) RESA_CAUCHY_CLIP="$2"; shift 2 ;;
        --resa-stream-weights) RESA_STREAM_WEIGHTS="$2"; shift 2 ;;
        --eval-dyn-groups) EVAL_DYN_GROUPS="$2"; shift 2 ;;
        --reference-path) REFERENCE_PATH="$2"; shift 2 ;;
        --result-tag)   RESULT_TAG="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --help|-h)      usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

require_arg() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "[ERROR] Missing required argument: ${name}" >&2
        usage >&2
        exit 2
    fi
}

require_arg "--run-dir" "${RUN_DIR}"
require_arg "--eval-root" "${EVAL_ROOT}"
require_arg "--score-mode" "${SCORE_MODE}"
require_arg "--split-manifest" "${SPLIT_MANIFEST}"
require_arg "--split-name" "${SPLIT_NAME}"
require_arg "--expected-ok" "${EXPECTED_OK}"
require_arg "--expected-ng" "${EXPECTED_NG}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[ERROR] RUN_DIR not found: ${RUN_DIR}" >&2
    exit 1
fi

if [[ ! -f "${RUN_DIR}/config.yaml" ]]; then
    echo "[ERROR] RUN_DIR has no config.yaml: ${RUN_DIR}" >&2
    exit 1
fi

if [[ ! -d "${EVAL_ROOT}" ]]; then
    echo "[ERROR] EVAL_ROOT not found: ${EVAL_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
    echo "[ERROR] SPLIT_MANIFEST not found: ${SPLIT_MANIFEST}" >&2
    exit 1
fi

CKPT_STEM="${CKPT_NAME%.pt}"
EVAL_NAME="$(basename "${EVAL_ROOT}")"
METRICS_DIR="${RUN_DIR}/eval_${EVAL_NAME}_${CKPT_STEM}_${SCORE_MODE}"
if [[ -n "${RESULT_TAG}" ]]; then
    if [[ ! "${RESULT_TAG}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
        echo "[ERROR] Invalid --result-tag: ${RESULT_TAG}" >&2
        exit 2
    fi
    METRICS_DIR="${METRICS_DIR}_${RESULT_TAG}"
fi
METRICS_FILE="${METRICS_DIR}/metrics.json"

if [[ -e "${METRICS_FILE}" ]]; then
    echo "[ERROR] Refusing to overwrite existing metrics: ${METRICS_FILE}" >&2
    exit 1
fi

if [[ -n "${CONDA_SH}" ]]; then
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "[ERROR] conda.sh not found: ${CONDA_SH}" >&2
        exit 1
    fi
    if [[ -z "${CONDA_ENV}" ]]; then
        echo "[ERROR] --conda-env is required when --conda-sh is used" >&2
        exit 1
    fi
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CMD=(
    python "${PROJECT_DIR}/evaluate.py"
    --root "${RUN_DIR}"
    --eval_root "${EVAL_ROOT}"
    --mode "${MODE}"
    --ckpt_name "${CKPT_NAME}"
    --device "${DEVICE}"
    --score_mode "${SCORE_MODE}"
    --split_manifest "${SPLIT_MANIFEST}"
    --split_name "${SPLIT_NAME}"
    --expected_ok "${EXPECTED_OK}"
    --expected_ng "${EXPECTED_NG}"
)

if [[ -n "${RESA_ENERGY}" ]]; then
    CMD+=(--resa_energy "${RESA_ENERGY}")
fi
if [[ -n "${RESA_CAUCHY_CLIP}" ]]; then
    CMD+=(--resa_cauchy_clip "${RESA_CAUCHY_CLIP}")
fi
if [[ -n "${RESA_STREAM_WEIGHTS}" ]]; then
    CMD+=(--resa_stream_weights "${RESA_STREAM_WEIGHTS}")
fi
if [[ -n "${EVAL_DYN_GROUPS}" ]]; then
    CMD+=(--eval_dyn_groups "${EVAL_DYN_GROUPS}")
fi
if [[ -n "${REFERENCE_PATH}" ]]; then
    CMD+=(--reference_path "${REFERENCE_PATH}")
fi
if [[ -n "${RESULT_TAG}" ]]; then
    CMD+=(--result_tag "${RESULT_TAG}")
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
    CMD+=(--eval_batch_size "${EVAL_BATCH_SIZE}")
fi

echo "========================================"
echo "[BOOT] NREM single-run evaluation Slurm task"
echo "[BOOT] Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[BOOT] Hostname: $(hostname)"
echo "[BOOT] Submit dir: ${SLURM_SUBMIT_DIR:-NA}"
echo "[BOOT] Current dir: $(pwd)"
echo "[BOOT] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[BOOT] Job name: ${SLURM_JOB_NAME:-NA}"
echo "[BOOT] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "[BOOT] Git commit: $(git rev-parse HEAD)"
echo "[BOOT] Conda env: ${CONDA_DEFAULT_ENV:-NA}"
echo "[BOOT] Python path: $(which python)"
python --version
echo "----------------------------------------"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Eval root: ${EVAL_ROOT}"
echo "[INFO] Score mode: ${SCORE_MODE}"
echo "[INFO] Split manifest: ${SPLIT_MANIFEST}"
echo "[INFO] Split name: ${SPLIT_NAME}"
echo "[INFO] Expected OK/NG: ${EXPECTED_OK}/${EXPECTED_NG}"
echo "[INFO] Checkpoint: ${CKPT_NAME}"
echo "[INFO] Result tag: ${RESULT_TAG:-default}"
echo "[INFO] REA energy override: ${RESA_ENERGY:-none}"
echo "[INFO] Cauchy clip override: ${RESA_CAUCHY_CLIP:-none}"
echo "[INFO] Stream weights override: ${RESA_STREAM_WEIGHTS:-none}"
echo "[INFO] CTM eval groups override: ${EVAL_DYN_GROUPS:-none}"
echo "[INFO] Reference path override: ${REFERENCE_PATH:-default}"
echo "[INFO] Expected metrics: ${METRICS_FILE}"
echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo
echo "========================================"

srun "${CMD[@]}"
