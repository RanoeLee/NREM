#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=NREM_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=2-02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  sbatch scripts/slurm/run_single_config.sbatch \
    --config CONFIG_YAML \
    --train-base TRAIN_BASE \
    --contam contam_XX \
    --seed SEED \
    --out OUT_ROOT \
    --session-name SESSION_NAME

Optional:
    --project-dir PROJECT_DIR
    --conda-sh CONDA_SH
    --conda-env CONDA_ENV
    --device DEVICE

This script runs exactly one NREM training task on one Slurm-allocated GPU.
USAGE
}

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_SH=""
CONDA_ENV=""
DEVICE="cuda"

CONFIG=""
TRAIN_BASE=""
CONTAM=""
SEED=""
OUT_ROOT=""
SESSION_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --train-base)   TRAIN_BASE="$2"; shift 2 ;;
        --contam)       CONTAM="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --out)          OUT_ROOT="$2"; shift 2 ;;
        --session-name) SESSION_NAME="$2"; shift 2 ;;
        --project-dir)  PROJECT_DIR="$2"; shift 2 ;;
        --conda-sh)     CONDA_SH="$2"; shift 2 ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
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

require_arg "--config" "${CONFIG}"
require_arg "--train-base" "${TRAIN_BASE}"
require_arg "--contam" "${CONTAM}"
require_arg "--seed" "${SEED}"
require_arg "--out" "${OUT_ROOT}"
require_arg "--session-name" "${SESSION_NAME}"

if [[ "${CONTAM}" != contam_* ]]; then
    echo "[ERROR] --contam must look like contam_XX, got: ${CONTAM}" >&2
    exit 2
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --seed must be an integer, got: ${SEED}" >&2
    exit 2
fi

TRAIN_ROOT="${TRAIN_BASE%/}/${CONTAM}"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${TRAIN_ROOT}" ]]; then
    echo "[ERROR] TRAIN_ROOT not found: ${TRAIN_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${OUT_ROOT}" ]]; then
    echo "[ERROR] OUT_ROOT not found: ${OUT_ROOT}" >&2
    exit 1
fi

EXP_NAME="$(
    awk '
        /^[[:space:]]*experiment:[[:space:]]*$/ { in_exp=1; next }
        /^[^[:space:]]/ { if ($0 !~ /^[[:space:]]*#/) in_exp=0 }
        in_exp && /^[[:space:]]*name:[[:space:]]*/ {
            sub(/^[[:space:]]*name:[[:space:]]*/, "", $0)
            gsub(/["'\'']/, "", $0)
            print $0
            exit
        }
    ' "${CONFIG}"
)"

if [[ -z "${EXP_NAME}" ]]; then
    echo "[ERROR] Could not parse experiment.name from ${CONFIG}" >&2
    exit 1
fi

RUN_DIR="${OUT_ROOT%/}/${SESSION_NAME}/seed_${SEED}/${EXP_NAME}/${CONTAM}"

if [[ -e "${RUN_DIR}" ]]; then
    echo "[ERROR] Refusing to overwrite existing output directory: ${RUN_DIR}" >&2
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
    python "${TRAIN_SCRIPT}"
    --config "${CONFIG}"
    --train_root "${TRAIN_ROOT}"
    --out "${OUT_ROOT}"
    --session_name "${SESSION_NAME}"
    --device "${DEVICE}"
    --seeds "${SEED}"
    --sweep_contam false
)

echo "========================================"
echo "[BOOT] NREM single-config Slurm task"
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
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Experiment: ${EXP_NAME}"
echo "[INFO] Train base: ${TRAIN_BASE}"
echo "[INFO] Contam: ${CONTAM}"
echo "[INFO] Train root: ${TRAIN_ROOT}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Out root: ${OUT_ROOT}"
echo "[INFO] Session name: ${SESSION_NAME}"
echo "[INFO] Expected run dir: ${RUN_DIR}"
echo "[INFO] Device: ${DEVICE}"
echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo
echo "========================================"

srun "${CMD[@]}"
