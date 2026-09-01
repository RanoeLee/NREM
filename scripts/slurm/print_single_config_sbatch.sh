#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/slurm/print_single_config_sbatch.sh \
    --config CONFIG_YAML \
    --train-base TRAIN_BASE \
    --contam contam_XX \
    --seed SEED \
    --out OUT_ROOT \
    --session-name SESSION_NAME \
    [--job-name JOB_NAME]

This helper prints an sbatch command only. It does not submit the job.
USAGE
}

CONFIG=""
TRAIN_BASE=""
CONTAM=""
SEED=""
OUT_ROOT=""
SESSION_NAME=""
JOB_NAME="NREM_single"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --train-base)   TRAIN_BASE="$2"; shift 2 ;;
        --contam)       CONTAM="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --out)          OUT_ROOT="$2"; shift 2 ;;
        --session-name) SESSION_NAME="$2"; shift 2 ;;
        --job-name)     JOB_NAME="$2"; shift 2 ;;
        --help|-h)      usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for pair in \
    "CONFIG:${CONFIG}" \
    "TRAIN_BASE:${TRAIN_BASE}" \
    "CONTAM:${CONTAM}" \
    "SEED:${SEED}" \
    "OUT_ROOT:${OUT_ROOT}" \
    "SESSION_NAME:${SESSION_NAME}"
do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ -z "${value}" ]]; then
        echo "[ERROR] Missing required argument: ${name}" >&2
        usage >&2
        exit 2
    fi
done

if [[ "${CONTAM}" != contam_* ]]; then
    echo "[ERROR] --contam must look like contam_XX, got: ${CONTAM}" >&2
    exit 2
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --seed must be an integer, got: ${SEED}" >&2
    exit 2
fi

printf 'sbatch --job-name=%q scripts/slurm/run_single_config.sbatch' "${JOB_NAME}"
printf ' --config %q' "${CONFIG}"
printf ' --train-base %q' "${TRAIN_BASE}"
printf ' --contam %q' "${CONTAM}"
printf ' --seed %q' "${SEED}"
printf ' --out %q' "${OUT_ROOT}"
printf ' --session-name %q' "${SESSION_NAME}"
printf '\n'
