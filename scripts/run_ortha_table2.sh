#!/bin/bash
# ============================================================
# OrthA Table 2 full pipeline: metrics → wide multi infer → summary
#
# Phase 0: recompute single metrics (style-only TA) — no GPU
# Phase 1: wide scaffold + cascade inference on 4 GPUs
# Phase 2: multi metrics with skip gates + Table 2 summary
#
# Usage (on GPU server):
#   bash scripts/run_ortha_table2.sh
#   SKIP_INFER=1 bash scripts/run_ortha_table2.sh   # metrics + report only
# ============================================================
set -uo pipefail

REGISTRY=${REGISTRY:-concept_registry.json}
EVAL_DIR=${EVAL_DIR:-outputs/eval_infer}
MULTI_ROOT=${MULTI_ROOT:-outputs/multi_concept}
DATASETS=${DATASETS:-Datasets}
NGPU=${NGPU:-4}
N_TRIPLES=${N_TRIPLES:-8}
SEEDS=${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61}
SCAFFOLD_INIT=${SCAFFOLD_INIT:-gray}
FORCE=${FORCE:-}
SKIP_INFER=${SKIP_INFER:-}

export PYTHONUNBUFFERED=1

echo "================================================================"
echo " OrthA Table 2 pipeline"
echo "================================================================"

echo "[0] Recompute single-concept metrics (style-only TA)…"
python scripts/compute_metrics.py \
    --eval_dir "${EVAL_DIR}" \
    --registry "${REGISTRY}" \
    --datasets_dir "${DATASETS}"

if [ -z "${SKIP_INFER}" ]; then
    force_flag=""
    [ -n "${FORCE}" ] && force_flag="FORCE=1"

    echo "[1] Scaffold wide seed sweep (${SEEDS})…"
    MODE=scaffold NGPU="${NGPU}" N_TRIPLES="${N_TRIPLES}" SEEDS="${SEEDS}" \
        SCAFFOLD_INIT="${SCAFFOLD_INIT}" DATASETS_DIR="${DATASETS}" \
        ${force_flag} bash scripts/run_multi_concept_all.sh

    echo "[2] Cascade wide seed sweep…"
    MODE=cascade NGPU="${NGPU}" N_TRIPLES="${N_TRIPLES}" SEEDS="${SEEDS}" \
    ${force_flag} bash scripts/run_multi_concept_all.sh
fi

METRIC_ARGS="--rank_metric id_min --min_per_concept_id 1.0 --min_ia 0.55 --min_ta 0.55"

echo "[3] Multi metrics (scaffold)…"
python scripts/compute_metrics_multi.py --multi_dir "${MULTI_ROOT}/scaffold" \
    --registry "${REGISTRY}" --datasets_dir "${DATASETS}" ${METRIC_ARGS}

echo "[4] Multi metrics (cascade)…"
python scripts/compute_metrics_multi.py --multi_dir "${MULTI_ROOT}/cascade" \
    --registry "${REGISTRY}" --datasets_dir "${DATASETS}" ${METRIC_ARGS}

echo "[5] Table 2 summary…"
python scripts/summarize_ortha_table2.py \
    --single_metrics "${EVAL_DIR}/metrics.json" \
    --multi_root "${MULTI_ROOT}"

echo "================================================================"
echo " Done. Table 2 written to outputs/ (see summarize_ortha_table2.py output)."
echo "================================================================"
