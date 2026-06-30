#!/bin/bash
# ============================================================
# Multi-GPU launcher for cascaded multi-concept eval inference.
# Shards triples across NGPU workers (one process per GPU).
#
# All workers compute the SAME deterministic triple set (fixed rng_seed);
# shard 0 writes outputs/multi_concept/triples.json, each worker renders
# only its slice (triple_idx % NGPU == worker_id).
#
# Usage:
#   bash scripts/run_multi_concept_all.sh
#   NGPU=4 N_TRIPLES=8 SEEDS="42 43 ... 61" bash scripts/run_multi_concept_all.sh
#   MODE=scaffold SCAFFOLD_INIT=ref_strip bash scripts/run_multi_concept_all.sh
#   FORCE=1 bash scripts/run_multi_concept_all.sh
# ============================================================
set -uo pipefail

NGPU=${NGPU:-4}
REGISTRY=${REGISTRY:-concept_registry.json}
METRICS=${METRICS:-outputs/eval_infer/metrics.json}
OUT_ROOT=${OUT_ROOT:-outputs/multi_concept}
MODE=${MODE:-cascade}          # cascade | scaffold
SCAFFOLD_INIT=${SCAFFOLD_INIT:-gray}  # gray | ref_strip (scaffold only)
DATASETS_DIR=${DATASETS_DIR:-Datasets}
N_TRIPLES=${N_TRIPLES:-8}
TRIPLES=${TRIPLES:-}           # explicit triples e.g. "thanos,gosling,margotrobbie"
SEEDS=${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61}
SCENES=${SCENES:-plain poker cyberpunk}
NUM_STEPS=${NUM_STEPS:-30}
CFG=${CFG:-4.0}
WIDTH=${WIDTH:-1536}
HEIGHT=${HEIGHT:-512}
FORCE=${FORCE:-}

mkdir -p logs/multi_concept

# Stream prints to the log files instead of block-buffering them.
export PYTHONUNBUFFERED=1

force_flag=""
[ -n "${FORCE}" ] && force_flag="--force"

echo "================================================================"
echo " Multi-concept ${MODE}: ${NGPU} GPU(s), n_triples=${N_TRIPLES}"
[ -n "${TRIPLES}" ] && echo "   explicit triples=${TRIPLES}"
echo "   seeds=(${SEEDS})  scenes=(${SCENES})  ${WIDTH}x${HEIGHT}"
[ "${MODE}" = "scaffold" ] && echo "   scaffold_init=${SCAFFOLD_INIT}"
echo "================================================================"

PIDS=()
for ((g=0; g<NGPU; g++)); do
    CUDA_VISIBLE_DEVICES="${g}" python -u scripts/inference_cascade.py \
        --registry "${REGISTRY}" \
        --metrics "${METRICS}" \
        --out_root "${OUT_ROOT}" \
        --mode "${MODE}" \
        --scaffold_init "${SCAFFOLD_INIT}" \
        --datasets_dir "${DATASETS_DIR}" \
        --n_triples "${N_TRIPLES}" \
        --triples "${TRIPLES}" \
        --seeds "${SEEDS}" \
        --scenes "${SCENES}" \
        --num_steps "${NUM_STEPS}" \
        --cfg "${CFG}" \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --shard_id "${g}" \
        --num_shards "${NGPU}" \
        ${force_flag} \
        >"logs/multi_concept/${MODE}_shard${g}.log" 2>&1 &
    PIDS+=($!)
    echo "[GPU${g}] launched (pid $!) -> logs/multi_concept/${MODE}_shard${g}.log"
done

wait

MULTI_DIR="${OUT_ROOT}/${MODE}"
echo "================================================================"
echo " All shards done (mode=${MODE}). Next:"
echo "   python scripts/compute_metrics_multi.py --multi_dir ${MULTI_DIR} \\"
echo "       --registry ${REGISTRY} --datasets_dir ${DATASETS_DIR} \\"
echo "       --rank_metric id_min --min_per_concept_id 1.0 --min_ia 0.55 --min_ta 0.55"
echo "================================================================"
