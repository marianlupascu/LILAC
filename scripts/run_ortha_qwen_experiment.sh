#!/bin/bash
# ============================================================
# OrthA-on-Qwen experiment (same-backbone baseline vs LILAC).
#
# Trains 8 orthogonal-adaptation LoRAs, MERGES random triples and generates
# each in a SINGLE pass (the real OrthA protocol), then scores with the same
# TA/IA/ID gates as Table 2 and prints OrthA vs cascade vs scaffold.
#
# Phases:
#   [0] verify single-concept metrics exist (concept pool + Table-2 S baseline)
#   [1] train 8 orthogonal LoRAs, one per GPU            (skip: SKIP_TRAIN=1)
#   [1b] numerical sanity check (orthogonality + recovery)
#   [2] merge-infer triples, sharded across GPUs         (skip: SKIP_INFER=1)
#   [3] multi metrics on outputs/multi_concept/ortha
#   [4] summarize OrthA vs cascade/scaffold + HTML + S3 hint
#
# Usage (8x H100):
#   bash scripts/run_ortha_qwen_experiment.sh
#   NGPU=8 N_TRIPLES=8 bash scripts/run_ortha_qwen_experiment.sh
#   SKIP_TRAIN=1 bash scripts/run_ortha_qwen_experiment.sh   # weights already trained
#   SKIP_INFER=1 bash scripts/run_ortha_qwen_experiment.sh   # metrics/report only
# ============================================================
set -uo pipefail

# Avoid collision with torchrun's RANK=0 environment variable.
unset RANK 2>/dev/null || true

REGISTRY=${REGISTRY:-concept_registry.json}
EVAL_DIR=${EVAL_DIR:-outputs/eval_infer}
MULTI_ROOT=${MULTI_ROOT:-outputs/multi_concept}
DATASETS=${DATASETS:-Datasets}
NGPU=${NGPU:-8}
# Optional: GPUS="0 2 3 4 5 6 7" to skip a busy GPU (default: 0..NGPU-1)
GPUS=${GPUS:-}
SUFFIX=${SUFFIX:-ortha}
# CONCEPTS env var (same as run_train_all.sh) takes precedence over ORTHA_CONCEPTS
ORTHA_CONCEPTS=${CONCEPTS:-${ORTHA_CONCEPTS:-"thanos gosling margotrobbie thor hulk lebron bradpitt jamiefoxx"}}
BASIS_SEED=${BASIS_SEED:-1234}
LORA_RANK=${LORA_RANK:-64}
N_TRIPLES=${N_TRIPLES:-8}
SEEDS=${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61}
SCENES=${SCENES:-plain poker cyberpunk}
NUM_STEPS=${NUM_STEPS:-30}
CFG=${CFG:-4.0}
# 1.0 = OrthA paper default; use -1 to inherit alpha/r from training_config (2.0 for a2x)
LORA_SCALE=${LORA_SCALE:-1.0}
WIDTH=${WIDTH:-1536}
HEIGHT=${HEIGHT:-512}
FORCE=${FORCE:-}
SKIP_TRAIN=${SKIP_TRAIN:-}
SKIP_INFER=${SKIP_INFER:-}

export PYTHONUNBUFFERED=1
ORTHA_DIR="${MULTI_ROOT}/ortha"
mkdir -p logs/ortha_merge

echo "================================================================"
echo " OrthA-on-Qwen experiment"
echo "   concepts : ${ORTHA_CONCEPTS}"
echo "   GPUs=${NGPU}  rank=${LORA_RANK}  basis_seed=${BASIS_SEED}"
echo "   n_triples=${N_TRIPLES}  seeds=(${SEEDS})  ${WIDTH}x${HEIGHT}"
echo "================================================================"

# ── [0] Single-concept metrics (needed for the Table-2 'Single' baseline) ──
if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
    echo "WARNING: ${EVAL_DIR}/metrics.json not found."
    echo "  The OrthA pool auto-detects from trained weights, so triples can still"
    echo "  form, but the Table-2 'Single' baseline (S→M Δ) needs single metrics."
    echo "  Produce them with:"
    echo "    bash scripts/run_eval_infer_all.sh"
    echo "    python scripts/compute_metrics.py --eval_dir ${EVAL_DIR}"
else
    echo "[0] Found ${EVAL_DIR}/metrics.json"
fi

# ── [1] Build registry if missing (using concept_map.tsv), then train ──
if [ -z "${SKIP_TRAIN}" ]; then
    if [ ! -f "${REGISTRY}" ]; then
        echo "ERROR: Registry '${REGISTRY}' not found."
        echo "  Build it with (from the repo root):"
        echo "    python scripts/build_concept_registry.py \\"
        echo "        --concept_map concept_map.tsv \\"
        echo "        --output ${REGISTRY}"
        echo "  Then verify all 8 OrthA concepts appear with correct type/anchors,"
        echo "  and re-run this script."
        exit 1
    fi
    # Verify the 8 OrthA concepts actually exist in the registry before spending
    # GPU time on training that will immediately fail.
    _missing=$(python - "${REGISTRY}" "${ORTHA_CONCEPTS}" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
have = {e.get("concept") for e in reg if e.get("concept") and e.get("type") != "unknown"}
want = sys.argv[2].split()
missing = [c for c in want if c not in have]
if missing:
    print(" ".join(missing))
PY
)
    if [ -n "${_missing}" ]; then
        echo "ERROR: These OrthA concepts are missing or still 'unknown' in ${REGISTRY}:"
        echo "  ${_missing}"
        echo ""
        echo "  The registry needs captioned metadata for each concept. Either:"
        echo "  1) Run captioning first:  bash scripts/run_recaption_all.sh"
        echo "     then rebuild:          python scripts/build_concept_registry.py \\"
        echo "                                --concept_map concept_map.tsv --output ${REGISTRY}"
        echo "  2) Or if you already have trained *_lora_ti_a2x weights, just"
        echo "     skip training:         SKIP_TRAIN=1 bash $0"
        exit 1
    fi
    echo "[1] Training orthogonal LoRAs…"
    force_env=""
    [ -n "${FORCE}" ] && force_env="FORCE=1"
    ORTHA_CONCEPTS="${ORTHA_CONCEPTS}" NGPU="${NGPU}" SUFFIX="${SUFFIX}" \
        LORA_RANK="${LORA_RANK}" BASIS_SEED="${BASIS_SEED}" REGISTRY="${REGISTRY}" \
        ${force_env} bash scripts/run_train_ortha_all.sh
    _train_rc=$?
    if [ "${_train_rc}" -ne 0 ]; then
        echo "ERROR: run_train_ortha_all.sh exited with code ${_train_rc}. Aborting."
        exit 1
    fi
else
    echo "[1] SKIP_TRAIN=1 — using existing *_lora_${SUFFIX} weights."
fi

# ── [1b] Numerical sanity check (skipped if weights not yet present) ──
# Count how many of the 8 concepts have final weights.
_n_weights=$(for c in ${ORTHA_CONCEPTS}; do
    [ -f "outputs/lora_weights/${c}_lora_${SUFFIX}/pytorch_lora_weights.safetensors" ] && echo ok
done | wc -l)
if [ "${_n_weights}" -lt 2 ]; then
    echo "[1b] SKIP sanity check — only ${_n_weights} OrthA weight(s) found (need >= 2)."
else
    echo "[1b] OrthA invariants (orthogonality + recovery) on ${_n_weights} concepts…"
    # recover_tol=5e-3: bf16 training (eps ~1.9e-3) introduces a fp32 round-trip
    # error of ~2-4e-3; a hard FAIL at 1e-3 is too tight for bf16 weights.
    python scripts/ortha_sanity_check.py \
        --weights_root outputs/lora_weights --suffix "${SUFFIX}" \
        --concepts "${ORTHA_CONCEPTS}" \
        --recover_tol 5e-3 || {
        echo "ERROR: OrthA sanity check failed — aborting before GPU inference."
        exit 1
    }
fi

# ── [2] Merge-infer triples, sharded across GPUs ──
# Abort early if no OrthA weights are present at all.
_n_weights_infer=$(for c in ${ORTHA_CONCEPTS}; do
    [ -f "outputs/lora_weights/${c}_lora_${SUFFIX}/pytorch_lora_weights.safetensors" ] && echo ok
done | wc -l)
if [ "${_n_weights_infer}" -lt 3 ] && [ -z "${SKIP_INFER}" ]; then
    echo "ERROR: Only ${_n_weights_infer} OrthA weight(s) found (need >= 3 to form a triple)."
    echo "  Training must complete first. Aborting."
    exit 1
fi

if [ -z "${SKIP_INFER}" ]; then
    echo "[2] Merge inference (single-pass), ${NGPU} shards (${_n_weights_infer} concepts ready)…"
    force_flag=""
    [ -n "${FORCE}" ] && force_flag="--force"
    PIDS=()
    # Build the GPU list: explicit GPUS env var or 0..NGPU-1
    if [ -n "${GPUS}" ]; then
        GPU_LIST=(${GPUS})
    else
        GPU_LIST=()
        for ((i=0; i<NGPU; i++)); do GPU_LIST+=($i); done
    fi
    N_SHARDS=${#GPU_LIST[@]}
    echo "[2] Merge inference (single-pass), ${N_SHARDS} GPU(s): ${GPU_LIST[*]}"

    PIDS=()
    for ((g=0; g<N_SHARDS; g++)); do
        gpu="${GPU_LIST[$g]}"
        CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/inference_ortha_merge.py \
            --registry "${REGISTRY}" \
            --metrics "${EVAL_DIR}/metrics.json" \
            --weights_root outputs/lora_weights \
            --suffix "${SUFFIX}" \
            --out_root "${MULTI_ROOT}" \
            --concepts "${ORTHA_CONCEPTS}" \
            --n_triples "${N_TRIPLES}" \
            --seeds "${SEEDS}" \
            --scenes "${SCENES}" \
            --num_steps "${NUM_STEPS}" \
            --cfg "${CFG}" \
            --lora_scale "${LORA_SCALE}" \
            --width "${WIDTH}" \
            --height "${HEIGHT}" \
            --shard_id "${g}" \
            --num_shards "${N_SHARDS}" \
            ${force_flag} \
            >"logs/ortha_merge/gpu${gpu}_shard${g}.log" 2>&1 &
        PIDS+=($!)
        echo "[GPU${gpu} shard${g}] launched (pid $!) -> logs/ortha_merge/gpu${gpu}_shard${g}.log"
    done
    wait
    echo "    all merge shards done."
else
    echo "[2] SKIP_INFER=1 — using existing ${ORTHA_DIR} images."
fi

# ── [3] Multi metrics with Table-2 gates ──
if [ ! -f "${ORTHA_DIR}/triples.json" ]; then
    echo "ERROR: ${ORTHA_DIR}/triples.json not found — merge inference did not produce output."
    echo "  Check logs/ortha_merge/shard0.log for errors."
    exit 1
fi
METRIC_ARGS="--rank_metric id_min --min_per_concept_id 1.0 --min_ia 0.55 --min_ta 0.55"
echo "[3] Multi metrics (ortha)…"
python scripts/compute_metrics_multi.py --multi_dir "${ORTHA_DIR}" \
    --registry "${REGISTRY}" --datasets_dir "${DATASETS}" ${METRIC_ARGS}

# ── [4] Summary (OrthA vs cascade/scaffold if present) + HTML ──
if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
    echo "[4] SKIP summary — ${EVAL_DIR}/metrics.json still missing (run eval infer first)."
    echo "    Generate with:"
    echo "      bash scripts/run_eval_infer_all.sh"
    echo "      python scripts/compute_metrics.py --eval_dir ${EVAL_DIR}"
else
    echo "[4] Summary table (OrthA vs cascade/scaffold)…"
    python scripts/summarize_ortha_table2.py \
        --single_metrics "${EVAL_DIR}/metrics.json" \
        --multi_root "${MULTI_ROOT}" \
        --methods ortha,cascade,scaffold \
        --output outputs/ortha_qwen_table.json
fi

echo "================================================================"
echo " Done. OrthA-on-Qwen results:"
echo "   table : outputs/ortha_qwen_table.json / .csv"
echo "   images: ${ORTHA_DIR}/<triple>/<scene>/seed*.png"
echo "================================================================"
