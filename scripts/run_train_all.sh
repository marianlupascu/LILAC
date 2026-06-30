#!/bin/bash
# ============================================================
# Overnight TI+LoRA training for ALL concepts in the registry,
# spread across N GPUs (default 4). Uses the fixed gosling/swift
# recipe for every concept:
#   rank=64, alpha=2*rank=128 ("x2" baked in), LR=1e-4, TI_LR=5e-4,
#   1500 steps, source_dropout=1.0, fresh '<id_concept>' TI token,
#   few/generic attribute anchors (so the TI token carries identity).
#
# Input : concept_registry.json (produced by build_concept_registry.py,
#         then REVIEWED by you).
# Output: outputs/lora_weights/<concept>_lora_ti_a2x/  (one per concept)
#         logs/train_all/<concept>.log
#
# GPU scheduling: N background workers; worker g trains the concepts
# whose registry index satisfies (index % N == g). Each GPU runs its
# concepts sequentially. ~21 jobs / 4 GPUs at ~100 min each ≈ 8-9h.
#
# Resumable: a concept whose final pytorch_lora_weights.safetensors
# already exists is SKIPPED (set FORCE=1 to retrain anyway).
#
# Usage:
#   bash scripts/run_train_all.sh
#   NGPU=4 LORA_RANK=64 ALPHA_MULT=2 STEPS=1500 bash scripts/run_train_all.sh
#   REGISTRY=concept_registry.json bash scripts/run_train_all.sh
#   CONCEPTS="trump swift dog" bash scripts/run_train_all.sh   # subset only
#   FORCE=1 bash scripts/run_train_all.sh                      # ignore existing weights
#   DRY_RUN=1 bash scripts/run_train_all.sh                    # print plan, train nothing
# ============================================================

# NOTE: deliberately NOT using `set -e`. One concept failing must not
# abort the whole overnight run; failures are logged and we move on.
set -uo pipefail

REGISTRY=${REGISTRY:-concept_registry.json}
NGPU=${NGPU:-4}
# Use LORA_RANK (not RANK) to avoid collision with the distributed-training
# environment variable RANK=0 that torchrun injects into the shell session.
unset RANK 2>/dev/null || true
LORA_RANK=${LORA_RANK:-64}
ALPHA_MULT=${ALPHA_MULT:-2}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_RANK * ALPHA_MULT))}
LR=${LR:-1e-4}
TI_LR=${TI_LR:-5e-4}
STEPS=${STEPS:-1500}
SOURCE_DROPOUT=${SOURCE_DROPOUT:-1.0}
REPEATS=${REPEATS:-10}
CKPT_STEPS=${CKPT_STEPS:-250}
SEED=${SEED:-42}
PRETRAINED=${PRETRAINED:-Qwen/Qwen-Image-Edit}
CONCEPTS=${CONCEPTS:-}        # optional space-separated allow-list
# Optional: GPUS="3 4 5 6 7" to use specific GPU IDs (default: 0..NGPU-1)
GPUS=${GPUS:-}
FORCE=${FORCE:-}
DRY_RUN=${DRY_RUN:-}

if [ ! -f "${REGISTRY}" ]; then
    echo "ERROR: registry '${REGISTRY}' not found."
    echo "Run first:  python scripts/build_concept_registry.py"
    exit 1
fi

mkdir -p logs/train_all

# ---- Parse registry -> tab-separated rows: dataset_dir \t concept \t trigger \t anchors(|-joined)
# Skips entries without a concept, with type 'unknown', or no images.
# (use a while-read loop, not `mapfile`, for bash 3.2 portability)
ROWS=()
while IFS= read -r _line; do
    [ -n "${_line}" ] && ROWS+=("${_line}")
done < <(python - "${REGISTRY}" "${CONCEPTS}" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
allow = set((sys.argv[2] or "").split()) if len(sys.argv) > 2 else set()
for e in reg:
    concept = e.get("concept")
    if not concept:
        continue
    if e.get("type") == "unknown":
        continue
    if not e.get("num_images"):
        continue
    if allow and concept not in allow:
        continue
    anchors = e.get("attribute_anchors") or []
    if not anchors:
        anchors = [f"a {e.get('class_noun') or concept}"]
    trigger = e.get("trigger") or f"<id_{concept}>"
    print("\t".join([e["dataset_dir"], concept, trigger, "|".join(anchors)]))
PY
)

NJOBS=${#ROWS[@]}
if [ "${NJOBS}" -eq 0 ]; then
    echo "No trainable concepts found in ${REGISTRY} (need concept + type != unknown + images)."
    exit 1
fi

echo "============================================================"
echo " Overnight training plan"
echo "   registry : ${REGISTRY}"
echo "   concepts : ${NJOBS}"
echo "   GPUs     : ${NGPU}"
echo "   recipe   : rank=${LORA_RANK} alpha=${LORA_ALPHA} (${ALPHA_MULT}x) LR=${LR}"
echo "              TI_LR=${TI_LR} steps=${STEPS} sd=${SOURCE_DROPOUT}"
echo "============================================================"
for ((i = 0; i < NJOBS; i++)); do
    IFS=$'\t' read -r d c t a <<<"${ROWS[$i]}"
    printf "  [GPU %d] %-16s %-14s anchors=[%s]\n" "$((i % NGPU))" "${c}" "${t}" "${a//|/, }"
done
echo "============================================================"

if [ -n "${DRY_RUN}" ]; then
    echo "DRY_RUN set - nothing trained."
    exit 0
fi

# ---- Train one concept (runs inside a GPU worker) ----
train_one() {
    local gpu="$1" dataset_dir="$2" concept="$3" trigger="$4"
    shift 4
    local -a ATTRS=("$@")
    local out="outputs/lora_weights/${concept}_lora_ti_a2x"
    local log="logs/train_all/${concept}.log"
    local final="${out}/pytorch_lora_weights.safetensors"

    if [ -f "${final}" ] && [ -z "${FORCE}" ]; then
        echo "[GPU${gpu}] SKIP ${concept} (final weights already exist: ${final})"
        return 0
    fi

    echo "[GPU${gpu}] START ${concept}  ->  ${out}  (log: ${log})"
    {
        echo "================================================================"
        echo " concept=${concept} trigger=${trigger}"
        echo " gpu=${gpu} rank=${LORA_RANK} alpha=${LORA_ALPHA} lr=${LR} ti_lr=${TI_LR}"
        echo " steps=${STEPS} sd=${SOURCE_DROPOUT} anchors=(${ATTRS[*]})"
        echo " started: $(date)"
        echo "================================================================"
    } >"${log}"

    # Step 1: regenerate metadata with the TI trigger + anchors
    python scripts/regen_metadata_ti.py \
        --dataset_dir "${dataset_dir}" \
        --concept "${concept}" \
        --new_token "${trigger}" \
        --attributes "${ATTRS[@]}" \
        --output_json "${dataset_dir}/metadata.jsonl" >>"${log}" 2>&1
    if [ $? -ne 0 ]; then
        echo "[GPU${gpu}] FAIL ${concept} (regen metadata) - see ${log}"
        return 1
    fi

    # Step 2: train LoRA + TI
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_lora_qwen_edit.py \
        --pretrained_model "${PRETRAINED}" \
        --dataset_jsonl "${dataset_dir}/metadata.jsonl" \
        --output_dir "${out}" \
        --concept_name "${concept}" \
        --trigger_token "${trigger}" \
        --rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --learning_rate "${LR}" \
        --max_train_steps "${STEPS}" \
        --resolution 512 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --mixed_precision bf16 \
        --gradient_checkpointing \
        --use_8bit_adam \
        --seed "${SEED}" \
        --repeats "${REPEATS}" \
        --checkpointing_steps "${CKPT_STEPS}" \
        --source_dropout "${SOURCE_DROPOUT}" \
        --use_textual_inversion \
        --new_token "${trigger}" \
        --ti_learning_rate "${TI_LR}" \
        --ti_init_attrs "${ATTRS[@]}" \
        --attribute_anchors "${ATTRS[@]}" >>"${log}" 2>&1
    local rc=$?

    if [ "${rc}" -eq 0 ] && [ -f "${final}" ]; then
        echo "[GPU${gpu}] DONE  ${concept}  ($(date))"
    else
        echo "[GPU${gpu}] FAIL  ${concept} (rc=${rc}) - see ${log}"
    fi
    return "${rc}"
}

# ---- Build GPU list (explicit GPUS env var or 0..NGPU-1) ----
if [ -n "${GPUS}" ]; then
    GPU_LIST=(${GPUS})
else
    GPU_LIST=()
    for ((i=0; i<NGPU; i++)); do GPU_LIST+=($i); done
fi
NWORKERS=${#GPU_LIST[@]}

# ---- GPU worker: processes its modulo slice sequentially ----
run_worker() {
    local slot="$1"          # slot index 0..NWORKERS-1
    local gpu="${GPU_LIST[$slot]}"
    for ((i = slot; i < NJOBS; i += NWORKERS)); do
        IFS=$'\t' read -r dataset_dir concept trigger anchors <<<"${ROWS[$i]}"
        IFS='|' read -r -a ATTRS <<<"${anchors}"
        train_one "${gpu}" "${dataset_dir}" "${concept}" "${trigger}" "${ATTRS[@]}"
    done
    echo "[GPU${gpu}] worker finished all assigned concepts."
}

# ---- Launch one background worker per GPU ----
WPIDS=()
for ((g = 0; g < NWORKERS; g++)); do
    run_worker "${g}" &
    WPIDS+=($!)
done

echo "Launched ${NWORKERS} GPU workers on [${GPU_LIST[*]}] (pids: ${WPIDS[*]}). Waiting..."
wait

echo "============================================================"
echo " All workers finished. Summary:"
echo "============================================================"
DONE=0
MISSING=0
for ((i = 0; i < NJOBS; i++)); do
    IFS=$'\t' read -r d c t a <<<"${ROWS[$i]}"
    f="outputs/lora_weights/${c}_lora_ti_a2x/pytorch_lora_weights.safetensors"
    if [ -f "${f}" ]; then
        printf "  OK    %-16s %s\n" "${c}" "${f}"
        DONE=$((DONE + 1))
    else
        printf "  FAIL  %-16s (no final weights - check logs/train_all/%s.log)\n" "${c}" "${c}"
        MISSING=$((MISSING + 1))
    fi
done
echo "------------------------------------------------------------"
echo "  ${DONE}/${NJOBS} trained, ${MISSING} missing."
echo ""
echo "NEXT (morning sanity): bash scripts/sanity_check_all.sh"
echo "============================================================"
