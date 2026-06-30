#!/bin/bash
# ============================================================
# OrthA (Orthogonal Adaptation, Po et al. CVPR 2024) training for the
# fixed 8-concept set, one concept per GPU (8x H100).
#
# Each concept is trained with the SAME recipe as run_train_all.sh
# (rank=64, alpha=128, TI on, 1500 steps) PLUS the orthogonal-basis
# constraint: its up-projection B is frozen to a disjoint column block
# of a globally-shared random orthogonal basis. The block index is the
# concept's position in ORTHA_CONCEPTS, so any subset (triple) of these
# LoRAs has mutually orthogonal column spaces and can be merged by
# summation with minimal interference.
#
# CRITICAL: --ortha_basis_seed and --ortha_num_concepts MUST be identical
# across all 8 runs (they define the shared basis). Do not change the
# ORTHA_CONCEPTS ordering after training without retraining.
#
# Output: outputs/lora_weights/<concept>_lora_ortha/  (one per concept)
#         logs/train_ortha/<concept>.log
#
# Usage:
#   bash scripts/run_train_ortha_all.sh
#   NGPU=8 STEPS=1500 bash scripts/run_train_ortha_all.sh
#   FORCE=1 bash scripts/run_train_ortha_all.sh      # retrain existing
#   DRY_RUN=1 bash scripts/run_train_ortha_all.sh    # print plan only
# ============================================================
set -uo pipefail

# The 8 recommended concepts (fixed ordering => fixed orthogonal block index).
ORTHA_CONCEPTS=${ORTHA_CONCEPTS:-"thanos gosling margotrobbie thor hulk lebron bradpitt jamiefoxx"}

REGISTRY=${REGISTRY:-concept_registry.json}
NGPU=${NGPU:-8}
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
SUFFIX=${SUFFIX:-ortha}
BASIS_SEED=${BASIS_SEED:-1234}
# Optional: GPUS="3 4 5 6 7" to use specific GPU IDs (default: 0..NGPU-1)
GPUS=${GPUS:-}
FORCE=${FORCE:-}
DRY_RUN=${DRY_RUN:-}

if [ ! -f "${REGISTRY}" ]; then
    echo "ERROR: registry '${REGISTRY}' not found."
    echo "Run first:  python scripts/build_concept_registry.py"
    exit 1
fi

mkdir -p logs/train_ortha

# ---- Build ordered rows for ORTHA_CONCEPTS: dataset_dir \t concept \t trigger \t anchors(|) ----
# Index in the output == orthogonal block index. Fails if a concept is missing.
ROWS=()
while IFS= read -r _line; do
    [ -n "${_line}" ] && ROWS+=("${_line}")
done < <(python - "${REGISTRY}" "${ORTHA_CONCEPTS}" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
wanted = sys.argv[2].split()
by_concept = {e.get("concept"): e for e in reg if e.get("concept")}
missing = [c for c in wanted if c not in by_concept]
if missing:
    sys.stderr.write("ERROR: concepts not in registry: %s\n" % ", ".join(missing))
    sys.exit(2)
for c in wanted:  # preserve ORTHA_CONCEPTS order -> stable block index
    e = by_concept[c]
    anchors = e.get("attribute_anchors") or []
    if not anchors:
        anchors = [f"a {e.get('class_noun') or c}"]
    trigger = e.get("trigger") or f"<id_{c}>"
    print("\t".join([e["dataset_dir"], c, trigger, "|".join(anchors)]))
PY
)

NJOBS=${#ROWS[@]}
NUM_CONCEPTS=${NJOBS}
if [ "${NJOBS}" -eq 0 ]; then
    echo "No OrthA concepts resolved. Check ORTHA_CONCEPTS and the registry."
    exit 1
fi

echo "============================================================"
echo " OrthA orthogonal-LoRA training plan"
echo "   registry     : ${REGISTRY}"
echo "   concepts     : ${NJOBS}  (block index = position below)"
echo "   GPUs         : ${NGPU}"
echo "   basis        : seed=${BASIS_SEED} width=${NUM_CONCEPTS}*${LORA_RANK}=$((NUM_CONCEPTS * LORA_RANK)) cols"
echo "   recipe       : rank=${LORA_RANK} alpha=${LORA_ALPHA} (${ALPHA_MULT}x) LR=${LR}"
echo "                  TI_LR=${TI_LR} steps=${STEPS} sd=${SOURCE_DROPOUT} suffix=${SUFFIX}"
echo "============================================================"
for ((i = 0; i < NJOBS; i++)); do
    IFS=$'\t' read -r d c t a <<<"${ROWS[$i]}"
    printf "  [GPU %d] block=%d  %-16s %-16s anchors=[%s]\n" \
        "$((i % NGPU))" "$i" "${c}" "${t}" "${a//|/, }"
done
echo "============================================================"

if [ -n "${DRY_RUN}" ]; then
    echo "DRY_RUN set - nothing trained."
    exit 0
fi

# ---- Train one orthogonal-LoRA concept (runs inside a GPU worker) ----
train_one() {
    local gpu="$1" dataset_dir="$2" concept="$3" trigger="$4" block_idx="$5"
    shift 5
    local -a ATTRS=("$@")
    local out="outputs/lora_weights/${concept}_lora_${SUFFIX}"
    local log="logs/train_ortha/${concept}.log"
    local final="${out}/pytorch_lora_weights.safetensors"

    if [ -f "${final}" ] && [ -z "${FORCE}" ]; then
        echo "[GPU${gpu}] SKIP ${concept} (final weights already exist: ${final})"
        return 0
    fi

    echo "[GPU${gpu}] START ${concept} (block ${block_idx})  ->  ${out}  (log: ${log})"
    {
        echo "================================================================"
        echo " concept=${concept} trigger=${trigger} ortha_block=${block_idx}"
        echo " gpu=${gpu} rank=${LORA_RANK} alpha=${LORA_ALPHA} lr=${LR} ti_lr=${TI_LR}"
        echo " steps=${STEPS} sd=${SOURCE_DROPOUT} basis_seed=${BASIS_SEED}"
        echo " num_concepts=${NUM_CONCEPTS} anchors=(${ATTRS[*]})"
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

    # Step 2: train orthogonal LoRA + TI
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
        --attribute_anchors "${ATTRS[@]}" \
        --ortha_orthogonal \
        --ortha_concept_index "${block_idx}" \
        --ortha_num_concepts "${NUM_CONCEPTS}" \
        --ortha_basis_seed "${BASIS_SEED}" >>"${log}" 2>&1
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
    local slot="$1"
    local gpu="${GPU_LIST[$slot]}"
    for ((i = slot; i < NJOBS; i += NWORKERS)); do
        IFS=$'\t' read -r dataset_dir concept trigger anchors <<<"${ROWS[$i]}"
        IFS='|' read -r -a ATTRS <<<"${anchors}"
        train_one "${gpu}" "${dataset_dir}" "${concept}" "${trigger}" "${i}" "${ATTRS[@]}"
    done
    echo "[GPU${gpu}] worker finished all assigned concepts."
}

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
    f="outputs/lora_weights/${c}_lora_${SUFFIX}/pytorch_lora_weights.safetensors"
    if [ -f "${f}" ]; then
        printf "  OK    block=%d %-16s %s\n" "$i" "${c}" "${f}"
        DONE=$((DONE + 1))
    else
        printf "  FAIL  block=%d %-16s (no final weights - check logs/train_ortha/%s.log)\n" "$i" "${c}" "${c}"
        MISSING=$((MISSING + 1))
    fi
done
echo "------------------------------------------------------------"
echo "  ${DONE}/${NJOBS} trained, ${MISSING} missing."
echo ""
echo "NEXT: bash scripts/run_ortha_qwen_experiment.sh   (merge-infer + metrics + table)"
echo "============================================================"
