#!/bin/bash
# ============================================================
# Single-concept evaluation inference for all trained LoRAs.
# Generates BOTH T2I (gray canvas) and I2I (real ref image) outputs.
#
# MODE=both (default) : t2i + i2i
# MODE=t2i            : gray canvas only
# MODE=i2i            : reference image source only
#
# Outputs:
#   outputs/eval_infer/t2i/<concept>/<prompt_slug>/seed<S>.png
#   outputs/eval_infer/i2i/<concept>/<prompt_slug>/seed<S>.png
#   logs/eval_infer/<concept>.log
#
# Usage:
#   bash scripts/run_eval_infer_all.sh
#   NGPU=4 SEEDS="42 43" bash scripts/run_eval_infer_all.sh
#   CONCEPTS="trump messi" bash scripts/run_eval_infer_all.sh
#   MODE=i2i FORCE=1 CONCEPTS="swift thor" bash scripts/run_eval_infer_all.sh
#   DRY_RUN=1 bash scripts/run_eval_infer_all.sh
# ============================================================
set -uo pipefail

REGISTRY=${REGISTRY:-concept_registry.json}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-outputs/lora_weights}
OUT_ROOT=${OUT_ROOT:-outputs/eval_infer}
NGPU=${NGPU:-4}
SEEDS=${SEEDS:-42 43}
SUFFIX=${SUFFIX:-ti_a2x}
NUM_STEPS=${NUM_STEPS:-30}
CFG=${CFG:-4.0}
CONCEPTS=${CONCEPTS:-}
# Optional: GPUS="3 4 5 6 7" to use specific GPU IDs (default: 0..NGPU-1)
GPUS=${GPUS:-}
DRY_RUN=${DRY_RUN:-}
MODE=${MODE:-both}   # t2i | i2i | both

if [ ! -f "${REGISTRY}" ]; then
    echo "ERROR: registry '${REGISTRY}' not found."; exit 1
fi

mkdir -p logs/eval_infer

# ---- Build per-concept rows: concept \t type \t trigger \t anchors(|-sep) \t dataset_dir ----
ROWS=()
while IFS= read -r _l; do [ -n "${_l}" ] && ROWS+=("${_l}"); done < <(
python - "${REGISTRY}" "${CONCEPTS}" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
allow = set((sys.argv[2] or "").split()) if len(sys.argv) > 2 else set()
for e in reg:
    c = e.get("concept")
    if not c or e.get("type","unknown") == "unknown": continue
    if allow and c not in allow: continue
    t   = e.get("trigger") or f"<id_{c}>"
    typ = e.get("type","person")
    anc = e.get("attribute_anchors") or []
    d   = e.get("dataset_dir","")
    print("\t".join([c, typ, t, "|".join(anc), d]))
PY
)

NJOBS=${#ROWS[@]}
if [ "${NJOBS}" -eq 0 ]; then
    echo "No concepts found in ${REGISTRY}."; exit 1
fi

echo "================================================================"
echo " Eval inference: ${NJOBS} concept(s), ${NGPU} GPU(s), MODE=${MODE}"
echo "   seeds=${SEEDS}  steps=${NUM_STEPS}  cfg=${CFG}"
echo "================================================================"
for ((i=0;i<NJOBS;i++)); do
    IFS=$'\t' read -r c typ t a _d <<<"${ROWS[$i]}"
    printf "  [GPU %d] %-16s %-8s %s\n" "$((i%NGPU))" "${c}" "${typ}" "${t}"
done
echo "================================================================"
[ -n "${DRY_RUN}" ] && echo "DRY_RUN — nothing generated." && exit 0

# ---- Prompt templates (paper-aligned, per type) ----
person_prompts=(
    "portrait|a photo of %TRIGGER%, %ANCHORS%, neutral background, studio lighting"
    "cyberpunk|a photo of %TRIGGER%, in the style of Cyberpunk 2077, 4K, ultra-realistic"
    "vangogh|a painting of %TRIGGER%, in the style of Van Gogh, oil painting, expressive brush strokes"
    "forest|a photo of %TRIGGER%, in a forest, natural daylight, bokeh background"
    "times_square|a photo of %TRIGGER%, in Times Square, street photography, neon lights"
    "superhero|a photo of %TRIGGER%, wearing a superhero costume, dramatic lighting"
    "poker|a photo of %TRIGGER%, playing poker, cinematic lighting, dramatic"
    "class_only|a photo of a %CLASS%, neutral background"
)
animal_prompts=(
    "portrait|a photo of %TRIGGER%, close-up portrait, neutral background, studio lighting"
    "cyberpunk|a photo of %TRIGGER%, in the style of Cyberpunk 2077, 4K, ultra-realistic, neon"
    "pixar|a photo of %TRIGGER%, in the style of Pixar animation, 4K, colorful"
    "forest|a photo of %TRIGGER%, in a forest, natural daylight, bokeh background"
    "beach|a photo of %TRIGGER%, at the beach, sunny day, golden hour"
    "watercolor|a painting of %TRIGGER%, watercolor style, soft colors"
    "class_only|a photo of a %CLASS%, neutral background"
)
object_prompts=(
    "portrait|a photo of %TRIGGER%, product shot, white background, studio lighting"
    "cyberpunk|a photo of %TRIGGER%, in the style of Cyberpunk 2077, 4K, ultra-realistic, neon"
    "watercolor|a painting of %TRIGGER%, watercolor style, soft colors"
    "forest|a photo of %TRIGGER%, placed in a forest, natural daylight"
    "fashion|a photo of %TRIGGER%, worn by a model, fashion photography, editorial"
    "class_only|a photo of a %CLASS%, neutral background"
)

# Find first image in a dataset directory (for i2i reference source).
find_ref_image() {
    local dir="$1"
    [ -z "${dir}" ] || [ ! -d "${dir}" ] && echo "" && return
    ls "${dir}"/*.jpg "${dir}"/*.jpeg "${dir}"/*.png "${dir}"/*.webp 2>/dev/null \
        | sort | head -1
}

infer_one() {
    local gpu="$1" concept="$2" ctype="$3" trigger="$4" dataset_dir="$5"
    shift 5
    local -a ANCS=("$@")
    local anchors_csv; anchors_csv=$(IFS=', '; echo "${ANCS[*]}")
    local class_word="${ANCS[0]:-${ctype}}"
    local bare_class="${class_word#a }"; bare_class="${bare_class#an }"
    local lora="${WEIGHTS_ROOT}/${concept}_lora_${SUFFIX}"
    local log="logs/eval_infer/${concept}.log"

    if [ ! -f "${lora}/pytorch_lora_weights.safetensors" ]; then
        echo "[GPU${gpu}] SKIP ${concept} — no weights at ${lora}"
        return 0
    fi

    echo "[GPU${gpu}] INFER ${concept} (${ctype}) mode=${MODE}"
    { echo "concept=${concept} type=${ctype} trigger=${trigger} mode=${MODE}";
      echo "anchors=(${ANCS[*]})"; echo "started=$(date)"; } >"${log}"

    # Find reference image once per concept (used for every i2i prompt).
    local ref_img
    ref_img=$(find_ref_image "${dataset_dir}")
    if [ "${MODE}" != "t2i" ] && [ -z "${ref_img}" ]; then
        echo "  WARN: no ref image found in '${dataset_dir}' — i2i will be skipped" >>"${log}"
    else
        echo "  ref_image=${ref_img}" >>"${log}"
    fi

    local -n prompt_list="${ctype}_prompts" 2>/dev/null || {
        local -n prompt_list="person_prompts"; }

    for entry in "${prompt_list[@]}"; do
        local slug="${entry%%|*}"
        local tpl="${entry#*|}"
        local prompt="${tpl//%TRIGGER%/${trigger}}"
        prompt="${prompt//%ANCHORS%/${anchors_csv}}"
        prompt="${prompt//%CLASS%/${bare_class}}"

        # ---- T2I: gray 512×512 canvas → pure text-to-image ----
        if [ "${MODE}" = "t2i" ] || [ "${MODE}" = "both" ]; then
            local t2i_out="${OUT_ROOT}/t2i/${concept}/${slug}"
            mkdir -p "${t2i_out}"
            for seed in ${SEEDS}; do
                local img="${t2i_out}/seed${seed}.png"
                if [ -f "${img}" ] && [ -z "${FORCE:-}" ]; then continue; fi
                CUDA_VISIBLE_DEVICES="${gpu}" python scripts/inference_lora.py \
                    --lora_path "${lora}" \
                    --prompt "${prompt}" \
                    --output_dir "${t2i_out}" \
                    --seed "${seed}" \
                    --num_images 1 \
                    --num_steps "${NUM_STEPS}" \
                    --cfg_scale "${CFG}" >>"${log}" 2>&1
                mv -f "${t2i_out}/output_seed${seed}.png" "${img}" 2>/dev/null || true
            done
        fi

        # ---- I2I: first training image as source → image editing ----
        if [ "${MODE}" = "i2i" ] || [ "${MODE}" = "both" ]; then
            if [ -z "${ref_img}" ]; then
                echo "  SKIP i2i ${concept}/${slug} — no ref image" >>"${log}"
            else
                local i2i_out="${OUT_ROOT}/i2i/${concept}/${slug}"
                mkdir -p "${i2i_out}"
                for seed in ${SEEDS}; do
                    local img="${i2i_out}/seed${seed}.png"
                    if [ -f "${img}" ] && [ -z "${FORCE:-}" ]; then continue; fi
                    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/inference_lora.py \
                        --lora_path "${lora}" \
                        --prompt "${prompt}" \
                        --source_image "${ref_img}" \
                        --output_dir "${i2i_out}" \
                        --seed "${seed}" \
                        --num_images 1 \
                        --num_steps "${NUM_STEPS}" \
                        --cfg_scale "${CFG}" >>"${log}" 2>&1
                    mv -f "${i2i_out}/output_seed${seed}.png" "${img}" 2>/dev/null || true
                done
            fi
        fi
    done

    if [ $? -eq 0 ]; then
        echo "[GPU${gpu}] DONE  ${concept}"
    else
        echo "[GPU${gpu}] FAIL  ${concept} — see ${log}"
    fi
}

# ---- Build GPU list (explicit GPUS env var or 0..NGPU-1) ----
if [ -n "${GPUS}" ]; then
    GPU_LIST=(${GPUS})
else
    GPU_LIST=()
    for ((i=0; i<NGPU; i++)); do GPU_LIST+=($i); done
fi
NWORKERS=${#GPU_LIST[@]}

run_worker() {
    local slot="$1"
    local gpu="${GPU_LIST[$slot]}"
    for ((i=slot; i<NJOBS; i+=NWORKERS)); do
        IFS=$'\t' read -r concept ctype trigger anchors dataset_dir <<<"${ROWS[$i]}"
        IFS='|' read -r -a ANCS <<<"${anchors}"
        infer_one "${gpu}" "${concept}" "${ctype}" "${trigger}" "${dataset_dir}" "${ANCS[@]}"
    done
}

WPIDS=()
for ((g=0;g<NWORKERS;g++)); do run_worker "${g}" & WPIDS+=($!); done
wait

echo "================================================================"
echo " Inference done. Next: compute single-concept metrics:"
echo "   python scripts/compute_metrics.py --eval_dir ${OUT_ROOT} \\"
echo "       --registry ${REGISTRY} --datasets_dir ${DATASETS_DIR:-Datasets}"
echo "================================================================"
