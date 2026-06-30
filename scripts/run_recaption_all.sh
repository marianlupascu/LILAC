#!/bin/bash
# ============================================================
# (Re)caption every dataset listed in concept_map.tsv using
# caption_dataset.py (Qwen2.5-VL, 'scene' mode by default), spread
# across N GPUs. Produces a fresh metadata.jsonl per dataset, which
# build_concept_registry.py can then verify + auto-type.
#
# Why 'scene' mode: it describes pose/setting only, NOT the face/hair/
# clothing, so the identity is carried by the trigger token (the same
# lesson as the TI anchors). Override with CAPTION_MODE=detailed if you
# want full-appearance captions.
#
# Input : concept_map.tsv  (from scripts/make_concept_map.py, with the
#         TODO concept cells filled in by you).
# Output: <dataset>/metadata.jsonl  +  logs/recaption_all/<concept>.log
#
# GPU scheduling: N background workers, worker g handles rows where
# (row_index % N == g). Resumable: a dataset that already has
# metadata.jsonl is skipped (set FORCE=1 to recaption anyway).
#
# Usage:
#   python scripts/make_concept_map.py            # make + then edit concept_map.tsv
#   bash scripts/run_recaption_all.sh
#   NGPU=4 CAPTION_MODE=scene bash scripts/run_recaption_all.sh
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/run_recaption_all.sh
#   FORCE=1 bash scripts/run_recaption_all.sh     # recaption even if metadata exists
#   DRY_RUN=1 bash scripts/run_recaption_all.sh
# ============================================================
set -uo pipefail

MAP=${MAP:-concept_map.tsv}
NGPU=${NGPU:-4}
CAPTION_MODE=${CAPTION_MODE:-scene}
CAPTIONER=${CAPTIONER:-qwen}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
FORCE=${FORCE:-}
DRY_RUN=${DRY_RUN:-}

if [ ! -f "${MAP}" ]; then
    echo "ERROR: map '${MAP}' not found."
    echo "Run first:  python scripts/make_concept_map.py   (then fill TODO concepts)"
    exit 1
fi

mkdir -p logs/recaption_all

# Parse map: skip comments/blank, skip has_images==0, skip concept==TODO.
ROWS=()
while IFS= read -r _line; do
    [ -n "${_line}" ] && ROWS+=("${_line}")
done < <(awk -F'\t' '
    /^[[:space:]]*#/ {next}
    NF < 2 {next}
    {
        dataset=$1; concept=$2; trigger=($4==""?"ohwx":$4); has=$5+0;
        if (concept=="" || concept=="TODO") next;
        if (has==0) next;
        print dataset "\t" concept "\t" trigger;
    }' "${MAP}")

NJOBS=${#ROWS[@]}
if [ "${NJOBS}" -eq 0 ]; then
    echo "No captionable rows in ${MAP} (need a real concept + has_images>0)."
    echo "Did you fill the TODO cells?"
    exit 1
fi

echo "============================================================"
echo " Recaption plan: ${NJOBS} dataset(s) on ${NGPU} GPU(s)"
echo "   mode=${CAPTION_MODE} captioner=${CAPTIONER} model=${MODEL}"
echo "============================================================"
for ((i = 0; i < NJOBS; i++)); do
    IFS=$'\t' read -r d c t <<<"${ROWS[$i]}"
    printf "  [GPU %d] %-14s trigger=%-8s %s\n" "$((i % NGPU))" "${c}" "${t}" "${d}"
done
echo "============================================================"

if [ -n "${DRY_RUN}" ]; then
    echo "DRY_RUN set - nothing captioned."
    exit 0
fi

caption_one() {
    local gpu="$1" dataset="$2" concept="$3" trigger="$4"
    local meta="${dataset}/metadata.jsonl"
    local log="logs/recaption_all/${concept}.log"

    if [ -f "${meta}" ] && [ -z "${FORCE}" ]; then
        echo "[GPU${gpu}] SKIP ${concept} (metadata exists: ${meta})"
        return 0
    fi
    echo "[GPU${gpu}] CAPTION ${concept} -> ${meta}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/caption_dataset.py \
        --dataset_dir "${dataset}" \
        --concept_name "${concept}" \
        --trigger_token "${trigger}" \
        --captioner "${CAPTIONER}" \
        --caption_mode "${CAPTION_MODE}" \
        --model_name "${MODEL}" \
        --output_json "${meta}" >"${log}" 2>&1
    if [ $? -eq 0 ] && [ -f "${meta}" ]; then
        echo "[GPU${gpu}] OK   ${concept}"
    else
        echo "[GPU${gpu}] FAIL ${concept} - see ${log}"
    fi
}

run_worker() {
    local g="$1"
    for ((i = g; i < NJOBS; i += NGPU)); do
        IFS=$'\t' read -r dataset concept trigger <<<"${ROWS[$i]}"
        caption_one "${g}" "${dataset}" "${concept}" "${trigger}"
    done
}

WPIDS=()
for ((g = 0; g < NGPU; g++)); do
    run_worker "${g}" &
    WPIDS+=($!)
done
wait

echo "============================================================"
echo " Recaption finished. Next:"
echo "   python scripts/build_concept_registry.py   # now detects all"
echo "   bash scripts/run_train_all.sh"
echo "============================================================"
