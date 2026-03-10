#!/bin/bash

set -euo pipefail

IMAGE_ROOT="/app/blender"
VQA_JSON="/app/blender/output/sizevar/both/vqa_obj1.json"
VARIANTS="obj1_closer obj2_closer obj1_farther obj2_farther"

# model_path suffix label gpu
MODELS=(
    "Efficient-Large-Model/NVILA-Lite-2B||NVILA-Lite-2B|0"
    "Zhoues/RoboRefer-2B-SFT|_roborefer|RoboRefer-2B-SFT|1"
    "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632|_2m|NVILA-Lite-2B-2M|2"
    "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221|_400k|NVILA-Lite-2B-400K|3"
    "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221|_80k|NVILA-Lite-2B-80K|4"
    "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221|_800k|NVILA-Lite-2B-800K|5"
)

run_model() {
    local MODEL_PATH="$1"
    local SUFFIX="$2"
    local LABEL="$3"
    local GPU="$4"

    local BASE_CSV="scale${SUFFIX}.csv"

    echo "========== ${LABEL} | sizevar (all variants) | GPU ${GPU} =========="
    CUDA_VISIBLE_DEVICES=$GPU python llava/cli/infer_logit_vqa.py \
        --model-path "$MODEL_PATH" \
        --vqa-json "$VQA_JSON" \
        --image-root "$IMAGE_ROOT" \
        --output-csv "$BASE_CSV"
}

# Launch all models in parallel, one GPU each
pids=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_PATH SUFFIX LABEL GPU <<< "$entry"
    run_model "$MODEL_PATH" "$SUFFIX" "$LABEL" "$GPU" \
        > "output_scale${SUFFIX}.log" 2>&1 &
    pids+=($!)
    echo "Launched ${LABEL} on GPU ${GPU} (PID: ${pids[-1]})"
done

echo "All jobs launched (PIDs: ${pids[*]}). Waiting..."

failed=0
for i in "${!MODELS[@]}"; do
    IFS='|' read -r _ SUFFIX LABEL _ <<< "${MODELS[$i]}"
    if wait "${pids[$i]}"; then
        echo "=== Done: ${LABEL} ==="
    else
        echo "=== FAILED: ${LABEL} (see output_scale${SUFFIX}.log) ===" >&2
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "Some runs failed." >&2
    exit 1
fi

echo "All done."
