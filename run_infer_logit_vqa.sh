#!/bin/bash

set -e

IMAGE_ROOT="/app/blender"
VQA_JSON="/app/blender/output/sizevar/both/vqa_obj1.json"

run_model() {
    local MODEL_PATH="$1"
    local SUFFIX="$2"

    local OUT_CSV="logit_results_vqa${SUFFIX}.csv"

    echo "========== ${MODEL_PATH} | vqa_obj1 =========="
    python llava/cli/infer_logit_vqa.py \
        --model-path "$MODEL_PATH" \
        --vqa-json "$VQA_JSON" \
        --image-root "$IMAGE_ROOT" \
        --output-csv "$OUT_CSV"

    echo "========== Plotting logit_vqa_size${SUFFIX}.png =========="
    python plot_logit_vqa_size.py \
        --input "$OUT_CSV" \
        --output "logit_vqa_size${SUFFIX}.png" \
        --title "Logit Diff — ${MODEL_PATH}"
}

run_model "Efficient-Large-Model/NVILA-Lite-2B" ""
run_model "Zhoues/RoboRefer-2B-SFT" "_roborefer"
# run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632" "_2m"
# run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221" "_400k"
# run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221" "_80k"
# run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221" "_800k"

echo "All done."
