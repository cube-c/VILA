#!/bin/bash

set -e

IMAGE_ROOT="/app/blender"
VQA_JSON="/app/blender/output/phasevar/0/vqa_obj1.json"

run_model() {
    local MODEL_PATH="$1"
    local SUFFIX="$2"
    local LABEL="$3"

    local OUT_CSV="logit_results_vqa_phase${SUFFIX}.csv"

    # echo "========== ${LABEL} | phasevar =========="
    # python llava/cli/infer_logit_vqa.py \
        # --model-path "$MODEL_PATH" \
        # --vqa-json "$VQA_JSON" \
        # --image-root "$IMAGE_ROOT" \
        # --output-csv "$OUT_CSV"

    echo "========== Plotting heatmap =========="
    python plot_heatmap_vqa.py \
        --input "$OUT_CSV" \
        --output "logit_heatmap_vqa_phase${SUFFIX}.png" \
        --title "${LABEL} — Phase Var Logit Diff"
}

run_model "Efficient-Large-Model/NVILA-Lite-2B" "" "NVILA-Lite-2B"
run_model "Zhoues/RoboRefer-2B-SFT" "_roborefer" "RoboRefer-2B-SFT"
run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632" "_2m" "NVILA-Lite-2B-2M"
run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221" "_400k" "NVILA-Lite-2B-400K"
run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221" "_80k" "NVILA-Lite-2B-80K"
run_model "/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221" "_800k" "NVILA-Lite-2B-800K"

echo "All done."
