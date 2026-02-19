#!/bin/bash

set -e

TEXT="Is the red sphere closer to the camera than the blue sphere? Answer with yes or no."

declare -A MODELS=(
    ["80k"]="/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221"
    ["400k"]="/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221"
    ["800k"]="/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221"
    ["2m"]="/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632"
)

for SCALE in 80k 400k 800k 2m; do
    MODEL_PATH="${MODELS[$SCALE]}"
    for SPLIT in h11 h12 h13; do
        echo "========== NVILA-Lite-2B-${SCALE^^} (local) | $SPLIT =========="
        python llava/cli/infer_logit.py \
            --model-path "$MODEL_PATH" \
            --text "$TEXT" \
            --media-dir "/app/blender/$SPLIT" \
            --output-csv "logit_results_${SCALE}_${SPLIT}.csv"
    done
done

echo "All done."
