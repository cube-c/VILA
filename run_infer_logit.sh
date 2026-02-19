#!/bin/bash

set -e

TEXT="Is the red sphere closer to the camera than the blue sphere? Answer with yes or no."

for SPLIT in h11 h12 h13; do
    echo "========== NVILA-Lite-2B | $SPLIT =========="
    python llava/cli/infer_logit.py \
        --model-path "Efficient-Large-Model/NVILA-Lite-2B" \
        --text "$TEXT" \
        --media-dir "/app/blender/$SPLIT" \
        --output-csv "logit_results_${SPLIT}.csv"
    echo "========== RoboRefer-2B-SFT | $SPLIT =========="
    python llava/cli/infer_logit.py \
        --model-path "Zhoues/RoboRefer-2B-SFT" \
        --text "$TEXT" \
        --media-dir "/app/blender/$SPLIT" \
        --output-csv "logit_results_roborefer_${SPLIT}.csv"
done

echo "All done."
