#!/bin/bash

set -e

for SPLIT in h11 h12 h13; do
    python plot_heatmap.py \
        --input "logit_results_${SPLIT}.csv" \
        --output "logit_heatmap_${SPLIT}.png" \
        --title "NVILA-Lite-2B | $SPLIT — Yes - No Logit Diff"

    python plot_heatmap.py \
        --input "logit_results_roborefer_${SPLIT}.csv" \
        --output "logit_heatmap_roborefer_${SPLIT}.png" \
        --title "RoboRefer-2B-SFT | $SPLIT — Yes - No Logit Diff"
done

echo "All done."
