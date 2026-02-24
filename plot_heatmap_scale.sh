#!/bin/bash

set -e

for SCALE in 2m 80k 400k 800k ; do
    for SPLIT in h11 h12 h13; do
        python plot_heatmap.py \
            --input "logit_results_${SCALE}_${SPLIT}.csv" \
            --output "logit_heatmap_${SCALE}_${SPLIT}.png" \
            --title "NVILA-Lite-2B-${SCALE^^} (local) | $SPLIT — Yes - No Logit Diff"
    done
done

echo "All done."
