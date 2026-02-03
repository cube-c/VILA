echo ""
echo "Running VILA inference with hook..."
echo "=================================="

# Layer range to process (default: 0-27 for all layers)
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-27}"

python llava/cli/infer_with_hook.py \
    --model-path "Zhoues/RoboRefer-2B-SFT" \
    --media demo_images/embspatial_300.jpg \
    --text "What is the counter in relation to the towel? Answer with left, right, on or under." \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END"
