echo ""
echo "Running VILA inference with hook..."
echo "=================================="

# Replace with your actual model path
MODEL_PATH="${MODEL_PATH:-Efficient-Large-Model/NVILA-Lite-2B}"

# Layer range to process (default: 0-27 for all layers)
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-27}"

python llava/cli/infer_with_hook.py \
    --model-path "$MODEL_PATH" \
    --media demo_images/embspatial_300.jpg \
    --text "What is the counter in relation to the towel? Answer with left, right, on or under" \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END"
