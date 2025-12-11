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
    --media demo_images/embspatial_100.jpg \
    --text "What is the spatial arrangement of doll and bench in the image concerning each other? A) The doll is on the right side of the bench. B) The doll is blocking the bench. C) The doll is left of the bench. D) The doll is beneath the bench. Please answer directly with only the letter of the correct option and nothing else." \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END"
