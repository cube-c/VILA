LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-27}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
OUTPUT="${OUTPUT:-output/attention_ratios_cvbench.csv}"

MODELS=(
    "Efficient-Large-Model/NVILA-Lite-2B"
    "Zhoues/RoboRefer-2B-SFT"
    "ch-min/NVILA-Lite-2B-SINGLE_REFSPATIAL_80K-20251107_104236"
    "ch-min/NVILA-Lite-2B-SINGLE_SAT_80K-20251107_104236"
)

TASKS=("Count" "Relation")

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "=== $model / $task ==="
        python llava/cli/attn_ratio_cvbench.py \
            --model-path "$model" \
            --layer-start "$LAYER_START" \
            --layer-end "$LAYER_END" \
            --output "$OUTPUT" \
            --task "$task" \
            --max-examples "$MAX_EXAMPLES"
    done
done