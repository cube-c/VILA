import argparse
import csv
import os
import sys
import tempfile

import torch

# Ensure llava/cli is on sys.path for custom_qwen2_patch (same as attn_ratio.py)
sys.path.insert(0, os.path.dirname(__file__))
import custom_qwen2_patch

from termcolor import colored
from datasets import load_dataset

import llava
from llava.media import Image

from llava.cli.attn_ratio import (
    act_hook,
    add_act_hooks,
    remove_act_hooks,
    analyze_token_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=27, help="Ending layer index (default: 27)")
    parser.add_argument("--output", type=str, default="output/attention_ratios_cvbench.csv", help="CSV output path")
    parser.add_argument("--task", type=str, default=None, choices=["Count", "Relation"], help="Filter by task type (Count or Relation)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Load model once
    model = llava.load(args.model_path, model_base=None)
    model.llm.config.use_cache = True
    model.llm.config.output_attentions = True

    # Load CV-Bench 2D dataset
    dataset = load_dataset("nyu-visionx/CV-Bench", "2D", split="test")
    if args.task is not None:
        dataset = dataset.filter(lambda x: x["task"] == args.task)
        print(f"Filtered to task={args.task}: {len(dataset)} examples")
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    print(f"Loaded {len(dataset)} examples from CV-Bench 2D")

    # Prepare CSV
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "model", "dataset_idx", "task", "source", "answer", "media",
        "prompt", "response", "layer_idx", "image_ratio", "prompt_ratio", "gen_ratio",
    ]
    write_header = not os.path.exists(output_path)

    for ex_idx, example in enumerate(dataset):
        print(f"\n{'='*60}")
        print(f"Example {ex_idx}/{len(dataset)}: task={example.get('task', '')}, source={example.get('source', '')}")

        # Save PIL image to temp file
        pil_image = example["image"]
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp_file.name

        pil_image.save(tmp_path)
        tmp_file.close()

        # Build prompt
        prompt_text = example["prompt"]
        prompt = [Image(tmp_path), prompt_text]

        # Analyze token positions
        token_info = analyze_token_positions(model, prompt)

        # Add hooks on all layers
        hooks = []
        for layer_idx in range(args.layer_start, args.layer_end + 1):
            hooks.append(add_act_hooks(model, layer_idx))

        # Generate response
        response = model.generate_content(prompt)
        print(colored(f"  Response: {response}", "cyan"))

        # Extract attention ratios per layer
        csv_rows = []
        for layer_idx in range(args.layer_start, args.layer_end + 1):
            llm_layer = model.llm.model.layers[layer_idx].self_attn
            acts_list = llm_layer.acts_list
            hook = hooks[layer_idx - args.layer_start]

            first_tensor = acts_list[0]
            batch_size, num_heads, base_seq_len, _ = first_tensor.shape

            image_embedding_size = base_seq_len - len(token_info["text_token_positions"])

            ar_tensor = acts_list[1]  # [batch, heads, 1, base_seq_len+1]
            attn_row = ar_tensor[0].mean(dim=0).squeeze(0).cpu()  # [base_seq_len+1]

            image_ratio = float(attn_row[:image_embedding_size].sum()) if image_embedding_size > 0 else 0.0
            prompt_ratio = float(attn_row[image_embedding_size:base_seq_len].sum())
            gen_ratio = float(attn_row[base_seq_len:base_seq_len + 1].sum())

            csv_rows.append({
                "model": args.model_path,
                "dataset_idx": ex_idx,
                "task": example.get("task", ""),
                "source": example.get("source", ""),
                "answer": example.get("answer", ""),
                "media": tmp_path,
                "prompt": prompt_text,
                "response": response,
                "layer_idx": layer_idx,
                "image_ratio": image_ratio,
                "prompt_ratio": prompt_ratio,
                "gen_ratio": gen_ratio,
            })

            remove_act_hooks(model, layer_idx, hook)

        # Append rows to CSV
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(csv_rows)

    print(f"\nDone. CSV at {output_path}")


if __name__ == "__main__":
    main()
