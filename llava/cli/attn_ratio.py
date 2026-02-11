import argparse
import csv
import os
import torch

import custom_qwen2_patch
from termcolor import colored

import llava
from llava.media import Image

from typing import Tuple
from torch import Tensor


def act_hook(module, in_values: Tuple[Tensor], out_values: Tuple[Tensor]) -> None:
    module.acts_list.append(out_values[1])


def add_act_hooks(model, layer_index):
    llm_layer = model.llm.model.layers[layer_index].self_attn
    setattr(llm_layer, "acts_list", [])
    hook = llm_layer.register_forward_hook(act_hook)
    return hook


def remove_act_hooks(model, layer_index, hook):
    """Remove hooks and clear activation lists."""
    hook.remove()
    llm_layer = model.llm.model.layers[layer_index].self_attn
    if hasattr(llm_layer, "acts_list"):
        delattr(llm_layer, "acts_list")


def analyze_token_positions(model, prompt):
    """Analyze and return information about token positions in the input."""
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation
    from llava.constants import DEFAULT_IMAGE_TOKEN

    conversation = [{"from": "human", "value": prompt}]
    media = extract_media(conversation, config=model.config)

    input_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True)

    image_token_id = model.tokenizer.media_token_ids.get("image", None)

    print(f"\n=== Input Token Structure ===")
    print(f"Total input tokens (before image embedding): {len(input_ids)}")
    print(f"Image token ID: {image_token_id}")

    image_token_positions = []
    text_token_positions = []
    for i, token_id in enumerate(input_ids):
        if token_id == image_token_id:
            image_token_positions.append(i)
        else:
            text_token_positions.append(i)

    print(f"Image token positions in input_ids: {image_token_positions}")
    print(f"Number of text tokens in input_ids: {len(text_token_positions)}")

    return {
        "input_ids": input_ids,
        "image_token_id": image_token_id,
        "image_token_positions": image_token_positions,
        "text_token_positions": text_token_positions,
        "total_input_tokens": len(input_ids)
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=27, help="Ending layer index (default: 27)")
    parser.add_argument("--output", type=str, default="output/attention_ratios.csv", help="CSV output path")
    args = parser.parse_args()

    # Load model
    model = llava.load(args.model_path, model_base=None)

    # Prepare multi-modal prompt
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    # Build prompt text and media path for CSV
    prompt_text = args.text or ""
    media_text = ",".join(args.media) if args.media else ""

    # Analyze token positions before generation
    token_info = analyze_token_positions(model, prompt)

    # Configure model for attention capture
    model.llm.config.use_cache = False
    model.llm.config.output_attentions = True

    # Add hooks on all layers in range
    hooks = []
    for layer_idx in range(args.layer_start, args.layer_end + 1):
        hooks.append(add_act_hooks(model, layer_idx))

    # Generate response once
    response = model.generate_content(prompt)
    print(colored(response, "cyan", attrs=["bold"]))

    # Collect CSV rows
    csv_rows = []

    for layer_idx in range(args.layer_start, args.layer_end + 1):
        llm_layer = model.llm.model.layers[layer_idx].self_attn
        acts_list = llm_layer.acts_list
        hook = hooks[layer_idx - args.layer_start]

        if len(acts_list) == 0:
            remove_act_hooks(model, layer_idx, hook)
            continue

        print(f"\n=== Processing Layer {layer_idx} ===")
        print(f"Number of attention tensors captured: {len(acts_list)}")

        first_tensor = acts_list[0]
        batch_size, num_heads, base_seq_len, _ = first_tensor.shape

        # Estimate image embedding size
        num_image_tokens_in_input = len(token_info["image_token_positions"])
        num_text_tokens_in_input = len(token_info["text_token_positions"])

        if num_image_tokens_in_input > 0:
            image_embedding_size = base_seq_len - num_text_tokens_in_input
        else:
            image_embedding_size = 0

        print(f"  Image tokens: 0..{image_embedding_size-1} ({image_embedding_size})")
        print(f"  Prompt tokens: {image_embedding_size}..{base_seq_len-1} ({num_text_tokens_in_input})")

        # First generated token: use acts_list[1] which is [batch, heads, 1, base_seq_len+1]
        if len(acts_list) < 2:
            remove_act_hooks(model, layer_idx, hook)
            continue

        ar_tensor = acts_list[1]  # [batch, heads, 1, base_seq_len+1]
        attn_row = ar_tensor[0].mean(dim=0).squeeze(0).cpu()  # [base_seq_len+1]

        image_ratio = float(attn_row[:image_embedding_size].sum()) if image_embedding_size > 0 else 0.0
        prompt_ratio = float(attn_row[image_embedding_size:base_seq_len].sum())
        gen_ratio = float(attn_row[base_seq_len:base_seq_len + 1].sum())  # self-attention

        csv_rows.append({
            "model": args.model_path,
            "media": media_text,
            "prompt": prompt_text,
            "response": response,
            "layer_idx": layer_idx,
            "image_ratio": image_ratio,
            "prompt_ratio": prompt_ratio,
            "gen_ratio": gen_ratio,
        })

        # Remove hooks for this layer
        remove_act_hooks(model, layer_idx, hook)
        print(f"  Done layer {layer_idx}")

    # Write CSV
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = ["model", "media", "prompt", "response", "layer_idx", "image_ratio", "prompt_ratio", "gen_ratio"]
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV written to {output_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
