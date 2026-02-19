import argparse
import csv
import glob
import os
from collections import defaultdict

import torch

import llava
from llava import conversation as clib
from llava.media import Image
from llava.mm_utils import process_images
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on those settings."""

    # get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        print("Num look close:", num_look_close)
        num_look_close = int(num_look_close)
        model.num_look_close = num_look_close
    if num_token_look_close is not None:
        print("Num token look close:", num_token_look_close)
        num_token_look_close = int(num_token_look_close)
        model.num_token_look_close = num_token_look_close
    if select_num_each_scale is not None:
        print("Select num each scale:", select_num_each_scale)
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        print("Look close mode:", look_close_mode)
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        print("Smooth selection prob:", smooth_selection_prob)
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, num_token_look_close // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


@torch.inference_mode()
def get_yes_no_logits(model, prompt):
    """Run a forward pass and return logits for 'Yes' and 'No' tokens."""
    tokenizer = model.tokenizer

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # Build conversation
    conversation = [{"from": "human", "value": prompt}]

    # Process media (assumes exactly 1 image)
    media_config = defaultdict(dict)
    media = extract_media(conversation, config=model.config)
    assert len(media.get("image", [])) == 1, "Expected exactly 1 image"

    images = process_images(media["image"], model.vision_tower.image_processor, model.config).half()
    media["image"] = [image for image in images]

    # Tokenize
    input_ids = tokenize_conversation(conversation, tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

    # Forward pass via _embed + LLM
    inputs_embeds, _, attention_mask = model._embed(input_ids, media, media_config, None, None)
    outputs = model.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    # Get logits for the last position
    last_logits = outputs.logits[0, -1, :]

    return last_logits[yes_id].item(), last_logits[no_id].item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--media-dir", type=str, required=True,
                        help="Directory containing images (*.png)")
    parser.add_argument("--output-csv", "-o", type=str, default="logit_results.csv")
    args = parser.parse_args()

    model = llava.load(args.model_path, model_base=None)
    configure_ps3_and_context_length(model)
    clib.default_conversation = clib.conv_templates["auto"].copy()

    image_paths = sorted(glob.glob(os.path.join(args.media_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {args.media_dir}")
        return

    print(f"Found {len(image_paths)} images in {args.media_dir}")

    results = []
    for i, image_path in enumerate(image_paths):
        prompt = [Image(image_path), args.text]
        yes_logit, no_logit = get_yes_no_logits(model, prompt)
        results.append({
            "model_path": args.model_path,
            "text": args.text,
            "image": os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path)),
            "Yes_logit": yes_logit,
            "No_logit": no_logit,
        })
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}: "
              f"Yes={yes_logit:.4f}, No={no_logit:.4f}")

    fieldnames = ["model_path", "text", "image", "Yes_logit", "No_logit"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
