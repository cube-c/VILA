import argparse
import csv
import json
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


def build_yes_no_question(entry):
    """Build a yes/no question from a VQA entry.

    Returns (question_text, ground_truth) where ground_truth is 'Yes' or 'No'.
    """
    obj1 = entry["obj1"]
    obj2 = entry["obj2"]
    question = (
        f"Is the {obj1['color']} {obj1['shape']} closer to the camera "
        f"than the {obj2['color']} {obj2['shape']}? Answer with yes or no."
    )
    gt = "Yes" if entry["answer"] == "closer" else "No"
    return question, gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--vqa-json", type=str, required=True,
                        help="Path to vqa.json file")
    parser.add_argument("--image-root", type=str, default="/app/blender",
                        help="Root directory for resolving image paths in vqa.json")
    parser.add_argument("--output-csv", "-o", type=str, default="logit_results_vqa.csv")
    args = parser.parse_args()

    model = llava.load(args.model_path, model_base=None)
    configure_ps3_and_context_length(model)
    clib.default_conversation = clib.conv_templates["auto"].copy()

    with open(args.vqa_json) as f:
        vqa_data = json.load(f)

    print(f"Loaded {len(vqa_data)} VQA entries from {args.vqa_json}")

    results = []
    correct = 0
    for i, entry in enumerate(vqa_data):
        image_path = os.path.join(args.image_root, entry["image"])
        question, gt = build_yes_no_question(entry)

        prompt = [Image(image_path), question]
        yes_logit, no_logit = get_yes_no_logits(model, prompt)
        pred = "Yes" if yes_logit > no_logit else "No"
        is_correct = pred == gt

        if is_correct:
            correct += 1

        results.append({
            "model_path": args.model_path,
            "image": entry["image"],
            "question": question,
            "ground_truth": gt,
            "prediction": pred,
            "Yes_logit": yes_logit,
            "No_logit": no_logit,
            "correct": is_correct,
        })
        print(f"[{i+1}/{len(vqa_data)}] {os.path.basename(entry['image'])}: "
              f"Yes={yes_logit:.4f}, No={no_logit:.4f} | pred={pred} gt={gt} {'OK' if is_correct else 'WRONG'}")

    acc = correct / len(vqa_data) * 100 if vqa_data else 0
    print(f"\nAccuracy: {correct}/{len(vqa_data)} ({acc:.1f}%)")

    fieldnames = ["model_path", "image", "question", "ground_truth", "prediction",
                  "Yes_logit", "No_logit", "correct"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
