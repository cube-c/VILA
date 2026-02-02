import argparse
import importlib.util
import json
import os
import re
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation
from llava.mm_utils import process_image, process_images
from llava.constants import DEFAULT_IMAGE_TOKEN


def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    """Replace time tokens in text with actual timestamps."""
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    excess_pattern = re.compile(rf"<t(\d+)>")
    matches = excess_pattern.findall(text)
    for match in matches:
        t = int(match)
        if t >= num_time_tokens:
            timestamp = round(duration, 2)  # Map to the end of the video
            text = text.replace(f"<t{t}>", f"<{timestamp}>")

    return text


def find_image_token_ranges(input_ids, media_token_ids):
    """
    Find the ranges of image tokens in the input sequence.
    Returns a list of tuples (start_idx, end_idx) for each contiguous image token sequence.
    """
    image_token_id = media_token_ids.get("image", None)
    if image_token_id is None:
        return []

    # Find all positions of image tokens
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].cpu().tolist()

    if not image_positions:
        return []

    # Group consecutive positions into ranges
    ranges = []
    start = image_positions[0]
    prev = start

    for pos in image_positions[1:]:
        if pos != prev + 1:
            # New range started
            ranges.append((start, prev + 1))
            start = pos
        prev = pos

    # Add the last range
    ranges.append((start, prev + 1))

    return ranges


def map_input_to_embedding_positions(input_ids, media_embeds_dict, tokenizer):
    """
    Map input token positions to embedding positions.
    This is crucial because after embedding, image tokens are replaced with image embeddings
    which can have different lengths.

    Returns:
        - embedding_to_input_map: dict mapping embedding index to (token_type, position)
        - image_embedding_ranges: list of (start, end) tuples for image embeddings
    """
    batch_idx = 0
    input_seq = input_ids[batch_idx]

    embedding_to_input_map = []
    image_embedding_ranges = []
    current_embed_pos = 0

    # Build inverse mapping from token ID to media name
    media_tokens = {}
    for name, token_id in tokenizer.media_token_ids.items():
        media_tokens[token_id] = name

    for token_pos, token_id in enumerate(input_seq):
        if token_id.item() in media_tokens:
            # This is a media token - it will be replaced by media embeddings
            media_type = media_tokens[token_id.item()]
            if media_type in media_embeds_dict and len(media_embeds_dict[media_type]) > 0:
                # Get the number of embeddings for this media token
                media_embed = media_embeds_dict[media_type][0]
                num_embeds = media_embed.shape[0]

                start_pos = current_embed_pos
                for i in range(num_embeds):
                    embedding_to_input_map.append(('image', token_pos, i))
                current_embed_pos += num_embeds

                image_embedding_ranges.append((start_pos, current_embed_pos))
        else:
            # Regular text token
            embedding_to_input_map.append(('text', token_pos, 0))
            current_embed_pos += 1

    return embedding_to_input_map, image_embedding_ranges


def visualize_attention_for_layer(attention, layer_idx, image_ranges, save_dir, num_heads_to_show=4):
    """
    Visualize attention map for a specific layer.

    Args:
        attention: Attention weights tensor [batch, num_heads, seq_len, seq_len]
        layer_idx: Layer index
        image_ranges: List of (start, end) tuples for image token positions
        save_dir: Directory to save visualizations
        num_heads_to_show: Number of attention heads to visualize
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size, num_heads, seq_len, _ = attention.shape

    # Check for NaN values
    if torch.isnan(attention).any():
        print(f"WARNING: Layer {layer_idx} contains NaN values in attention! Replacing with zeros.")
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)

    # Check for inf values
    if torch.isinf(attention).any():
        print(f"WARNING: Layer {layer_idx} contains inf values in attention! Replacing with zeros.")
        attention = torch.where(torch.isinf(attention), torch.zeros_like(attention), attention)

    # Visualize selected attention heads
    heads_to_viz = min(num_heads_to_show, num_heads)
    head_indices = np.linspace(0, num_heads - 1, heads_to_viz, dtype=int)

    for head_idx in head_indices:
        # Get attention for this head [seq_len, seq_len]
        attn = attention[0, head_idx].cpu().detach().float().numpy()

        # Double-check for NaN in numpy array
        if np.isnan(attn).any():
            print(f"WARNING: NaN detected in layer {layer_idx}, head {head_idx} after conversion to numpy")
            attn = np.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)

        # Create full attention map
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(attn, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=attn.max() if attn.max() > 0 else 1.0)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add lines to mark image token regions
        for start, end in image_ranges:
            # Horizontal and vertical lines to show image token regions
            ax.axhline(y=start, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=end, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=start, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=end, color='r', linestyle='--', alpha=0.5, linewidth=1)

            # Add shaded regions
            ax.axhspan(start, end, alpha=0.1, color='red')
            ax.axvspan(start, end, alpha=0.1, color='red')

        ax.set_xlabel('Key Position (attending to)', fontsize=10)
        ax.set_ylabel('Query Position (attending from)', fontsize=10)
        ax.set_title(f'Layer {layer_idx} - Head {head_idx}\nAttention Map (Red regions: Image Tokens)', fontsize=12)

        save_path = os.path.join(save_dir, f'layer{layer_idx:02d}_head{head_idx:02d}_full.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved attention map: {save_path}")

    # Create average attention map across all heads
    attn_avg = attention[0].mean(dim=0).cpu().detach().float().numpy()

    # Check for NaN in averaged attention
    if np.isnan(attn_avg).any():
        print(f"WARNING: NaN detected in averaged attention for layer {layer_idx}")
        attn_avg = np.nan_to_num(attn_avg, nan=0.0, posinf=1.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=attn_avg.max() if attn_avg.max() > 0 else 1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for start, end in image_ranges:
        ax.axhline(y=start, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=end, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=start, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=end, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhspan(start, end, alpha=0.1, color='red')
        ax.axvspan(start, end, alpha=0.1, color='red')

    ax.set_xlabel('Key Position (attending to)', fontsize=10)
    ax.set_ylabel('Query Position (attending from)', fontsize=10)
    ax.set_title(f'Layer {layer_idx} - Average Across All Heads\nAttention Map (Red regions: Image Tokens)', fontsize=12)

    save_path = os.path.join(save_dir, f'layer{layer_idx:02d}_avg_all_heads.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved average attention map: {save_path}")

    # Visualize 2D spatial attention for image patches
    visualize_2d_image_attention(attention, layer_idx, image_ranges, save_dir)


def visualize_2d_image_attention(attention, layer_idx, image_ranges, save_dir):
    """
    Visualize attention to/from image tokens in 2D spatial layout.

    Args:
        attention: Attention weights [batch, num_heads, seq_len, seq_len]
        layer_idx: Layer index
        image_ranges: List of (start, end) tuples for image token positions
        save_dir: Directory to save visualizations
    """
    if not image_ranges:
        return

    # Check for NaN values
    if torch.isnan(attention).any():
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)

    # Average across all heads for clarity
    attention_avg = attention[0].mean(dim=0)  # [seq_len, seq_len]

    for img_idx, (start, end) in enumerate(image_ranges):
        num_image_tokens = end - start

        # Check for CLS token or other special tokens
        # If num_image_tokens = perfect_square + K, likely has K special tokens
        has_special_tokens = False
        num_spatial_tokens = num_image_tokens
        num_special = 0

        # Try subtracting potential special tokens (commonly 1 CLS token)
        for n_special in [1, 2, 3, 4]:  # Try 1-4 special tokens
            remaining = num_image_tokens - n_special
            patch_size_test = int(np.sqrt(remaining))
            if patch_size_test * patch_size_test == remaining and remaining > 0:
                has_special_tokens = True
                num_spatial_tokens = remaining
                num_special = n_special
                print(f"🔍 Detected {n_special} special token(s) (likely CLS). Total: {num_image_tokens}, Spatial: {num_spatial_tokens}")
                break

        # Try to infer patch grid size (assume square)
        patch_size = int(np.sqrt(num_spatial_tokens))

        if patch_size * patch_size != num_spatial_tokens:
            # Not a perfect square, try common aspect ratios
            # Common vision transformer patch counts
            common_sizes = {
                196: (14, 14),
                256: (16, 16),
                576: (24, 24),
                1024: (32, 32),
                729: (27, 27),
                784: (28, 28),
                121: (11, 11),  # 122 - 1 = 121 = 11x11
                144: (12, 12),
                169: (13, 13),
                225: (15, 15),
            }

            if num_spatial_tokens in common_sizes:
                patch_h, patch_w = common_sizes[num_spatial_tokens]
            else:
                # Try to find closest factorization
                factors = []
                for i in range(1, int(np.sqrt(num_spatial_tokens)) + 1):
                    if num_spatial_tokens % i == 0:
                        factors.append((i, num_spatial_tokens // i))
                # Pick the factorization closest to square
                if factors:
                    patch_h, patch_w = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    print(f"❌ Cannot determine 2D layout for {num_spatial_tokens} spatial tokens")
                    continue
        else:
            patch_h = patch_w = patch_size

        print(f"✓ Image {img_idx}: {num_image_tokens} total tokens -> {patch_h}x{patch_w} spatial grid ({num_spatial_tokens} patches)")

        # Extract attention, skipping special tokens (usually at the beginning)
        # Assumption: special tokens are at start of the sequence
        spatial_start = start + num_special
        spatial_end = end

        attn_from_image = attention_avg[spatial_start:spatial_end, :].cpu().detach().float().numpy()  # [num_spatial_tokens, seq_len]
        attn_to_image = attention_avg[:, spatial_start:spatial_end].cpu().detach().float().numpy()  # [seq_len, num_spatial_tokens]

        # Handle NaN
        attn_from_image = np.nan_to_num(attn_from_image, nan=0.0)
        attn_to_image = np.nan_to_num(attn_to_image, nan=0.0)

        # Self-attention within spatial image tokens only
        attn_self = attention_avg[spatial_start:spatial_end, spatial_start:spatial_end].cpu().detach().float().numpy()  # [num_spatial_tokens, num_spatial_tokens]
        attn_self = np.nan_to_num(attn_self, nan=0.0)

        # Reshape to 2D spatial grid
        try:
            attn_self_2d = attn_self.reshape(patch_h, patch_w, patch_h, patch_w)
        except:
            print(f"WARNING: Cannot reshape {attn_self.shape} to ({patch_h}, {patch_w}, {patch_h}, {patch_w})")
            continue

        # Average attention from each patch to all other patches
        attn_2d_avg = attn_self_2d.mean(axis=(2, 3))  # [patch_h, patch_w]

        # Visualize 2D attention map
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Debug: verify shapes
        print(f"  2D visualization shapes - patch_h: {patch_h}, patch_w: {patch_w}")
        print(f"  attn_2d_avg shape: {attn_2d_avg.shape}")

        # 1. Average self-attention per patch
        im0 = axes[0].imshow(attn_2d_avg, cmap='hot', interpolation='nearest', origin='upper', aspect='equal')
        axes[0].set_title(f'Layer {layer_idx} - Image {img_idx}\nAverage Self-Attention per Patch')
        axes[0].set_xlabel('Patch X')
        axes[0].set_ylabel('Patch Y')
        # Set proper ticks
        axes[0].set_xticks(np.arange(0, patch_w, max(1, patch_w // 8)))
        axes[0].set_yticks(np.arange(0, patch_h, max(1, patch_h // 8)))
        axes[0].set_xlim(-0.5, patch_w - 0.5)
        axes[0].set_ylim(patch_h - 0.5, -0.5)  # Inverted for image coordinates
        plt.colorbar(im0, ax=axes[0])

        # 2. Attention FROM image patches (aggregated over spatial dimensions)
        attn_from_spatial = attn_from_image.mean(axis=1).reshape(patch_h, patch_w)  # Average over all attending-to positions
        print(f"  attn_from_spatial shape: {attn_from_spatial.shape}")
        im1 = axes[1].imshow(attn_from_spatial, cmap='hot', interpolation='nearest', origin='upper', aspect='equal')
        axes[1].set_title(f'Layer {layer_idx} - Image {img_idx}\nAvg Attention FROM Each Patch')
        axes[1].set_xlabel('Patch X')
        axes[1].set_ylabel('Patch Y')
        axes[1].set_xticks(np.arange(0, patch_w, max(1, patch_w // 8)))
        axes[1].set_yticks(np.arange(0, patch_h, max(1, patch_h // 8)))
        axes[1].set_xlim(-0.5, patch_w - 0.5)
        axes[1].set_ylim(patch_h - 0.5, -0.5)
        plt.colorbar(im1, ax=axes[1])

        # 3. Attention TO image patches (aggregated over spatial dimensions)
        attn_to_spatial = attn_to_image.mean(axis=0).reshape(patch_h, patch_w)  # Average over all attending-from positions
        print(f"  attn_to_spatial shape: {attn_to_spatial.shape}")
        im2 = axes[2].imshow(attn_to_spatial, cmap='hot', interpolation='nearest', origin='upper', aspect='equal')
        axes[2].set_title(f'Layer {layer_idx} - Image {img_idx}\nAvg Attention TO Each Patch')
        axes[2].set_xlabel('Patch X')
        axes[2].set_ylabel('Patch Y')
        axes[2].set_xticks(np.arange(0, patch_w, max(1, patch_w // 8)))
        axes[2].set_yticks(np.arange(0, patch_h, max(1, patch_h // 8)))
        axes[2].set_xlim(-0.5, patch_w - 0.5)
        axes[2].set_ylim(patch_h - 0.5, -0.5)
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'layer{layer_idx:02d}_image{img_idx}_2d_spatial.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved 2D spatial attention map: {save_path}")

        # Also create a detailed self-attention visualization
        # Show attention from center patch to all patches as 2D map
        center_h, center_w = patch_h // 2, patch_w // 2
        center_idx = center_h * patch_w + center_w

        if 0 <= center_idx < num_image_tokens:
            center_attn = attn_self[center_idx, :].reshape(patch_h, patch_w)
            print(f"  center_attn shape: {center_attn.shape}, center patch: [{center_h}, {center_w}]")

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(center_attn, cmap='hot', interpolation='nearest', origin='upper', aspect='equal')
            ax.set_title(f'Layer {layer_idx} - Image {img_idx}\nAttention FROM Center Patch [{center_h}, {center_w}]')
            ax.set_xlabel('Patch X')
            ax.set_ylabel('Patch Y')

            # Set proper ticks
            ax.set_xticks(np.arange(0, patch_w, max(1, patch_w // 8)))
            ax.set_yticks(np.arange(0, patch_h, max(1, patch_h // 8)))
            ax.set_xlim(-0.5, patch_w - 0.5)
            ax.set_ylim(patch_h - 0.5, -0.5)

            # Mark center patch with a star
            ax.plot(center_w, center_h, 'g*', markersize=20, label='Center Patch', markeredgecolor='white', markeredgewidth=1.5)
            ax.legend(loc='upper right')

            plt.colorbar(im, ax=ax)

            save_path = os.path.join(save_dir, f'layer{layer_idx:02d}_image{img_idx}_center_attn_2d.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved center patch attention: {save_path}")


def save_attention_statistics(attentions, image_ranges, save_dir):
    """
    Save statistics about attention to image tokens.
    """
    stats = {}

    for layer_idx, attention in enumerate(attentions):
        # attention shape: [batch, num_heads, seq_len, seq_len]
        layer_stats = {}

        # Check for NaN/inf
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            print(f"WARNING: Layer {layer_idx} contains NaN/inf in statistics computation")
            attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
            attention = torch.where(torch.isinf(attention), torch.zeros_like(attention), attention)

        for img_idx, (start, end) in enumerate(image_ranges):
            # Calculate average attention TO image tokens (from all tokens)
            attn_to_image = attention[0, :, :, start:end].mean()

            # Calculate average attention FROM image tokens (to all tokens)
            attn_from_image = attention[0, :, start:end, :].mean()

            # Calculate self-attention within image tokens
            attn_self_image = attention[0, :, start:end, start:end].mean()

            # Convert to float and handle any remaining NaN
            to_val = float(attn_to_image.cpu())
            from_val = float(attn_from_image.cpu())
            self_val = float(attn_self_image.cpu())

            layer_stats[f'image_{img_idx}'] = {
                'range': f'[{start}:{end}]',
                'num_tokens': end - start,
                'avg_attention_to_image': to_val if not np.isnan(to_val) else 0.0,
                'avg_attention_from_image': from_val if not np.isnan(from_val) else 0.0,
                'avg_self_attention': self_val if not np.isnan(self_val) else 0.0,
                'has_nan': np.isnan(to_val) or np.isnan(from_val) or np.isnan(self_val),
            }

        stats[f'layer_{layer_idx}'] = layer_stats

    # Save as JSON
    stats_path = os.path.join(save_dir, 'attention_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved attention statistics: {stats_path}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--lora-path", "-l", type=str, default=None)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")
    parser.add_argument("--num_video_frames", "-nf", type=int, default=-1)
    parser.add_argument("--video_max_tiles", "-vm", type=int, default=-1)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    parser.add_argument("--visualize-attention", action="store_true", help="Visualize and save attention maps")
    parser.add_argument("--attention-output-dir", type=str, default="attention_maps", help="Directory to save attention visualizations")
    parser.add_argument("--attention-heads-per-layer", type=int, default=4, help="Number of attention heads to visualize per layer")
    args = parser.parse_args()

    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    if args.lora_path is None:
        model = llava.load(args.model_path, model_base=None)
    else:
        model = llava.load(args.lora_path, model_base=args.model_path)
    print("="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(model)

    # Override num_video_frames and video_max_tiles
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames

    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    # Configure PS3 and adjust context length
    configure_ps3_and_context_length(model)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Prepare multi-modal prompt
    has_video = False
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
            elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                cap = cv2.VideoCapture(media)
                duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                media = Video(media)
                has_video = True
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    if not args.visualize_attention:
        # Standard inference without attention visualization
        response = model.generate_content(prompt, response_format=response_format)
        print(colored(response, "cyan", attrs=["bold"]))
        print(type(model))
    else:
        # Inference with attention visualization
        print("\n" + "="*80)
        print("ATTENTION VISUALIZATION MODE")
        print("="*80 + "\n")

        # Prepare conversation and media
        conversation = [{"from": "human", "value": prompt}]

        # Extract and process media
        media_dict = extract_media(conversation, config=model.config)
        media_config = defaultdict(dict)

        # See generate_content method in llava/model/llava_arch.py
        for name in media_dict:
            if name == "image":
                if len(media_dict["image"]) == 1 and model.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    model.config.image_processor = model.vision_tower.image_processor
                    if model.config.image_aspect_ratio == "dynamic":
                        images = process_image(media_dict["image"][0], model.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(model.config.s2_scales) is str:
                            model.config.s2_scales = list(map(int, model.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media_dict["image"][0], model.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media_dict["image"], model.vision_tower.image_processor, model.config).half()
                media_dict[name] = [image for image in images]
            elif name == "video":
                media_dict[name] = [
                    process_images(imgs, model.vision_tower.image_processor, model.config).half()
                    for imgs in media_dict[name]
                ]

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

        print(f"Input sequence length: {input_ids.shape[1]}")
        print(f"Number of images: {len(media_dict.get('image', []))}")

        # Get image token positions in input_ids
        image_token_ranges_input = find_image_token_ranges(input_ids, model.tokenizer.media_token_ids)
        print(f"Image token ranges in input: {image_token_ranges_input}")

        # Embed the inputs to get the actual embedding sequence
        inputs_embeds, _, attention_mask = model._embed(input_ids, media_dict, media_config, None, None)

        print(f"Embedding sequence length: {inputs_embeds.shape[1]}")

        # Get media embeddings to understand their sizes
        from collections import deque
        media_embeds_for_mapping = defaultdict(deque)
        for name in media_dict:
            embeds = model.encoders[name](media_dict[name], media_config[name])
            media_embeds_for_mapping[name] = deque(embeds)

        # Map positions
        embedding_map, image_embedding_ranges = map_input_to_embedding_positions(
            input_ids, media_embeds_for_mapping, model.tokenizer
        )

        print(f"Image embedding ranges: {image_embedding_ranges}")
        for start, end in image_embedding_ranges:
            print(f"  Image tokens in embedding space: positions {start} to {end} ({end-start} tokens)")

        # Validate inputs before forward pass
        print("\nValidating inputs...")
        if torch.isnan(inputs_embeds).any():
            print("❌ ERROR: inputs_embeds contains NaN values!")
            nan_count = torch.isnan(inputs_embeds).sum().item()
            print(f"   Number of NaN values: {nan_count}")
        else:
            print("✓ inputs_embeds: OK (no NaN)")

        if torch.isinf(inputs_embeds).any():
            print("❌ WARNING: inputs_embeds contains inf values!")

        # Ensure model is in eval mode
        model.eval()
        model.llm.eval()
        print("✓ Model set to eval mode")

        # Create position IDs
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        print(f"✓ position_ids shape: {position_ids.shape}")

        # Prepare attention mask - let the model handle it internally
        # Modern transformers (LLaMA, etc.) handle 2D->4D expansion and causal masking internally
        # Run generation with attention output
        print("\nRunning generation with attention output...")
        with torch.no_grad():
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
        print("✓ Generation completed successfully")

        # Extract generated tokens and decode to get response
        generated_ids = outputs.sequences[0]
        response_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("\n" + "="*80)
        print("GENERATED RESPONSE:")
        print("="*80)
        print(colored(response_text, "cyan", attrs=["bold"]))
        print("="*80 + "\n")

        # Extract attentions - for generate(), attentions are nested: (generation_step, layer)
        # We'll use the attentions from the first generation step
        attentions = outputs.attentions[0] if outputs.attentions else []
        print(f"\n✓ Number of layers with attention: {len(attentions)}")
        print()
        if len(attentions) > 0:
            print(f"✓ Attention shape per layer: {attentions[0].shape}")  # [batch, num_heads, seq_len, seq_len]

            # Check ALL layers for NaN to identify where it starts
            print("\n" + "="*80)
            print("Per-Layer NaN Analysis")
            print("="*80)
            first_nan_layer = None
            for layer_idx, attn in enumerate(attentions):
                has_nan = torch.isnan(attn).any()
                has_inf = torch.isinf(attn).any()

                if has_nan or has_inf:
                    nan_ratio = torch.isnan(attn).sum().item() / attn.numel() if has_nan else 0
                    inf_ratio = torch.isinf(attn).sum().item() / attn.numel() if has_inf else 0
                    status = "❌ NaN" if has_nan else "⚠️ Inf"
                    print(f"Layer {layer_idx:2d}: {status} (NaN: {nan_ratio:.2%}, Inf: {inf_ratio:.2%})")
                    if first_nan_layer is None and has_nan:
                        first_nan_layer = layer_idx
                else:
                    min_val, max_val = attn.min().item(), attn.max().item()
                    mean_val = attn.mean().item()
                    print(f"Layer {layer_idx:2d}: ✓ OK - Range: [{min_val:.6f}, {max_val:.6f}], Mean: {mean_val:.6f}")
            print("="*80 + "\n")

        else:
            print("❌ ERROR: No attention outputs received!")

        # # Save attention visualizations for each layer
        # print(f"Saving attention visualizations to: {args.attention_output_dir}")
        # os.makedirs(args.attention_output_dir, exist_ok=True)

        # for layer_idx, attention in enumerate(attentions):
            # layer_dir = os.path.join(args.attention_output_dir, f'layer_{layer_idx:02d}')
            # visualize_attention_for_layer(
                # attention,
                # layer_idx,
                # image_embedding_ranges,
                # layer_dir,
                # num_heads_to_show=args.attention_heads_per_layer
            # )

        # # Save attention statistics
        # save_attention_statistics(attentions, image_embedding_ranges, args.attention_output_dir)


if __name__ == "__main__":
    main()
