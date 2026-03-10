# VLM Spatial Reasoning Evaluation — Phase Variation Experiment

## Overview

This experiment evaluates how Vision Language Models (VLMs) understand spatial depth relationships between objects, and how their accuracy varies with the vertical phase (position) of objects in the scene. We test whether models rely on visual shortcuts (e.g., "higher in image = farther away") rather than true depth understanding.

## Models Tested

| Model | Variants | Location |
|-------|----------|----------|
| **NVILA-Lite-2B** (VILA) | base, 80k, 400k, 800k, 2m, roborefer | `/app/VILA/` |
| **Molmo-7B** | base, 80k, 400k, 800k, 2m | `/app/molmo/` |
| **Qwen2.5-VL-7B** | base, 80k, 400k, 800k, 2m, 3b | `/app/qwen/` |
| **Qwen3-VL-235B** | 235b | `/app/qwen/results_small/` |

The variant suffixes (80k, 400k, 800k, 2m) refer to the amount of spatial reasoning training data used for finetuning. "roborefer" uses the RoboRefer-2B-SFT model. "3b" is a 3B parameter variant.

## Dataset

**Source**: Blender-generated synthetic scenes at `/app/blender/output/phasevar_5/`

- **12 scenes** x **256 phase combinations** (16 phA x 16 phB) = **3,072 images**
- Each image contains **two objects** (obj1, obj2) whose vertical positions follow sinusoidal trajectories controlled by phase parameters phA (obj1) and phB (obj2)
- Phase indices 0-15 map to angles 0-337.5 degrees (step 22.5)
- Ground truth depth is computed from Blender's 3D coordinates

**VQA JSON**: `/app/blender/output/phasevar_5/vqa_obj1.json`
- Contains: image path, question, answer ("closer"/"farther"), obj1/obj2 metadata (color, shape, bounding box)

**Cell Classification**: `/app/blender/output/phasevar_5/cell_class.json`
- Precomputed mapping of each (phA, phB) cell to "consistent" or "counter"
- **Consistent** (118 cells): obj1 is visually higher in the image (obj1_cy < obj2_cy) — visual position matches typical depth heuristic
- **Counter** (116 cells): obj1 is visually lower (obj1_cy > obj2_cy) — visual position contradicts the heuristic
- 22 mixed cells excluded (different scenes within the same cell disagree)

## Pipeline

### 1. Blender Rendering (`/app/blender/`)

```
python batch_render_phasevar.py    # Render phase-varied scenes
python make_vqa.py                 # Generate VQA JSON with questions + bounding boxes
```

### 2. Inference — Extract Yes/No Logits

**VILA**: `python llava/cli/infer_logit_vqa.py`
**Molmo**: `python /app/molmo/infer_logit_vqa.py`
**Qwen**: `python /app/qwen/infer_logit_vqa.py`

For each image, the model is asked 4 question variants:

| Variant | Question Template | GT when closer | GT when farther |
|---------|-------------------|----------------|-----------------|
| `obj1_closer` | "Is the {obj1} closer to the camera than the {obj2}?" | Yes | No |
| `obj2_closer` | "Is the {obj2} closer to the camera than the {obj1}?" | No | Yes |
| `obj1_farther` | "Is the {obj1} farther from the camera than the {obj2}?" | No | Yes |
| `obj2_farther` | "Is the {obj2} farther from the camera than the {obj1}?" | Yes | No |

We extract the logits for "Yes" and "No" tokens at the last position and compute:
- `P(Yes) = sigmoid(Yes_logit - No_logit)`
- `P(correct) = P(Yes)` when GT=Yes, `1 - P(Yes)` when GT=No

**Output**: One CSV per variant with columns: `model_path, variant, image, question, ground_truth, prediction, Yes_logit, No_logit, correct`

**Orchestration**: `run_infer_logit_vqa_phase.sh` — runs all models in parallel across GPUs

### 3. Per-Variant Heatmap (`plot_heatmap_vqa.py`)

Plots a 16x16 heatmap of mean P(Yes) per (phA, phB) cell for a single variant CSV. Shows how the model's "Yes" probability varies across phase space.

### 4. Aggregated Heatmap (`plot_heatmap_vqa_agg.py`)

Combines all 4 variant CSVs into a single heatmap of mean P(correct) (accuracy). Axes show degrees (0-337.5). Includes a metrics panel with:
- Mean accuracy, RMSE
- Vertical consistency (top vs bottom phase regions)
- Horizontal consistency (left vs right phase regions)
- Overlap corner deltas

**Output**: PNG heatmap + TSV metrics file

### 5. Consistent vs Counter Analysis (`plot_heatmap_vqa_counter.py`)

Uses `cell_class.json` to split data into consistent and counter subsets, then reports:
- Mean accuracy for each subset
- Difference (consistent - counter)

A positive difference indicates the model performs better when visual position aligns with the typical "higher = farther" heuristic.

**Output**: PNG with 4 panels (cell split map, consistent heatmap, counter heatmap, metrics) + TSV

### 6. Cell Classification (`save_cell_class.py`)

Generates `cell_class.json` from the VQA JSON bounding boxes:
```
python save_cell_class.py \
  --vqa-json /app/blender/output/phasevar_5/vqa_obj1.json \
  --output /app/blender/output/phasevar_5/cell_class.json
```

## Key Findings

All models show higher accuracy on **consistent** images (where visual position matches depth) than on **counter** images, confirming reliance on the "higher in image = farther" heuristic.

| Model | Variant | Mean Acc | Acc Consistent | Acc Counter | Delta |
|-------|---------|----------|---------------|-------------|-------|
| VILA | base | 0.488 | 0.504 | 0.471 | +0.033 |
| VILA | 400k | 0.669 | 0.804 | 0.538 | +0.267 |
| VILA | 2m | 0.812 | 0.875 | 0.749 | +0.127 |
| VILA | roborefer | 0.793 | 0.816 | 0.770 | +0.046 |
| Molmo | base | 0.528 | 0.565 | 0.487 | +0.078 |
| Molmo | 800k | 0.531 | 0.628 | 0.430 | +0.198 |
| Molmo | 2m | 0.666 | 0.703 | 0.630 | +0.073 |
| Qwen | base | 0.570 | 0.776 | 0.360 | +0.416 |
| Qwen | 2m | 0.500 | 0.648 | 0.353 | +0.295 |
| Qwen | 3b | 0.512 | 0.595 | 0.428 | +0.166 |
| Qwen-S | 235b | 0.908 | 0.948 | 0.880 | +0.068 |

Notable patterns:
- **Qwen base** has the largest gap (+0.416), suggesting heavy reliance on visual heuristics
- **Finetuning** generally increases overall accuracy but often increases the gap too (especially at 400k-800k), then narrows at higher data scales
- **RoboRefer** and **Qwen-Small 235b** show the smallest gaps, suggesting better spatial grounding

## File Structure

```
/app/VILA/
  llava/cli/infer_logit_vqa.py       # VILA inference script
  plot_heatmap_vqa.py                 # Per-variant P(Yes) heatmap
  plot_heatmap_vqa_agg.py             # Aggregated P(correct) heatmap
  plot_heatmap_vqa_counter.py         # Consistent vs counter analysis
  save_cell_class.py                  # Generate cell classification JSON
  run_infer_logit_vqa_phase.sh        # Orchestration script
  logit_results_vqa_phase*.csv        # Result CSVs
  logit_heatmap_vqa_phase*.png        # Generated heatmaps

/app/molmo/
  infer_logit_vqa.py                  # Molmo inference script
  run_infer_logit_vqa_phase.sh
  logit_results_molmo_vqa_phase*.csv
  logit_heatmap_molmo_vqa_phase*.png

/app/qwen/
  infer_logit_vqa.py                  # Qwen inference script
  run_infer_logit_vqa_phase.sh
  logit_results_qwen_vqa_phase*.csv
  logit_heatmap_qwen_vqa_phase*.png
  results_small/                      # Qwen-Small (235B) results

/app/blender/
  output/phasevar_5/                  # Dataset
    vqa_obj1.json                     # VQA entries with bounding boxes
    cell_class.json                   # Precomputed consistent/counter labels
    *.png                             # Rendered images
```
