# CLAUDE.md — VLM Spatial Reasoning Experiment

## Project Overview

This repo (VILA) is one of three model repos used in a VLM spatial reasoning evaluation experiment. The experiment tests whether Vision Language Models rely on visual shortcuts (e.g., "higher in image = farther away") rather than true depth understanding by varying the vertical positions of two objects across a 16x16 phase grid.

### Models Tested

| Model | Repo | Variants |
|-------|------|----------|
| NVILA-Lite-2B | `/app/VILA/` | base, 80k, 400k, 800k, 2m, roborefer |
| Molmo-7B | `/app/molmo/` | base, 80k, 400k, 800k, 2m |
| Qwen2.5-VL-7B | `/app/qwen/` | base, 80k, 400k, 800k, 2m, 3b |
| Qwen3-VL-235B | `/app/qwen/results_small/` | 235b |

Variant suffixes (80k, 400k, 800k, 2m) = amount of spatial reasoning training data for finetuning. "roborefer" = RoboRefer-2B-SFT model. "3b" = 3B parameter Qwen variant.

## Dataset

- **Source**: `/app/blender/output/phasevar_5/`
- **12 scenes** x **256 phase combinations** (16 phA x 16 phB) = **3,072 images**
- Phase indices 0-15 map to angles 0-337.5 degrees (step 22.5)
- **VQA JSON**: `/app/blender/output/phasevar_5/vqa_obj1.json`
- **Cell classification**: `/app/blender/output/phasevar_5/cell_class.json`
  - 118 consistent cells (obj1 visually higher, matching "higher=farther" heuristic)
  - 116 counter cells (obj1 visually lower, contradicting heuristic)
  - 22 mixed cells excluded (scenes within same cell disagree)
  - Cell (8,8) forced to "consistent" (was equal+consistent mix)

## Key Concepts

### P(correct) Computation
```
logit_diff = Yes_logit - No_logit
P(Yes) = sigmoid(logit_diff)
P(correct) = P(Yes) when GT=Yes, 1-P(Yes) when GT=No
```

### 4 Question Variants
Each image gets 4 questions (obj1_closer, obj2_closer, obj1_farther, obj2_farther) to eliminate question-wording bias.

### Consistent vs Counter
- **Consistent**: obj1_cy < obj2_cy (obj1 visually higher) — visual position matches typical depth heuristic
- **Counter**: obj1_cy > obj2_cy (obj1 visually lower) — contradicts the heuristic
- Classification depends only on (phA, phB) phase indices, NOT image paths — reusable across datasets

## Pipeline Scripts

### 1. Inference — Extract Yes/No Logits
```bash
# VILA
CUDA_VISIBLE_DEVICES=0 python llava/cli/infer_logit_vqa.py \
    --model-path Efficient-Large-Model/NVILA-Lite-2B \
    --vqa-json /app/blender/output/phasevar_5/vqa_obj1.json \
    --image-root /app/blender \
    --output-csv logit_results_vqa_phase.csv
```
Output: one CSV per variant with columns: `model_path, variant, image, question, ground_truth, prediction, Yes_logit, No_logit, correct`

### 2. Orchestration
```bash
bash run_infer_logit_vqa_phase.sh
```
Runs all VILA model variants in parallel across GPUs. Currently inference is commented out (already completed); only runs plotting steps.

### 3. Cell Classification
```bash
python save_cell_class.py \
    --vqa-json /app/blender/output/phasevar_5/vqa_obj1.json \
    --output /app/blender/output/phasevar_5/cell_class.json
```
Generates `cell_class.json` from VQA JSON bounding boxes. Forces (8,8) to "consistent" and removes mixed cells.

### 4. Per-Variant Heatmap
```bash
python plot_heatmap_vqa.py \
    --input logit_results_vqa_phase_obj1_closer.csv \
    --output logit_heatmap_vqa_phase_obj1_closer.png \
    --title "NVILA-Lite-2B | obj1_closer"
```
16x16 heatmap of mean P(Yes) for a single variant CSV.

### 5. Aggregated Accuracy Heatmap
```bash
python plot_heatmap_vqa_agg.py \
    --inputs logit_results_vqa_phase_obj1_closer.csv \
             logit_results_vqa_phase_obj2_closer.csv \
             logit_results_vqa_phase_obj1_farther.csv \
             logit_results_vqa_phase_obj2_farther.csv \
    --output logit_heatmap_vqa_phase_agg.png \
    --title "NVILA-Lite-2B | Mean accuracy"
```
Combines 4 variants into mean P(correct) heatmap. Axes in degrees (0-337.5), major ticks at 0/45/90/.../315. Outputs PNG + TSV with metrics (mean/min/max accuracy, RMSE, region deltas).

### 6. Argmax Accuracy Heatmap
```bash
python plot_heatmap_vqa_acc.py \
    --inputs [same 4 CSVs] \
    --output logit_heatmap_vqa_phase_acc.png \
    --title "NVILA-Lite-2B | Accuracy"
```
Uses argmax (correct column) instead of P(correct).

### 7. Consistent vs Counter Analysis
```bash
python plot_heatmap_vqa_counter.py \
    --inputs [4 variant CSVs] \
    --cell-class /app/blender/output/phasevar_5/cell_class.json \
    --output logit_heatmap_vqa_counter.png \
    --title "Consistent vs Counter"
```
4-panel plot: cell split map, consistent heatmap, counter heatmap, metrics. Reports acc_consistent, acc_counter, and delta. Outputs PNG + TSV.

### 8. Side-by-Side Model Comparison
```bash
python plot_heatmap_compare.py \
    --prefixes logit_results_vqa_phase logit_results_vqa_phase_400k logit_results_vqa_phase_2m \
    --labels "base" "400k" "2m" \
    --cell-class /app/blender/output/phasevar_5/cell_class.json \
    --output heatmap_compare_nvila.png \
    --suptitle "NVILA-Lite-2B"
```
Outputs **two separate images**: `{base}_consistent.png` and `{base}_counter.png`. Each shows a 1xN row of heatmaps (one per model variant) with gray masking for non-relevant cells. Uses theta labels, degree axes (major ticks at 0/90/180/270), large fonts (20pt labels/ticks, 30pt suptitle).

### 9. Accuracy Line Charts
```bash
python plot_accuracy_lines.py
```
Hardcoded data from `counter_summary.csv`. Plots accuracy vs training data scale for each model (overall, consistent, counter lines).

## CSV Naming Convention

| Pattern | Description |
|---------|-------------|
| `logit_results_vqa_phase_obj1_closer.csv` | VILA base, obj1_closer variant |
| `logit_results_vqa_phase_400k_obj1_closer.csv` | VILA 400k, obj1_closer variant |
| `logit_results_vqa_phase_roborefer_obj1_closer.csv` | VILA roborefer |
| `/app/molmo/logit_results_molmo_vqa_phase_obj1_closer.csv` | Molmo base |
| `/app/molmo/logit_results_molmo_vqa_phase_400k_obj1_closer.csv` | Molmo 400k |
| `/app/qwen/logit_results_qwen_vqa_phase_obj1_closer.csv` | Qwen base |
| `/app/qwen/logit_results_qwen_vqa_phase_400k_obj1_closer.csv` | Qwen 400k |

## Key Model Paths

```
# VILA finetuned models
/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632
/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221
/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221
/app/DATA/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221

# HuggingFace models
Efficient-Large-Model/NVILA-Lite-2B        # VILA base
Zhoues/RoboRefer-2B-SFT                    # RoboRefer
```

## Key Results

All models show higher accuracy on consistent images than counter images, confirming reliance on the "higher in image = farther" heuristic.

| Model | Variant | Mean Acc | Consistent | Counter | Delta |
|-------|---------|----------|------------|---------|-------|
| VILA | base | 0.488 | 0.504 | 0.471 | +0.033 |
| VILA | 400k | 0.669 | 0.804 | 0.538 | +0.267 |
| VILA | 2m | 0.812 | 0.875 | 0.749 | +0.127 |
| VILA | roborefer | 0.793 | 0.816 | 0.770 | +0.046 |
| Molmo | base | 0.528 | 0.565 | 0.487 | +0.078 |
| Molmo | 2m | 0.666 | 0.703 | 0.630 | +0.073 |
| Qwen | base | 0.570 | 0.776 | 0.360 | +0.416 |
| Qwen | 2m | 0.500 | 0.648 | 0.353 | +0.295 |
| Qwen-S | 235b | 0.908 | 0.948 | 0.880 | +0.068 |

Notable:
- Qwen base has the largest gap (+0.416) — heavy reliance on visual heuristics
- Finetuning increases overall accuracy but often increases the gap at mid-scale (400k-800k), then narrows at 2m
- RoboRefer and Qwen-Small 235b show smallest gaps — better spatial grounding

## File Structure

```
/app/VILA/
  CLAUDE.md                                 # This file
  EXPERIMENT_README.md                      # Full experiment documentation
  llava/cli/infer_logit_vqa.py              # VILA inference (Yes/No logit extraction)
  run_infer_logit_vqa_phase.sh              # Orchestration (all VILA models)
  save_cell_class.py                        # Generate cell_class.json
  plot_heatmap_vqa.py                       # Per-variant P(Yes) heatmap
  plot_heatmap_vqa_agg.py                   # Aggregated P(correct) heatmap (degree axes)
  plot_heatmap_vqa_acc.py                   # Argmax accuracy heatmap
  plot_heatmap_vqa_counter.py               # Consistent vs counter analysis (uses cell_class.json)
  plot_heatmap_compare.py                   # Side-by-side comparison (2 images: consistent + counter)
  plot_accuracy_lines.py                    # Accuracy vs training scale line charts
  counter_summary.csv                       # All models' consistent/counter metrics
  logit_results_vqa_phase*.csv              # Result CSVs
  logit_heatmap_vqa_phase*.png              # Generated heatmaps

/app/molmo/
  infer_logit_vqa.py                        # Molmo inference
  run_infer_logit_vqa_phase.sh              # Orchestration
  logit_results_molmo_vqa_phase*.csv        # Results

/app/qwen/
  infer_logit_vqa.py                        # Qwen inference
  run_infer_logit_vqa_phase.sh              # Orchestration
  logit_results_qwen_vqa_phase*.csv         # Results
  results_small/                            # Qwen-Small (235B) results

/app/blender/
  output/phasevar_5/
    vqa_obj1.json                           # VQA entries with bounding boxes
    cell_class.json                         # Precomputed consistent/counter labels (118+116)
    *.png                                   # Rendered images (3,072 total)
```

## Notion

Results are documented on Notion:
- Main page (Molmo Consistency Issue): `3139c310-69ec-802f-b522-f2a7aa3b5cb3`
- Experiment Details sub-page: `3189c310-69ec-8130-befe-e64888510402`

## Design Decisions & Gotchas

1. **Cell classification is phase-based, not path-based**: Classification depends only on (phA, phB) indices, not image file paths. This allows reuse across datasets (e.g., phasevar_4 vs phasevar_5) and models that may have different path prefixes in their CSVs.

2. **Cell (8,8) exception**: This cell had both "equal" and "consistent" labels across its 12 scenes. It's forced to "consistent" in `save_cell_class.py`.

3. **22 mixed cells removed**: Cells where different scenes within the same (phA, phB) disagree on consistent/counter classification are excluded entirely.

4. **Qwen base used phasevar_4 paths**: The base Qwen model CSVs reference `phasevar_4/` paths while others use `phasevar_5/`. This doesn't affect analysis since cell classification uses phase indices, not paths.

5. **plot_heatmap_vqa_agg.py updates**: Removed cyan Rectangle annotations, added degree-based axes (extent mapping), renamed "Mean P(correct)" to "Mean accuracy", major ticks at 0/45/90/.../315.

6. **plot_heatmap_compare.py outputs 2 images**: Originally a single 2x3 grid, refactored to output `{base}_consistent.png` and `{base}_counter.png` separately. Gray masking for non-relevant cells. Uses constrained_layout, theta labels, 90-degree major ticks.

7. **BBox format**: `[x1, y1, x2, y2]`, center_y = (y1+y2)/2. Lower cy = higher in image.
