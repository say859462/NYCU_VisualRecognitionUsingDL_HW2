# NYCU Visual Recognition using Deep Learning

#### ***Student:*** 劉哲良

#### ***Student ID:*** 314551113


---

## Introduction

This project addresses the **digit detection** task in Homework 2 of NYCU *Visual Recognition using Deep Learning*. The input is an RGB street-view image and the target is a COCO-style list of digit bounding boxes and classes. The dataset contains **30,062 / 3,340 / 13,068** images for train / validation / test.

The homework requires:

- a **DETR-family** detector
- a **ResNet-50** backbone
- **no external data**
- COCO-style final prediction file `pred.json`

After a long experiment sequence, the final direction converged to **RT-DETRv2 with a fixed rectangular input shape (`320x640`) plus an auxiliary digit-classification branch**. The strongest public-test score in our experiments was **0.41 mAP**, achieved by:

- `EXP60`: aggressive inference-time auxiliary fusion
- `EXP64`: a more robust variant of `EXP60` with family-specific attenuation on `{1,4,7}`

---

## Core Findings

The most important conclusions from this homework are:

- The main bottleneck was **not** post-processing, raw query count, or train-time multi-scale resizing.
- The strongest gains came from improving **class ranking** and **digit discriminability**.
- `EXP60` was the first experiment that clearly exceeded the previous honest-validation ceiling:
  - `AP 0.4790 / AP50 0.9361 / AP75 0.421`
- `EXP64 attn015` preserved almost all of that gain while making predictions cleaner:
  - `AP 0.4788 / AP50 0.9360 / AP75 0.422`
- On the public test set, both `EXP60` and `EXP64` reached **0.41 mAP**.
- `EXP65` (targeted confusion loss) improved validation but dropped on test:
  - validation `0.4775`, test `0.40`
- `EXP66` (queries `300 -> 1000`) did not improve test:
  - validation `0.4710`, test `0.40`

In short, **auxiliary fusion was the most reliable positive direction**, while hard-coded confusion loss and simply adding more queries were not strong test-oriented solutions.

---

## Method Summary

### Final Mainline Detector

- Backbone: `ResNet-50`
- Detector: `RT-DETRv2`
- Model source: `PekingU/rtdetr_v2_r50vd`
- Load strategy: Preserve ResNet 50 pretrained weight only , initialize all other components
- Input size: fixed `320x640`
- Queries: `300`
- Post-process: official Hugging Face RT-DETRv2 pipeline

### Training Recipe

- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Backbone learning rate: `1e-5`
- Weight decay: `5e-4`
- Scheduler: `OneCycle`
- EMA: enabled
- AMP: enabled
- Train batch size: `24`

### Main Augmentations

- Color jitter
- Gamma augmentation
- Light affine augmentation
- Fixed rectangular resize


---

## Environment Setup

### Option 1 — Existing virtual environment

If you already have the homework environment:

```powershell
.venv\Scripts\Activate.ps1
```

### Option 2 — Create a fresh virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision transformers scipy pillow matplotlib tqdm pycocotools numpy
```

### Recommended Environment

- Python `3.11`
- PyTorch `2.x`
- torchvision
- transformers
- scipy
- pillow
- matplotlib
- tqdm
- pycocotools

---


## Usage

### 1. Train 

```powershell
python main.py --config .\config.json
```


### 2. Validation / Inference

Evaluate :

```powershell
python test.py --config .\config.json --split val --model_path .\Model_Weight\best_model.pth --output_json .\Prediction\val.json
```
Evaluate with attenuation `0.15`:

```powershell
python test.py --config .\config.json --split val --model_path .\Model_Weight\best_model.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_json .\Prediction\val_attn015.json
```

Generate test predictions:

```powershell
python test.py --config .\config.json --split test --model_path .\Model_Weight\best_model.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_json .\Prediction\pred.json
```

### 3. Visualization / Error Analysis

Run detailed validation analysis for `EXP64 attn015`:

```powershell
python visualize.py --config .\config.json --split val --model_path .\Model_Weight\best_model.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_dir .\Plot\val_visualization
```

---

## Performance Snapshot

### Main RT-DETRv2 Experiments

| Experiment | Main Idea | Val AP | AP50 | AP75 | public-Test mAP | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `EXP54` | fixed `320x640` RT-DETRv2 baseline | `0.4725` | `0.9275` | `0.414` | - | strong baseline |
| `EXP56` | train-only auxiliary digit head | `0.4729` | `0.9298` | `0.411` | - | cleaner predictions, no direct test evidence alone |
| `EXP60` | inference-time auxiliary fusion | `0.4790` | `0.9361` | `0.421` | `0.41` | best validation peak |
| `EXP64` | `EXP60` + attenuation on `{1,4,7}` | `0.4788` | `0.9360` | `0.422` | `0.41` | strongest robust submission-oriented variant |
| `EXP65` | targeted confusion loss | `0.4775` | `0.9365` | `0.422` | `0.40` | validation gain did not transfer to test |
| `EXP66` | queries `300 -> 1000` | `0.4710` | `0.9298` | `0.408` | `0.40` | more queries did not solve the main bottleneck |
| `EXP68` | `EXP60 + EXP64` WBF ensemble | `0.4767` | `0.9355` | `0.416` | - | ensemble regressed |
| `EXP68b` | `EXP60 + EXP64` concat + NMS ensemble | `0.4768` | `0.9359` | `0.416` | - | ensemble regressed |


---

## References

- RT-DETR official repository: https://github.com/lyuwenyu/RT-DETR
- Hugging Face checkpoint: `PekingU/rtdetr_v2_r50vd`
- DETR: End-to-End Object Detection with Transformers
- RT-DETRv2: Improved Baseline for Real-Time Detection Transformer
- Deformable detr: Deformable transformers for end-to-end object detection
- Relation detr: Exploring explicit position relation prior for object detection
## Snapshot
<img width="1107" height="53" alt="image" src="https://github.com/user-attachments/assets/316fae41-3952-439c-90b5-8ac89ab9f684" />


