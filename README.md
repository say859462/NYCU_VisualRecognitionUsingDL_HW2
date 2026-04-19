# HW2 — DETR-Based Street-View Digit Detection

#### ***Course:*** Visual Recognition using Deep Learning (NYCU)

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

After a long experiment sequence, the final direction converged to **RT-DETRv2 with a fixed rectangular input shape (`320x640`) plus an auxiliary digit-classification branch**. The strongest hidden-test score in our experiments was **0.41 mAP**, achieved by:

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
- On the hidden test set, both `EXP60` and `EXP64` reached **0.41 mAP**.
- `EXP65` (targeted confusion loss) improved validation but dropped on test:
  - validation `0.4775`, test `0.40`
- `EXP66` (queries `300 -> 1000`) did not improve test:
  - validation `0.4710`, test `0.40`

In short, **auxiliary fusion was the most reliable positive direction**, while hard-coded confusion loss and simply adding more queries were not strong test-oriented solutions.

---

## Method Summary

### Final Mainline Detector

- Backbone: `ResNet-50-VD`
- Detector: `RT-DETRv2`
- Model source: `PekingU/rtdetr_v2_r50vd`
- Load strategy: `pretrained_reset_transformer`
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

### Best Additional Idea

The most effective idea was to add an **auxiliary digit classifier** during training and use it during inference to refine the detector logits. This produced:

- higher GT-side correctness in `EXP60`
- a more robust version with fewer side effects in `EXP64`

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

## Project Layout

```text
HW2/
├── Dataset/                                # train / valid / test images and COCO jsons
├── Files/                                  # reports and experiment logs
├── Model_Weight/                           # checkpoints
├── Plot/                                   # training curves and validation analysis
├── Prediction/                             # prediction json outputs
├── AGENT.md                                # condensed experiment conclusions
├── main.py                                 # training entry
├── test.py                                 # inference / validation entry
├── visualize.py                            # validation analysis / visualization
├── dataset.py                              # dataset and augmentation pipeline
├── model.py                                # detector implementations and adapters
├── utils.py                                # postprocess, evaluation, saving helpers
├── train.py                                # one-epoch training loop
├── val.py                                  # validation loop
└── config_exp*.json                        # experiment configs
```

---

## Usage

### 1. Train a Baseline or New Experiment

Example: train the strong RT-DETRv2 baseline (`EXP54`)

```powershell
python main.py --config .\config_exp54_rtdetrv2_full_reset_auxaug.json
```

Example: train the auxiliary-branch model (`EXP56`)

```powershell
python main.py --config .\config_exp56_rtdetrv2_full_reset_auxcls_trainonly.json
```

Example: train layout-shift augmentation experiment (`EXP69a`)

```powershell
python main.py --config .\config_exp69a_rtdetrv2_auxcls_layout_shift_train.json
```

### 2. Validation / Inference

Evaluate `EXP60`:

```powershell
python test.py --config .\config_exp60_rtdetrv2_auxcls_gated_fusion_eval.json --split val --model_path .\Model_Weight\56\best_model_rtdetrv2_full_reset_auxcls_trainonly.pth --output_json .\Prediction\exp60_val_gated020.json
```

Evaluate `EXP64` with attenuation `0.15`:

```powershell
python test.py --config .\config_exp64_rtdetrv2_auxcls_family_attenuation_eval.json --split val --model_path .\Model_Weight\56\best_model_rtdetrv2_full_reset_auxcls_trainonly.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_json .\Prediction\exp64_val_attn015.json
```

Generate test predictions:

```powershell
python test.py --config .\config_exp64_rtdetrv2_auxcls_family_attenuation_eval.json --split test --model_path .\Model_Weight\56\best_model_rtdetrv2_full_reset_auxcls_trainonly.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_json .\Prediction\pred.json
```

### 3. Visualization / Error Analysis

Run detailed validation analysis for `EXP64 attn015`:

```powershell
python visualize.py --config .\config_exp64_rtdetrv2_auxcls_family_attenuation_eval.json --split val --model_path .\Model_Weight\56\best_model_rtdetrv2_full_reset_auxcls_trainonly.pth --aux_digit_family_attenuation_weights '{"1,4,7": 0.15}' --output_dir .\Plot\64\val_visualization_attn015
```

### 4. Preview Train-Time Augmentations

For `EXP69a`, preview horizontal layout-shift augmentation before training:

```powershell
python .\preview_train_augmentations.py --config .\config_exp69a_rtdetrv2_auxcls_layout_shift_train.json --output_dir .\Plot\69a\train_aug_preview --num_samples 8
```

---

## Performance Snapshot

### Main RT-DETRv2 Experiments

| Experiment | Main Idea | Val AP | AP50 | AP75 | Hidden-Test mAP | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `EXP54` | fixed `320x640` RT-DETRv2 baseline | `0.4725` | `0.9275` | `0.414` | - | strong baseline |
| `EXP56` | train-only auxiliary digit head | `0.4729` | `0.9298` | `0.411` | - | cleaner predictions, no direct test evidence alone |
| `EXP60` | inference-time auxiliary fusion | `0.4790` | `0.9361` | `0.421` | `0.41` | best validation peak |
| `EXP64` | `EXP60` + attenuation on `{1,4,7}` | `0.4788` | `0.9360` | `0.422` | `0.41` | strongest robust submission-oriented variant |
| `EXP65` | targeted confusion loss | `0.4775` | `0.9365` | `0.422` | `0.40` | validation gain did not transfer to test |
| `EXP66` | queries `300 -> 1000` | `0.4710` | `0.9298` | `0.408` | `0.40` | more queries did not solve the main bottleneck |
| `EXP68` | `EXP60 + EXP64` WBF ensemble | `0.4767` | `0.9355` | `0.416` | - | ensemble regressed |
| `EXP68b` | `EXP60 + EXP64` concat + NMS ensemble | `0.4768` | `0.9359` | `0.416` | - | ensemble regressed |

### Final Recommendation

If the goal is the **best validation peak**, use:

- `EXP60`

If the goal is a **more robust submission-oriented variant** with the same hidden-test score in our experiments, use:

- `EXP64 attn015`

---

## Analysis Workflow

This project relies heavily on validation visualization and structured error summaries.

Important analysis artifacts include:

- training curves
- confusion matrices
- query coverage curves
- per-digit error breakdowns
- representative qualitative samples

Examples:

- [Plot/54/val_visualization](/E:/School/Course/VisualRecognitionUsingDL/HW2/Plot/54/val_visualization)
- [Plot/60/val_visualization_gated020](/E:/School/Course/VisualRecognitionUsingDL/HW2/Plot/60/val_visualization_gated020)
- [Plot/64/val_visualization_attn015](/E:/School/Course/VisualRecognitionUsingDL/HW2/Plot/64/val_visualization_attn015)
- [Plot/65/val_visualization](/E:/School/Course/VisualRecognitionUsingDL/HW2/Plot/65/val_visualization)

The final report also uses these outputs to justify every important conclusion with evidence instead of only qualitative claims.

---

## Submission Notes

- Final competition submission must be a COCO-style `pred.json`
- Do **not** include dataset files or model checkpoints in the final zip
- The report must be written in **English**
- The final report format required by the homework is **PDF**

---

## References

- RT-DETR official repository: https://github.com/lyuwenyu/RT-DETR
- Hugging Face checkpoint: `PekingU/rtdetr_v2_r50vd`
- DETR: End-to-End Object Detection with Transformers
- RT-DETRv2: Improved Baseline for Real-Time Detection Transformer

