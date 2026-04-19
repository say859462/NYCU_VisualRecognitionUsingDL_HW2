"""Validation visualization and detailed error analysis utilities."""

import argparse
from functools import partial
import json
import math
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from dataset import DetectionDataset, DetectionTransform, collate_fn
from model import adapt_checkpoint_state_dict, build_model_from_config
from utils import (
    build_inference_postprocess_kwargs,
    collect_coco_predictions,
    collect_coco_predictions_debug,
)


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor back to displayable RGB tensor."""
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


def build_prediction_map(predictions: list[dict]) -> dict[int, list[dict]]:
    """Group predictions by image id."""
    pred_map = {}
    for pred in predictions:
        image_id = int(pred["image_id"])
        if image_id not in pred_map:
            pred_map[image_id] = []
        pred_map[image_id].append(pred)
    return pred_map


def build_passthrough_debug_stage_map(pred_map: dict[int, list[dict]]) -> dict[int, dict]:
    """Create synthetic stage traces when predictions come from an external json."""
    stage_names = [
        "pre_threshold",
        "post_threshold",
        "post_topk_pre_class_nms",
        "post_class_nms",
        "post_topk_post_class_nms",
        "post_agnostic_nms",
        "post_cross_class_clean",
        "post_topk_final",
        "post_topk",
        "final",
    ]
    debug_map = {}
    for image_id, predictions in pred_map.items():
        stage_blob = {"_meta": {"topk_stage": "external_prediction_json"}}
        for stage_name in stage_names:
            stage_blob[stage_name] = [dict(prediction) for prediction in predictions]
        debug_map[int(image_id)] = stage_blob
    return debug_map


def predictions_to_tensors(predictions: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert prediction dict list to tensors."""
    if not predictions:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0,), dtype=torch.float32),
        )

    boxes = []
    labels = []
    scores = []
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(pred["category_id"]) - 1)
        scores.append(float(pred["score"]))

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(scores, dtype=torch.float32),
    )


def analyze_sample(target: dict, predictions: list[dict], iou_match_threshold: float) -> dict:
    """Assign one primary error type to a sample using heuristic rules."""
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes, pred_labels, pred_scores = predictions_to_tensors(predictions)

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]

    if num_gt == 0 and num_pred == 0:
        return {"sample_type": "correct", "details": {"num_gt": 0, "num_pred": 0}}
    if num_gt == 0 and num_pred > 0:
        return {
            "sample_type": "duplicate",
            "details": {"num_gt": 0, "num_pred": num_pred},
        }
    if num_pred == 0:
        return {"sample_type": "missed", "details": {"num_gt": num_gt, "num_pred": 0}}

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    pred_best_iou, pred_best_gt = iou_matrix.max(dim=1)
    gt_best_iou, gt_best_pred = iou_matrix.max(dim=0)

    correct_match_matrix = (iou_matrix >= iou_match_threshold) & (pred_labels[:, None] == gt_labels[None, :])
    gt_match_counts = correct_match_matrix.sum(dim=0)
    pred_match_counts = correct_match_matrix.sum(dim=1)

    duplicate = bool((gt_match_counts > 1).any().item() or (pred_match_counts > 1).any().item())
    classification_error = bool(
        (
            (pred_best_iou >= iou_match_threshold)
            & (pred_labels != gt_labels[pred_best_gt])
        ).any().item()
    )
    localization_error = bool(
        (
            (pred_labels == gt_labels[pred_best_gt])
            & (pred_best_iou >= 0.10)
            & (pred_best_iou < iou_match_threshold)
        ).any().item()
    )
    missed = bool((gt_match_counts == 0).any().item())

    grouped = False
    if num_pred > 0 and num_gt > 1:
        grouped = bool((iou_matrix >= 0.20).sum(dim=1).max().item() >= 2)
    if not grouped and num_pred > 0 and num_gt > 0:
        matched_gt = gt_boxes[pred_best_gt]
        pred_area = (
            (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp_min(1.0)
            * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp_min(1.0)
        )
        gt_area = (
            (matched_gt[:, 2] - matched_gt[:, 0]).clamp_min(1.0)
            * (matched_gt[:, 3] - matched_gt[:, 1]).clamp_min(1.0)
        )
        area_ratio = pred_area / gt_area
        grouped = bool(
            (
                (pred_best_iou >= 0.20)
                & ((area_ratio > 1.8) | (area_ratio < 0.55))
            ).any().item()
        )

    correct = (
        num_pred == num_gt
        and num_gt > 0
        and bool((gt_match_counts == 1).all().item())
        and bool((pred_match_counts == 1).all().item())
        and not duplicate
    )

    if correct:
        sample_type = "correct"
    elif duplicate:
        sample_type = "duplicate"
    elif classification_error:
        sample_type = "cls_error"
    elif grouped:
        sample_type = "grouped"
    elif localization_error:
        sample_type = "loc_error"
    elif missed:
        sample_type = "missed"
    else:
        sample_type = "loc_error"

    return {
        "sample_type": sample_type,
        "details": {
            "num_gt": num_gt,
            "num_pred": num_pred,
            "best_iou_mean": float(pred_best_iou.mean().item()) if num_pred > 0 else 0.0,
            "best_score_mean": float(pred_scores.mean().item()) if num_pred > 0 else 0.0,
        },
    }


def _sort_error_tags(tags: set[str] | list[str]) -> list[str]:
    priority = {
        "duplicate": 0,
        "cls_error": 1,
        "grouped": 2,
        "loc_error": 3,
        "missed": 4,
        "background_fp": 5,
    }
    return sorted(set(tags), key=lambda tag: (priority.get(tag, 99), tag))


def _select_primary_type(sample_tags: list[str], is_clean: bool) -> str:
    if is_clean:
        return "correct"
    if "duplicate" in sample_tags:
        return "duplicate"
    if "cls_error" in sample_tags:
        return "cls_error"
    if "grouped" in sample_tags:
        return "grouped"
    if "loc_error" in sample_tags:
        return "loc_error"
    if "missed" in sample_tags:
        return "missed"
    if "background_fp" in sample_tags:
        return "background_fp"
    return "loc_error"


def _greedy_match_pairs(
    iou_matrix: torch.Tensor,
    min_iou: float,
    valid_mask: torch.Tensor | None = None,
) -> list[tuple[int, int, float]]:
    if iou_matrix.numel() == 0:
        return []

    if valid_mask is None:
        valid_mask = torch.ones_like(iou_matrix, dtype=torch.bool)

    sorted_indices = torch.argsort(iou_matrix.reshape(-1), descending=True)
    used_pred_indices: set[int] = set()
    used_gt_indices: set[int] = set()
    num_gt = iou_matrix.shape[1]
    matches: list[tuple[int, int, float]] = []

    for flat_index in sorted_indices.tolist():
        pred_idx = flat_index // num_gt
        gt_idx = flat_index % num_gt
        iou_value = float(iou_matrix[pred_idx, gt_idx].item())
        if iou_value < min_iou:
            break
        if not bool(valid_mask[pred_idx, gt_idx].item()):
            continue
        if pred_idx in used_pred_indices or gt_idx in used_gt_indices:
            continue
        used_pred_indices.add(pred_idx)
        used_gt_indices.add(gt_idx)
        matches.append((pred_idx, gt_idx, iou_value))

    return matches


def _increment_digit_stat(table: dict[str, dict[str, int]], digit: int, key: str, amount: int = 1) -> None:
    digit_key = str(int(digit))
    if digit_key not in table:
        table[digit_key] = {}
    table[digit_key][key] = table[digit_key].get(key, 0) + int(amount)


def analyze_sample_v2(
    target: dict,
    predictions: list[dict],
    iou_match_threshold: float,
    loose_iou_threshold: float = 0.10,
    grouped_iou_threshold: float = 0.25,
) -> dict:
    """Return multi-label tags plus GT-centric / prediction-centric summaries."""
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes, pred_labels, pred_scores = predictions_to_tensors(predictions)

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]

    gt_stats = {
        "gt_total": int(num_gt),
        "gt_matched_correct": 0,
        "gt_wrong_class": 0,
        "gt_loc_error": 0,
        "gt_missed": 0,
    }
    pred_stats = {
        "pred_total": int(num_pred),
        "pred_true_positive": 0,
        "pred_grouped": 0,
        "pred_duplicate": 0,
        "pred_wrong_class": 0,
        "pred_loc_error": 0,
        "pred_background_fp": 0,
    }
    gt_digit_stats: dict[str, dict[str, int]] = {}
    pred_digit_stats: dict[str, dict[str, int]] = {}

    for gt_label in gt_labels.tolist():
        _increment_digit_stat(gt_digit_stats, int(gt_label), "gt_total")
    for pred_label in pred_labels.tolist():
        _increment_digit_stat(pred_digit_stats, int(pred_label), "pred_total")

    if num_gt == 0 and num_pred == 0:
        return {
            "primary_type": "correct",
            "sample_tags": [],
            "is_clean": True,
            "gt_stats": gt_stats,
            "pred_stats": pred_stats,
            "details": {
                "num_gt": 0,
                "num_pred": 0,
                "best_iou_mean": 0.0,
                "best_score_mean": 0.0,
                "matched_iou_mean": 0.0,
                "matched_pairs": 0,
                "gt_digit_stats": gt_digit_stats,
                "pred_digit_stats": pred_digit_stats,
            },
        }

    if num_gt == 0:
        pred_stats["pred_background_fp"] = int(num_pred)
        for pred_label in pred_labels.tolist():
            _increment_digit_stat(pred_digit_stats, int(pred_label), "pred_background_fp")
        return {
            "primary_type": "background_fp",
            "sample_tags": ["background_fp"],
            "is_clean": False,
            "gt_stats": gt_stats,
            "pred_stats": pred_stats,
            "details": {
                "num_gt": 0,
                "num_pred": int(num_pred),
                "best_iou_mean": 0.0,
                "best_score_mean": float(pred_scores.mean().item()) if num_pred > 0 else 0.0,
                "matched_iou_mean": 0.0,
                "matched_pairs": 0,
                "gt_digit_stats": gt_digit_stats,
                "pred_digit_stats": pred_digit_stats,
            },
        }

    if num_pred == 0:
        gt_stats["gt_missed"] = int(num_gt)
        for gt_label in gt_labels.tolist():
            _increment_digit_stat(gt_digit_stats, int(gt_label), "gt_missed")
        return {
            "primary_type": "missed",
            "sample_tags": ["missed"],
            "is_clean": False,
            "gt_stats": gt_stats,
            "pred_stats": pred_stats,
            "details": {
                "num_gt": int(num_gt),
                "num_pred": 0,
                "best_iou_mean": 0.0,
                "best_score_mean": 0.0,
                "matched_iou_mean": 0.0,
                "matched_pairs": 0,
                "gt_digit_stats": gt_digit_stats,
                "pred_digit_stats": pred_digit_stats,
            },
        }

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    pred_best_iou, pred_best_gt = iou_matrix.max(dim=1)
    same_label_matrix = pred_labels[:, None] == gt_labels[None, :]

    true_positive_matches = _greedy_match_pairs(iou_matrix, iou_match_threshold, same_label_matrix)
    matched_pred_indices = {pred_idx for pred_idx, _, _ in true_positive_matches}
    matched_gt_indices = {gt_idx for _, gt_idx, _ in true_positive_matches}
    gt_stats["gt_matched_correct"] = len(true_positive_matches)
    pred_stats["pred_true_positive"] = len(true_positive_matches)
    for pred_idx, gt_idx, _ in true_positive_matches:
        _increment_digit_stat(gt_digit_stats, int(gt_labels[gt_idx].item()), "gt_matched_correct")
        _increment_digit_stat(pred_digit_stats, int(pred_labels[pred_idx].item()), "pred_true_positive")

    sample_tag_set: set[str] = set()

    for gt_idx in range(num_gt):
        if gt_idx in matched_gt_indices:
            continue
        gt_label = int(gt_labels[gt_idx].item())
        gt_ious = iou_matrix[:, gt_idx]
        candidate_pred_indices = [pred_idx for pred_idx in range(num_pred) if pred_idx not in matched_pred_indices]

        has_wrong_class = any(
            float(gt_ious[pred_idx].item()) >= iou_match_threshold and int(pred_labels[pred_idx].item()) != gt_label
            for pred_idx in candidate_pred_indices
        )
        has_loc_error = any(
            loose_iou_threshold <= float(gt_ious[pred_idx].item()) < iou_match_threshold
            and int(pred_labels[pred_idx].item()) == gt_label
            for pred_idx in candidate_pred_indices
        )

        if has_wrong_class:
            gt_stats["gt_wrong_class"] += 1
            _increment_digit_stat(gt_digit_stats, gt_label, "gt_wrong_class")
            sample_tag_set.add("cls_error")
        elif has_loc_error:
            gt_stats["gt_loc_error"] += 1
            _increment_digit_stat(gt_digit_stats, gt_label, "gt_loc_error")
            sample_tag_set.add("loc_error")
        else:
            gt_stats["gt_missed"] += 1
            _increment_digit_stat(gt_digit_stats, gt_label, "gt_missed")
            sample_tag_set.add("missed")

    for pred_idx in range(num_pred):
        if pred_idx in matched_pred_indices:
            continue

        overlaps = iou_matrix[pred_idx]
        pred_label = int(pred_labels[pred_idx].item())
        overlap_count = int((overlaps >= grouped_iou_threshold).sum().item())
        same_label_overlap_mask = gt_labels == pred_label
        same_label_ious = overlaps[same_label_overlap_mask]
        has_duplicate = bool((same_label_ious >= iou_match_threshold).any().item()) if same_label_ious.numel() > 0 else False
        is_grouped = overlap_count >= 2
        best_iou = float(pred_best_iou[pred_idx].item())
        best_gt_label = int(gt_labels[pred_best_gt[pred_idx]].item())
        has_wrong_class = best_iou >= iou_match_threshold and pred_label != best_gt_label
        same_label_best_iou = float(same_label_ious.max().item()) if same_label_ious.numel() > 0 else 0.0

        if is_grouped:
            sample_tag_set.add("grouped")
        if has_duplicate:
            sample_tag_set.add("duplicate")

        if has_duplicate:
            pred_stats["pred_duplicate"] += 1
            _increment_digit_stat(pred_digit_stats, pred_label, "pred_duplicate")
        elif is_grouped:
            pred_stats["pred_grouped"] += 1
            _increment_digit_stat(pred_digit_stats, pred_label, "pred_grouped")
        elif has_wrong_class:
            pred_stats["pred_wrong_class"] += 1
            _increment_digit_stat(pred_digit_stats, pred_label, "pred_wrong_class")
            sample_tag_set.add("cls_error")
        elif same_label_best_iou >= loose_iou_threshold:
            pred_stats["pred_loc_error"] += 1
            _increment_digit_stat(pred_digit_stats, pred_label, "pred_loc_error")
            sample_tag_set.add("loc_error")
        else:
            pred_stats["pred_background_fp"] += 1
            _increment_digit_stat(pred_digit_stats, pred_label, "pred_background_fp")
            sample_tag_set.add("background_fp")

    matched_iou_mean = (
        float(sum(match_iou for _, _, match_iou in true_positive_matches) / len(true_positive_matches))
        if true_positive_matches
        else 0.0
    )
    is_clean = (
        gt_stats["gt_matched_correct"] == gt_stats["gt_total"]
        and pred_stats["pred_true_positive"] == pred_stats["pred_total"]
        and not sample_tag_set
    )
    sample_tags = _sort_error_tags(sample_tag_set)
    primary_type = _select_primary_type(sample_tags, is_clean)

    return {
        "primary_type": primary_type,
        "sample_tags": sample_tags,
        "is_clean": is_clean,
        "gt_stats": gt_stats,
        "pred_stats": pred_stats,
        "details": {
            "num_gt": int(num_gt),
            "num_pred": int(num_pred),
            "best_iou_mean": float(pred_best_iou.mean().item()),
            "best_score_mean": float(pred_scores.mean().item()),
            "matched_iou_mean": matched_iou_mean,
            "matched_pairs": len(true_positive_matches),
            "gt_digit_stats": gt_digit_stats,
            "pred_digit_stats": pred_digit_stats,
        },
    }


def _has_candidate(
    predictions: list[dict],
    gt_box: torch.Tensor,
    gt_label: int,
    min_iou: float,
    require_same_label: bool,
) -> bool:
    pred_boxes, pred_labels, _ = predictions_to_tensors(predictions)
    if pred_boxes.numel() == 0:
        return False
    ious = box_iou(pred_boxes, gt_box.unsqueeze(0)).squeeze(1)
    if require_same_label:
        return bool(((ious >= min_iou) & (pred_labels == gt_label)).any().item())
    return bool((ious >= min_iou).any().item())


def analyze_missed_sources(
    target: dict,
    final_predictions: list[dict],
    stage_predictions: dict[str, list[dict]],
    iou_match_threshold: float,
    diagnostic_iou: float = 0.30,
) -> dict:
    """Break missed GTs into proposal/class/postprocess causes."""
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    final_boxes, final_labels, _ = predictions_to_tensors(final_predictions)
    stage_meta = stage_predictions.get("_meta", {}) if isinstance(stage_predictions, dict) else {}
    topk_stage = str(stage_meta.get("topk_stage", "pre_class_nms"))
    pre_threshold_stage = stage_predictions.get("pre_threshold", [])
    post_threshold_stage = stage_predictions.get("post_threshold", pre_threshold_stage)
    post_topk_stage = stage_predictions.get("post_topk", post_threshold_stage)
    post_class_nms_stage = stage_predictions.get("post_class_nms", post_topk_stage)
    post_agnostic_nms_stage = stage_predictions.get("post_agnostic_nms", post_class_nms_stage)
    post_cross_class_clean_stage = stage_predictions.get("post_cross_class_clean", post_agnostic_nms_stage)
    final_stage = stage_predictions.get("final", post_cross_class_clean_stage)

    counts = {
        "missed_total_gt": 0,
        "missed_no_candidate": 0,
        "missed_wrong_class_from_start": 0,
        "missed_filtered_by_threshold": 0,
        "missed_filtered_by_topk": 0,
        "missed_filtered_by_class_nms": 0,
        "missed_filtered_by_agnostic_nms": 0,
        "missed_filtered_by_cross_class_clean": 0,
        "missed_localized_but_unmatched": 0,
    }

    if gt_boxes.numel() == 0:
        return counts

    if final_boxes.numel() == 0:
        matched_gt_mask = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
    else:
        final_iou = box_iou(final_boxes, gt_boxes)
        final_match = (final_iou >= iou_match_threshold) & (final_labels[:, None] == gt_labels[None, :])
        matched_gt_mask = final_match.any(dim=0)

    for gt_idx in range(gt_boxes.shape[0]):
        if matched_gt_mask[gt_idx]:
            continue

        gt_box = gt_boxes[gt_idx]
        gt_label = int(gt_labels[gt_idx].item())
        counts["missed_total_gt"] += 1

        if not _has_candidate(pre_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=False):
            counts["missed_no_candidate"] += 1
        elif not _has_candidate(pre_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_wrong_class_from_start"] += 1
        elif not _has_candidate(post_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_threshold"] += 1
        elif topk_stage == "pre_class_nms" and not _has_candidate(post_topk_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_topk"] += 1
        elif not _has_candidate(post_class_nms_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_class_nms"] += 1
        elif topk_stage == "post_class_nms" and not _has_candidate(post_topk_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_topk"] += 1
        elif topk_stage == "final" and not _has_candidate(post_agnostic_nms_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_agnostic_nms"] += 1
        elif topk_stage == "final" and not _has_candidate(post_cross_class_clean_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_cross_class_clean"] += 1
        elif topk_stage == "final" and not _has_candidate(final_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_topk"] += 1
        elif topk_stage != "final" and not _has_candidate(final_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
            counts["missed_filtered_by_agnostic_nms"] += 1
        else:
            counts["missed_localized_but_unmatched"] += 1

    return counts


def accumulate_confusion_matrix(
    target: dict,
    predictions: list[dict],
    iou_match_threshold: float,
    confusion_matrix: torch.Tensor,
    confusion_score_sum: torch.Tensor | None = None,
    confusion_iou_sum: torch.Tensor | None = None,
):
    """Accumulate one-to-one GT->prediction confusion pairs."""
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes, pred_labels, pred_scores = predictions_to_tensors(predictions)

    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    if iou_matrix.numel() == 0:
        return

    sorted_indices = torch.argsort(iou_matrix.reshape(-1), descending=True)
    used_pred_indices: set[int] = set()
    used_gt_indices: set[int] = set()
    num_gt = gt_boxes.shape[0]

    for flat_index in sorted_indices.tolist():
        pred_idx = flat_index // num_gt
        gt_idx = flat_index % num_gt
        if pred_idx in used_pred_indices or gt_idx in used_gt_indices:
            continue

        iou_value = float(iou_matrix[pred_idx, gt_idx].item())
        if iou_value < iou_match_threshold:
            break

        gt_label = int(gt_labels[gt_idx].item())
        pred_label = int(pred_labels[pred_idx].item())
        confusion_matrix[gt_label, pred_label] += 1
        if confusion_score_sum is not None:
            confusion_score_sum[gt_label, pred_label] += float(pred_scores[pred_idx].item())
        if confusion_iou_sum is not None:
            confusion_iou_sum[gt_label, pred_label] += iou_value

        used_pred_indices.add(pred_idx)
        used_gt_indices.add(gt_idx)


def _iou_bucket_name(iou_value: float) -> str:
    if iou_value >= 0.90:
        return "0.90-1.00"
    if iou_value >= 0.75:
        return "0.75-0.90"
    return "0.50-0.75"


def _init_iou_bucket_summary() -> dict:
    return {
        "0.50-0.75": {"matched_total": 0, "correct": 0, "wrong_class": 0},
        "0.75-0.90": {"matched_total": 0, "correct": 0, "wrong_class": 0},
        "0.90-1.00": {"matched_total": 0, "correct": 0, "wrong_class": 0},
    }


def accumulate_classification_iou_diagnostics(
    target: dict,
    predictions: list[dict],
    iou_match_threshold: float,
    iou_bucket_summary: dict,
    high_iou_confusion_matrix: torch.Tensor,
    high_iou_threshold: float = 0.75,
):
    """Track whether classification errors persist when matched boxes are already tight."""
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes, pred_labels, _ = predictions_to_tensors(predictions)

    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    if iou_matrix.numel() == 0:
        return

    sorted_indices = torch.argsort(iou_matrix.reshape(-1), descending=True)
    used_pred_indices: set[int] = set()
    used_gt_indices: set[int] = set()
    num_gt = gt_boxes.shape[0]

    for flat_index in sorted_indices.tolist():
        pred_idx = flat_index // num_gt
        gt_idx = flat_index % num_gt
        if pred_idx in used_pred_indices or gt_idx in used_gt_indices:
            continue

        iou_value = float(iou_matrix[pred_idx, gt_idx].item())
        if iou_value < iou_match_threshold:
            break

        gt_label = int(gt_labels[gt_idx].item())
        pred_label = int(pred_labels[pred_idx].item())
        bucket_name = _iou_bucket_name(iou_value)
        bucket_stats = iou_bucket_summary[bucket_name]
        bucket_stats["matched_total"] += 1
        if gt_label == pred_label:
            bucket_stats["correct"] += 1
        else:
            bucket_stats["wrong_class"] += 1
            if iou_value >= high_iou_threshold:
                high_iou_confusion_matrix[gt_label, pred_label] += 1

        used_pred_indices.add(pred_idx)
        used_gt_indices.add(gt_idx)


def finalize_iou_bucket_summary(iou_bucket_summary: dict) -> dict:
    finalized = {}
    for bucket_name, bucket_stats in iou_bucket_summary.items():
        matched_total = int(bucket_stats["matched_total"])
        correct = int(bucket_stats["correct"])
        wrong_class = int(bucket_stats["wrong_class"])
        finalized[bucket_name] = {
            "matched_total": matched_total,
            "correct": correct,
            "wrong_class": wrong_class,
            "correct_rate": _safe_divide(correct, matched_total),
            "wrong_class_rate": _safe_divide(wrong_class, matched_total),
        }
    return finalized


def extract_top_confusions(confusion_matrix: torch.Tensor, topk: int = 10) -> list[dict]:
    confusions = []
    num_classes = confusion_matrix.shape[0]
    for gt_label in range(num_classes):
        for pred_label in range(num_classes):
            if gt_label == pred_label:
                continue
            count = int(confusion_matrix[gt_label, pred_label].item())
            if count <= 0:
                continue
            confusions.append({"gt": gt_label, "pred": pred_label, "count": count})

    confusions.sort(key=lambda item: item["count"], reverse=True)
    return confusions[:topk]


def save_confusion_matrix_figure(confusion_matrix: torch.Tensor, save_path: str):
    counts = confusion_matrix.to(torch.float32)
    row_sum = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = counts / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    digit_labels = [str(idx) for idx in range(confusion_matrix.shape[0])]

    im0 = axes[0].imshow(counts.numpy(), cmap="Blues")
    axes[0].set_title("Digit Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_xticks(range(len(digit_labels)))
    axes[0].set_yticks(range(len(digit_labels)))
    axes[0].set_xticklabels(digit_labels)
    axes[0].set_yticklabels(digit_labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(normalized.numpy(), cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title("Digit Confusion Matrix (Row Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground Truth")
    axes[1].set_xticks(range(len(digit_labels)))
    axes[1].set_yticks(range(len(digit_labels)))
    axes[1].set_xticklabels(digit_labels)
    axes[1].set_yticklabels(digit_labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax, values, formatter in [
        (axes[0], counts, lambda value: f"{int(value)}"),
        (axes[1], normalized, lambda value: f"{value:.2f}"),
    ]:
        max_value = float(values.max().item()) if values.numel() > 0 else 0.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = float(values[row, col].item())
                if row == col and value == 0.0:
                    continue
                if row != col and value <= 0.0:
                    continue
                text_color = "white" if value > max_value * 0.5 and max_value > 0 else "black"
                ax.text(col, row, formatter(value), ha="center", va="center", fontsize=8, color=text_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def merge_count_dict(total: dict[str, int], partial: dict[str, int]) -> None:
    for key, value in partial.items():
        total[key] = total.get(key, 0) + int(value)


def _compute_box_gap(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1_a, y1_a, x2_a, y2_a = [float(value) for value in box_a.tolist()]
    x1_b, y1_b, x2_b, y2_b = [float(value) for value in box_b.tolist()]
    gap_x = max(x1_b - x2_a, x1_a - x2_b, 0.0)
    gap_y = max(y1_b - y2_a, y1_a - y2_b, 0.0)
    return math.hypot(gap_x, gap_y)


def compute_gt_crowd_slices(
    target: dict,
    crowded_gap_ratio_threshold: float = 0.75,
) -> list[dict]:
    gt_boxes = target["boxes"].cpu()
    num_gt = int(gt_boxes.shape[0])
    results: list[dict] = []

    for gt_idx in range(num_gt):
        gt_box = gt_boxes[gt_idx]
        width = max(1.0, float(gt_box[2].item() - gt_box[0].item()))
        height = max(1.0, float(gt_box[3].item() - gt_box[1].item()))
        scale = max(1.0, math.sqrt(width * height))

        if num_gt <= 1:
            results.append(
                {
                    "slice": "singleton",
                    "nearest_gap_px": None,
                    "nearest_gap_ratio": None,
                    "box_scale": scale,
                }
            )
            continue

        nearest_gap_px = None
        for other_idx in range(num_gt):
            if other_idx == gt_idx:
                continue
            gap_px = _compute_box_gap(gt_box, gt_boxes[other_idx])
            if nearest_gap_px is None or gap_px < nearest_gap_px:
                nearest_gap_px = gap_px

        nearest_gap_ratio = float(nearest_gap_px / scale) if nearest_gap_px is not None else None
        slice_name = "crowded" if nearest_gap_ratio is not None and nearest_gap_ratio <= float(crowded_gap_ratio_threshold) else "isolated"
        results.append(
            {
                "slice": slice_name,
                "nearest_gap_px": float(nearest_gap_px) if nearest_gap_px is not None else None,
                "nearest_gap_ratio": nearest_gap_ratio,
                "box_scale": scale,
            }
        )

    return results


def analyze_gt_outcomes_v3(
    target: dict,
    predictions: list[dict],
    iou_match_threshold: float,
    loose_iou_threshold: float = 0.10,
) -> list[dict]:
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes, pred_labels, pred_scores = predictions_to_tensors(predictions)

    num_gt = int(gt_boxes.shape[0])
    outcomes: list[dict] = []
    if num_gt == 0:
        return outcomes

    if pred_boxes.numel() == 0:
        for gt_idx in range(num_gt):
            outcomes.append(
                {
                    "gt_index": gt_idx,
                    "label": int(gt_labels[gt_idx].item()),
                    "outcome": "missed",
                    "matched_correct": False,
                    "best_any_iou": 0.0,
                    "best_same_label_iou": 0.0,
                    "best_wrong_class_iou": 0.0,
                    "best_pred_score": 0.0,
                    "best_same_label_score": 0.0,
                }
            )
        return outcomes

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    same_label_matrix = pred_labels[:, None] == gt_labels[None, :]
    true_positive_matches = _greedy_match_pairs(iou_matrix, iou_match_threshold, same_label_matrix)
    matched_gt_indices = {gt_idx for _, gt_idx, _ in true_positive_matches}

    for gt_idx in range(num_gt):
        gt_label = int(gt_labels[gt_idx].item())
        gt_ious = iou_matrix[:, gt_idx]
        best_any_iou, best_pred_idx = gt_ious.max(dim=0)
        best_pred_score = float(pred_scores[best_pred_idx].item())

        same_label_mask = pred_labels == gt_label
        if bool(same_label_mask.any().item()):
            same_label_ious = gt_ious[same_label_mask]
            same_label_scores = pred_scores[same_label_mask]
            best_same_label_iou = float(same_label_ious.max().item())
            best_same_label_score = float(same_label_scores[same_label_ious.argmax()].item())
        else:
            best_same_label_iou = 0.0
            best_same_label_score = 0.0

        wrong_class_mask = pred_labels != gt_label
        if bool(wrong_class_mask.any().item()):
            best_wrong_class_iou = float(gt_ious[wrong_class_mask].max().item())
        else:
            best_wrong_class_iou = 0.0

        if gt_idx in matched_gt_indices:
            outcome = "matched_correct"
        elif best_wrong_class_iou >= float(iou_match_threshold):
            outcome = "wrong_class"
        elif best_same_label_iou >= float(loose_iou_threshold):
            outcome = "loc_error"
        else:
            outcome = "missed"

        outcomes.append(
            {
                "gt_index": gt_idx,
                "label": gt_label,
                "outcome": outcome,
                "matched_correct": outcome == "matched_correct",
                "best_any_iou": float(best_any_iou.item()),
                "best_same_label_iou": best_same_label_iou,
                "best_wrong_class_iou": best_wrong_class_iou,
                "best_pred_score": best_pred_score,
                "best_same_label_score": best_same_label_score,
            }
        )

    return outcomes


def _classify_missed_source_single_gt(
    gt_box: torch.Tensor,
    gt_label: int,
    stage_predictions: dict[str, list[dict]],
    diagnostic_iou: float,
) -> str:
    stage_meta = stage_predictions.get("_meta", {}) if isinstance(stage_predictions, dict) else {}
    topk_stage = str(stage_meta.get("topk_stage", "pre_class_nms"))
    pre_threshold_stage = stage_predictions.get("pre_threshold", [])
    post_threshold_stage = stage_predictions.get("post_threshold", pre_threshold_stage)
    post_topk_stage = stage_predictions.get("post_topk", post_threshold_stage)
    post_class_nms_stage = stage_predictions.get("post_class_nms", post_topk_stage)
    post_agnostic_nms_stage = stage_predictions.get("post_agnostic_nms", post_class_nms_stage)
    post_cross_class_clean_stage = stage_predictions.get("post_cross_class_clean", post_agnostic_nms_stage)
    final_stage = stage_predictions.get("final", post_cross_class_clean_stage)

    if not _has_candidate(pre_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=False):
        return "missed_no_candidate"
    if not _has_candidate(pre_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_wrong_class_from_start"
    if not _has_candidate(post_threshold_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_threshold"
    if topk_stage == "pre_class_nms" and not _has_candidate(post_topk_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_topk"
    if not _has_candidate(post_class_nms_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_class_nms"
    if topk_stage == "post_class_nms" and not _has_candidate(post_topk_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_topk"
    if topk_stage == "final" and not _has_candidate(post_agnostic_nms_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_agnostic_nms"
    if topk_stage == "final" and not _has_candidate(post_cross_class_clean_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_cross_class_clean"
    if topk_stage == "final" and not _has_candidate(final_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_topk"
    if topk_stage != "final" and not _has_candidate(final_stage, gt_box, gt_label, diagnostic_iou, require_same_label=True):
        return "missed_filtered_by_agnostic_nms"
    return "missed_localized_but_unmatched"


def summarize_v2_records(analysis_records: list[dict]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    legacy_summary: dict[str, int] = {}
    image_tag_summary: dict[str, int] = {}
    gt_summary: dict[str, int] = {}
    pred_summary: dict[str, int] = {}

    for record in analysis_records:
        legacy_type = record.get("legacy_sample_type", record["sample_type"])
        legacy_summary[legacy_type] = legacy_summary.get(legacy_type, 0) + 1

        if record.get("is_clean", False):
            image_tag_summary["correct"] = image_tag_summary.get("correct", 0) + 1
        elif record.get("sample_tags"):
            for tag in record["sample_tags"]:
                image_tag_summary[tag] = image_tag_summary.get(tag, 0) + 1
        else:
            image_tag_summary["uncategorized"] = image_tag_summary.get("uncategorized", 0) + 1

        details = record.get("details", {})
        gt_stats = details.get("gt_stats")
        pred_stats = details.get("pred_stats")
        if gt_stats:
            merge_count_dict(gt_summary, gt_stats)
        if pred_stats:
            merge_count_dict(pred_summary, pred_stats)

    return legacy_summary, image_tag_summary, gt_summary, pred_summary


def _init_slice_summary() -> dict[str, dict[str, int | float]]:
    return {
        slice_name: {
            "gt_total": 0,
            "gt_matched_correct": 0,
            "gt_wrong_class": 0,
            "gt_loc_error": 0,
            "gt_missed": 0,
            "avg_nearest_gap_px": 0.0,
            "avg_nearest_gap_ratio": 0.0,
            "_gap_count": 0,
        }
        for slice_name in ["crowded", "isolated", "singleton"]
    }


def finalize_slice_summary(slice_summary: dict[str, dict[str, int | float]]) -> dict[str, dict[str, int | float]]:
    finalized: dict[str, dict[str, int | float]] = {}
    for slice_name, item in slice_summary.items():
        gt_total = int(item["gt_total"])
        matched_correct = int(item["gt_matched_correct"])
        wrong_class = int(item["gt_wrong_class"])
        loc_error = int(item["gt_loc_error"])
        missed = int(item["gt_missed"])
        gap_count = int(item["_gap_count"])
        finalized[slice_name] = {
            "gt_total": gt_total,
            "gt_matched_correct": matched_correct,
            "gt_wrong_class": wrong_class,
            "gt_loc_error": loc_error,
            "gt_missed": missed,
            "gt_correct_rate": _safe_divide(matched_correct, gt_total),
            "gt_wrong_class_rate": _safe_divide(wrong_class, gt_total),
            "gt_loc_error_rate": _safe_divide(loc_error, gt_total),
            "gt_missed_rate": _safe_divide(missed, gt_total),
            "avg_nearest_gap_px": (_safe_divide(item["avg_nearest_gap_px"], gap_count) if gap_count > 0 else None),
            "avg_nearest_gap_ratio": (_safe_divide(item["avg_nearest_gap_ratio"], gap_count) if gap_count > 0 else None),
        }
    return finalized


def build_query_coverage_points(max_queries_hint: int) -> list[int]:
    default_points = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]
    points = {value for value in default_points if value <= max(1, int(max_queries_hint))}
    points.add(1)
    points.add(max(1, int(max_queries_hint)))
    return sorted(points)


def init_query_coverage_summary(curve_points: list[int]) -> dict:
    return {
        "curve_points": {
            int(k): {
                "gt_total": 0,
                "any_candidate_iou_0_30": 0,
                "same_label_candidate_iou_0_30": 0,
                "same_label_match_iou_0_50": 0,
            }
            for k in curve_points
        },
        "max_predictions_seen": 0,
        "source_stage": "pre_threshold",
    }


def accumulate_query_coverage_summary(
    summary: dict,
    target: dict,
    stage_predictions: dict[str, list[dict]],
    curve_points: list[int],
    iou_match_threshold: float,
    diagnostic_iou: float = 0.30,
) -> None:
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    if gt_boxes.numel() == 0:
        return

    raw_predictions = list(stage_predictions.get("pre_threshold", stage_predictions.get("final", [])))
    raw_predictions = sorted(raw_predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    summary["max_predictions_seen"] = max(int(summary["max_predictions_seen"]), len(raw_predictions))

    for k in curve_points:
        subset = raw_predictions[: min(int(k), len(raw_predictions))]
        curve_item = summary["curve_points"][int(k)]
        curve_item["gt_total"] += int(gt_boxes.shape[0])
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            label_int = int(gt_label.item())
            if _has_candidate(subset, gt_box, label_int, diagnostic_iou, require_same_label=False):
                curve_item["any_candidate_iou_0_30"] += 1
            if _has_candidate(subset, gt_box, label_int, diagnostic_iou, require_same_label=True):
                curve_item["same_label_candidate_iou_0_30"] += 1
            if _has_candidate(subset, gt_box, label_int, iou_match_threshold, require_same_label=True):
                curve_item["same_label_match_iou_0_50"] += 1


def finalize_query_coverage_summary(summary: dict) -> dict:
    curve_rows = []
    match_rates = []
    candidate_rates = []

    for k in sorted(summary["curve_points"].keys()):
        item = summary["curve_points"][k]
        gt_total = int(item["gt_total"])
        row = {
            "k": int(k),
            "gt_total": gt_total,
            "any_candidate_rate_iou_0_30": _safe_divide(item["any_candidate_iou_0_30"], gt_total),
            "same_label_candidate_rate_iou_0_30": _safe_divide(item["same_label_candidate_iou_0_30"], gt_total),
            "same_label_match_rate_iou_0_50": _safe_divide(item["same_label_match_iou_0_50"], gt_total),
        }
        curve_rows.append(row)
        candidate_rates.append(float(row["same_label_candidate_rate_iou_0_30"]))
        match_rates.append(float(row["same_label_match_rate_iou_0_50"]))

    max_candidate_rate = max(candidate_rates, default=0.0)
    max_match_rate = max(match_rates, default=0.0)
    saturation_candidate_k = None
    saturation_match_k = None
    for row in curve_rows:
        if saturation_candidate_k is None and max_candidate_rate > 0.0 and float(row["same_label_candidate_rate_iou_0_30"]) >= 0.95 * max_candidate_rate:
            saturation_candidate_k = int(row["k"])
        if saturation_match_k is None and max_match_rate > 0.0 and float(row["same_label_match_rate_iou_0_50"]) >= 0.95 * max_match_rate:
            saturation_match_k = int(row["k"])

    return {
        "source_stage": summary.get("source_stage", "pre_threshold"),
        "max_predictions_seen": int(summary.get("max_predictions_seen", 0)),
        "curve_points": curve_rows,
        "saturation_k_95_same_label_candidate": saturation_candidate_k,
        "saturation_k_95_same_label_match": saturation_match_k,
    }


def save_query_coverage_figure(query_coverage_summary: dict, save_path: str) -> None:
    curve_points = query_coverage_summary.get("curve_points", [])
    if not curve_points:
        return

    ks = [int(item["k"]) for item in curve_points]
    any_candidate = [float(item["any_candidate_rate_iou_0_30"]) for item in curve_points]
    same_label_candidate = [float(item["same_label_candidate_rate_iou_0_30"]) for item in curve_points]
    same_label_match = [float(item["same_label_match_rate_iou_0_50"]) for item in curve_points]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ks, any_candidate, marker="o", label="Any candidate @ IoU>=0.30")
    ax.plot(ks, same_label_candidate, marker="o", label="Same-label candidate @ IoU>=0.30")
    ax.plot(ks, same_label_match, marker="o", label="Same-label match @ IoU>=0.50")
    ax.set_xlabel("Top-K Raw Queries")
    ax.set_ylabel("GT Recall Rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Query Coverage Curve (V3)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def init_oracle_summary() -> dict[str, int]:
    return {
        "gt_total": 0,
        "final_correct_gt": 0,
        "oracle_class_recoverable_gt": 0,
        "oracle_box_recoverable_gt": 0,
        "oracle_postprocess_recoverable_gt": 0,
        "oracle_no_candidate_gt": 0,
        "oracle_combined_recoverable_gt": 0,
    }


def finalize_oracle_summary(summary: dict[str, int]) -> dict[str, int | float]:
    gt_total = int(summary["gt_total"])
    final_correct_gt = int(summary["final_correct_gt"])
    oracle_combined_recoverable_gt = int(summary["oracle_combined_recoverable_gt"])
    return {
        **{key: int(value) for key, value in summary.items()},
        "final_correct_rate": _safe_divide(final_correct_gt, gt_total),
        "oracle_class_recoverable_rate": _safe_divide(summary["oracle_class_recoverable_gt"], gt_total),
        "oracle_box_recoverable_rate": _safe_divide(summary["oracle_box_recoverable_gt"], gt_total),
        "oracle_postprocess_recoverable_rate": _safe_divide(summary["oracle_postprocess_recoverable_gt"], gt_total),
        "oracle_no_candidate_rate": _safe_divide(summary["oracle_no_candidate_gt"], gt_total),
        "oracle_combined_upper_bound_rate": _safe_divide(final_correct_gt + oracle_combined_recoverable_gt, gt_total),
    }


def build_postprocess_gap_report(missed_reason_summary: dict, gt_summary: dict) -> dict[str, int | float]:
    missed_total_gt = int(missed_reason_summary.get("missed_total_gt", 0))
    postprocess_filtered_total = int(
        missed_reason_summary.get("missed_filtered_by_threshold", 0)
        + missed_reason_summary.get("missed_filtered_by_topk", 0)
        + missed_reason_summary.get("missed_filtered_by_class_nms", 0)
        + missed_reason_summary.get("missed_filtered_by_agnostic_nms", 0)
        + missed_reason_summary.get("missed_filtered_by_cross_class_clean", 0)
    )
    gt_total = int(gt_summary.get("gt_total", 0))
    gt_correct = int(gt_summary.get("gt_matched_correct", 0))
    return {
        "gt_total": gt_total,
        "gt_matched_correct": gt_correct,
        "gt_correct_rate": _safe_divide(gt_correct, gt_total),
        "missed_total_gt": missed_total_gt,
        "missed_no_candidate": int(missed_reason_summary.get("missed_no_candidate", 0)),
        "missed_wrong_class_from_start": int(missed_reason_summary.get("missed_wrong_class_from_start", 0)),
        "missed_filtered_by_threshold": int(missed_reason_summary.get("missed_filtered_by_threshold", 0)),
        "missed_filtered_by_topk": int(missed_reason_summary.get("missed_filtered_by_topk", 0)),
        "missed_filtered_by_class_nms": int(missed_reason_summary.get("missed_filtered_by_class_nms", 0)),
        "missed_filtered_by_agnostic_nms": int(missed_reason_summary.get("missed_filtered_by_agnostic_nms", 0)),
        "missed_filtered_by_cross_class_clean": int(missed_reason_summary.get("missed_filtered_by_cross_class_clean", 0)),
        "missed_localized_but_unmatched": int(missed_reason_summary.get("missed_localized_but_unmatched", 0)),
        "postprocess_filtered_total": postprocess_filtered_total,
        "postprocess_filtered_rate_over_missed": _safe_divide(postprocess_filtered_total, missed_total_gt),
        "threshold_share_of_postprocess_gap": _safe_divide(missed_reason_summary.get("missed_filtered_by_threshold", 0), postprocess_filtered_total),
        "topk_share_of_postprocess_gap": _safe_divide(missed_reason_summary.get("missed_filtered_by_topk", 0), postprocess_filtered_total),
        "class_nms_share_of_postprocess_gap": _safe_divide(missed_reason_summary.get("missed_filtered_by_class_nms", 0), postprocess_filtered_total),
        "agnostic_nms_share_of_postprocess_gap": _safe_divide(missed_reason_summary.get("missed_filtered_by_agnostic_nms", 0), postprocess_filtered_total),
        "cross_class_clean_share_of_postprocess_gap": _safe_divide(missed_reason_summary.get("missed_filtered_by_cross_class_clean", 0), postprocess_filtered_total),
    }


def _init_digit_summary(num_classes: int, keys: list[str]) -> dict[str, dict[str, int | float]]:
    return {str(digit): {key: 0 for key in keys} for digit in range(num_classes)}


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0


def summarize_per_digit_records(
    analysis_records: list[dict],
    num_classes: int,
) -> tuple[dict[str, dict[str, int | float]], dict[str, dict[str, int | float]], dict[str, object]]:
    gt_keys = ["gt_total", "gt_matched_correct", "gt_wrong_class", "gt_loc_error", "gt_missed"]
    pred_keys = ["pred_total", "pred_true_positive", "pred_duplicate", "pred_grouped", "pred_wrong_class", "pred_loc_error", "pred_background_fp"]
    gt_digit_summary = _init_digit_summary(num_classes, gt_keys)
    pred_digit_summary = _init_digit_summary(num_classes, pred_keys)

    for record in analysis_records:
        details = record.get("details", {})
        for digit_key, digit_stats in details.get("gt_digit_stats", {}).items():
            for stat_key, stat_value in digit_stats.items():
                gt_digit_summary[digit_key][stat_key] = int(gt_digit_summary[digit_key].get(stat_key, 0)) + int(stat_value)
        for digit_key, digit_stats in details.get("pred_digit_stats", {}).items():
            for stat_key, stat_value in digit_stats.items():
                pred_digit_summary[digit_key][stat_key] = int(pred_digit_summary[digit_key].get(stat_key, 0)) + int(stat_value)

    for digit in range(num_classes):
        digit_key = str(digit)
        gt_total = int(gt_digit_summary[digit_key]["gt_total"])
        gt_digit_summary[digit_key]["gt_correct_rate"] = _safe_divide(gt_digit_summary[digit_key]["gt_matched_correct"], gt_total)
        gt_digit_summary[digit_key]["gt_wrong_class_rate"] = _safe_divide(gt_digit_summary[digit_key]["gt_wrong_class"], gt_total)
        gt_digit_summary[digit_key]["gt_loc_error_rate"] = _safe_divide(gt_digit_summary[digit_key]["gt_loc_error"], gt_total)
        gt_digit_summary[digit_key]["gt_missed_rate"] = _safe_divide(gt_digit_summary[digit_key]["gt_missed"], gt_total)
        gt_digit_summary[digit_key]["gt_error_rate"] = _safe_divide(
            int(gt_digit_summary[digit_key]["gt_wrong_class"]) + int(gt_digit_summary[digit_key]["gt_loc_error"]) + int(gt_digit_summary[digit_key]["gt_missed"]),
            gt_total,
        )

        pred_total = int(pred_digit_summary[digit_key]["pred_total"])
        pred_digit_summary[digit_key]["pred_true_positive_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_true_positive"], pred_total)
        pred_digit_summary[digit_key]["pred_duplicate_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_duplicate"], pred_total)
        pred_digit_summary[digit_key]["pred_grouped_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_grouped"], pred_total)
        pred_digit_summary[digit_key]["pred_wrong_class_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_wrong_class"], pred_total)
        pred_digit_summary[digit_key]["pred_loc_error_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_loc_error"], pred_total)
        pred_digit_summary[digit_key]["pred_background_fp_rate"] = _safe_divide(pred_digit_summary[digit_key]["pred_background_fp"], pred_total)

    ranked_digits = sorted(range(num_classes), key=lambda digit: int(gt_digit_summary[str(digit)]["gt_total"]), reverse=True)
    group_size = max(1, min(3, num_classes // 3 if num_classes >= 3 else num_classes))
    head_digits = ranked_digits[:group_size]
    tail_digits = list(reversed(ranked_digits[-group_size:]))
    head_mean_correct_rate = sum(float(gt_digit_summary[str(digit)]["gt_correct_rate"]) for digit in head_digits) / max(1, len(head_digits))
    tail_mean_correct_rate = sum(float(gt_digit_summary[str(digit)]["gt_correct_rate"]) for digit in tail_digits) / max(1, len(tail_digits))
    head_mean_error_rate = sum(float(gt_digit_summary[str(digit)]["gt_error_rate"]) for digit in head_digits) / max(1, len(head_digits))
    tail_mean_error_rate = sum(float(gt_digit_summary[str(digit)]["gt_error_rate"]) for digit in tail_digits) / max(1, len(tail_digits))

    long_tail_summary = {
        "head_digits_by_gt_frequency": head_digits,
        "tail_digits_by_gt_frequency": tail_digits,
        "head_mean_correct_rate": head_mean_correct_rate,
        "tail_mean_correct_rate": tail_mean_correct_rate,
        "head_mean_error_rate": head_mean_error_rate,
        "tail_mean_error_rate": tail_mean_error_rate,
        "tail_minus_head_error_rate_gap": tail_mean_error_rate - head_mean_error_rate,
        "tail_minus_head_correct_rate_gap": tail_mean_correct_rate - head_mean_correct_rate,
    }

    return gt_digit_summary, pred_digit_summary, long_tail_summary


def save_per_digit_error_figure(
    gt_digit_summary: dict[str, dict[str, int | float]],
    pred_digit_summary: dict[str, dict[str, int | float]],
    save_path: str,
) -> None:
    gt_rate_keys = ["gt_correct_rate", "gt_wrong_class_rate", "gt_loc_error_rate", "gt_missed_rate"]
    pred_rate_keys = ["pred_true_positive_rate", "pred_duplicate_rate", "pred_grouped_rate", "pred_wrong_class_rate", "pred_loc_error_rate", "pred_background_fp_rate"]
    gt_labels = ["correct", "wrong_cls", "loc", "missed"]
    pred_labels = ["tp", "dup", "grouped", "wrong_cls", "loc", "bg_fp"]
    digit_labels = sorted(gt_digit_summary.keys(), key=lambda item: int(item))

    gt_matrix = torch.tensor([[float(gt_digit_summary[digit][key]) for key in gt_rate_keys] for digit in digit_labels], dtype=torch.float32)
    pred_matrix = torch.tensor([[float(pred_digit_summary[digit][key]) for key in pred_rate_keys] for digit in digit_labels], dtype=torch.float32)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, values, column_labels, title in [
        (axes[0], gt_matrix, gt_labels, "Per-Digit GT-Centric Outcome Rates"),
        (axes[1], pred_matrix, pred_labels, "Per-Digit Prediction-Centric Outcome Rates"),
    ]:
        im = ax.imshow(values.numpy(), cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Digit")
        ax.set_xticks(range(len(column_labels)))
        ax.set_yticks(range(len(digit_labels)))
        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(digit_labels)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = float(values[row, col].item())
                text_color = "white" if value >= 0.50 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_sample_label(record: dict) -> str:
    if record.get("is_clean", False):
        return "correct | tags=clean"

    tag_text = ",".join(record.get("sample_tags", [])) if record.get("sample_tags") else "none"
    legacy_type = record.get("legacy_sample_type")
    if legacy_type is not None:
        return f"{record['sample_type']} | tags={tag_text} | legacy={legacy_type}"
    return f"{record['sample_type']} | tags={tag_text}"


def clone_target_to_orig_geometry(target: dict) -> dict:
    """Clone a target and express boxes in original-image coordinates."""
    cloned = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in target.items()}
    if "boxes" not in cloned or "size" not in cloned or "orig_size" not in cloned:
        return cloned

    boxes = cloned["boxes"]
    size_tensor = cloned["size"].to(dtype=torch.float32)
    orig_size_tensor = cloned["orig_size"].to(dtype=torch.float32)
    current_height = max(float(size_tensor[0].item()), 1.0)
    current_width = max(float(size_tensor[1].item()), 1.0)
    orig_height = max(float(orig_size_tensor[0].item()), 1.0)
    orig_width = max(float(orig_size_tensor[1].item()), 1.0)

    if (
        torch.is_tensor(boxes)
        and boxes.numel() > 0
        and (abs(current_height - orig_height) > 1e-6 or abs(current_width - orig_width) > 1e-6)
    ):
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= orig_width / current_width
        boxes[:, [1, 3]] *= orig_height / current_height
        cloned["boxes"] = boxes
        if "area" in cloned and torch.is_tensor(cloned["area"]):
            cloned["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    cloned["size"] = cloned["orig_size"].clone()
    return cloned


def draw_gt_boxes(ax, target: dict):
    """Draw GT boxes in green."""
    boxes = target["boxes"].cpu()
    labels = target["labels"].cpu()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 3),
            f"GT:{int(label.item())}",
            color="lime",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )


def draw_pred_boxes(ax, predictions: list[dict]):
    """Draw predicted boxes in red."""
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            max(0, y - 3),
            f"P:{int(pred['category_id']) - 1} {float(pred['score']):.2f}",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )


def save_single_visualization(image_tensor, target, image_predictions, sample_type: str, save_path: str):
    """Save one side-by-side GT / Prediction figure."""
    image_np = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    draw_gt_boxes(axes[0], target)

    axes[1].imshow(image_np)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    draw_pred_boxes(axes[1], image_predictions)

    image_id = int(target["image_id"].item())
    file_name = target.get("file_name", "")
    fig.suptitle(f"{sample_type} | image_id={image_id} | {file_name}", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_only_visualization(image_tensor, target, image_predictions, sample_type: str, save_path: str):
    """Save one side-by-side image / prediction figure for unlabeled test samples."""
    image_np = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    draw_pred_boxes(axes[1], image_predictions)

    image_id = int(target["image_id"].item())
    file_name = target.get("file_name", "")
    fig.suptitle(f"{sample_type} | image_id={image_id} | {file_name}", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(saved_paths: list[str], output_path: str, cols: int = 4, thumb_size=(320, 160)):
    """Save a contact sheet from saved visualization images."""
    if not saved_paths:
        return

    images = []
    for path in saved_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize(thumb_size)
        images.append(image)

    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), color=(255, 255, 255))

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        sheet.paste(image, (col * thumb_size[0], row * thumb_size[1]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)


def select_samples(analysis_records: list[dict], args) -> list[dict]:
    """Select a balanced mix of clean and error-tagged samples."""
    rng = random.Random(args.seed)
    buckets = {
        "correct": [],
        "duplicate": [],
        "cls_error": [],
        "loc_error": [],
        "missed": [],
        "grouped": [],
        "background_fp": [],
    }
    for record in analysis_records:
        if record.get("is_clean", False):
            buckets["correct"].append(record)
        for tag in record.get("sample_tags", []):
            if tag in buckets:
                buckets[tag].append(record)
        if not record.get("is_clean", False) and not record.get("sample_tags"):
            buckets["loc_error"].append(record)

    for records in buckets.values():
        rng.shuffle(records)

    requested = {
        "correct": args.num_correct,
        "duplicate": args.num_duplicate,
        "cls_error": args.num_cls_error,
        "loc_error": args.num_loc_error,
        "missed": args.num_missed,
        "grouped": args.num_grouped,
    }

    selected = []
    used_ids = set()
    for sample_type in ["correct", "duplicate", "cls_error", "loc_error", "missed", "grouped"]:
        for record in buckets[sample_type][:requested[sample_type]]:
            selected.append(record)
            used_ids.add(record["dataset_index"])

    if len(selected) < args.num_total:
        for sample_type in ["duplicate", "cls_error", "loc_error", "missed", "grouped", "correct"]:
            for record in buckets[sample_type]:
                if record["dataset_index"] in used_ids:
                    continue
                selected.append(record)
                used_ids.add(record["dataset_index"])
                if len(selected) >= args.num_total:
                    break
            if len(selected) >= args.num_total:
                break

    if len(selected) < args.num_total:
        for record in buckets["background_fp"]:
            if record["dataset_index"] in used_ids:
                continue
            selected.append(record)
            used_ids.add(record["dataset_index"])
            if len(selected) >= args.num_total:
                break

    return selected[:args.num_total]


def select_test_samples(analysis_records: list[dict], args) -> list[dict]:
    """Select test samples with priority on images that have predictions."""
    rng = random.Random(args.seed)
    with_predictions = [record for record in analysis_records if record["sample_type"] == "has_pred"]
    without_predictions = [record for record in analysis_records if record["sample_type"] == "no_pred"]
    rng.shuffle(with_predictions)
    rng.shuffle(without_predictions)

    selected = with_predictions[:args.num_total]
    if len(selected) < args.num_total:
        selected.extend(without_predictions[: args.num_total - len(selected)])
    return selected[:args.num_total]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DETR predictions on validation or test set")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prediction_json", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_total", type=int, default=100)
    parser.add_argument("--num_correct", type=int, default=20)
    parser.add_argument("--num_duplicate", type=int, default=20)
    parser.add_argument("--num_cls_error", type=int, default=20)
    parser.add_argument("--num_loc_error", type=int, default=15)
    parser.add_argument("--num_missed", type=int, default=15)
    parser.add_argument("--num_grouped", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou_match_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--class_score_thresholds", type=str, default=None)
    parser.add_argument("--topk_per_image", type=int, default=None)
    parser.add_argument("--postprocess_topk_stage", type=str, default=None)
    parser.add_argument("--use_nms", type=str, default=None)
    parser.add_argument("--nms_iou_threshold", type=float, default=None)
    parser.add_argument("--use_class_containment_suppression", type=str, default=None)
    parser.add_argument("--class_containment_threshold", type=float, default=None)
    parser.add_argument("--use_agnostic_nms", type=str, default=None)
    parser.add_argument("--agnostic_nms_iou_threshold", type=float, default=None)
    parser.add_argument("--agnostic_containment_threshold", type=float, default=None)
    parser.add_argument("--aux_digit_classifier_fusion_weight", type=float, default=None)
    parser.add_argument("--use_aux_digit_classifier_gated_fusion", type=str, default=None)
    parser.add_argument("--aux_digit_gate_top1_threshold", type=float, default=None)
    parser.add_argument("--aux_digit_gate_margin_threshold", type=float, default=None)
    parser.add_argument("--use_aux_digit_confusion_family_selective_fusion", type=str, default=None)
    parser.add_argument("--use_aux_digit_confusion_family_attenuation", type=str, default=None)
    parser.add_argument("--aux_digit_confusion_families", type=str, default=None)
    parser.add_argument("--aux_digit_family_fusion_weights", type=str, default=None)
    parser.add_argument("--aux_digit_family_attenuation_weights", type=str, default=None)
    parser.add_argument("--missed_diag_iou", type=float, default=0.30)
    parser.add_argument("--crowded_gap_ratio_threshold", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    if args.aux_digit_classifier_fusion_weight is not None:
        config["aux_digit_classifier_fusion_weight"] = float(args.aux_digit_classifier_fusion_weight)
    if args.use_aux_digit_classifier_gated_fusion is not None:
        config["use_aux_digit_classifier_gated_fusion"] = (
            str(args.use_aux_digit_classifier_gated_fusion).lower() in {"1", "true", "yes", "y"}
        )
    if args.aux_digit_gate_top1_threshold is not None:
        config["aux_digit_gate_top1_threshold"] = float(args.aux_digit_gate_top1_threshold)
    if args.aux_digit_gate_margin_threshold is not None:
        config["aux_digit_gate_margin_threshold"] = float(args.aux_digit_gate_margin_threshold)
    if args.use_aux_digit_confusion_family_selective_fusion is not None:
        config["use_aux_digit_confusion_family_selective_fusion"] = (
            str(args.use_aux_digit_confusion_family_selective_fusion).lower() in {"1", "true", "yes", "y"}
        )
    if args.use_aux_digit_confusion_family_attenuation is not None:
        config["use_aux_digit_confusion_family_attenuation"] = (
            str(args.use_aux_digit_confusion_family_attenuation).lower() in {"1", "true", "yes", "y"}
        )
    if args.aux_digit_confusion_families is not None:
        config["aux_digit_confusion_families"] = json.loads(args.aux_digit_confusion_families)
    if args.aux_digit_family_fusion_weights is not None:
        config["aux_digit_family_fusion_weights"] = json.loads(args.aux_digit_family_fusion_weights)
    if args.aux_digit_family_attenuation_weights is not None:
        config["aux_digit_family_attenuation_weights"] = json.loads(args.aux_digit_family_attenuation_weights)
    inference_postprocess_kwargs = build_inference_postprocess_kwargs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]
    prediction_json_path = args.prediction_json
    if args.score_threshold is not None:
        inference_postprocess_kwargs["score_threshold"] = float(args.score_threshold)
    if args.class_score_thresholds is not None:
        inference_postprocess_kwargs["class_score_thresholds"] = json.loads(
            args.class_score_thresholds
        )
    if args.topk_per_image is not None:
        inference_postprocess_kwargs["topk_per_image"] = int(args.topk_per_image)
    if args.postprocess_topk_stage is not None:
        inference_postprocess_kwargs["postprocess_topk_stage"] = str(args.postprocess_topk_stage)
    if args.use_nms is not None:
        inference_postprocess_kwargs["use_nms"] = args.use_nms.lower() in {"1", "true", "yes", "y"}
    if args.nms_iou_threshold is not None:
        inference_postprocess_kwargs["nms_iou_threshold"] = float(args.nms_iou_threshold)
    if args.use_class_containment_suppression is not None:
        inference_postprocess_kwargs["use_class_containment_suppression"] = (
            args.use_class_containment_suppression.lower() in {"1", "true", "yes", "y"}
        )
    if args.class_containment_threshold is not None:
        inference_postprocess_kwargs["class_containment_threshold"] = float(args.class_containment_threshold)
    if args.use_agnostic_nms is not None:
        inference_postprocess_kwargs["use_agnostic_nms"] = args.use_agnostic_nms.lower() in {"1", "true", "yes", "y"}
    if args.agnostic_nms_iou_threshold is not None:
        inference_postprocess_kwargs["agnostic_nms_iou_threshold"] = float(args.agnostic_nms_iou_threshold)
    if args.agnostic_containment_threshold is not None:
        inference_postprocess_kwargs["agnostic_containment_threshold"] = float(args.agnostic_containment_threshold)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "./Plot/val_visualization" if args.split == "val" else "./Plot/test_visualization"
    os.makedirs(output_dir, exist_ok=True)

    dataset_split = "val" if args.split == "val" else "test"
    image_dir = config["val_image_dir"] if args.split == "val" else config["test_image_dir"]
    annotation_path = config["val_ann_path"] if args.split == "val" else config.get("test_ann_path", None)
    if args.split == "test" and annotation_path is not None and (not os.path.exists(annotation_path)):
        annotation_path = None

    # External prediction JSONs are expected to be in original image
    # coordinates. Build the analysis dataset in raw image geometry so GT boxes
    # live in the same coordinate system.
    use_raw_geometry = prediction_json_path is not None
    data_transform = DetectionTransform(
        split=dataset_split,
        fixed_image_size=None if use_raw_geometry else config.get("inference_fixed_image_size", None),
        max_image_size=None if use_raw_geometry else config.get("inference_max_image_size", config.get("max_image_size", 640)),
        use_color_jitter=False,
        use_gaussian_blur=False,
        allow_upscale=False if use_raw_geometry else bool(config.get("inference_allow_upscale", False)),
        max_upscale_ratio=1.0 if use_raw_geometry else float(config.get("inference_max_upscale_ratio", 1.0)),
    )
    inference_dataset = DetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        split=dataset_split,
        transform=data_transform,
    )
    visualization_dataset = DetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        split=dataset_split,
        transform=DetectionTransform(
            split=dataset_split,
            max_image_size=None,
            use_color_jitter=False,
            use_gaussian_blur=False,
            allow_upscale=False,
            max_upscale_ratio=1.0,
        ),
    )
    if args.batch_size is None:
        default_eval_batch_size = int(
            config.get(
                "val_batch_size" if args.split == "val" else "test_batch_size",
                config.get("eval_batch_size", config["batch_size"]),
            )
        )
    else:
        default_eval_batch_size = int(args.batch_size)
    eval_num_workers = int(
        config.get(
            "val_num_workers" if args.split == "val" else "test_num_workers",
            config.get("eval_num_workers", args.num_workers),
        )
    )
    eval_pin_memory = bool(
        config.get(
            "val_pin_memory" if args.split == "val" else "test_pin_memory",
            config.get("eval_pin_memory", config.get("pin_memory", True)),
        )
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=default_eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        collate_fn=partial(
            collate_fn,
            pad_size_divisor=int(
                config.get(
                    "pad_size_divisor",
                    32 if str(config.get("model_backend", "")).lower() in {"hf_rtdetr_v2", "hf_rtdetr_v2_aux"} else 1,
                )
            ),
        ),
    )

    model = None
    external_pred_map = None
    external_debug_stage_map = None
    if prediction_json_path is None:
        model = build_model_from_config(config, pretrained_backbone_override=False).to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        incompatible = model.load_state_dict(adapt_checkpoint_state_dict(model, state_dict), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"Checkpoint compatibility | missing={len(incompatible.missing_keys)} "
                f"| unexpected={len(incompatible.unexpected_keys)}"
            )
        model.eval()
    else:
        with open(prediction_json_path, "r", encoding="utf-8") as file:
            external_predictions = json.load(file)
        external_pred_map = build_prediction_map(external_predictions)
        external_debug_stage_map = build_passthrough_debug_stage_map(external_pred_map)

    analysis_records = []
    dataset_index = 0
    confusion_matrix = torch.zeros((int(config["num_classes"]), int(config["num_classes"])), dtype=torch.int64)
    confusion_score_sum = torch.zeros_like(confusion_matrix, dtype=torch.float32)
    confusion_iou_sum = torch.zeros_like(confusion_matrix, dtype=torch.float32)
    high_iou_confusion_matrix = torch.zeros_like(confusion_matrix, dtype=torch.int64)
    cls_error_iou_bucket_summary = _init_iou_bucket_summary()
    query_coverage_points = build_query_coverage_points(int(config.get("num_queries", 20)))
    query_coverage_summary_accum = init_query_coverage_summary(query_coverage_points)
    crowded_slice_summary_accum = _init_slice_summary()
    oracle_summary_accum = init_oracle_summary()

    with torch.no_grad():
        pbar = tqdm(
            inference_loader,
            desc="Analyzing Validation" if args.split == "val" else "Analyzing Test",
            leave=False,
        )
        for images, masks, targets in pbar:
            if model is not None:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                targets_on_device = [
                    {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(images, masks)
                if args.split == "val":
                    predictions, debug_stage_map = collect_coco_predictions_debug(
                        outputs=outputs,
                        targets=targets_on_device,
                        **inference_postprocess_kwargs,
                    )
                else:
                    predictions = collect_coco_predictions(
                        outputs=outputs,
                        targets=targets_on_device,
                        **inference_postprocess_kwargs,
                    )
                    debug_stage_map = {}
            else:
                predictions = []
                debug_stage_map = {}
            pred_map = build_prediction_map(predictions) if model is not None else {}

            for batch_idx, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                analysis_target = clone_target_to_orig_geometry(target)
                if model is not None:
                    image_predictions = pred_map.get(image_id, [])
                else:
                    image_predictions = external_pred_map.get(image_id, []) if external_pred_map is not None else []
                    debug_stage_map = external_debug_stage_map if external_debug_stage_map is not None else {}
                if args.split == "val":
                    stage_prediction_blob = debug_stage_map.get(
                        image_id,
                        {
                            "_meta": {"topk_stage": "pre_class_nms"},
                            "pre_threshold": [],
                            "post_threshold": [],
                            "post_topk_pre_class_nms": [],
                            "post_topk_post_class_nms": [],
                            "post_topk": [],
                            "post_class_nms": [],
                            "post_agnostic_nms": [],
                            "post_cross_class_clean": [],
                            "post_topk_final": [],
                            "final": [],
                        },
                    )
                    accumulate_confusion_matrix(
                        target=analysis_target,
                        predictions=image_predictions,
                        iou_match_threshold=args.iou_match_threshold,
                        confusion_matrix=confusion_matrix,
                        confusion_score_sum=confusion_score_sum,
                        confusion_iou_sum=confusion_iou_sum,
                    )
                    accumulate_classification_iou_diagnostics(
                        target=analysis_target,
                        predictions=image_predictions,
                        iou_match_threshold=args.iou_match_threshold,
                        iou_bucket_summary=cls_error_iou_bucket_summary,
                        high_iou_confusion_matrix=high_iou_confusion_matrix,
                        high_iou_threshold=0.75,
                    )
                    legacy_analysis = analyze_sample(
                        target=analysis_target,
                        predictions=image_predictions,
                        iou_match_threshold=args.iou_match_threshold,
                    )
                    analysis_v2 = analyze_sample_v2(
                        target=analysis_target,
                        predictions=image_predictions,
                        iou_match_threshold=args.iou_match_threshold,
                    )
                    sample_type = analysis_v2["primary_type"]
                    details = dict(analysis_v2["details"])
                    missed_reason_counts = analyze_missed_sources(
                        target=analysis_target,
                        final_predictions=image_predictions,
                        stage_predictions=stage_prediction_blob,
                        iou_match_threshold=args.iou_match_threshold,
                        diagnostic_iou=float(args.missed_diag_iou),
                    )
                    accumulate_query_coverage_summary(
                        summary=query_coverage_summary_accum,
                        target=analysis_target,
                        stage_predictions=stage_prediction_blob,
                        curve_points=query_coverage_points,
                        iou_match_threshold=args.iou_match_threshold,
                        diagnostic_iou=float(args.missed_diag_iou),
                    )
                    gt_outcomes_v3 = analyze_gt_outcomes_v3(
                        target=analysis_target,
                        predictions=image_predictions,
                        iou_match_threshold=args.iou_match_threshold,
                        loose_iou_threshold=float(args.missed_diag_iou),
                    )
                    raw_gt_outcomes_v3 = analyze_gt_outcomes_v3(
                        target=analysis_target,
                        predictions=stage_prediction_blob.get("pre_threshold", image_predictions),
                        iou_match_threshold=args.iou_match_threshold,
                        loose_iou_threshold=float(args.missed_diag_iou),
                    )
                    raw_gt_outcomes_by_idx = {int(item["gt_index"]): item for item in raw_gt_outcomes_v3}
                    gt_crowd_slices = compute_gt_crowd_slices(
                        target=analysis_target,
                        crowded_gap_ratio_threshold=float(args.crowded_gap_ratio_threshold),
                    )
                    gt_boxes_cpu = analysis_target["boxes"].cpu()
                    gt_labels_cpu = analysis_target["labels"].cpu()
                    for gt_outcome, crowd_info in zip(gt_outcomes_v3, gt_crowd_slices):
                        slice_item = crowded_slice_summary_accum[crowd_info["slice"]]
                        slice_item["gt_total"] += 1
                        if gt_outcome["outcome"] == "matched_correct":
                            slice_item["gt_matched_correct"] += 1
                        elif gt_outcome["outcome"] == "wrong_class":
                            slice_item["gt_wrong_class"] += 1
                        elif gt_outcome["outcome"] == "loc_error":
                            slice_item["gt_loc_error"] += 1
                        else:
                            slice_item["gt_missed"] += 1
                        if crowd_info["nearest_gap_px"] is not None:
                            slice_item["avg_nearest_gap_px"] += float(crowd_info["nearest_gap_px"])
                            slice_item["avg_nearest_gap_ratio"] += float(crowd_info["nearest_gap_ratio"])
                            slice_item["_gap_count"] += 1

                        oracle_summary_accum["gt_total"] += 1
                        if gt_outcome["matched_correct"]:
                            oracle_summary_accum["final_correct_gt"] += 1
                            continue

                        gt_idx = int(gt_outcome["gt_index"])
                        raw_gt_outcome = raw_gt_outcomes_by_idx.get(gt_idx, {})
                        gt_box = gt_boxes_cpu[gt_idx]
                        gt_label = int(gt_labels_cpu[gt_idx].item())
                        missed_reason = _classify_missed_source_single_gt(
                            gt_box=gt_box,
                            gt_label=gt_label,
                            stage_predictions=stage_prediction_blob,
                            diagnostic_iou=float(args.missed_diag_iou),
                        )
                        class_recoverable = (
                            float(raw_gt_outcome.get("best_any_iou", 0.0)) >= float(args.iou_match_threshold)
                            and float(raw_gt_outcome.get("best_same_label_iou", 0.0)) < float(args.iou_match_threshold)
                        )
                        box_recoverable = (
                            float(raw_gt_outcome.get("best_same_label_iou", 0.0)) >= float(args.missed_diag_iou)
                            and float(raw_gt_outcome.get("best_same_label_iou", 0.0)) < float(args.iou_match_threshold)
                        )
                        postprocess_recoverable = missed_reason in {
                            "missed_filtered_by_threshold",
                            "missed_filtered_by_topk",
                            "missed_filtered_by_class_nms",
                            "missed_filtered_by_agnostic_nms",
                            "missed_filtered_by_cross_class_clean",
                        }
                        if class_recoverable:
                            oracle_summary_accum["oracle_class_recoverable_gt"] += 1
                        if box_recoverable:
                            oracle_summary_accum["oracle_box_recoverable_gt"] += 1
                        if postprocess_recoverable:
                            oracle_summary_accum["oracle_postprocess_recoverable_gt"] += 1
                        if missed_reason == "missed_no_candidate":
                            oracle_summary_accum["oracle_no_candidate_gt"] += 1
                        if class_recoverable or box_recoverable or postprocess_recoverable:
                            oracle_summary_accum["oracle_combined_recoverable_gt"] += 1
                    details["legacy_sample_type"] = legacy_analysis["sample_type"]
                    details["legacy_details"] = legacy_analysis["details"]
                    details["gt_stats"] = analysis_v2["gt_stats"]
                    details["pred_stats"] = analysis_v2["pred_stats"]
                    details["missed_reason_counts"] = missed_reason_counts
                else:
                    sample_type = "has_pred" if len(image_predictions) > 0 else "no_pred"
                    details = {
                        "num_pred": len(image_predictions),
                        "best_score": max((float(pred["score"]) for pred in image_predictions), default=0.0),
                    }
                analysis_records.append(
                    {
                        "dataset_index": dataset_index,
                        "image_id": image_id,
                        "file_name": target.get("file_name", ""),
                        "sample_type": sample_type,
                        "legacy_sample_type": details.get("legacy_sample_type", sample_type),
                        "sample_tags": analysis_v2["sample_tags"] if args.split == "val" else [],
                        "is_clean": analysis_v2["is_clean"] if args.split == "val" else False,
                        "details": details,
                        "predictions": image_predictions,
                    }
                )
                dataset_index += 1

    selected_records = select_samples(analysis_records, args) if args.split == "val" else select_test_samples(analysis_records, args)
    saved_paths = []

    for output_idx, record in enumerate(selected_records):
        image_tensor, target = visualization_dataset[record["dataset_index"]]
        sample_label = build_sample_label(record) if args.split == "val" else record["sample_type"]
        save_path = os.path.join(
            output_dir,
            f"{record['sample_type']}_{output_idx:03d}_img_{record['image_id']}.png",
        )
        if args.split == "val":
            save_single_visualization(
                image_tensor=image_tensor,
                target=target,
                image_predictions=record["predictions"],
                sample_type=sample_label,
                save_path=save_path,
            )
        else:
            save_prediction_only_visualization(
                image_tensor=image_tensor,
                target=target,
                image_predictions=record["predictions"],
                sample_type=sample_label,
                save_path=save_path,
            )
        saved_paths.append(save_path)

    contact_sheet_path = os.path.join(output_dir, "contact_sheet.png")
    save_contact_sheet(saved_paths, contact_sheet_path, cols=4, thumb_size=(320, 160))

    summary = {}
    missed_reason_summary = {}
    for record in analysis_records:
        summary[record["sample_type"]] = summary.get(record["sample_type"], 0) + 1
        missed_reason_counts = record["details"].get("missed_reason_counts")
        if missed_reason_counts:
            for reason, count in missed_reason_counts.items():
                missed_reason_summary[reason] = missed_reason_summary.get(reason, 0) + int(count)

    if args.split == "val":
        legacy_summary, image_tag_summary, gt_summary, pred_summary = summarize_v2_records(analysis_records)
        confusion_save_path = os.path.join(output_dir, "digit_confusion_matrix.png")
        per_digit_save_path = os.path.join(output_dir, "digit_error_breakdown_v2.png")
        save_confusion_matrix_figure(confusion_matrix, confusion_save_path)
        gt_digit_summary, pred_digit_summary, long_tail_summary = summarize_per_digit_records(
            analysis_records=analysis_records,
            num_classes=int(config["num_classes"]),
        )
        save_per_digit_error_figure(gt_digit_summary, pred_digit_summary, per_digit_save_path)
        top_confusions = extract_top_confusions(confusion_matrix, topk=10)
        high_iou_top_confusions = extract_top_confusions(high_iou_confusion_matrix, topk=10)
        cls_error_iou_bucket_summary_final = finalize_iou_bucket_summary(cls_error_iou_bucket_summary)
        query_coverage_summary_final = finalize_query_coverage_summary(query_coverage_summary_accum)
        crowded_slice_summary_final = finalize_slice_summary(crowded_slice_summary_accum)
        oracle_summary_final = finalize_oracle_summary(oracle_summary_accum)
        postprocess_gap_report = build_postprocess_gap_report(missed_reason_summary, gt_summary)
        total_matched = int(confusion_matrix.sum().item())
        total_correct_matched = int(confusion_matrix.diag().sum().item())
        high_iou_wrong_class_total = int(high_iou_confusion_matrix.sum().item())
        query_coverage_save_path = os.path.join(output_dir, "query_coverage_curve_v3.png")
        save_query_coverage_figure(query_coverage_summary_final, query_coverage_save_path)

        print("Validation sample summary (legacy primary label):")
        for sample_type in ["correct", "duplicate", "cls_error", "loc_error", "missed", "grouped"]:
            print(f"  {sample_type}: {legacy_summary.get(sample_type, 0)}")
        print("Validation sample summary (v2 multi-label image tags):")
        for sample_type in ["correct", "duplicate", "cls_error", "grouped", "loc_error", "missed", "background_fp"]:
            print(f"  {sample_type}: {image_tag_summary.get(sample_type, 0)}")
        print("GT-centric summary (v2):")
        for key in ["gt_total", "gt_matched_correct", "gt_wrong_class", "gt_loc_error", "gt_missed"]:
            print(f"  {key}: {gt_summary.get(key, 0)}")
        print("Prediction-centric summary (v2):")
        for key in ["pred_total", "pred_true_positive", "pred_duplicate", "pred_grouped", "pred_wrong_class", "pred_loc_error", "pred_background_fp"]:
            print(f"  {key}: {pred_summary.get(key, 0)}")
        print("Per-digit GT-centric summary (v2):")
        for digit in range(int(config["num_classes"])):
            item = gt_digit_summary[str(digit)]
            print(
                "  "
                f"digit {digit}: total={int(item['gt_total'])}, correct={int(item['gt_matched_correct'])}, "
                f"wrong_cls={int(item['gt_wrong_class'])}, loc={int(item['gt_loc_error'])}, missed={int(item['gt_missed'])}, "
                f"correct_rate={float(item['gt_correct_rate']):.3f}, error_rate={float(item['gt_error_rate']):.3f}"
            )
        print("Per-digit prediction-centric summary (v2):")
        for digit in range(int(config["num_classes"])):
            item = pred_digit_summary[str(digit)]
            print(
                "  "
                f"digit {digit}: total={int(item['pred_total'])}, tp={int(item['pred_true_positive'])}, dup={int(item['pred_duplicate'])}, "
                f"grouped={int(item['pred_grouped'])}, wrong_cls={int(item['pred_wrong_class'])}, loc={int(item['pred_loc_error'])}, "
                f"bg_fp={int(item['pred_background_fp'])}, tp_rate={float(item['pred_true_positive_rate']):.3f}"
            )
        print("Long-tail check (v2):")
        print(f"  head_digits_by_gt_frequency: {long_tail_summary['head_digits_by_gt_frequency']}")
        print(f"  tail_digits_by_gt_frequency: {long_tail_summary['tail_digits_by_gt_frequency']}")
        print(f"  head_mean_correct_rate: {float(long_tail_summary['head_mean_correct_rate']):.3f}")
        print(f"  tail_mean_correct_rate: {float(long_tail_summary['tail_mean_correct_rate']):.3f}")
        print(f"  head_mean_error_rate: {float(long_tail_summary['head_mean_error_rate']):.3f}")
        print(f"  tail_mean_error_rate: {float(long_tail_summary['tail_mean_error_rate']):.3f}")
        print(f"  tail_minus_head_error_rate_gap: {float(long_tail_summary['tail_minus_head_error_rate_gap']):.3f}")
        print("Missed GT reason summary:")
        for reason in [
            "missed_total_gt",
            "missed_no_candidate",
            "missed_wrong_class_from_start",
            "missed_filtered_by_threshold",
            "missed_filtered_by_topk",
            "missed_filtered_by_class_nms",
            "missed_filtered_by_agnostic_nms",
            "missed_filtered_by_cross_class_clean",
            "missed_localized_but_unmatched",
        ]:
            print(f"  {reason}: {missed_reason_summary.get(reason, 0)}")
        print("Digit confusion summary:")
        print(f"  matched_pairs: {total_matched}")
        print(f"  matched_correct: {total_correct_matched}")
        print(f"  matched_wrong: {total_matched - total_correct_matched}")
        for item in top_confusions:
            gt_label = item["gt"]
            pred_label = item["pred"]
            count = item["count"]
            avg_score = float(confusion_score_sum[gt_label, pred_label].item()) / max(1, count)
            avg_iou = float(confusion_iou_sum[gt_label, pred_label].item()) / max(1, count)
            print(f"  {gt_label}->{pred_label}: count={count}, avg_score={avg_score:.3f}, avg_iou={avg_iou:.3f}")
        print("Classification by IoU bucket:")
        for bucket_name in ["0.50-0.75", "0.75-0.90", "0.90-1.00"]:
            item = cls_error_iou_bucket_summary_final[bucket_name]
            print(
                "  "
                f"{bucket_name}: matched_total={int(item['matched_total'])}, correct={int(item['correct'])}, "
                f"wrong_class={int(item['wrong_class'])}, wrong_class_rate={float(item['wrong_class_rate']):.3f}"
            )
        print("High-IoU (>=0.75) wrong-class confusions:")
        print(f"  high_iou_wrong_class_total: {high_iou_wrong_class_total}")
        for item in high_iou_top_confusions:
            print(f"  {item['gt']}->{item['pred']}: count={int(item['count'])}")
        print("V3 query coverage summary:")
        coverage_print_ks = []
        for preferred_k in [1, 4, 8, 12, 20, int(config.get("num_queries", 20))]:
            if any(int(item["k"]) == preferred_k for item in query_coverage_summary_final["curve_points"]):
                coverage_print_ks.append(preferred_k)
        coverage_print_ks = sorted(set(coverage_print_ks))
        coverage_map = {int(item["k"]): item for item in query_coverage_summary_final["curve_points"]}
        for k in coverage_print_ks:
            item = coverage_map[int(k)]
            print(
                "  "
                f"K={k}: any@0.30={float(item['any_candidate_rate_iou_0_30']):.3f}, "
                f"same_label@0.30={float(item['same_label_candidate_rate_iou_0_30']):.3f}, "
                f"same_label@0.50={float(item['same_label_match_rate_iou_0_50']):.3f}"
            )
        print(f"  saturation_k_95_same_label_candidate: {query_coverage_summary_final['saturation_k_95_same_label_candidate']}")
        print(f"  saturation_k_95_same_label_match: {query_coverage_summary_final['saturation_k_95_same_label_match']}")
        print("V3 crowded-vs-isolated GT summary:")
        for slice_name in ["crowded", "isolated", "singleton"]:
            item = crowded_slice_summary_final[slice_name]
            print(
                "  "
                f"{slice_name}: total={int(item['gt_total'])}, correct_rate={float(item['gt_correct_rate']):.3f}, "
                f"wrong_cls_rate={float(item['gt_wrong_class_rate']):.3f}, loc_rate={float(item['gt_loc_error_rate']):.3f}, "
                f"missed_rate={float(item['gt_missed_rate']):.3f}"
            )
        print("V3 oracle summary:")
        print(f"  final_correct_rate: {float(oracle_summary_final['final_correct_rate']):.3f}")
        print(f"  oracle_class_recoverable_gt: {int(oracle_summary_final['oracle_class_recoverable_gt'])}")
        print(f"  oracle_box_recoverable_gt: {int(oracle_summary_final['oracle_box_recoverable_gt'])}")
        print(f"  oracle_postprocess_recoverable_gt: {int(oracle_summary_final['oracle_postprocess_recoverable_gt'])}")
        print(f"  oracle_combined_upper_bound_rate: {float(oracle_summary_final['oracle_combined_upper_bound_rate']):.3f}")
        print("V3 postprocess gap summary:")
        print(f"  postprocess_filtered_total: {int(postprocess_gap_report['postprocess_filtered_total'])}")
        print(f"  postprocess_filtered_rate_over_missed: {float(postprocess_gap_report['postprocess_filtered_rate_over_missed']):.3f}")

        summary_save_path = os.path.join(output_dir, "analysis_summary_v2.json")
        summary_v3_save_path = os.path.join(output_dir, "analysis_summary_v3.json")
        with open(summary_save_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "legacy_summary": legacy_summary,
                    "image_tag_summary": image_tag_summary,
                    "gt_summary": gt_summary,
                    "pred_summary": pred_summary,
                    "gt_digit_summary": gt_digit_summary,
                    "pred_digit_summary": pred_digit_summary,
                    "long_tail_summary": long_tail_summary,
                    "missed_reason_summary": missed_reason_summary,
                    "top_confusions": top_confusions,
                    "high_iou_confusions": high_iou_top_confusions,
                    "cls_error_iou_bucket_summary": cls_error_iou_bucket_summary_final,
                    "high_iou_wrong_class_total": high_iou_wrong_class_total,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
        with open(summary_v3_save_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "query_coverage_summary": query_coverage_summary_final,
                    "crowded_slice_meta": {
                        "crowded_gap_ratio_threshold": float(args.crowded_gap_ratio_threshold),
                    },
                    "crowded_slice_summary": crowded_slice_summary_final,
                    "oracle_report": oracle_summary_final,
                    "postprocess_gap_report": postprocess_gap_report,
                    "per_digit_error_breakdown": {
                        "gt_digit_summary": gt_digit_summary,
                        "pred_digit_summary": pred_digit_summary,
                    },
                    "source_meta": {
                        "prediction_json": prediction_json_path,
                        "postprocess_topk_stage": str(inference_postprocess_kwargs.get("postprocess_topk_stage", "pre_class_nms")),
                        "score_threshold": float(inference_postprocess_kwargs.get("score_threshold", 0.0)),
                        "topk_per_image": int(inference_postprocess_kwargs.get("topk_per_image", 0)),
                        "class_logit_bias": inference_postprocess_kwargs.get("class_logit_bias", {}),
                        "query_coverage_source_stage": str(query_coverage_summary_final.get("source_stage", "pre_threshold")),
                    },
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
    else:
        total_predictions = sum(len(record["predictions"]) for record in analysis_records)
        print("Test sample summary:")
        print(f"  has_pred: {summary.get('has_pred', 0)}")
        print(f"  no_pred: {summary.get('no_pred', 0)}")
        print(f"  total_predictions: {total_predictions}")
    print(f"Saved {len(saved_paths)} visualization images to: {output_dir}")
    print(f"Contact sheet saved to: {contact_sheet_path}")
    if args.split == "val":
        print(f"Digit confusion matrix saved to: {confusion_save_path}")
        print(f"Per-digit error breakdown saved to: {per_digit_save_path}")
        print(f"V2 analysis summary saved to: {summary_save_path}")
        print(f"V3 query coverage figure saved to: {query_coverage_save_path}")
        print(f"V3 analysis summary saved to: {summary_v3_save_path}")


if __name__ == "__main__":
    main()



