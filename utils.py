"""Training, evaluation, and postprocess utilities."""

import json
import math
import os
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_iou, nms

from model import box_cxcywh_to_xyxy, rescale_boxes_to_pixels

try:
    from transformers import RTDetrImageProcessor
except Exception:  # pragma: no cover - fallback only if transformers import breaks
    RTDetrImageProcessor = None


_RTDETR_IMAGE_PROCESSOR = None


def _is_hf_rtdetr_backend(model_backend: str | None) -> bool:
    return str(model_backend or "").lower() in {
        "hf_rtdetr_v2",
        "hf_rtdetr_v2_aux",
        "hf_rtdetr_v2_qs",
    }


def plot_training_curves(
    train_loss,
    val_loss,
    val_map,
    save_path="./Plot/training_curves_detr_baseline.png",
):
    """Plot training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = list(range(1, len(train_loss) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_map, label="Val mAP", linewidth=2)
    best_epoch = int(torch.tensor(val_map).argmax().item()) + \
        1 if val_map else 1
    best_map = max(val_map) if val_map else 0.0
    axes[1].scatter([best_epoch], [best_map], marker="*", s=250)
    axes[1].set_title("Validation mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + cosine annealing scheduler."""

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        eta_min=1e-6,
        last_epoch=-1,
    ):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(0, warmup_steps), self.total_steps - 1)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / \
            max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


class ModelEMA:
    """Official-style EMA with warmup ramp for stabler validation/test."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        warmups: int = 2000,
    ):
        self.decay = float(decay)
        self.warmups = max(0, int(warmups))
        self.updates = 0
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def _get_decay(self) -> float:
        if self.warmups <= 0:
            return self.decay
        return self.decay * (
            1.0 - math.exp(-float(self.updates) / float(self.warmups))
        )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        current_decay = self._get_decay()
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(current_decay).add_(model_value, alpha=1.0 - current_decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "warmups": self.warmups,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if (
            isinstance(state_dict, dict)
            and "module" in state_dict
            and isinstance(state_dict["module"], dict)
        ):
            self.module.load_state_dict(state_dict["module"])
            self.updates = int(state_dict.get("updates", 0))
            self.decay = float(state_dict.get("decay", self.decay))
            self.warmups = max(0, int(state_dict.get("warmups", self.warmups)))
            return
        # Backward compatibility with older checkpoints that stored only module weights.
        self.module.load_state_dict(state_dict)


def build_inference_postprocess_kwargs(config: dict) -> dict:
    """Build shared inference kwargs for val/test/visualization."""
    model_backend = str(config.get("model_backend", "custom_relation_detr")).lower()
    return {
        "model_backend": model_backend,
        "use_official_backend_postprocess": bool(
            config.get(
                "use_official_backend_postprocess",
                _is_hf_rtdetr_backend(model_backend),
            )
        ),
        "rtdetr_postprocess_variant": str(
            config.get("rtdetr_postprocess_variant", "official_hf")
        ).lower(),
        "score_threshold": float(config.get("score_threshold", 0.20)),
        "topk_per_image": int(config.get("topk_per_image", 10)),
        "postprocess_topk_stage": str(
            config.get("postprocess_topk_stage", "pre_class_nms")
        ),
        "class_logit_bias": config.get("class_logit_bias", {}),
        "use_nms": bool(config.get("use_nms", False)),
        "nms_iou_threshold": float(config.get("nms_iou_threshold", 0.5)),
        "use_class_containment_suppression": bool(
            config.get("use_class_containment_suppression", False)
        ),
        "class_containment_threshold": float(
            config.get("class_containment_threshold", 0.9)
        ),
        "use_agnostic_nms": bool(config.get("use_agnostic_nms", False)),
        "agnostic_nms_iou_threshold": float(
            config.get("agnostic_nms_iou_threshold", 0.7)
        ),
        "agnostic_containment_threshold": float(
            config.get("agnostic_containment_threshold", 0.85)
        ),
        "use_cross_class_overlap_suppression": bool(
            config.get("use_cross_class_overlap_suppression", False)
        ),
        "cross_class_overlap_iou_threshold": float(
            config.get("cross_class_overlap_iou_threshold", 0.9)
        ),
        "cross_class_overlap_containment_threshold": float(
            config.get("cross_class_overlap_containment_threshold", 0.95)
        ),
        "cross_class_overlap_score_margin": float(
            config.get("cross_class_overlap_score_margin", 0.08)
        ),
    }


def _to_loss_tensor(value, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.tensor(float(value), device=device)


def resolve_training_loss(
    outputs: dict,
    targets: list[dict],
    criterion=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    official_loss = outputs.get("official_loss")
    if official_loss is None:
        if criterion is None:
            raise ValueError(
                "Custom backend requires a criterion when official_loss "
                "is unavailable."
            )
        loss_dict = criterion(outputs, targets)
        return loss_dict["loss"], loss_dict

    device = official_loss.device
    official_loss_dict = {
        key: _to_loss_tensor(value, device)
        for key, value in (outputs.get("official_loss_dict") or {}).items()
    }
    if criterion is None:
        zero = official_loss.detach() * 0.0
        extra_loss_dict = {
            "loss_targeted_confusion_margin": zero,
            "loss_aux_digit_cls": zero,
            "loss_query_quality": zero,
            "loss_targeted_confusion": zero,
            "loss_aux_digit_cls_weighted": zero,
            "loss_query_quality_weighted": zero,
            "loss_extra_custom": zero,
        }
    else:
        extra_loss_dict = criterion.compute_additional_losses(outputs, targets)
    total_loss = official_loss + extra_loss_dict["loss_extra_custom"]

    zero = official_loss.detach() * 0.0
    classification_loss = zero
    for loss_key in ("loss_ce", "loss_vfl", "loss_cls", "loss_focal"):
        if loss_key in official_loss_dict:
            classification_loss = official_loss_dict[loss_key]
            break
    loss_dict: dict[str, torch.Tensor] = dict(official_loss_dict)
    loss_dict["loss_ce"] = classification_loss
    loss_dict["loss_bbox"] = official_loss_dict.get("loss_bbox", zero)
    loss_dict["loss_giou"] = official_loss_dict.get("loss_giou", zero)
    loss_dict["loss_objectness"] = zero
    loss_dict["loss_main"] = official_loss
    loss_dict["loss_aux"] = zero
    loss_dict["loss_group"] = zero
    loss_dict["loss_enc"] = zero
    loss_dict["loss_dn"] = zero
    loss_dict["loss_hybrid"] = zero
    loss_dict.update(extra_loss_dict)
    loss_dict["loss"] = total_loss
    return total_loss, loss_dict


def parse_tta_scales(raw_value) -> list[float]:
    """Parse TTA scale factors from config/CLI into a clean float list."""
    if raw_value is None:
        return [1.0]

    if isinstance(raw_value, (int, float)):
        scale = float(raw_value)
        return [scale] if scale > 0.0 else [1.0]

    if isinstance(raw_value, str):
        candidates = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        candidates = list(raw_value)
    else:
        return [1.0]

    parsed = []
    for item in candidates:
        try:
            scale = float(item)
        except (TypeError, ValueError):
            continue
        if scale > 0.0:
            parsed.append(scale)

    if not parsed:
        return [1.0]

    deduped = []
    seen = set()
    for scale in parsed:
        key = round(scale, 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(scale)
    return deduped


def _apply_agnostic_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    agnostic_nms_iou_threshold: float,
    agnostic_containment_threshold: float,
) -> torch.Tensor:
    if boxes.shape[0] <= 1:
        return torch.arange(boxes.shape[0], device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    areas = ((boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6))

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[remaining]
        ious = box_iou(current_box, remaining_boxes).squeeze(0)

        inter_x1 = torch.maximum(current_box[:, 0], remaining_boxes[:, 0])
        inter_y1 = torch.maximum(current_box[:, 1], remaining_boxes[:, 1])
        inter_x2 = torch.minimum(current_box[:, 2], remaining_boxes[:, 2])
        inter_y2 = torch.minimum(current_box[:, 3], remaining_boxes[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        inter_area = inter_w * inter_h
        containment = inter_area / torch.minimum(areas[current], areas[remaining]).clamp_min(1e-6)

        suppress = (ious > agnostic_nms_iou_threshold) | (containment > agnostic_containment_threshold)
        order = remaining[~suppress]

    return torch.stack(keep) if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def _apply_same_label_containment_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    containment_threshold: float,
) -> torch.Tensor:
    if boxes.shape[0] <= 1:
        return torch.arange(boxes.shape[0], device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    areas = ((boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6))

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[remaining]
        same_label = labels[remaining] == labels[current]

        inter_x1 = torch.maximum(current_box[:, 0], remaining_boxes[:, 0])
        inter_y1 = torch.maximum(current_box[:, 1], remaining_boxes[:, 1])
        inter_x2 = torch.minimum(current_box[:, 2], remaining_boxes[:, 2])
        inter_y2 = torch.minimum(current_box[:, 3], remaining_boxes[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        inter_area = inter_w * inter_h
        containment = inter_area / torch.minimum(areas[current], areas[remaining]).clamp_min(1e-6)

        suppress = same_label & (containment > containment_threshold)
        order = remaining[~suppress]

    return torch.stack(keep) if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def _apply_cross_class_overlap_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    containment_threshold: float,
    score_margin: float,
) -> torch.Tensor:
    if boxes.shape[0] <= 1:
        return torch.arange(boxes.shape[0], device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    areas = ((boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6))

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[remaining]
        different_label = labels[remaining] != labels[current]
        if not bool(different_label.any().item()):
            order = remaining
            continue

        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        inter_x1 = torch.maximum(current_box[:, 0], remaining_boxes[:, 0])
        inter_y1 = torch.maximum(current_box[:, 1], remaining_boxes[:, 1])
        inter_x2 = torch.minimum(current_box[:, 2], remaining_boxes[:, 2])
        inter_y2 = torch.minimum(current_box[:, 3], remaining_boxes[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        inter_area = inter_w * inter_h
        containment = inter_area / torch.minimum(areas[current], areas[remaining]).clamp_min(1e-6)
        score_gap = scores[current] - scores[remaining]

        suppress = different_label & (
            ((ious > iou_threshold) & (score_gap <= score_margin))
            | (containment > containment_threshold)
        )
        order = remaining[~suppress]

    return torch.stack(keep) if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def _format_predictions(
    image_id: int,
    image_size: list[float] | tuple[float, float],
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
) -> list[dict]:
    image_height, image_width = image_size
    formatted = []
    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        formatted.append(
            {
                "image_id": image_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item()),
                "category_id": int(label.item()) + 1,
            }
        )
    return formatted


def _sort_predictions_by_score(
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() <= 1:
        return scores, labels, boxes
    order = scores.argsort(descending=True)
    return scores[order], labels[order], boxes[order]


def _apply_topk(
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    topk_per_image: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        return scores, labels, boxes
    topk = min(max(0, int(topk_per_image)), scores.numel())
    if topk <= 0 or topk >= scores.numel():
        return _sort_predictions_by_score(scores, labels, boxes)
    scores, indices = scores.topk(topk)
    labels = labels[indices]
    boxes = boxes[indices]
    return _sort_predictions_by_score(scores, labels, boxes)


def _prediction_dicts_to_tensors(predictions: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def merge_coco_prediction_sets(
    prediction_sets: list[list[dict]],
    image_sizes_by_id: dict[int, list[float] | tuple[float, float]],
    score_threshold: float = 0.20,
    topk_per_image: int = 10,
    postprocess_topk_stage: str = "final",
    class_logit_bias: dict | list | tuple | None = None,
    tta_merge_method: str = "nms",
    tta_wbf_iou_threshold: float = 0.65,
    tta_wbf_skip_box_threshold: float | None = None,
    use_nms: bool = False,
    nms_iou_threshold: float = 0.5,
    use_class_containment_suppression: bool = False,
    class_containment_threshold: float = 0.9,
    use_agnostic_nms: bool = False,
    agnostic_nms_iou_threshold: float = 0.7,
    agnostic_containment_threshold: float = 0.85,
    use_cross_class_overlap_suppression: bool = False,
    cross_class_overlap_iou_threshold: float = 0.9,
    cross_class_overlap_containment_threshold: float = 0.95,
    cross_class_overlap_score_margin: float = 0.08,
) -> list[dict]:
    """Merge per-scale prediction sets with the same suppression logic used in inference."""
    # TTA merge always performs a single final top-k after cross-scale suppression.
    # Accept the argument for API compatibility with shared inference kwargs.
    _ = postprocess_topk_stage, class_logit_bias

    grouped_predictions: dict[int, list[tuple[dict, int]]] = {}
    for set_index, prediction_set in enumerate(prediction_sets):
        for prediction in prediction_set:
            image_id = int(prediction["image_id"])
            grouped_predictions.setdefault(image_id, []).append((prediction, set_index))

    merged_predictions: list[dict] = []
    for image_id, image_predictions_with_source in grouped_predictions.items():
        image_size = image_sizes_by_id.get(image_id)
        if image_size is None:
            continue
        image_predictions = [prediction for prediction, _ in image_predictions_with_source]

        if str(tta_merge_method).lower() == "wbf":
            skip_box_threshold = float(score_threshold if tta_wbf_skip_box_threshold is None else tta_wbf_skip_box_threshold)
            fused_predictions = _merge_prediction_sets_wbf_single_image(
                image_predictions_with_source=image_predictions_with_source,
                num_prediction_sets=max(1, len(prediction_sets)),
                iou_threshold=float(tta_wbf_iou_threshold),
                skip_box_threshold=skip_box_threshold,
            )
            boxes, labels, scores = _prediction_dicts_to_tensors(fused_predictions)
        else:
            boxes, labels, scores = _prediction_dicts_to_tensors(image_predictions)
            if scores.numel() == 0:
                continue

            keep = scores >= float(score_threshold)
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

            if use_nms and scores.numel() > 0:
                kept_indices = []
                for class_label in labels.unique():
                    class_mask = labels == class_label
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
                    selected = nms(class_boxes, class_scores, nms_iou_threshold)
                    kept_indices.append(class_indices[selected])
                if kept_indices:
                    kept_indices = torch.cat(kept_indices, dim=0)
                    scores = scores[kept_indices]
                    labels = labels[kept_indices]
                    boxes = boxes[kept_indices]
                    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

        if scores.numel() == 0:
            continue

        if use_class_containment_suppression and scores.numel() > 1:
            kept_indices = _apply_same_label_containment_suppression(
                boxes=boxes,
                scores=scores,
                labels=labels,
                containment_threshold=class_containment_threshold,
            )
            scores = scores[kept_indices]
            labels = labels[kept_indices]
            boxes = boxes[kept_indices]
            scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

        if use_agnostic_nms and scores.numel() > 1:
            kept_indices = _apply_agnostic_suppression(
                boxes=boxes,
                scores=scores,
                agnostic_nms_iou_threshold=agnostic_nms_iou_threshold,
                agnostic_containment_threshold=agnostic_containment_threshold,
            )
            scores = scores[kept_indices]
            labels = labels[kept_indices]
            boxes = boxes[kept_indices]
            scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

        if use_cross_class_overlap_suppression and scores.numel() > 1:
            kept_indices = _apply_cross_class_overlap_suppression(
                boxes=boxes,
                scores=scores,
                labels=labels,
                iou_threshold=cross_class_overlap_iou_threshold,
                containment_threshold=cross_class_overlap_containment_threshold,
                score_margin=cross_class_overlap_score_margin,
            )
            scores = scores[kept_indices]
            labels = labels[kept_indices]
            boxes = boxes[kept_indices]
            scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

        scores, labels, boxes = _apply_topk(scores, labels, boxes, topk_per_image)
        merged_predictions.extend(
            _format_predictions(
                image_id=image_id,
                image_size=image_size,
                scores=scores,
                labels=labels,
                boxes=boxes,
            )
        )

    return merged_predictions


def _merge_prediction_sets_wbf_single_image(
    image_predictions_with_source: list[tuple[dict, int]],
    num_prediction_sets: int,
    iou_threshold: float,
    skip_box_threshold: float,
) -> list[dict]:
    """Class-aware WBF for one image using per-scale predictions as sources."""
    if not image_predictions_with_source:
        return []

    image_id = int(image_predictions_with_source[0][0]["image_id"])
    grouped_by_label: dict[int, list[tuple[torch.Tensor, float, int]]] = {}
    for prediction, source_index in image_predictions_with_source:
        score = float(prediction["score"])
        if score < float(skip_box_threshold):
            continue
        x, y, w, h = prediction["bbox"]
        if w <= 0.0 or h <= 0.0:
            continue
        label = int(prediction["category_id"])
        box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        grouped_by_label.setdefault(label, []).append((box, score, int(source_index)))

    fused_predictions: list[dict] = []
    for label, entries in grouped_by_label.items():
        entries = sorted(entries, key=lambda item: item[1], reverse=True)
        clusters: list[dict] = []
        for box, score, source_index in entries:
            matched_cluster = None
            matched_iou = -1.0
            for cluster in clusters:
                cluster_box = cluster["box"].unsqueeze(0)
                candidate_iou = float(box_iou(cluster_box, box.unsqueeze(0)).item())
                if candidate_iou >= float(iou_threshold) and candidate_iou > matched_iou:
                    matched_iou = candidate_iou
                    matched_cluster = cluster
            if matched_cluster is None:
                clusters.append(
                    {
                        "boxes": [box],
                        "scores": [score],
                        "sources": {source_index},
                        "box": box.clone(),
                    }
                )
            else:
                matched_cluster["boxes"].append(box)
                matched_cluster["scores"].append(score)
                matched_cluster["sources"].add(source_index)
                weights = torch.tensor(matched_cluster["scores"], dtype=torch.float32).view(-1, 1)
                stacked_boxes = torch.stack(matched_cluster["boxes"], dim=0)
                matched_cluster["box"] = (stacked_boxes * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)

        for cluster in clusters:
            stacked_boxes = torch.stack(cluster["boxes"], dim=0)
            score_tensor = torch.tensor(cluster["scores"], dtype=torch.float32)
            weights = score_tensor.view(-1, 1)
            fused_box = (stacked_boxes * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)
            fused_score = float(score_tensor.mean().item())
            support_ratio = min(len(cluster["sources"]), int(num_prediction_sets)) / max(1, int(num_prediction_sets))
            fused_score *= float(support_ratio)
            x1, y1, x2, y2 = fused_box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            fused_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(fused_score),
                    "category_id": int(label),
                }
            )

    return fused_predictions


def _record_stage(
    stage_debug: dict,
    name: str,
    image_id: int,
    image_size: list[float] | tuple[float, float],
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
) -> None:
    stage_debug[name] = _format_predictions(image_id, image_size, scores, labels, boxes)




def _select_query_predictions(
    raw_logits: torch.Tensor,
    pred_objectness_logits: torch.Tensor | None,
    pred_quality_scores: torch.Tensor | None = None,
    class_logit_bias: dict | list | tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if class_logit_bias:
        calibrated_logits = raw_logits.clone()
        if isinstance(class_logit_bias, dict):
            for class_idx, bias in class_logit_bias.items():
                try:
                    class_index = int(class_idx)
                    bias_value = float(bias)
                except (TypeError, ValueError):
                    continue
                if 0 <= class_index < calibrated_logits.shape[-1]:
                    calibrated_logits[..., class_index] = calibrated_logits[..., class_index] + bias_value
        else:
            bias_tensor = torch.as_tensor(class_logit_bias, dtype=calibrated_logits.dtype, device=calibrated_logits.device)
            if bias_tensor.numel() == calibrated_logits.shape[-1]:
                calibrated_logits = calibrated_logits + bias_tensor.view(1, -1)
        raw_logits = calibrated_logits

    if pred_objectness_logits is None:
        probs = raw_logits.sigmoid()
    else:
        probs = raw_logits.softmax(dim=-1) * pred_objectness_logits.sigmoid()
    if pred_quality_scores is not None:
        quality_scores = pred_quality_scores.to(dtype=probs.dtype, device=probs.device)
        if quality_scores.dim() == 1:
            quality_scores = quality_scores.unsqueeze(-1)
        probs = probs * quality_scores.clamp(0.0, 1.0)
    return probs.max(dim=-1)


def _postprocess_single_image(
    raw_logits: torch.Tensor,
    pred_objectness_logits: torch.Tensor | None,
    pred_quality_scores: torch.Tensor | None,
    pred_boxes: torch.Tensor,
    target: dict,
    score_threshold: float,
    topk_per_image: int,
    postprocess_topk_stage: str,
    class_logit_bias: dict | list | tuple | None,
    use_nms: bool,
    nms_iou_threshold: float,
    use_class_containment_suppression: bool,
    class_containment_threshold: float,
    use_agnostic_nms: bool,
    agnostic_nms_iou_threshold: float,
    agnostic_containment_threshold: float,
    use_cross_class_overlap_suppression: bool,
    cross_class_overlap_iou_threshold: float,
    cross_class_overlap_containment_threshold: float,
    cross_class_overlap_score_margin: float,
) -> dict:
    image_id = int(target["image_id"].item())
    output_size = target.get("orig_size", target["size"])
    image_size = output_size.tolist()

    boxes = box_cxcywh_to_xyxy(pred_boxes)
    # Model boxes are normalized in [0, 1]. Always export predictions in the
    # original image coordinate system so evaluation remains correct even when
    # validation/test-time resizing or upscaling is enabled.
    boxes = rescale_boxes_to_pixels(boxes, output_size.unsqueeze(0)).squeeze(0)
    scores, labels = _select_query_predictions(
        raw_logits=raw_logits,
        pred_objectness_logits=pred_objectness_logits,
        pred_quality_scores=pred_quality_scores,
        class_logit_bias=class_logit_bias,
    )
    thresholds = torch.full_like(scores, score_threshold)
    if postprocess_topk_stage not in {"pre_class_nms", "post_class_nms", "final"}:
        postprocess_topk_stage = "pre_class_nms"

    stage_debug = {
        "_meta": {
            "topk_stage": postprocess_topk_stage,
            "class_logit_bias": class_logit_bias if class_logit_bias else {},
        },
    }
    _record_stage(stage_debug, "pre_threshold", image_id, image_size, scores, labels, boxes)

    keep = scores >= thresholds
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)
    _record_stage(stage_debug, "post_threshold", image_id, image_size, scores, labels, boxes)

    if postprocess_topk_stage == "pre_class_nms":
        scores, labels, boxes = _apply_topk(scores, labels, boxes, topk_per_image)
    _record_stage(stage_debug, "post_topk_pre_class_nms", image_id, image_size, scores, labels, boxes)

    if use_nms and scores.numel() > 0:
        kept_indices = []
        unique_labels = labels.unique()
        for class_label in unique_labels:
            class_mask = labels == class_label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            selected = nms(class_boxes, class_scores, nms_iou_threshold)
            kept_indices.append(class_indices[selected])

        if len(kept_indices) > 0:
            kept_indices = torch.cat(kept_indices, dim=0)
            scores = scores[kept_indices]
            labels = labels[kept_indices]
            boxes = boxes[kept_indices]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

    if use_class_containment_suppression and scores.numel() > 1:
        kept_indices = _apply_same_label_containment_suppression(
            boxes=boxes,
            scores=scores,
            labels=labels,
            containment_threshold=class_containment_threshold,
        )
        scores = scores[kept_indices]
        labels = labels[kept_indices]
        boxes = boxes[kept_indices]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)
    _record_stage(stage_debug, "post_class_nms", image_id, image_size, scores, labels, boxes)

    if postprocess_topk_stage == "post_class_nms":
        scores, labels, boxes = _apply_topk(scores, labels, boxes, topk_per_image)
    _record_stage(stage_debug, "post_topk_post_class_nms", image_id, image_size, scores, labels, boxes)

    if use_agnostic_nms and scores.numel() > 1:
        kept_indices = _apply_agnostic_suppression(
            boxes=boxes,
            scores=scores,
            agnostic_nms_iou_threshold=agnostic_nms_iou_threshold,
            agnostic_containment_threshold=agnostic_containment_threshold,
        )
        scores = scores[kept_indices]
        labels = labels[kept_indices]
        boxes = boxes[kept_indices]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)
    _record_stage(stage_debug, "post_agnostic_nms", image_id, image_size, scores, labels, boxes)

    if use_cross_class_overlap_suppression and scores.numel() > 1:
        kept_indices = _apply_cross_class_overlap_suppression(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=cross_class_overlap_iou_threshold,
            containment_threshold=cross_class_overlap_containment_threshold,
            score_margin=cross_class_overlap_score_margin,
        )
        scores = scores[kept_indices]
        labels = labels[kept_indices]
        boxes = boxes[kept_indices]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)
    _record_stage(stage_debug, "post_cross_class_clean", image_id, image_size, scores, labels, boxes)

    if postprocess_topk_stage == "final":
        scores, labels, boxes = _apply_topk(scores, labels, boxes, topk_per_image)
    _record_stage(stage_debug, "post_topk_final", image_id, image_size, scores, labels, boxes)

    if postprocess_topk_stage == "pre_class_nms":
        stage_debug["post_topk"] = stage_debug["post_topk_pre_class_nms"]
    elif postprocess_topk_stage == "post_class_nms":
        stage_debug["post_topk"] = stage_debug["post_topk_post_class_nms"]
    else:
        stage_debug["post_topk"] = stage_debug["post_topk_final"]
    _record_stage(stage_debug, "final", image_id, image_size, scores, labels, boxes)

    return {
        "predictions": stage_debug["final"],
        "debug": stage_debug,
    }


def _should_use_official_rtdetr_postprocess(
    model_backend: str | None,
    use_official_backend_postprocess: bool,
) -> bool:
    return bool(use_official_backend_postprocess) and _is_hf_rtdetr_backend(model_backend)


def _get_official_rtdetr_image_processor():
    global _RTDETR_IMAGE_PROCESSOR
    if RTDetrImageProcessor is None:
        return None
    if _RTDETR_IMAGE_PROCESSOR is None:
        _RTDETR_IMAGE_PROCESSOR = RTDetrImageProcessor()
    return _RTDETR_IMAGE_PROCESSOR


def _postprocess_single_image_rtdetr_official(
    raw_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target: dict,
    score_threshold: float,
):
    image_id = int(target["image_id"].item())
    output_size = target.get("orig_size", target["size"]).float()
    image_size = output_size.tolist()

    processor = _get_official_rtdetr_image_processor()
    if processor is not None:
        official_results = processor.post_process_object_detection(
            outputs=SimpleNamespace(
                logits=raw_logits.unsqueeze(0).float(),
                pred_boxes=pred_boxes.unsqueeze(0).float(),
            ),
            threshold=float(score_threshold),
            target_sizes=output_size.unsqueeze(0),
            use_focal_loss=True,
        )[0]
        scores = official_results["scores"]
        labels = official_results["labels"]
        boxes = official_results["boxes"]
    else:
        # Fallback clone of transformers RTDetrImageProcessor.post_process_object_detection().
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        boxes = rescale_boxes_to_pixels(boxes, output_size.unsqueeze(0)).squeeze(0)
        num_top_queries = int(raw_logits.shape[0])
        num_classes = int(raw_logits.shape[1])
        scores = torch.sigmoid(raw_logits.float())
        scores, index = torch.topk(scores.flatten(), k=min(num_top_queries, scores.numel()), dim=-1)
        labels = index % num_classes
        query_index = index // num_classes
        boxes = boxes[query_index]
        keep = scores > float(score_threshold)
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

    stage_debug = {
        "_meta": {
            "score_threshold": float(score_threshold),
            "source": "transformers.RTDetrImageProcessor.post_process_object_detection"
            if processor is not None
            else "fallback_clone_of_transformers_rt_detr_postprocess",
            "topk_stage": "official",
        }
    }
    _record_stage(stage_debug, "final", image_id, image_size, scores, labels, boxes)

    return {
        "predictions": stage_debug["final"],
        "debug": stage_debug,
    }


def _postprocess_single_image_rtdetr_official_local_clone(
    raw_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target: dict,
    score_threshold: float,
):
    image_id = int(target["image_id"].item())
    output_size = target.get("orig_size", target["size"]).float()
    image_size = output_size.tolist()

    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = rescale_boxes_to_pixels(boxes, output_size.unsqueeze(0)).squeeze(0)

    num_top_queries = int(raw_logits.shape[0])
    num_classes = int(raw_logits.shape[1])
    scores = torch.sigmoid(raw_logits.float())
    scores, index = torch.topk(scores.flatten(), k=min(num_top_queries, scores.numel()), dim=-1)
    labels = index % num_classes
    query_index = index // num_classes
    boxes = boxes[query_index]
    keep = scores > float(score_threshold)
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    scores, labels, boxes = _sort_predictions_by_score(scores, labels, boxes)

    stage_debug = {
        "_meta": {
            "score_threshold": float(score_threshold),
            "source": "official_local_clone",
            "topk_stage": "official",
        }
    }
    _record_stage(stage_debug, "final", image_id, image_size, scores, labels, boxes)
    return {
        "predictions": stage_debug["final"],
        "debug": stage_debug,
    }


def collect_coco_predictions(
    outputs,
    targets,
    model_backend=None,
    use_official_backend_postprocess=False,
    rtdetr_postprocess_variant="official_hf",
    score_threshold=0.20,
    topk_per_image=10,
    postprocess_topk_stage="pre_class_nms",
    class_logit_bias=None,
    use_nms=False,
    nms_iou_threshold=0.5,
    use_class_containment_suppression=False,
    class_containment_threshold=0.9,
    use_agnostic_nms=False,
    agnostic_nms_iou_threshold=0.7,
    agnostic_containment_threshold=0.85,
    use_cross_class_overlap_suppression=False,
    cross_class_overlap_iou_threshold=0.9,
    cross_class_overlap_containment_threshold=0.95,
    cross_class_overlap_score_margin=0.08,
):
    """Convert model outputs into COCO-style prediction dicts.

    RT-DETRv2 can optionally use either the direct Hugging Face postprocess or
    the local clone of the same official logic. Other backends keep using the
    project's original postprocess path.
    """
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    pred_objectness_logits = outputs.get("pred_objectness_logits")
    pred_quality_scores = outputs.get("pred_quality_scores")

    predictions = []
    batch_size = pred_logits.shape[0]
    use_official_postprocess = _should_use_official_rtdetr_postprocess(
        model_backend=model_backend,
        use_official_backend_postprocess=use_official_backend_postprocess,
    )

    for batch_idx in range(batch_size):
        if use_official_postprocess:
            variant = str(rtdetr_postprocess_variant or "official_hf").lower()
            if variant == "official_local_clone":
                postprocess = _postprocess_single_image_rtdetr_official_local_clone(
                    raw_logits=pred_logits[batch_idx],
                    pred_boxes=pred_boxes[batch_idx],
                    target=targets[batch_idx],
                    score_threshold=score_threshold,
                )
            else:
                postprocess = _postprocess_single_image_rtdetr_official(
                    raw_logits=pred_logits[batch_idx],
                    pred_boxes=pred_boxes[batch_idx],
                    target=targets[batch_idx],
                    score_threshold=score_threshold,
                )
        else:
            postprocess = _postprocess_single_image(
                raw_logits=pred_logits[batch_idx],
                pred_objectness_logits=None if pred_objectness_logits is None else pred_objectness_logits[batch_idx],
                pred_quality_scores=None if pred_quality_scores is None else pred_quality_scores[batch_idx],
                pred_boxes=pred_boxes[batch_idx],
                target=targets[batch_idx],
                score_threshold=score_threshold,
                topk_per_image=topk_per_image,
                postprocess_topk_stage=postprocess_topk_stage,
                class_logit_bias=class_logit_bias,
                use_nms=use_nms,
                nms_iou_threshold=nms_iou_threshold,
                use_class_containment_suppression=use_class_containment_suppression,
                class_containment_threshold=class_containment_threshold,
                use_agnostic_nms=use_agnostic_nms,
                agnostic_nms_iou_threshold=agnostic_nms_iou_threshold,
                agnostic_containment_threshold=agnostic_containment_threshold,
                use_cross_class_overlap_suppression=use_cross_class_overlap_suppression,
                cross_class_overlap_iou_threshold=cross_class_overlap_iou_threshold,
                cross_class_overlap_containment_threshold=cross_class_overlap_containment_threshold,
                cross_class_overlap_score_margin=cross_class_overlap_score_margin,
            )
        predictions.extend(postprocess["predictions"])

    return predictions


def collect_coco_predictions_debug(
    outputs,
    targets,
    model_backend=None,
    use_official_backend_postprocess=False,
    rtdetr_postprocess_variant="official_hf",
    score_threshold=0.20,
    topk_per_image=10,
    postprocess_topk_stage="pre_class_nms",
    class_logit_bias=None,
    use_nms=False,
    nms_iou_threshold=0.5,
    use_class_containment_suppression=False,
    class_containment_threshold=0.9,
    use_agnostic_nms=False,
    agnostic_nms_iou_threshold=0.7,
    agnostic_containment_threshold=0.85,
    use_cross_class_overlap_suppression=False,
    cross_class_overlap_iou_threshold=0.9,
    cross_class_overlap_containment_threshold=0.95,
    cross_class_overlap_score_margin=0.08,
):
    """Return final predictions plus intermediate postprocess stages for debugging."""
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    pred_objectness_logits = outputs.get("pred_objectness_logits")
    pred_quality_scores = outputs.get("pred_quality_scores")

    predictions = []
    debug_by_image = {}
    batch_size = pred_logits.shape[0]
    use_official_postprocess = _should_use_official_rtdetr_postprocess(
        model_backend=model_backend,
        use_official_backend_postprocess=use_official_backend_postprocess,
    )

    for batch_idx in range(batch_size):
        image_id = int(targets[batch_idx]["image_id"].item())
        if use_official_postprocess:
            variant = str(rtdetr_postprocess_variant or "official_hf").lower()
            if variant == "official_local_clone":
                postprocess = _postprocess_single_image_rtdetr_official_local_clone(
                    raw_logits=pred_logits[batch_idx],
                    pred_boxes=pred_boxes[batch_idx],
                    target=targets[batch_idx],
                    score_threshold=score_threshold,
                )
            else:
                postprocess = _postprocess_single_image_rtdetr_official(
                    raw_logits=pred_logits[batch_idx],
                    pred_boxes=pred_boxes[batch_idx],
                    target=targets[batch_idx],
                    score_threshold=score_threshold,
                )
        else:
            postprocess = _postprocess_single_image(
                raw_logits=pred_logits[batch_idx],
                pred_objectness_logits=None if pred_objectness_logits is None else pred_objectness_logits[batch_idx],
                pred_quality_scores=None if pred_quality_scores is None else pred_quality_scores[batch_idx],
                pred_boxes=pred_boxes[batch_idx],
                target=targets[batch_idx],
                score_threshold=score_threshold,
                topk_per_image=topk_per_image,
                postprocess_topk_stage=postprocess_topk_stage,
                class_logit_bias=class_logit_bias,
                use_nms=use_nms,
                nms_iou_threshold=nms_iou_threshold,
                use_class_containment_suppression=use_class_containment_suppression,
                class_containment_threshold=class_containment_threshold,
                use_agnostic_nms=use_agnostic_nms,
                agnostic_nms_iou_threshold=agnostic_nms_iou_threshold,
                agnostic_containment_threshold=agnostic_containment_threshold,
                use_cross_class_overlap_suppression=use_cross_class_overlap_suppression,
                cross_class_overlap_iou_threshold=cross_class_overlap_iou_threshold,
                cross_class_overlap_containment_threshold=cross_class_overlap_containment_threshold,
                cross_class_overlap_score_margin=cross_class_overlap_score_margin,
            )
        predictions.extend(postprocess["predictions"])
        debug_by_image[image_id] = postprocess["debug"]

    return predictions, debug_by_image


def evaluate_coco_map(ann_path: str, predictions: list[dict]) -> dict:
    """Evaluate predictions with pycocotools if available."""
    metrics = {"AP": 0.0, "AP50": 0.0}
    if ann_path is None or (not os.path.exists(ann_path)):
        return metrics

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools is not installed. Skip COCO mAP evaluation.")
        return metrics

    coco_gt = COCO(ann_path)
    if len(predictions) == 0:
        return metrics

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        json.dump(predictions, tmp_file)
        tmp_pred_path = tmp_file.name

    try:
        coco_dt = coco_gt.loadRes(tmp_pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics["AP"] = float(coco_eval.stats[0])
        metrics["AP50"] = float(coco_eval.stats[1])
    finally:
        if os.path.exists(tmp_pred_path):
            os.remove(tmp_pred_path)

    return metrics


def save_checkpoint(
    checkpoint_path,
    epoch_idx,
    model,
    ema_model,
    optimizer,
    scheduler,
    best_map,
    best_val_loss,
    history,
    epochs_no_improve,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None if ema_model is None else ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_map": best_map,
            "best_val_loss": best_val_loss,
            "history": history,
            "epochs_no_improve": epochs_no_improve,
        },
        checkpoint_path,
    )
