"""Validation loop helpers."""

from __future__ import annotations

import torch
from tqdm import tqdm

from utils import collect_coco_predictions, evaluate_coco_map, resolve_training_loss


def _loss_item(loss_dict: dict, key: str, device: torch.device) -> float:
    """Read a loss term as float while tolerating missing optional keys."""
    default = torch.tensor(0.0, device=device)
    return float(loss_dict.get(key, default).item())


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    ann_path,
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
    use_amp=True,
):
    """Validate one epoch and compute COCO-style mAP."""
    model.eval()
    if criterion is not None:
        criterion.eval()

    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    running_loss_objectness = 0.0
    running_loss_targeted_confusion = 0.0
    total_batches = 0
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for images, masks, targets in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = [
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]

            amp_enabled = bool(use_amp) and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images, masks, targets)
                loss, loss_dict = resolve_training_loss(outputs, targets, criterion)

            if not torch.isfinite(loss):
                print("[val] non-finite loss detected; skip batch")
                continue

            running_loss += loss.item()
            running_loss_ce += loss_dict["loss_ce"].item()
            running_loss_bbox += loss_dict["loss_bbox"].item()
            running_loss_giou += loss_dict["loss_giou"].item()
            running_loss_objectness += _loss_item(
                loss_dict,
                "loss_objectness",
                loss.device,
            )
            running_loss_targeted_confusion += _loss_item(
                loss_dict,
                "loss_targeted_confusion",
                loss.device,
            )
            total_batches += 1

            batch_predictions = collect_coco_predictions(
                outputs=outputs,
                targets=targets,
                model_backend=model_backend,
                use_official_backend_postprocess=use_official_backend_postprocess,
                rtdetr_postprocess_variant=rtdetr_postprocess_variant,
                score_threshold=score_threshold,
                topk_per_image=topk_per_image,
                postprocess_topk_stage=postprocess_topk_stage,
                class_logit_bias=class_logit_bias,
                use_nms=use_nms,
                nms_iou_threshold=nms_iou_threshold,
                use_class_containment_suppression=(
                    use_class_containment_suppression
                ),
                class_containment_threshold=class_containment_threshold,
                use_agnostic_nms=use_agnostic_nms,
                agnostic_nms_iou_threshold=agnostic_nms_iou_threshold,
                agnostic_containment_threshold=agnostic_containment_threshold,
                use_cross_class_overlap_suppression=(
                    use_cross_class_overlap_suppression
                ),
                cross_class_overlap_iou_threshold=(
                    cross_class_overlap_iou_threshold
                ),
                cross_class_overlap_containment_threshold=(
                    cross_class_overlap_containment_threshold
                ),
                cross_class_overlap_score_margin=(
                    cross_class_overlap_score_margin
                ),
            )
            all_predictions.extend(batch_predictions)

    coco_metrics = evaluate_coco_map(ann_path=ann_path, predictions=all_predictions)
    normalizer = max(1, total_batches)
    return {
        "loss": running_loss / normalizer,
        "loss_ce": running_loss_ce / normalizer,
        "loss_bbox": running_loss_bbox / normalizer,
        "loss_giou": running_loss_giou / normalizer,
        "loss_objectness": running_loss_objectness / normalizer,
        "loss_targeted_confusion": running_loss_targeted_confusion / normalizer,
        "map": coco_metrics.get("AP"),
        "map50": coco_metrics.get("AP50"),
        "predictions": all_predictions,
    }
