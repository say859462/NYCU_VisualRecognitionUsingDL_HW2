"""Inference and validation entry point."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import DetectionDataset, DetectionTransform, collate_fn
from model import adapt_checkpoint_state_dict, build_model_from_config
from utils import (
    build_inference_postprocess_kwargs,
    collect_coco_predictions,
    evaluate_coco_map,
    merge_coco_prediction_sets,
    parse_tta_scales,
)

HF_RTDETR_BACKENDS = {"hf_rtdetr_v2", "hf_rtdetr_v2_aux"}


def get_pad_size_divisor(config: dict) -> int:
    """Use RT-DETR-friendly padding for Hugging Face backends."""
    model_backend = str(config.get("model_backend", "")).lower()
    default_divisor = 32 if model_backend in HF_RTDETR_BACKENDS else 1
    return int(config.get("pad_size_divisor", default_divisor))


def parse_optional_bool(value: str | bool | None) -> bool | None:
    """Parse a CLI-style boolean override while preserving None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_optional_json(value: str | None):
    """Parse a JSON CLI override while preserving None."""
    if value is None:
        return None
    return json.loads(value)


def build_loader(
    image_dir: str,
    annotation_path: str | None,
    split: str,
    max_image_size: int,
    fixed_image_size: list[int] | tuple[int, int] | None,
    allow_upscale: bool,
    max_upscale_ratio: float,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool = True,
    pad_size_divisor: int = 1,
) -> DataLoader:
    """Build an evaluation dataloader with deterministic transforms."""
    dataset = DetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        split=split,
        transform=DetectionTransform(
            split=split,
            max_image_size=max_image_size,
            fixed_image_size=fixed_image_size,
            use_color_jitter=False,
            use_random_grayscale=False,
            use_gaussian_blur=False,
            allow_upscale=allow_upscale,
            max_upscale_ratio=max_upscale_ratio,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_fn, pad_size_divisor=pad_size_divisor),
    )


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register evaluation-only CLI overrides."""
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--topk_per_image", type=int, default=None)
    parser.add_argument("--postprocess_topk_stage", type=str, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--class_score_thresholds", type=str, default=None)
    parser.add_argument("--use_nms", type=str, default=None)
    parser.add_argument("--nms_iou_threshold", type=float, default=None)
    parser.add_argument(
        "--use_class_containment_suppression",
        type=str,
        default=None,
    )
    parser.add_argument("--class_containment_threshold", type=float, default=None)
    parser.add_argument("--use_agnostic_nms", type=str, default=None)
    parser.add_argument("--agnostic_nms_iou_threshold", type=float, default=None)
    parser.add_argument("--agnostic_containment_threshold", type=float, default=None)
    parser.add_argument("--tta_scales", type=str, default=None)
    parser.add_argument("--tta_merge_method", type=str, default=None)
    parser.add_argument("--tta_wbf_iou_threshold", type=float, default=None)
    parser.add_argument("--tta_wbf_skip_box_threshold", type=float, default=None)
    parser.add_argument(
        "--aux_digit_classifier_fusion_weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use_aux_digit_classifier_gated_fusion",
        type=str,
        default=None,
    )
    parser.add_argument("--aux_digit_gate_top1_threshold", type=float, default=None)
    parser.add_argument("--aux_digit_gate_margin_threshold", type=float, default=None)
    parser.add_argument(
        "--use_aux_digit_confusion_family_selective_fusion",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_aux_digit_confusion_family_attenuation",
        type=str,
        default=None,
    )
    parser.add_argument("--aux_digit_confusion_families", type=str, default=None)
    parser.add_argument("--aux_digit_family_fusion_weights", type=str, default=None)
    parser.add_argument(
        "--aux_digit_family_attenuation_weights",
        type=str,
        default=None,
    )


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply non-postprocess CLI overrides directly to the loaded config."""
    float_overrides = {
        "aux_digit_classifier_fusion_weight": args.aux_digit_classifier_fusion_weight,
        "aux_digit_gate_top1_threshold": args.aux_digit_gate_top1_threshold,
        "aux_digit_gate_margin_threshold": args.aux_digit_gate_margin_threshold,
    }
    for key, value in float_overrides.items():
        if value is not None:
            config[key] = float(value)

    bool_overrides = {
        "use_aux_digit_classifier_gated_fusion": (
            args.use_aux_digit_classifier_gated_fusion
        ),
        "use_aux_digit_confusion_family_selective_fusion": (
            args.use_aux_digit_confusion_family_selective_fusion
        ),
        "use_aux_digit_confusion_family_attenuation": (
            args.use_aux_digit_confusion_family_attenuation
        ),
    }
    for key, value in bool_overrides.items():
        parsed = parse_optional_bool(value)
        if parsed is not None:
            config[key] = parsed

    json_overrides = {
        "aux_digit_confusion_families": args.aux_digit_confusion_families,
        "aux_digit_family_fusion_weights": args.aux_digit_family_fusion_weights,
        "aux_digit_family_attenuation_weights": (
            args.aux_digit_family_attenuation_weights
        ),
    }
    for key, value in json_overrides.items():
        parsed = parse_optional_json(value)
        if parsed is not None:
            config[key] = parsed

    return config


def resolve_eval_settings(
    config: dict,
    split: str,
    default_num_workers: int,
) -> tuple[int, bool, int, str, str | None]:
    """Resolve split-specific loader and dataset settings."""
    batch_size = int(
        config.get(
            f"{split}_batch_size",
            config.get("eval_batch_size", config["batch_size"]),
        )
    )
    pin_memory = bool(
        config.get(
            f"{split}_pin_memory",
            config.get("eval_pin_memory", config.get("pin_memory", True)),
        )
    )
    num_workers = int(
        config.get(
            f"{split}_num_workers",
            config.get("eval_num_workers", default_num_workers),
        )
    )
    image_dir = config[f"{split}_image_dir"]
    annotation_path = config.get(f"{split}_ann_path")
    if annotation_path is not None and not os.path.exists(annotation_path):
        annotation_path = None
    return batch_size, pin_memory, num_workers, image_dir, annotation_path


def apply_postprocess_overrides(
    inference_kwargs: dict,
    args: argparse.Namespace,
) -> dict:
    """Apply CLI overrides to the postprocess settings."""
    if args.score_threshold is not None:
        inference_kwargs["score_threshold"] = float(args.score_threshold)
    if args.class_score_thresholds is not None:
        inference_kwargs["class_score_thresholds"] = parse_optional_json(
            args.class_score_thresholds
        )
    if args.topk_per_image is not None:
        inference_kwargs["topk_per_image"] = int(args.topk_per_image)
    if args.postprocess_topk_stage is not None:
        inference_kwargs["postprocess_topk_stage"] = str(
            args.postprocess_topk_stage
        )
    if args.nms_iou_threshold is not None:
        inference_kwargs["nms_iou_threshold"] = float(args.nms_iou_threshold)
    if args.class_containment_threshold is not None:
        inference_kwargs["class_containment_threshold"] = float(
            args.class_containment_threshold
        )
    if args.agnostic_nms_iou_threshold is not None:
        inference_kwargs["agnostic_nms_iou_threshold"] = float(
            args.agnostic_nms_iou_threshold
        )
    if args.agnostic_containment_threshold is not None:
        inference_kwargs["agnostic_containment_threshold"] = float(
            args.agnostic_containment_threshold
        )

    bool_overrides = {
        "use_nms": args.use_nms,
        "use_class_containment_suppression": (
            args.use_class_containment_suppression
        ),
        "use_agnostic_nms": args.use_agnostic_nms,
    }
    for key, value in bool_overrides.items():
        parsed = parse_optional_bool(value)
        if parsed is not None:
            inference_kwargs[key] = parsed

    return inference_kwargs


def _scale_fixed_image_size(
    fixed_image_size: list[int] | tuple[int, int] | None,
    scale: float,
) -> tuple[int, int] | None:
    if fixed_image_size is None:
        return None
    target_height = max(1, int(round(float(fixed_image_size[0]) * float(scale))))
    target_width = max(1, int(round(float(fixed_image_size[1]) * float(scale))))
    return target_height, target_width


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation/test inference")
    add_cli_arguments(parser)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    config = apply_cli_overrides(config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    split = args.split
    model_path = (
        args.model_path
        if args.model_path is not None
        else config["best_model_path"]
    )
    output_json = (
        args.output_json if args.output_json is not None else config["prediction_path"]
    )

    batch_size, pin_memory, num_workers, image_dir, annotation_path = (
        resolve_eval_settings(config, split, args.num_workers)
    )

    model = build_model_from_config(
        config,
        pretrained_backbone_override=False,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    incompatible = model.load_state_dict(
        adapt_checkpoint_state_dict(model, state_dict),
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Checkpoint compatibility | missing={len(incompatible.missing_keys)} "
            f"| unexpected={len(incompatible.unexpected_keys)}"
        )
    model.eval()

    base_max_image_size = int(
        config.get("inference_max_image_size", config.get("max_image_size", 640))
    )
    base_fixed_image_size = config.get("inference_fixed_image_size")
    allow_upscale = bool(config.get("inference_allow_upscale", False))
    max_upscale_ratio = float(config.get("inference_max_upscale_ratio", 1.0))
    pad_size_divisor = get_pad_size_divisor(config)

    base_loader = build_loader(
        image_dir=image_dir,
        annotation_path=annotation_path,
        split=split,
        max_image_size=base_max_image_size,
        fixed_image_size=base_fixed_image_size,
        allow_upscale=allow_upscale,
        max_upscale_ratio=max_upscale_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pad_size_divisor=pad_size_divisor,
    )

    inference_postprocess_kwargs = build_inference_postprocess_kwargs(config)
    inference_postprocess_kwargs = apply_postprocess_overrides(
        inference_postprocess_kwargs,
        args,
    )

    tta_enabled = bool(config.get("use_tta", False))
    tta_scales = parse_tta_scales(
        args.tta_scales
        if args.tta_scales is not None
        else config.get("tta_scales", [1.0])
    )
    if not tta_enabled:
        tta_scales = [1.0]

    tta_merge_method = str(
        args.tta_merge_method
        if args.tta_merge_method is not None
        else config.get("tta_merge_method", "nms")
    ).lower()
    tta_wbf_iou_threshold = float(
        args.tta_wbf_iou_threshold
        if args.tta_wbf_iou_threshold is not None
        else config.get("tta_wbf_iou_threshold", 0.65)
    )
    raw_wbf_skip_threshold = (
        args.tta_wbf_skip_box_threshold
        if args.tta_wbf_skip_box_threshold is not None
        else config.get("tta_wbf_skip_box_threshold")
    )
    tta_wbf_skip_box_threshold = (
        None
        if raw_wbf_skip_threshold is None
        else float(raw_wbf_skip_threshold)
    )

    image_sizes_by_id: dict[int, list[float] | tuple[float, float]] = {}
    prediction_sets: list[list[dict]] = []

    with torch.no_grad():
        for scale in tta_scales:
            scaled_max_image_size = max(
                32,
                int(round(base_max_image_size * float(scale))),
            )
            scaled_fixed_image_size = _scale_fixed_image_size(
                base_fixed_image_size,
                scale,
            )
            use_base_loader = (
                scaled_max_image_size == base_max_image_size
                and scaled_fixed_image_size == base_fixed_image_size
                and len(tta_scales) == 1
            )
            scale_loader = base_loader if use_base_loader else build_loader(
                image_dir=image_dir,
                annotation_path=annotation_path,
                split=split,
                max_image_size=scaled_max_image_size,
                fixed_image_size=scaled_fixed_image_size,
                allow_upscale=allow_upscale,
                max_upscale_ratio=max_upscale_ratio,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                pad_size_divisor=pad_size_divisor,
            )
            if len(tta_scales) > 1:
                print(
                    f"Running TTA scale {scale:.2f} | "
                    f"max_image_size={scaled_max_image_size}"
                )

            scale_predictions: list[dict] = []
            for images, masks, targets in scale_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                for target in targets:
                    output_size = target.get("orig_size", target["size"])
                    image_sizes_by_id[int(target["image_id"].item())] = (
                        output_size.tolist()
                        if torch.is_tensor(output_size)
                        else list(output_size)
                    )

                targets = [
                    {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(images, masks)

                scale_predictions.extend(
                    collect_coco_predictions(
                        outputs=outputs,
                        targets=targets,
                        **inference_postprocess_kwargs,
                    )
                )
            prediction_sets.append(scale_predictions)

    if len(prediction_sets) == 1:
        predictions = prediction_sets[0]
    else:
        predictions = merge_coco_prediction_sets(
            prediction_sets=prediction_sets,
            image_sizes_by_id=image_sizes_by_id,
            tta_merge_method=tta_merge_method,
            tta_wbf_iou_threshold=tta_wbf_iou_threshold,
            tta_wbf_skip_box_threshold=tta_wbf_skip_box_threshold,
            **inference_postprocess_kwargs,
        )

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(predictions, file)

    print(f"Prediction JSON saved to: {output_json}")
    if split == "val" and annotation_path is not None:
        metrics = evaluate_coco_map(
            ann_path=annotation_path,
            predictions=predictions,
        )
        print(f"Validation AP: {metrics.get('AP', 0.0):.4f}")
        print(f"Validation AP50: {metrics.get('AP50', 0.0):.4f}")


if __name__ == "__main__":
    main()
