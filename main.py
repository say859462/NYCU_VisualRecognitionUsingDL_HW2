"""Training entry point for the active detection experiments."""

import argparse
from functools import partial
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    AspectRatioBatchSampler,
    DetectionDataset,
    DetectionTransform,
    collate_fn,
)
from model import (
    HungarianMatcher,
    SetCriterion,
    adapt_checkpoint_state_dict,
    build_model_from_config,
)
from train import train_one_epoch
from utils import (
    ModelEMA,
    WarmUpCosineAnnealingLR,
    build_inference_postprocess_kwargs,
    plot_training_curves,
    save_checkpoint,
)
from val import validate_one_epoch

HF_RTDETR_BACKENDS = {"hf_rtdetr_v2", "hf_rtdetr_v2_aux", "hf_rtdetr_v2_qs"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path when needed."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def get_pad_size_divisor(config: dict) -> int:
    """Use RT-DETR-friendly padding for Hugging Face backends."""
    model_backend = str(config.get("model_backend", "")).lower()
    default_divisor = 32 if model_backend in HF_RTDETR_BACKENDS else 1
    return int(config.get("pad_size_divisor", default_divisor))


def build_train_detection_transform(config: dict) -> DetectionTransform:
    """Build the standard train-time transform from config."""
    use_dynamic_multi_scale = bool(config.get("use_dynamic_multi_scale", False))
    train_max_image_size = config.get(
        "train_max_image_size",
        config.get("max_image_size", 640),
    )
    train_image_size_setting = (
        config.get("train_max_image_sizes", train_max_image_size)
        if use_dynamic_multi_scale
        else train_max_image_size
    )

    return DetectionTransform(
        split="train",
        max_image_size=train_image_size_setting,
        fixed_image_size=config.get("train_fixed_image_size"),
        fixed_image_sizes=config.get("train_fixed_image_sizes"),
        use_color_jitter=bool(config.get("use_color_jitter", False)),
        brightness_jitter=float(config.get("brightness_jitter", 0.10)),
        contrast_jitter=float(config.get("contrast_jitter", 0.10)),
        saturation_jitter=float(config.get("saturation_jitter", 0.05)),
        hue_jitter=float(config.get("hue_jitter", 0.0)),
        use_random_grayscale=bool(config.get("use_random_grayscale", False)),
        random_grayscale_prob=float(config.get("random_grayscale_prob", 0.0)),
        use_gamma_aug=bool(config.get("use_gamma_aug", False)),
        gamma_min=float(config.get("gamma_min", 0.8)),
        gamma_max=float(config.get("gamma_max", 1.2)),
        use_gaussian_blur=bool(config.get("use_gaussian_blur", False)),
        gaussian_blur_prob=float(config.get("gaussian_blur_prob", 0.15)),
        gaussian_blur_radius=float(config.get("gaussian_blur_radius", 0.5)),
        use_light_affine_aug=bool(config.get("use_light_affine_aug", False)),
        max_rotation_degree=float(config.get("max_rotation_degree", 5.0)),
        max_translation_ratio=float(config.get("max_translation_ratio", 0.02)),
        min_scale=float(config.get("min_affine_scale", 0.95)),
        max_scale=float(config.get("max_affine_scale", 1.05)),
        max_shear_degree=float(config.get("max_shear_degree", 4.0)),
        use_position_debias_aug=bool(config.get("use_position_debias_aug", False)),
        max_canvas_shift_ratio=float(config.get("max_canvas_shift_ratio", 0.10)),
        use_horizontal_layout_shift_aug=bool(
            config.get("use_horizontal_layout_shift_aug", False)
        ),
        horizontal_layout_shift_prob=float(
            config.get("horizontal_layout_shift_prob", 0.5)
        ),
        max_horizontal_layout_shift_ratio=float(
            config.get("max_horizontal_layout_shift_ratio", 0.10)
        ),
        horizontal_layout_shift_allow_truncation=bool(
            config.get("horizontal_layout_shift_allow_truncation", False)
        ),
        use_mild_truncation_aug=bool(config.get("use_mild_truncation_aug", True)),
        truncation_prob=float(config.get("truncation_prob", 0.15)),
        enable_dynamic_multi_scale=use_dynamic_multi_scale,
        dynamic_multi_scale_max_upscale_ratio=float(
            config.get("dynamic_multi_scale_max_upscale_ratio", 1.0)
        ),
        use_synthetic_digit_style_aug=bool(
            config.get("use_synthetic_digit_style_aug", False)
        ),
        synthetic_digit_style_prob=float(
            config.get("synthetic_digit_style_prob", 0.35)
        ),
        synthetic_digit_style_max_regions=int(
            config.get("synthetic_digit_style_max_regions", 2)
        ),
        synthetic_digit_style_min_scale=float(
            config.get("synthetic_digit_style_min_scale", 0.85)
        ),
        synthetic_digit_style_max_scale=float(
            config.get("synthetic_digit_style_max_scale", 1.20)
        ),
        synthetic_digit_style_max_shift_ratio=float(
            config.get("synthetic_digit_style_max_shift_ratio", 0.10)
        ),
        synthetic_digit_style_gamma_min=float(
            config.get("synthetic_digit_style_gamma_min", 0.75)
        ),
        synthetic_digit_style_gamma_max=float(
            config.get("synthetic_digit_style_gamma_max", 1.25)
        ),
        synthetic_digit_style_blur_prob=float(
            config.get("synthetic_digit_style_blur_prob", 0.25)
        ),
        synthetic_digit_style_blur_radius=float(
            config.get("synthetic_digit_style_blur_radius", 0.7)
        ),
    )


def build_eval_detection_transform(config: dict, split: str) -> DetectionTransform:
    """Build the deterministic validation transform from config."""
    return DetectionTransform(
        split=split,
        max_image_size=config.get(
            "inference_max_image_size",
            config.get("max_image_size", 640),
        ),
        fixed_image_size=config.get("inference_fixed_image_size"),
        use_color_jitter=False,
        use_random_grayscale=False,
        use_gaussian_blur=False,
        allow_upscale=bool(config.get("inference_allow_upscale", False)),
        max_upscale_ratio=float(config.get("inference_max_upscale_ratio", 1.0)),
    )


def build_optimizer(model, config):
    return optim.AdamW(
        model.get_parameter_groups(
            lr_base=float(config.get("learning_rate", 1e-4)),
            lr_backbone=float(config.get("backbone_learning_rate", 1e-5)),
        ),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, config, steps_per_epoch: int):
    scheduler_type = str(config.get("scheduler_type", "warmup_cosine")).lower()
    total_steps = int(config["num_epochs"]) * max(1, int(steps_per_epoch))

    if scheduler_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                float(config.get("backbone_learning_rate", 1e-5)),
                float(config.get("learning_rate", 1e-4)),
            ],
            total_steps=total_steps,
            pct_start=float(config.get("onecycle_pct_start", 0.2)),
            anneal_strategy=str(config.get("onecycle_anneal_strategy", "cos")),
            div_factor=float(config.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(config.get("onecycle_final_div_factor", 1e4)),
        )

    warmup_steps = int(config.get("warmup_epochs", 3)) * max(1, int(steps_per_epoch))
    return WarmUpCosineAnnealingLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=float(config.get("eta_min", 1e-6)),
    )


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    batch_sampler=None,
    generator=None,
    pin_memory=True,
    persistent_workers=None,
    pad_size_divisor: int = 1,
):
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(persistent_workers) and num_workers > 0,
            collate_fn=partial(collate_fn, pad_size_divisor=pad_size_divisor),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers) and num_workers > 0,
        collate_fn=partial(collate_fn, pad_size_divisor=pad_size_divisor),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for output_path in (
        config["checkpoint_path"],
        config["best_model_path"],
        config["best_loss_model_path"],
        config["training_curve_path"],
    ):
        ensure_parent_dir(output_path)

    train_transform = build_train_detection_transform(config)
    val_transform = build_eval_detection_transform(config, split="val")

    train_dataset = DetectionDataset(
        image_dir=config["train_image_dir"],
        annotation_path=config["train_ann_path"],
        split="train",
        transform=train_transform,
    )
    val_dataset = DetectionDataset(
        image_dir=config["val_image_dir"],
        annotation_path=config["val_ann_path"],
        split="val",
        transform=val_transform,
    )

    train_batch_size = int(config.get("train_batch_size", config["batch_size"]))
    val_batch_size = int(
        config.get(
            "val_batch_size",
            config.get("eval_batch_size", config["batch_size"]),
        )
    )
    train_num_workers = int(
        config.get("train_num_workers", config.get("num_workers", 4))
    )
    val_num_workers = int(
        config.get(
            "val_num_workers",
            config.get("eval_num_workers", config.get("num_workers", 4)),
        )
    )
    train_pin_memory = bool(
        config.get("train_pin_memory", config.get("pin_memory", True))
    )
    val_pin_memory = bool(
        config.get(
            "val_pin_memory",
            config.get("eval_pin_memory", config.get("pin_memory", True)),
        )
    )
    train_persistent_workers = bool(
        config.get("train_persistent_workers", train_num_workers > 0)
    )
    val_persistent_workers = bool(
        config.get(
            "val_persistent_workers",
            config.get("eval_persistent_workers", val_num_workers > 0),
        )
    )

    train_batch_sampler = None
    if bool(config.get("use_bucket_sampler", False)):
        train_max_image_size = config.get(
            "train_max_image_size",
            config.get("max_image_size", 640),
        )
        train_image_size_setting = (
            config.get("train_max_image_sizes", train_max_image_size)
            if bool(config.get("use_dynamic_multi_scale", False))
            else train_max_image_size
        )
        train_batch_sampler = AspectRatioBatchSampler(
            dataset=train_dataset,
            batch_size=train_batch_size,
            drop_last=bool(config.get("bucket_drop_last", False)),
            num_bins=int(config.get("bucket_num_bins", 4)),
            bucket_mode=str(config.get("bucket_mode", "aspect_ratio")),
            max_image_size=train_image_size_setting,
            max_upscale_ratio=float(
                config.get("dynamic_multi_scale_max_upscale_ratio", 1.0)
            ),
        )

    model_backend = str(config.get("model_backend", "custom_relation_detr")).lower()
    pad_size_divisor = get_pad_size_divisor(config)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=train_batch_sampler is None,
        batch_sampler=train_batch_sampler,
        generator=dataloader_generator,
        pin_memory=train_pin_memory,
        persistent_workers=train_persistent_workers,
        pad_size_divisor=pad_size_divisor,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        shuffle=False,
        generator=dataloader_generator,
        pin_memory=val_pin_memory,
        persistent_workers=val_persistent_workers,
        pad_size_divisor=pad_size_divisor,
    )

    model = build_model_from_config(config).to(device)
    if (
        hasattr(model, "encoder")
        and hasattr(model.encoder, "layers")
        and len(model.encoder.layers) > 0
    ):
        encoder_attn = model.encoder.layers[0].self_attn
        if getattr(encoder_attn, "official_impl", None) is not None:
            print("Using official MSDeformAttn CUDA op.")
        else:
            status = getattr(
                encoder_attn,
                "msda_status",
                {"available": False, "error": "unknown"},
            )
            print(
                "Using PyTorch deformable attention fallback. "
                f"CUDA op status: {status.get('error', '')}"
            )
    else:
        print(
            "Using model backend: "
            f"{str(config.get('model_backend', 'custom_relation_detr'))}"
        )

    needs_extra_hf_losses = (
        abs(float(config.get("targeted_confusion_margin_loss_weight", 0.0))) > 1e-8
    )

    if model_backend == "hf_rtdetr_v2" and not needs_extra_hf_losses:
        criterion = None
    else:
        # Keep the matcher/criterion path for custom backends and for any
        # extra losses layered on top of the Hugging Face RT-DETR variants.
        matcher = HungarianMatcher(
            cost_class=float(config.get("set_cost_class", 1.0)),
            cost_bbox=float(config.get("set_cost_bbox", 5.0)),
            cost_giou=float(config.get("set_cost_giou", 2.0)),
            focal_alpha=float(config.get("focal_alpha", 0.25)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            iou_cost_type=str(config.get("matcher_iou_cost_type", "giou")),
            class_cost_type=str(config.get("matcher_class_cost_type", "auto")),
        )
        criterion = SetCriterion(
            num_classes=int(config["num_classes"]),
            matcher=matcher,
            loss_ce_weight=float(config.get("loss_ce_weight", 1.0)),
            loss_bbox_weight=float(config.get("loss_bbox_weight", 5.0)),
            loss_giou_weight=float(config.get("loss_giou_weight", 2.0)),
            enc_loss_weight=float(config.get("enc_loss_weight", 0.5)),
            focal_alpha=float(config.get("focal_alpha", 0.25)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            loss_objectness_weight=float(config.get("loss_objectness_weight", 1.0)),
            exp32_aux_positive_topk=int(config.get("exp32_aux_positive_topk", 0)),
            exp32_aux_positive_weight=float(
                config.get("exp32_aux_positive_weight", 0.0)
            ),
            exp32_aux_min_iou=float(config.get("exp32_aux_min_iou", 0.15)),
            box_iou_loss_type=str(config.get("box_iou_loss_type", "giou")),
            targeted_confusion_margin_loss_weight=float(
                config.get("targeted_confusion_margin_loss_weight", 0.0)
            ),
            targeted_confusion_margin_rules=config.get(
                "targeted_confusion_margin_rules",
                [],
            ),
            aux_digit_classifier_loss_weight=float(
                config.get("aux_digit_classifier_loss_weight", 0.0)
            ),
            query_quality_loss_weight=float(
                config.get("query_quality_loss_weight", 0.0)
            ),
        ).to(device)

    optimizer = build_optimizer(model, config)
    gradient_accumulation_steps = max(
        1,
        int(config.get("gradient_accumulation_steps", 1)),
    )
    steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / gradient_accumulation_steps),
    )
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema_model = None
    if bool(config.get("use_ema", False)):
        ema_model = ModelEMA(
            model=model,
            decay=float(config.get("ema_decay", 0.9999)),
            warmups=int(config.get("ema_warmups", 2000)),
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "val_loss_raw": [],
        "val_map_raw": [],
        "val_loss_ema": [],
        "val_map_ema": [],
    }
    start_epoch = 0
    best_map = -1.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    init_model_path = config.get("init_model_path")
    if init_model_path and os.path.exists(init_model_path):
        init_state_dict = torch.load(
            init_model_path,
            map_location=device,
            weights_only=False,
        )
        if isinstance(init_state_dict, dict) and "model_state_dict" in init_state_dict:
            init_state_dict = init_state_dict["model_state_dict"]
        elif isinstance(init_state_dict, dict) and "model" in init_state_dict:
            init_state_dict = init_state_dict["model"]
        adapted_init_state = adapt_checkpoint_state_dict(model, init_state_dict)
        model_state = model.state_dict()
        compatible_init_state = {}
        skipped_mismatch_keys = []
        skipped_prefix_keys = []
        init_skip_prefixes = tuple(
            str(prefix) for prefix in config.get("init_skip_prefixes", [])
            if str(prefix)
        )
        init_keep_prefixes = tuple(
            str(prefix) for prefix in config.get("init_keep_prefixes", [])
            if str(prefix)
        )
        for key, value in adapted_init_state.items():
            if init_keep_prefixes and not key.startswith(init_keep_prefixes):
                skipped_prefix_keys.append(key)
                continue
            if init_skip_prefixes and key.startswith(init_skip_prefixes):
                skipped_prefix_keys.append(key)
                continue
            if key not in model_state:
                compatible_init_state[key] = value
                continue
            target_value = model_state[key]
            if (
                hasattr(value, "shape")
                and hasattr(target_value, "shape")
                and value.shape != target_value.shape
            ):
                skipped_mismatch_keys.append(
                    (key, tuple(value.shape), tuple(target_value.shape))
                )
                continue
            compatible_init_state[key] = value

        incompatible = model.load_state_dict(compatible_init_state, strict=False)
        print(
            f"Initialized model from {init_model_path} | "
            f"missing={len(incompatible.missing_keys)} | "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        if skipped_mismatch_keys:
            preview = ", ".join(
                f"{key}: {src_shape}->{dst_shape}"
                for key, src_shape, dst_shape in skipped_mismatch_keys[:5]
            )
            print(
                f"Skipped {len(skipped_mismatch_keys)} mismatched init keys "
                f"(showing up to 5): {preview}"
            )
        if skipped_prefix_keys:
            preview = ", ".join(skipped_prefix_keys[:5])
            print(
                f"Skipped {len(skipped_prefix_keys)} init keys by prefix filter "
                f"(showing up to 5): {preview}"
            )

    if bool(config.get("resume_training", False)) and os.path.exists(
        config["checkpoint_path"]
    ):
        checkpoint = torch.load(
            config["checkpoint_path"],
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            adapt_checkpoint_state_dict(model, checkpoint["model_state_dict"])
        )
        resume_epoch = int(checkpoint["epoch"]) + 1
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        elif ema_model is not None:
            ema_model = ModelEMA(
                model=model,
                decay=float(config.get("ema_decay", 0.9999)),
                warmups=int(config.get("ema_warmups", 2000)),
            )
        start_epoch = resume_epoch
        best_map = float(checkpoint.get("best_map", -1.0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = checkpoint.get("history", history)
        history.setdefault("val_loss_raw", [])
        history.setdefault("val_map_raw", [])
        history.setdefault("val_loss_ema", [])
        history.setdefault("val_map_ema", [])
        epochs_no_improve = int(checkpoint.get("epochs_no_improve", 0))

    training_start_time = time.time()
    num_epochs = int(config["num_epochs"])
    early_stopping_patience = int(config.get("early_stopping_patience", 8))
    inference_postprocess_kwargs = build_inference_postprocess_kwargs(config)

    try:
        for epoch_idx in range(start_epoch, num_epochs):
            epoch = epoch_idx + 1
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            train_stats = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                ema_model=ema_model,
                use_amp=amp_enabled,
                gradient_accumulation_steps=gradient_accumulation_steps,
                enable_timing=bool(config.get("enable_train_timing", True)),
                timing_warmup_steps=int(
                    config.get("train_timing_warmup_steps", 10)
                ),
                timing_max_steps=int(config.get("train_timing_max_steps", 100)),
            )
            eval_model = ema_model.module if ema_model is not None else model
            val_stats = validate_one_epoch(
                model=eval_model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                ann_path=config["val_ann_path"],
                use_amp=amp_enabled,
                **inference_postprocess_kwargs,
            )

            history["train_loss"].append(train_stats["loss"])
            history["val_loss"].append(val_stats["loss"])
            history["val_map"].append(
                val_stats["map"] if val_stats["map"] is not None else 0.0
            )
            history["val_loss_raw"].append(float("nan"))
            history["val_map_raw"].append(float("nan"))
            history["val_loss_ema"].append(val_stats["loss"])
            history["val_map_ema"].append(
                val_stats["map"] if val_stats["map"] is not None else 0.0
            )

            print(
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                "Val mAP: "
                f"{0.0 if val_stats['map'] is None else val_stats['map']:.4f} | "
                "Val AP50: "
                f"{0.0 if val_stats['map50'] is None else val_stats['map50']:.4f}"
            )

            improved = False
            current_map = val_stats["map"] if val_stats["map"] is not None else -1.0
            if current_map > best_map:
                best_map = current_map
                torch.save(eval_model.state_dict(), config["best_model_path"])
                print(f"Best mAP model saved ({best_map:.4f})")
                improved = True

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save(eval_model.state_dict(), config["best_loss_model_path"])
                print(f"Best loss model saved ({best_val_loss:.4f})")

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement! {epochs_no_improve}/{early_stopping_patience}")

            save_checkpoint(
                checkpoint_path=config["checkpoint_path"],
                epoch_idx=epoch_idx,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_map=best_map,
                best_val_loss=best_val_loss,
                history=history,
                epochs_no_improve=epochs_no_improve,
            )

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        elapsed_minutes = (time.time() - training_start_time) / 60.0
        print(f"Training finished in {elapsed_minutes:.2f} minutes")
        if history["train_loss"] or history["val_loss"] or history["val_map"]:
            plot_training_curves(
                train_loss=history["train_loss"],
                val_loss=history["val_loss"],
                val_map=history["val_map"],
                save_path=config["training_curve_path"],
            )
            print(f"Training curves saved to {config['training_curve_path']}")
        else:
            print("No training history available. Skipping training curve export.")


if __name__ == "__main__":
    main()

