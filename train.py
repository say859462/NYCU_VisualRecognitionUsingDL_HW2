"""Training loop helpers."""

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from utils import resolve_training_loss


def _loss_item(loss_dict: dict, key: str, device: torch.device) -> float:
    """Read a loss term as float while tolerating missing optional keys."""
    default = torch.tensor(0.0, device=device)
    return float(loss_dict.get(key, default).item())


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    ema_model=None,
    use_amp=True,
    gradient_accumulation_steps=1,
    enable_timing=False,
    timing_warmup_steps=10,
    timing_max_steps=100,
):
    """Run one training epoch and return aggregated loss statistics."""
    model.train()
    if criterion is not None:
        criterion.train()

    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    running_loss_objectness = 0.0
    running_loss_main = 0.0
    running_loss_aux = 0.0
    running_loss_group = 0.0
    running_loss_enc = 0.0
    running_loss_dn = 0.0
    running_loss_hybrid = 0.0
    running_loss_targeted_confusion = 0.0
    running_loss_aux_digit_cls = 0.0
    total_batches = 0

    timing_sums = {
        "data": 0.0,
        "to_device": 0.0,
        "forward_loss": 0.0,
        "backward_optim": 0.0,
        "iter": 0.0,
    }
    timing_count = 0

    accumulation_steps = max(1, int(gradient_accumulation_steps))
    amp_enabled = bool(use_amp) and device.type == "cuda"
    progress_bar = tqdm(train_loader, desc="Train", leave=False)
    last_iter_end = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step_idx, (images, masks, targets) in enumerate(progress_bar):
        iter_start = time.perf_counter()
        data_time = iter_start - last_iter_end

        to_device_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        to_device_time = time.perf_counter() - to_device_start

        forward_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images, masks, targets)
            loss, loss_dict = resolve_training_loss(outputs, targets, criterion)
            backward_loss = loss / accumulation_steps
        forward_time = time.perf_counter() - forward_start

        if not torch.isfinite(loss):
            print("[train] non-finite loss detected; skip batch")
            optimizer.zero_grad(set_to_none=True)
            last_iter_end = time.perf_counter()
            continue

        backward_start = time.perf_counter()
        scaler.scale(backward_loss).backward()

        should_step = (
            (step_idx + 1) % accumulation_steps == 0
            or (step_idx + 1) == len(train_loader)
        )
        optimizer_was_skipped = False
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            if amp_enabled:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_skipped = scaler.get_scale() < previous_scale
            else:
                optimizer.step()

            if ema_model is not None and not optimizer_was_skipped:
                ema_model.update(model)

            if scheduler is not None and not optimizer_was_skipped:
                optimizer._opt_called = True
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        backward_time = time.perf_counter() - backward_start
        iter_time = time.perf_counter() - iter_start

        running_loss += loss.item()
        running_loss_ce += loss_dict["loss_ce"].item()
        running_loss_bbox += loss_dict["loss_bbox"].item()
        running_loss_giou += loss_dict["loss_giou"].item()
        running_loss_objectness += _loss_item(loss_dict, "loss_objectness", loss.device)
        running_loss_main += _loss_item(loss_dict, "loss_main", loss.device)
        running_loss_aux += _loss_item(loss_dict, "loss_aux", loss.device)
        running_loss_group += _loss_item(loss_dict, "loss_group", loss.device)
        running_loss_enc += _loss_item(loss_dict, "loss_enc", loss.device)
        running_loss_dn += _loss_item(loss_dict, "loss_dn", loss.device)
        running_loss_hybrid += _loss_item(loss_dict, "loss_hybrid", loss.device)
        running_loss_targeted_confusion += _loss_item(
            loss_dict,
            "loss_targeted_confusion",
            loss.device,
        )
        running_loss_aux_digit_cls += _loss_item(
            loss_dict,
            "loss_aux_digit_cls_weighted",
            loss.device,
        )
        total_batches += 1

        if (
            enable_timing
            and step_idx >= timing_warmup_steps
            and timing_count < timing_max_steps
        ):
            timing_sums["data"] += data_time
            timing_sums["to_device"] += to_device_time
            timing_sums["forward_loss"] += forward_time
            timing_sums["backward_optim"] += backward_time
            timing_sums["iter"] += iter_time
            timing_count += 1

        progress_bar.set_postfix(
            {"loss": f"{running_loss / max(1, total_batches):.4f}"}
        )
        last_iter_end = time.perf_counter()

    if enable_timing and timing_count > 0:
        print(
            "[train timing] "
            f"avg over {timing_count} steps | "
            f"data={timing_sums['data'] / timing_count:.4f}s | "
            f"to_device={timing_sums['to_device'] / timing_count:.4f}s | "
            f"forward+loss={timing_sums['forward_loss'] / timing_count:.4f}s | "
            "backward+optim="
            f"{timing_sums['backward_optim'] / timing_count:.4f}s | "
            f"iter={timing_sums['iter'] / timing_count:.4f}s"
        )

    normalizer = max(1, total_batches)
    timing_normalizer = max(1, timing_count)
    return {
        "loss": running_loss / normalizer,
        "loss_ce": running_loss_ce / normalizer,
        "loss_bbox": running_loss_bbox / normalizer,
        "loss_giou": running_loss_giou / normalizer,
        "loss_objectness": running_loss_objectness / normalizer,
        "loss_main": running_loss_main / normalizer,
        "loss_aux": running_loss_aux / normalizer,
        "loss_group": running_loss_group / normalizer,
        "loss_enc": running_loss_enc / normalizer,
        "loss_dn": running_loss_dn / normalizer,
        "loss_hybrid": running_loss_hybrid / normalizer,
        "loss_targeted_confusion": running_loss_targeted_confusion / normalizer,
        "loss_aux_digit_cls_weighted": running_loss_aux_digit_cls / normalizer,
        "timing_count": timing_count,
        "timing_data": timing_sums["data"] / timing_normalizer,
        "timing_to_device": timing_sums["to_device"] / timing_normalizer,
        "timing_forward_loss": timing_sums["forward_loss"] / timing_normalizer,
        "timing_backward_optim": (
            timing_sums["backward_optim"] / timing_normalizer
        ),
        "timing_iter": timing_sums["iter"] / timing_normalizer,
    }
