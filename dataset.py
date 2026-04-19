"""Dataset and transform utilities for digit detection."""

import json
import math
import os
from collections import defaultdict
from typing import Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import BatchSampler, Dataset


def _build_bucket_boundaries(values: list[float], num_bins: int) -> list[float]:
    """Build quantile-based bucket boundaries."""
    if len(values) <= 1 or num_bins <= 1:
        return []

    sorted_values = sorted(float(value) for value in values)
    boundaries = []
    for bin_idx in range(1, num_bins):
        position = math.ceil(len(sorted_values) * bin_idx / num_bins) - 1
        position = min(max(position, 0), len(sorted_values) - 1)
        boundaries.append(sorted_values[position])
    return boundaries


def _assign_bucket_id(value: float, boundaries: list[float]) -> int:
    """Assign a value to a bucket according to boundaries."""
    for bucket_id, boundary in enumerate(boundaries):
        if value <= boundary:
            return bucket_id
    return len(boundaries)


def _normalize_image_size_candidates(
    max_image_size: int | list[int] | None,
) -> list[int] | None:
    """Normalize max-image-size inputs into a unique sorted candidate list."""
    if max_image_size is None:
        return None
    if isinstance(max_image_size, int):
        return [int(max_image_size)]

    candidates = sorted({int(size) for size in max_image_size if int(size) > 0})
    return candidates or None


def _resolve_reference_max_image_size(
    max_image_size: int | list[int] | None,
) -> int | None:
    """Pick the largest configured image size as the reference scale."""
    candidates = _normalize_image_size_candidates(max_image_size)
    if candidates is None:
        return None
    return int(max(candidates))


def _normalize_fixed_image_size(
    fixed_image_size: list[int] | tuple[int, int] | None,
) -> tuple[int, int] | None:
    if fixed_image_size is None:
        return None
    if len(fixed_image_size) != 2:
        raise ValueError(
            "fixed_image_size must contain exactly 2 integers: [height, width]."
        )
    target_height = max(1, int(fixed_image_size[0]))
    target_width = max(1, int(fixed_image_size[1]))
    return (target_height, target_width)


def _normalize_fixed_image_size_candidates(
    fixed_image_sizes: (
        list[list[int] | tuple[int, int]]
        | tuple[list[int] | tuple[int, int], ...]
        | None
    ),
) -> list[tuple[int, int]] | None:
    if fixed_image_sizes is None:
        return None
    candidates: list[tuple[int, int]] = []
    for candidate in fixed_image_sizes:
        normalized = _normalize_fixed_image_size(candidate)
        if normalized is not None and normalized not in candidates:
            candidates.append(normalized)
    return candidates or None


class DetectionTransform:
    """Lightweight transform for detection data."""

    def __init__(
        self,
        split: str = "train",
        max_image_size: int | list[int] | None = None,
        fixed_image_size: list[int] | tuple[int, int] | None = None,
        fixed_image_sizes: list[list[int] | tuple[int, int]] | tuple[list[int] | tuple[int, int], ...] | None = None,
        use_color_jitter: bool = False,
        brightness_jitter: float = 0.10,
        contrast_jitter: float = 0.10,
        saturation_jitter: float = 0.05,
        hue_jitter: float = 0.0,
        use_random_grayscale: bool = False,
        random_grayscale_prob: float = 0.0,
        use_gamma_aug: bool = False,
        gamma_min: float = 0.8,
        gamma_max: float = 1.2,
        use_gaussian_blur: bool = False,
        gaussian_blur_prob: float = 0.15,
        gaussian_blur_radius: float = 0.5,
        use_image_noise_aug: bool = False,
        image_noise_prob: float = 0.20,
        image_noise_std: float = 0.02,
        use_light_affine_aug: bool = False,
        max_rotation_degree: float = 5.0,
        max_translation_ratio: float = 0.02,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
        max_shear_degree: float = 4.0,
        use_position_debias_aug: bool = False,
        max_canvas_shift_ratio: float = 0.10,
        use_horizontal_layout_shift_aug: bool = False,
        horizontal_layout_shift_prob: float = 0.5,
        max_horizontal_layout_shift_ratio: float = 0.10,
        horizontal_layout_shift_allow_truncation: bool = False,
        use_mild_truncation_aug: bool = True,
        truncation_prob: float = 0.15,
        enable_dynamic_multi_scale: bool = False,
        dynamic_multi_scale_max_upscale_ratio: float = 1.0,
        allow_upscale: bool = False,
        max_upscale_ratio: float = 1.0,
        use_synthetic_digit_style_aug: bool = False,
        synthetic_digit_style_prob: float = 0.35,
        synthetic_digit_style_max_regions: int = 2,
        synthetic_digit_style_min_scale: float = 0.85,
        synthetic_digit_style_max_scale: float = 1.20,
        synthetic_digit_style_max_shift_ratio: float = 0.10,
        synthetic_digit_style_gamma_min: float = 0.75,
        synthetic_digit_style_gamma_max: float = 1.25,
        synthetic_digit_style_blur_prob: float = 0.25,
        synthetic_digit_style_blur_radius: float = 0.7,
        return_debug_metadata: bool = False,
    ):
        self.split = split
        self.max_image_size = max_image_size
        self.max_image_size_candidates = _normalize_image_size_candidates(max_image_size)
        self.max_image_size_reference = _resolve_reference_max_image_size(max_image_size)
        self.fixed_image_size = _normalize_fixed_image_size(fixed_image_size)
        self.fixed_image_size_candidates = _normalize_fixed_image_size_candidates(
            fixed_image_sizes
        )
        self.use_color_jitter = use_color_jitter
        self.brightness_jitter = max(0.0, float(brightness_jitter))
        self.contrast_jitter = max(0.0, float(contrast_jitter))
        self.saturation_jitter = max(0.0, float(saturation_jitter))
        self.hue_jitter = max(0.0, float(hue_jitter))
        self.use_random_grayscale = bool(use_random_grayscale)
        self.random_grayscale_prob = max(0.0, float(random_grayscale_prob))
        self.use_gamma_aug = bool(use_gamma_aug)
        self.gamma_min = max(1e-3, float(gamma_min))
        self.gamma_max = max(self.gamma_min, float(gamma_max))
        self.use_gaussian_blur = use_gaussian_blur
        self.gaussian_blur_prob = max(0.0, float(gaussian_blur_prob))
        self.gaussian_blur_radius = max(0.0, float(gaussian_blur_radius))
        self.use_image_noise_aug = bool(use_image_noise_aug)
        self.image_noise_prob = max(0.0, float(image_noise_prob))
        self.image_noise_std = max(0.0, float(image_noise_std))
        self.use_light_affine_aug = use_light_affine_aug
        self.max_rotation_degree = max_rotation_degree
        self.max_translation_ratio = max_translation_ratio
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_shear_degree = max_shear_degree
        self.use_position_debias_aug = use_position_debias_aug
        self.max_canvas_shift_ratio = max_canvas_shift_ratio
        self.use_horizontal_layout_shift_aug = bool(use_horizontal_layout_shift_aug)
        self.horizontal_layout_shift_prob = max(
            0.0,
            float(horizontal_layout_shift_prob),
        )
        self.max_horizontal_layout_shift_ratio = max(
            0.0,
            float(max_horizontal_layout_shift_ratio),
        )
        self.horizontal_layout_shift_allow_truncation = bool(
            horizontal_layout_shift_allow_truncation
        )
        self.use_mild_truncation_aug = use_mild_truncation_aug
        self.truncation_prob = truncation_prob
        self.enable_dynamic_multi_scale = bool(enable_dynamic_multi_scale)
        self.dynamic_multi_scale_max_upscale_ratio = max(
            1.0,
            float(dynamic_multi_scale_max_upscale_ratio),
        )
        self.allow_upscale = bool(allow_upscale)
        self.max_upscale_ratio = max(1.0, float(max_upscale_ratio))
        self.use_synthetic_digit_style_aug = bool(use_synthetic_digit_style_aug)
        self.synthetic_digit_style_prob = max(0.0, float(synthetic_digit_style_prob))
        self.synthetic_digit_style_max_regions = max(0, int(synthetic_digit_style_max_regions))
        self.synthetic_digit_style_min_scale = max(0.25, float(synthetic_digit_style_min_scale))
        self.synthetic_digit_style_max_scale = max(
            self.synthetic_digit_style_min_scale,
            float(synthetic_digit_style_max_scale),
        )
        self.synthetic_digit_style_max_shift_ratio = max(
            0.0,
            float(synthetic_digit_style_max_shift_ratio),
        )
        self.synthetic_digit_style_gamma_min = max(
            1e-3,
            float(synthetic_digit_style_gamma_min),
        )
        self.synthetic_digit_style_gamma_max = max(
            self.synthetic_digit_style_gamma_min,
            float(synthetic_digit_style_gamma_max),
        )
        self.synthetic_digit_style_blur_prob = max(
            0.0,
            float(synthetic_digit_style_blur_prob),
        )
        self.synthetic_digit_style_blur_radius = max(
            0.0,
            float(synthetic_digit_style_blur_radius),
        )
        self.return_debug_metadata = bool(return_debug_metadata)

    def _sample_target_long_side(self) -> int | None:
        if self.max_image_size_reference is None:
            return None
        if (
            self.split == "train"
            and self.enable_dynamic_multi_scale
            and self.max_image_size_candidates is not None
            and len(self.max_image_size_candidates) > 1
        ):
            sampled_idx = int(torch.randint(len(self.max_image_size_candidates), (1,)).item())
            return int(self.max_image_size_candidates[sampled_idx])
        return int(self.max_image_size_reference)

    def _resize_keep_ratio(self, image: Image.Image, boxes: torch.Tensor):
        """
        Resize the image that exceeds the maximum size while keeping the aspect ratio, and adjust the bounding boxes accordingly.
        """
        target_long_side = self._sample_target_long_side()
        if target_long_side is None:
            return image, boxes

        width, height = image.size
        long_side = max(width, height)
        scale = float(target_long_side) / max(float(long_side), 1.0)
        if scale >= 1.0:
            if self.split == "train" and self.enable_dynamic_multi_scale:
                scale = min(scale, self.dynamic_multi_scale_max_upscale_ratio)
            elif self.allow_upscale:
                scale = min(scale, self.max_upscale_ratio)
            else:
                return image, boxes
            if scale <= 1.0 + 1e-6:
                return image, boxes
        elif abs(scale - 1.0) <= 1e-6:
            return image, boxes
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = image.resize((new_width, new_height), Image.BILINEAR)

        if boxes.numel() > 0:
            boxes = boxes * scale
        return image, boxes

    def _resize_fixed_size(self, image: Image.Image, boxes: torch.Tensor):
        target_size = self.fixed_image_size
        if (
            self.split == "train"
            and self.fixed_image_size_candidates is not None
            and len(self.fixed_image_size_candidates) > 0
        ):
            sampled_idx = int(torch.randint(len(self.fixed_image_size_candidates), (1,)).item())
            target_size = self.fixed_image_size_candidates[sampled_idx]

        if target_size is None:
            return image, boxes

        target_height, target_width = target_size
        width, height = image.size
        if width == target_width and height == target_height:
            return image, boxes

        scale_x = float(target_width) / max(float(width), 1.0)
        scale_y = float(target_height) / max(float(height), 1.0)
        image = image.resize((target_width, target_height), Image.BILINEAR)
        if boxes.numel() > 0:
            scale = boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])
            boxes = boxes * scale
        return image, boxes

    @staticmethod
    def _apply_affine_to_boxes(
        boxes: torch.Tensor,
        width: int,
        height: int,
        angle: float,
        translate: tuple[int, int],
        scale: float,
        shear: tuple[float, float],
    ) -> torch.Tensor:
        """Apply the same affine transform used on image to bounding boxes."""
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)

        center = [width * 0.5, height * 0.5]
        inverse_matrix = TF._get_inverse_affine_matrix(  # type: ignore[attr-defined]
            center=center,
            angle=angle,
            translate=list(translate),
            scale=scale,
            shear=list(shear),
        )
        inverse_matrix = torch.tensor(
            [
                [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
                [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
                [0.0, 0.0, 1.0],
            ],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        forward_matrix = torch.linalg.inv(inverse_matrix)

        corners = torch.stack(
            [
                torch.stack([boxes[:, 0], boxes[:, 1], torch.ones_like(boxes[:, 0])], dim=-1),
                torch.stack([boxes[:, 2], boxes[:, 1], torch.ones_like(boxes[:, 0])], dim=-1),
                torch.stack([boxes[:, 2], boxes[:, 3], torch.ones_like(boxes[:, 0])], dim=-1),
                torch.stack([boxes[:, 0], boxes[:, 3], torch.ones_like(boxes[:, 0])], dim=-1),
            ],
            dim=1,
        )
        transformed = torch.matmul(corners, forward_matrix.T)[..., :2]

        min_xy = transformed.min(dim=1).values
        max_xy = transformed.max(dim=1).values
        transformed_boxes = torch.cat([min_xy, max_xy], dim=-1)
        transformed_boxes[:, 0::2] = transformed_boxes[:, 0::2].clamp(0.0, float(width))
        transformed_boxes[:, 1::2] = transformed_boxes[:, 1::2].clamp(0.0, float(height))
        keep = (transformed_boxes[:, 2] > transformed_boxes[:, 0]) & (transformed_boxes[:, 3] > transformed_boxes[:, 1])
        return transformed_boxes[keep]

    @staticmethod
    def _shift_boxes(boxes: torch.Tensor, width: int, height: int, shift_x: int, shift_y: int) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4), torch.zeros((0,), dtype=torch.bool, device=boxes.device)

        shifted_boxes = boxes + boxes.new_tensor([shift_x, shift_y, shift_x, shift_y])
        shifted_boxes[:, 0::2] = shifted_boxes[:, 0::2].clamp(0.0, float(width))
        shifted_boxes[:, 1::2] = shifted_boxes[:, 1::2].clamp(0.0, float(height))
        keep = (shifted_boxes[:, 2] > shifted_boxes[:, 0]) & (shifted_boxes[:, 3] > shifted_boxes[:, 1])
        return shifted_boxes[keep], keep

    def _apply_position_debias(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        iscrowd: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shift content inside the canvas to weaken layout shortcuts while keeping resize-keep-ratio intact."""
        width, height = image.size
        max_shift_x = int(round(width * self.max_canvas_shift_ratio))
        max_shift_y = int(round(height * self.max_canvas_shift_ratio))
        if max_shift_x <= 0 and max_shift_y <= 0:
            return image, boxes, labels, iscrowd

        allow_truncation = self.use_mild_truncation_aug and torch.rand(1).item() < self.truncation_prob
        if boxes.numel() > 0 and not allow_truncation:
            safe_min_x = max(-max_shift_x, int(math.ceil(-boxes[:, 0].min().item())))
            safe_max_x = min(max_shift_x, int(math.floor(width - boxes[:, 2].max().item())))
            safe_min_y = max(-max_shift_y, int(math.ceil(-boxes[:, 1].min().item())))
            safe_max_y = min(max_shift_y, int(math.floor(height - boxes[:, 3].max().item())))
        else:
            safe_min_x, safe_max_x = -max_shift_x, max_shift_x
            safe_min_y, safe_max_y = -max_shift_y, max_shift_y

        if safe_min_x > safe_max_x or safe_min_y > safe_max_y:
            return image, boxes, labels, iscrowd

        shift_x = int(torch.randint(safe_min_x, safe_max_x + 1, (1,)).item()) if safe_min_x != safe_max_x else safe_min_x
        shift_y = int(torch.randint(safe_min_y, safe_max_y + 1, (1,)).item()) if safe_min_y != safe_max_y else safe_min_y
        if shift_x == 0 and shift_y == 0:
            return image, boxes, labels, iscrowd

        shifted_image = TF.affine(
            image,
            angle=0.0,
            translate=[shift_x, shift_y],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )
        shifted_boxes, keep = self._shift_boxes(boxes, width, height, shift_x, shift_y)
        if not allow_truncation and shifted_boxes.shape[0] != boxes.shape[0]:
            return image, boxes, labels, iscrowd

        if keep.numel() == 0:
            return shifted_image, shifted_boxes, labels[:0], iscrowd[:0]
        return shifted_image, shifted_boxes, labels[keep], iscrowd[keep]

    def _apply_horizontal_layout_shift(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        iscrowd: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        """Shift content horizontally inside the canvas to mimic layout-shifted street-view compositions."""
        width, height = image.size
        max_shift_x = int(round(width * self.max_horizontal_layout_shift_ratio))
        if max_shift_x <= 0:
            return image, boxes, labels, iscrowd, None

        allow_truncation = self.horizontal_layout_shift_allow_truncation
        if boxes.numel() > 0 and not allow_truncation:
            safe_min_x = max(-max_shift_x, int(math.ceil(-boxes[:, 0].min().item())))
            safe_max_x = min(max_shift_x, int(math.floor(width - boxes[:, 2].max().item())))
        else:
            safe_min_x, safe_max_x = -max_shift_x, max_shift_x

        if safe_min_x > safe_max_x:
            return image, boxes, labels, iscrowd, None

        shift_x = int(torch.randint(safe_min_x, safe_max_x + 1, (1,)).item()) if safe_min_x != safe_max_x else safe_min_x
        if shift_x == 0:
            return image, boxes, labels, iscrowd, None

        shifted_image = TF.affine(
            image,
            angle=0.0,
            translate=[shift_x, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )
        shifted_boxes, keep = self._shift_boxes(boxes, width, height, shift_x, 0)
        if not allow_truncation and shifted_boxes.shape[0] != boxes.shape[0]:
            return image, boxes, labels, iscrowd, None

        debug_metadata = {
            "name": "horizontal_layout_shift",
            "shift_x": int(shift_x),
            "shift_ratio": float(shift_x) / max(float(width), 1.0),
            "allow_truncation": bool(allow_truncation),
        }
        if keep.numel() == 0:
            return shifted_image, shifted_boxes, labels[:0], iscrowd[:0], debug_metadata
        return shifted_image, shifted_boxes, labels[keep], iscrowd[keep], debug_metadata

    @staticmethod
    def _center_crop_or_paste_patch(
        base_patch: Image.Image,
        styled_patch: Image.Image,
        shift_x: int = 0,
        shift_y: int = 0,
    ) -> Image.Image:
        base_width, base_height = base_patch.size
        styled_width, styled_height = styled_patch.size

        crop_left = max(0, (styled_width - base_width) // 2)
        crop_top = max(0, (styled_height - base_height) // 2)
        crop_right = crop_left + min(base_width, styled_width)
        crop_bottom = crop_top + min(base_height, styled_height)
        styled_patch = styled_patch.crop((crop_left, crop_top, crop_right, crop_bottom))

        canvas = base_patch.copy()
        paste_left = (base_width - styled_patch.size[0]) // 2 + int(shift_x)
        paste_top = (base_height - styled_patch.size[1]) // 2 + int(shift_y)
        paste_left = min(max(paste_left, 0), max(0, base_width - styled_patch.size[0]))
        paste_top = min(max(paste_top, 0), max(0, base_height - styled_patch.size[1]))
        canvas.paste(styled_patch, (paste_left, paste_top))
        return canvas

    def _apply_synthetic_digit_style(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        return_debug_metadata: bool = False,
    ):
        if (
            not self.use_synthetic_digit_style_aug
            or self.synthetic_digit_style_prob <= 0.0
            or self.synthetic_digit_style_max_regions <= 0
            or boxes.numel() == 0
            or torch.rand(1).item() >= self.synthetic_digit_style_prob
        ):
            return (image, []) if return_debug_metadata else image

        width, height = image.size
        if width <= 0 or height <= 0:
            return (image, []) if return_debug_metadata else image

        candidate_indices = []
        for box_idx, box in enumerate(boxes):
            x0, y0, x1, y1 = [float(value) for value in box.tolist()]
            patch_width = int(round(x1 - x0))
            patch_height = int(round(y1 - y0))
            if patch_width >= 8 and patch_height >= 8:
                candidate_indices.append(box_idx)
        if not candidate_indices:
            return (image, []) if return_debug_metadata else image

        max_regions = min(self.synthetic_digit_style_max_regions, len(candidate_indices))
        if max_regions <= 0:
            return (image, []) if return_debug_metadata else image
        perm = torch.randperm(len(candidate_indices))
        selected_count = int(torch.randint(1, max_regions + 1, (1,)).item())
        selected_indices = [candidate_indices[int(idx)] for idx in perm[:selected_count].tolist()]

        styled_image = image.copy()
        debug_regions = []
        local_brightness_jitter = max(self.brightness_jitter, 0.20)
        local_contrast_jitter = max(self.contrast_jitter, 0.20)
        local_saturation_jitter = max(self.saturation_jitter, 0.10)
        local_hue_jitter = max(self.hue_jitter, 0.02)

        for box_idx in selected_indices:
            x0, y0, x1, y1 = [float(value) for value in boxes[box_idx].tolist()]
            left = max(0, min(width - 1, int(math.floor(x0))))
            top = max(0, min(height - 1, int(math.floor(y0))))
            right = max(left + 1, min(width, int(math.ceil(x1))))
            bottom = max(top + 1, min(height, int(math.ceil(y1))))
            if right - left < 4 or bottom - top < 4:
                continue

            original_patch = styled_image.crop((left, top, right, bottom))
            transformed_patch = original_patch.copy()

            brightness_factor = 1.0 + float(torch.empty(1).uniform_(-local_brightness_jitter, local_brightness_jitter))
            transformed_patch = TF.adjust_brightness(
                transformed_patch,
                brightness_factor,
            )
            contrast_factor = 1.0 + float(torch.empty(1).uniform_(-local_contrast_jitter, local_contrast_jitter))
            transformed_patch = TF.adjust_contrast(
                transformed_patch,
                contrast_factor,
            )
            saturation_factor = 1.0 + float(torch.empty(1).uniform_(-local_saturation_jitter, local_saturation_jitter))
            transformed_patch = TF.adjust_saturation(
                transformed_patch,
                saturation_factor,
            )
            hue_delta = 0.0
            if local_hue_jitter > 0.0:
                hue_delta = float(torch.empty(1).uniform_(-local_hue_jitter, local_hue_jitter))
                transformed_patch = TF.adjust_hue(
                    transformed_patch,
                    hue_delta,
                )
            gamma_value = float(torch.empty(1).uniform_(self.synthetic_digit_style_gamma_min, self.synthetic_digit_style_gamma_max))
            transformed_patch = TF.adjust_gamma(
                transformed_patch,
                gamma=gamma_value,
            )

            patch_width, patch_height = transformed_patch.size
            scale_x = float(torch.empty(1).uniform_(self.synthetic_digit_style_min_scale, self.synthetic_digit_style_max_scale))
            scale_y = float(torch.empty(1).uniform_(self.synthetic_digit_style_min_scale, self.synthetic_digit_style_max_scale))
            resized_width = max(2, int(round(patch_width * scale_x)))
            resized_height = max(2, int(round(patch_height * scale_y)))
            transformed_patch = transformed_patch.resize((resized_width, resized_height), Image.BILINEAR)

            max_shift_x = int(round(patch_width * self.synthetic_digit_style_max_shift_ratio))
            max_shift_y = int(round(patch_height * self.synthetic_digit_style_max_shift_ratio))
            shift_x = (
                int(torch.randint(-max_shift_x, max_shift_x + 1, (1,)).item())
                if max_shift_x > 0
                else 0
            )
            shift_y = (
                int(torch.randint(-max_shift_y, max_shift_y + 1, (1,)).item())
                if max_shift_y > 0
                else 0
            )
            transformed_patch = self._center_crop_or_paste_patch(
                base_patch=original_patch,
                styled_patch=transformed_patch,
                shift_x=shift_x,
                shift_y=shift_y,
            )

            if self.synthetic_digit_style_blur_radius > 0.0 and torch.rand(1).item() < self.synthetic_digit_style_blur_prob:
                transformed_patch = transformed_patch.filter(
                    ImageFilter.GaussianBlur(radius=self.synthetic_digit_style_blur_radius)
                )
                blur_applied = True
            else:
                blur_applied = False

            styled_image.paste(transformed_patch, (left, top, right, bottom))
            debug_regions.append(
                {
                    "box_index": int(box_idx),
                    "paste_box": [left, top, right, bottom],
                    "brightness_factor": brightness_factor,
                    "contrast_factor": contrast_factor,
                    "saturation_factor": saturation_factor,
                    "hue_delta": hue_delta,
                    "gamma": gamma_value,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "shift_x": shift_x,
                    "shift_y": shift_y,
                    "blur_applied": blur_applied,
                }
            )

        return (styled_image, debug_regions) if return_debug_metadata else styled_image

    def __call__(self, image: Image.Image, target: dict):
        boxes = target["boxes"].clone()
        labels = target["labels"].clone()
        iscrowd = target["iscrowd"].clone()
        debug_aug_metadata: list[dict] = []
        if self.fixed_image_size is not None or self.fixed_image_size_candidates is not None:
            image, boxes = self._resize_fixed_size(image, boxes)
        else:
            image, boxes = self._resize_keep_ratio(image, boxes)

        if self.split == "train":
            if self.use_horizontal_layout_shift_aug and torch.rand(1).item() < self.horizontal_layout_shift_prob:
                image, boxes, labels, iscrowd, shift_debug = self._apply_horizontal_layout_shift(
                    image=image,
                    boxes=boxes,
                    labels=labels,
                    iscrowd=iscrowd,
                )
                if shift_debug is not None:
                    debug_aug_metadata.append(shift_debug)

            if self.use_position_debias_aug and torch.rand(1).item() < 0.5:
                image, boxes, labels, iscrowd = self._apply_position_debias(
                    image=image,
                    boxes=boxes,
                    labels=labels,
                    iscrowd=iscrowd,
                )

            if self.use_synthetic_digit_style_aug:
                image = self._apply_synthetic_digit_style(
                    image=image,
                    boxes=boxes,
                )

            if self.use_light_affine_aug and torch.rand(1).item() < 0.5:
                width, height = image.size
                angle = float(torch.empty(1).uniform_(-self.max_rotation_degree, self.max_rotation_degree))
                translate_x = int(round(width * float(torch.empty(1).uniform_(-self.max_translation_ratio, self.max_translation_ratio))))
                translate_y = int(round(height * float(torch.empty(1).uniform_(-self.max_translation_ratio, self.max_translation_ratio))))
                scale = float(torch.empty(1).uniform_(self.min_scale, self.max_scale))
                shear_x = float(torch.empty(1).uniform_(-self.max_shear_degree, self.max_shear_degree))
                shear_y = float(torch.empty(1).uniform_(-self.max_shear_degree, self.max_shear_degree))

                transformed_boxes = self._apply_affine_to_boxes(
                    boxes=boxes,
                    width=width,
                    height=height,
                    angle=angle,
                    translate=(translate_x, translate_y),
                    scale=scale,
                    shear=(shear_x, shear_y),
                )

                if transformed_boxes.shape[0] == boxes.shape[0]:
                    image = TF.affine(
                        image,
                        angle=angle,
                        translate=[translate_x, translate_y],
                        scale=scale,
                        shear=[shear_x, shear_y],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        fill=0,
                    )
                    boxes = transformed_boxes

            if self.use_color_jitter:
                image = TF.adjust_brightness(
                    image,
                    1.0 + float(torch.empty(1).uniform_(-self.brightness_jitter, self.brightness_jitter)),
                )
                image = TF.adjust_contrast(
                    image,
                    1.0 + float(torch.empty(1).uniform_(-self.contrast_jitter, self.contrast_jitter)),
                )
                image = TF.adjust_saturation(
                    image,
                    1.0 + float(torch.empty(1).uniform_(-self.saturation_jitter, self.saturation_jitter)),
                )
                if self.hue_jitter > 0.0:
                    image = TF.adjust_hue(
                        image,
                        float(torch.empty(1).uniform_(-self.hue_jitter, self.hue_jitter)),
                    )

            if self.use_random_grayscale and torch.rand(1).item() < self.random_grayscale_prob:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

            if self.use_gamma_aug:
                image = TF.adjust_gamma(
                    image,
                    gamma=float(torch.empty(1).uniform_(self.gamma_min, self.gamma_max)),
                )

            if self.use_gaussian_blur and torch.rand(1).item() < self.gaussian_blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(radius=self.gaussian_blur_radius))

        image_tensor = TF.to_tensor(image)
        if (
            self.split == "train"
            and self.use_image_noise_aug
            and self.image_noise_std > 0.0
            and torch.rand(1).item() < self.image_noise_prob
        ):
            noise = torch.randn_like(image_tensor) * self.image_noise_std
            image_tensor = (image_tensor + noise).clamp(0.0, 1.0)

        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        if boxes.numel() > 0:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["size"] = torch.tensor(
            [image_tensor.shape[1], image_tensor.shape[2]], dtype=torch.int64)
        if self.return_debug_metadata:
            target["debug_aug_metadata"] = debug_aug_metadata
        return image_tensor, target


class DetectionDataset(Dataset):
    """COCO-like detection dataset for train / valid / test."""

    def __init__(
        self,
        image_dir: str,
        annotation_path: str | None = None,
        split: str = "train",
        transform: Callable | None = None,
    ):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.split = split
        self.transform = transform

        self.images = []
        self.annotations_by_image = defaultdict(list)
        self._image_size_cache: dict[int, tuple[int, int]] = {}

        if annotation_path is not None and os.path.exists(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.images = sorted(data.get("images", []), key=lambda x: x["id"])
            for idx, image_info in enumerate(self.images):
                width = image_info.get("width")
                height = image_info.get("height")
                if width is not None and height is not None:
                    self._image_size_cache[idx] = (int(width), int(height))

            for ann in data.get("annotations", []):
                self.annotations_by_image[ann["image_id"]].append(ann)
        else:
            file_names = [
                file_name
                for file_name in os.listdir(image_dir)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            file_names.sort()
            self.images = [
                {
                    "id": int(os.path.splitext(file_name)[0]),
                    "file_name": file_name,
                }
                for file_name in file_names
            ]

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)

    def get_image_size(self, idx: int) -> tuple[int, int]:
        """Return image size as (width, height)."""
        if idx in self._image_size_cache:
            return self._image_size_cache[idx]

        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        with Image.open(image_path) as image:
            width, height = image.size
        self._image_size_cache[idx] = (width, height)
        return width, height

    def get_aspect_ratio(self, idx: int) -> float:
        """Return width / height."""
        width, height = self.get_image_size(idx)
        return float(width) / max(float(height), 1.0)

    def get_resized_image_size(
        self,
        idx: int,
        max_image_size: int | list[int] | None = None,
        max_upscale_ratio: float = 1.0,
    ) -> tuple[int, int]:
        """Return resized image size after applying max image-size constraint."""
        width, height = self.get_image_size(idx)
        reference_max_image_size = _resolve_reference_max_image_size(max_image_size)
        if reference_max_image_size is None:
            return width, height

        long_side = max(width, height)
        scale = float(reference_max_image_size) / max(float(long_side), 1.0)
        if scale >= 1.0:
            scale = min(scale, max(1.0, float(max_upscale_ratio)))
            if scale <= 1.0 + 1e-6:
                return width, height
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return resized_width, resized_height

    def get_bucket_value(
        self,
        idx: int,
        mode: str = "aspect_ratio",
        max_image_size: int | list[int] | None = None,
        max_upscale_ratio: float = 1.0,
    ) -> float:
        """Return the bucket key for a specific sample."""
        width, height = self.get_resized_image_size(
            idx,
            max_image_size=max_image_size,
            max_upscale_ratio=max_upscale_ratio,
        )

        if mode == "aspect_ratio":
            return float(width) / max(float(height), 1.0)
        if mode == "image_area":
            return float(width * height)
        if mode == "short_side":
            return float(min(width, height))
        if mode == "long_side":
            return float(max(width, height))
        if mode == "aspect_ratio_area":
            return float(width) / max(float(height), 1.0)
        raise ValueError(f"Unsupported bucket_mode: {mode}")

    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        ann_list = self.annotations_by_image.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in ann_list:
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(width), float(x + w))
            y2 = min(float(height), float(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]) - 1)
            areas.append(float((x2 - x1) * (y2 - y1)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
            "file_name": file_name,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)
            image = TF.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        return image, target


class AspectRatioBatchSampler(BatchSampler):
    """Batch sampler that groups similarly shaped images into the same batch."""

    def __init__(
        self,
        dataset: DetectionDataset,
        batch_size: int,
        drop_last: bool = False,
        num_bins: int = 4,
        bucket_mode: str = "aspect_ratio",
        max_image_size: int | list[int] | None = None,
        max_upscale_ratio: float = 1.0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(dataset) == 0:
            raise ValueError("dataset must not be empty")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_bins = max(1, int(num_bins))
        self.bucket_mode = bucket_mode
        self.max_image_size = max_image_size
        self.max_upscale_ratio = max(1.0, float(max_upscale_ratio))

        self.bucket_indices: dict[int, list[int]] = defaultdict(list)
        self.bucket_area_values: dict[int, list[tuple[int, float]]] | None = None

        if self.bucket_mode == "aspect_ratio_area":
            aspect_values = [
                self.dataset.get_bucket_value(
                    idx,
                    mode="aspect_ratio",
                    max_image_size=self.max_image_size,
                    max_upscale_ratio=self.max_upscale_ratio,
                )
                for idx in range(len(self.dataset))
            ]
            aspect_boundaries = _build_bucket_boundaries(aspect_values, self.num_bins)
            self.bucket_area_values = defaultdict(list)
            for idx, aspect_value in enumerate(aspect_values):
                bucket_id = _assign_bucket_id(aspect_value, aspect_boundaries)
                self.bucket_indices[bucket_id].append(idx)
                area_value = self.dataset.get_bucket_value(
                    idx,
                    mode="image_area",
                    max_image_size=self.max_image_size,
                    max_upscale_ratio=self.max_upscale_ratio,
                )
                self.bucket_area_values[bucket_id].append((idx, area_value))
        else:
            bucket_values = [
                self.dataset.get_bucket_value(
                    idx,
                    mode=self.bucket_mode,
                    max_image_size=self.max_image_size,
                    max_upscale_ratio=self.max_upscale_ratio,
                )
                for idx in range(len(self.dataset))
            ]
            boundaries = _build_bucket_boundaries(bucket_values, self.num_bins)

            for idx, bucket_value in enumerate(bucket_values):
                bucket_id = _assign_bucket_id(bucket_value, boundaries)
                self.bucket_indices[bucket_id].append(idx)

    def __iter__(self):
        batches = []
        for bucket_id, indices in self.bucket_indices.items():
            if self.bucket_mode == "aspect_ratio_area" and self.bucket_area_values is not None:
                area_pairs = list(self.bucket_area_values[bucket_id])
                area_order = torch.randperm(len(area_pairs)).tolist()
                shuffled_pairs = [area_pairs[position] for position in area_order]
                shuffled_pairs.sort(key=lambda item: item[1])
                shuffled_indices = [idx for idx, _ in shuffled_pairs]
            else:
                shuffled_order = torch.randperm(len(indices)).tolist()
                shuffled_indices = [indices[position] for position in shuffled_order]

            for start_idx in range(0, len(shuffled_indices), self.batch_size):
                batch = shuffled_indices[start_idx:start_idx + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if len(batches) > 1:
            batch_order = torch.randperm(len(batches)).tolist()
            batches = [batches[position] for position in batch_order]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += math.ceil(len(indices) / self.batch_size)
        return total_batches


def collate_fn(batch, pad_size_divisor: int = 1):
    """Pad images to the largest size in the batch and build masks."""
    images, targets = list(zip(*batch))

    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    pad_size_divisor = max(1, int(pad_size_divisor))
    if pad_size_divisor > 1:
        max_height = int(math.ceil(max_height / pad_size_divisor) * pad_size_divisor)
        max_width = int(math.ceil(max_width / pad_size_divisor) * pad_size_divisor)
    batch_size = len(images)

    padded_images = images[0].new_zeros((batch_size, 3, max_height, max_width))
    masks = torch.ones((batch_size, max_height, max_width), dtype=torch.bool)

    processed_targets = []
    for idx, (image, target) in enumerate(zip(images, targets)):
        _, height, width = image.shape
        padded_images[idx, :, :height, :width] = image
        masks[idx, :height, :width] = False

        new_target = dict(target)
        new_target["size"] = torch.tensor([height, width], dtype=torch.int64)
        processed_targets.append(new_target)

    return padded_images, masks, processed_targets
