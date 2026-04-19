"""Model definitions and adapters for the detection experiments."""

from collections import OrderedDict
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou, roi_align

try:
    from deformable_ops import MSDeformAttn as OfficialMSDeformAttn
    from deformable_ops import build_or_load_msda_extension
    from deformable_ops import get_msda_status
except Exception:
    OfficialMSDeformAttn = None
    build_or_load_msda_extension = None

    def get_msda_status():
        return {"available": False, "error": "deformable_ops import failed"}


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def get_valid_spatial_shape_from_mask(
    mask: torch.Tensor | None,
    default_height: int,
    default_width: int,
) -> tuple[int, int]:
    """Infer per-sample valid feature-map size from a bottom-right padding mask."""
    if mask is None:
        return int(default_height), int(default_width)

    mask = mask.to(torch.bool)
    valid_rows = (~mask).any(dim=1)
    valid_cols = (~mask).any(dim=0)
    valid_height = int(valid_rows.sum().item())
    valid_width = int(valid_cols.sum().item())
    if valid_height <= 0 or valid_width <= 0:
        return int(default_height), int(default_width)
    return valid_height, valid_width


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean(dim=1).sum() / max(1.0, num_boxes)


def vari_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gt_score: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid().detach()
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1.0 - alpha) * prob.pow(gamma) * (1.0 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(
        inputs, target_score, weight=weight, reduction="none")
    return loss.mean(dim=1).sum() / max(1.0, num_boxes)


def _get_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


class GenerateCDNQueries(nn.Module):
    def __init__(
        self,
        num_queries: int,
        num_classes: int,
        label_embed_dim: int,
        denoising_nums: int = 100,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.label_embed_dim = int(label_embed_dim)
        self.denoising_nums = int(denoising_nums)
        self.label_noise_prob = float(label_noise_prob)
        self.box_noise_scale = float(box_noise_scale)
        self.denoising_groups = 1
        self.label_encoder = nn.Embedding(
            self.num_classes, self.label_embed_dim)

    def apply_label_noise(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_noise_prob <= 0.0 or labels.numel() == 0:
            return labels
        noise_mask = torch.rand_like(labels.float()) < (
            self.label_noise_prob * 0.5)
        noised_labels = torch.randint_like(
            labels, low=0, high=self.num_classes)
        return torch.where(noise_mask, noised_labels, labels)

    def apply_box_noise(self, boxes: torch.Tensor) -> torch.Tensor:
        if self.box_noise_scale <= 0.0 or boxes.numel() == 0:
            return boxes
        num_boxes = len(boxes) // max(self.denoising_groups * 2, 1)
        positive_idx = torch.arange(
            num_boxes, dtype=torch.long, device=boxes.device)
        positive_idx = positive_idx.unsqueeze(
            0).repeat(self.denoising_groups, 1)
        positive_idx += (
            torch.arange(self.denoising_groups, dtype=torch.long,
                         device=boxes.device).unsqueeze(1) * num_boxes * 2
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_boxes

        diff = torch.zeros_like(boxes)
        diff[:, :2] = boxes[:, 2:] / 2
        diff[:, 2:] = boxes[:, 2:] / 2
        rand_sign = torch.randint_like(
            boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(boxes)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign
        xyxy_boxes = box_cxcywh_to_xyxy(boxes)
        xyxy_boxes = (xyxy_boxes + rand_part * diff *
                      self.box_noise_scale).clamp(min=0.0, max=1.0)
        return box_xyxy_to_cxcywh(xyxy_boxes)

    def generate_query_masks(self, padded_gt_count: int, device: torch.device) -> torch.Tensor:
        noised_query_nums = padded_gt_count * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.zeros((tgt_size, tgt_size),
                                device=device, dtype=torch.bool)
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for group_idx in range(self.denoising_groups):
            start = padded_gt_count * group_idx
            end = padded_gt_count * (group_idx + 1)
            attn_mask[start:end, :start] = True
            attn_mask[start:end, end:noised_query_nums] = True
        return attn_mask

    def forward(
        self,
        gt_labels_list: list[torch.Tensor],
        gt_boxes_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        gt_nums_per_image = [labels.numel() for labels in gt_labels_list]
        max_gt_num_per_image = max(
            gt_nums_per_image) if gt_nums_per_image else 0
        denoising_groups = self.denoising_nums * \
            max_gt_num_per_image // max(max_gt_num_per_image ** 2, 1)
        self.denoising_groups = max(int(denoising_groups), 1)

        if max_gt_num_per_image <= 0:
            device = self.label_encoder.weight.device
            return (
                torch.zeros((len(gt_labels_list), 0,
                            self.label_embed_dim), device=device),
                torch.zeros((len(gt_labels_list), 0, 4), device=device),
                torch.zeros((self.num_queries, self.num_queries),
                            device=device, dtype=torch.bool),
                self.denoising_groups,
                0,
            )

        gt_labels = torch.cat(gt_labels_list, dim=0)
        gt_boxes = torch.cat(gt_boxes_list, dim=0)
        gt_labels = gt_labels.repeat(self.denoising_groups * 2, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups * 2, 1)
        device = gt_labels.device
        batch_size = len(gt_labels_list)

        noised_labels = self.apply_label_noise(gt_labels)
        noised_boxes = inverse_sigmoid(self.apply_box_noise(gt_boxes))
        label_embedding = self.label_encoder(noised_labels)

        noised_query_nums = max_gt_num_per_image * self.denoising_groups * 2
        noised_label_queries = torch.zeros(
            (batch_size, noised_query_nums, self.label_embed_dim), device=device)
        noised_box_queries = torch.zeros(
            (batch_size, noised_query_nums, 4), device=device)

        batch_idx = torch.arange(batch_size, device=device)
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image, dtype=torch.long, device=device))
        batch_idx_per_group = batch_idx_per_instance.repeat(
            self.denoising_groups * 2, 1).flatten()

        valid_index_per_group = torch.cat(
            [torch.arange(num, device=device) for num in gt_nums_per_image], dim=0)
        valid_index_per_group = torch.cat(
            [
                valid_index_per_group + max_gt_num_per_image * group_idx
                for group_idx in range(self.denoising_groups * 2)
            ],
            dim=0,
        ).long()

        if batch_idx_per_group.numel() > 0:
            noised_label_queries[(batch_idx_per_group,
                                  valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group,
                                valid_index_per_group)] = noised_boxes

        attn_mask = self.generate_query_masks(2 * max_gt_num_per_image, device)
        return noised_label_queries, noised_box_queries, attn_mask, self.denoising_groups, max_gt_num_per_image * 2


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[idx], dims[idx + 1]) for idx in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class GroupedPredictionHead(nn.Module):
    """ResNeXt-style grouped bottleneck head for query classification."""

    def __init__(self, input_dim: int, output_dim: int, num_groups: int = 4, bottleneck_ratio: float = 0.5):
        super().__init__()
        self.num_groups = max(1, int(num_groups))
        if input_dim % self.num_groups != 0:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_groups={self.num_groups}")

        self.input_dim = input_dim
        self.group_dim = input_dim // self.num_groups
        self.bottleneck_dim = max(16, int(self.group_dim * bottleneck_ratio))
        self.norm = nn.LayerNorm(input_dim)
        self.group_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.group_dim, self.bottleneck_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.bottleneck_dim, self.group_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_groups)
            ]
        )
        self.out_proj = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.norm.bias, 0.0)
        nn.init.constant_(self.norm.weight, 1.0)
        for branch in self.group_mlps:
            for module in branch:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        grouped_inputs = residual.chunk(self.num_groups, dim=-1)
        grouped_outputs = [branch(chunk) for branch, chunk in zip(
            self.group_mlps, grouped_inputs)]
        fused = torch.cat(grouped_outputs, dim=-1) + residual
        return self.out_proj(fused)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Backbone(nn.Module):
    """ResNet-50 backbone returning layer2-4 or layer1-4 features."""

    DETECTOR_BACKBONE_DEFAULT_LOCAL_PATHS = {
        "relation_detr_backbone": [
            "./Model_Weight/pretrained/relation_detr_resnet50.pth",
            "./Model_Weight/pretrained/relation_detr_resnet50_12.pth",
            "./Model_Weight/pretrained/relation_detr_resnet50_24.pth",
            "./Model_Weight/pretrained/relation_detr_r50_12.pth",
            "./Model_Weight/pretrained/relation_detr_r50_24.pth",
            "./Model_Weight/pretrained/relation_detr_resnet50_backbone_only.pth",
            "./Pretrained/relation_detr_resnet50.pth",
            "./Pretrained/relation_detr_resnet50_12.pth",
            "./Pretrained/relation_detr_resnet50_24.pth",
            "./Pretrained/relation_detr_r50_12.pth",
            "./Pretrained/relation_detr_r50_24.pth",
            "./Pretrained/relation_detr_resnet50_backbone_only.pth",
        ],
    }

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stem_and_layer1: bool = False,
        use_fpn_features: bool = False,
        use_dc5: bool = False,
        backbone_pretrain_source: str = "imagenet",
        backbone_pretrain_checkpoint_path: str | None = None,
        backbone_pretrain_url: str | None = None,
    ):
        super().__init__()
        self.use_fpn_features = bool(use_fpn_features)
        self.backbone_pretrain_source = str(
            backbone_pretrain_source or "imagenet").lower()
        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained and self.backbone_pretrain_source == "imagenet"
            else None
        )
        backbone = torchvision.models.resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, bool(use_dc5)],
        )
        self.body = nn.ModuleDict(
            OrderedDict(
                [
                    ("conv1", backbone.conv1),
                    ("bn1", backbone.bn1),
                    ("relu", backbone.relu),
                    ("maxpool", backbone.maxpool),
                    ("layer1", backbone.layer1),
                    ("layer2", backbone.layer2),
                    ("layer3", backbone.layer3),
                    ("layer4", backbone.layer4),
                ]
            )
        )
        self.num_channels = [
            256, 512, 1024, 2048] if self.use_fpn_features else [512, 1024, 2048]

        detector_backbone_sources = {
            "detr_backbone",
            "detector_backbone",
            "deformable_detr_backbone",
            "dab_deformable_detr_backbone",
            "dn_dab_deformable_detr_backbone",
            "dino_backbone",
            "relation_detr_backbone",
        }
        if pretrained and self.backbone_pretrain_source in detector_backbone_sources:
            self._load_detr_backbone_pretrain(
                checkpoint_path=backbone_pretrain_checkpoint_path,
                checkpoint_url=backbone_pretrain_url,
            )

        if freeze_stem_and_layer1:
            for name in ["conv1", "bn1", "layer1"]:
                for param in self.body[name].parameters():
                    param.requires_grad = False

    @staticmethod
    def _unwrap_state_dict(state_dict: dict) -> dict:
        current = state_dict
        for key in ["model_state_dict", "state_dict", "model"]:
            if isinstance(current, dict) and key in current and isinstance(current[key], dict):
                current = current[key]
        while isinstance(current, dict) and "module" in current and isinstance(current["module"], dict):
            current = current["module"]
        return current

    def _extract_detr_backbone_state_dict(self, state_dict: dict) -> dict:
        state_dict = self._unwrap_state_dict(state_dict)
        mapped_state = {}
        allowed_roots = ("conv1.", "bn1.", "layer1.",
                         "layer2.", "layer3.", "layer4.")
        prefix_mappings = [
            "backbone.",
            "backbone.0.body.",
            "backbone.body.",
            "backbone.backbone.0.body.",
            "backbone.backbone.body.",
            "model.backbone.",
            "model.backbone.0.body.",
            "model.backbone.body.",
            "model.backbone.conv_encoder.model.",
            "backbone.conv_encoder.model.",
            "module.backbone.0.body.",
            "module.backbone.body.",
            "module.model.backbone.0.body.",
            "module.model.backbone.body.",
            "module.model.backbone.conv_encoder.model.",
            "module.backbone.conv_encoder.model.",
            "body.",
        ]

        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            while key.startswith("module."):
                key = key[len("module."):]
            mapped_key = None
            for source_prefix in prefix_mappings:
                if key.startswith(source_prefix):
                    suffix = key[len(source_prefix):]
                    if suffix.startswith("fc."):
                        mapped_key = None
                        break
                    if not suffix.startswith(allowed_roots):
                        mapped_key = None
                        break
                    mapped_key = "body." + suffix
                    break
            if mapped_key is None:
                continue
            mapped_state[mapped_key] = value
        return mapped_state

    def _load_detr_backbone_pretrain(
        self,
        checkpoint_path: str | None,
        checkpoint_url: str | None,
    ) -> None:
        resolved_path, resolved_url = self._resolve_detector_backbone_location(
            checkpoint_path=checkpoint_path,
            checkpoint_url=checkpoint_url,
        )
        state_dict = None
        if resolved_path:
            state_dict = torch.load(
                resolved_path, map_location="cpu", weights_only=False)
        else:
            if not resolved_url:
                raise ValueError(
                    "DETR backbone preload requires either checkpoint_path or checkpoint_url.")
            state_dict = torch.hub.load_state_dict_from_url(
                resolved_url,
                map_location="cpu",
                check_hash=False,
                progress=True,
            )

        backbone_state = self._extract_detr_backbone_state_dict(state_dict)
        if not backbone_state:
            raise RuntimeError(
                "Failed to extract ResNet-50 backbone weights from DETR checkpoint.")

        incompatible = self.load_state_dict(backbone_state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key]
        missing = [key for key in incompatible.missing_keys if not key.endswith(
            "num_batches_tracked")]
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys while loading DETR backbone preload: {unexpected[:10]}")
        if missing:
            raise RuntimeError(
                f"Missing keys while loading DETR backbone preload: {missing[:10]}")

    def _resolve_detector_backbone_location(
        self,
        checkpoint_path: str | None,
        checkpoint_url: str | None,
    ) -> tuple[str | None, str | None]:
        if checkpoint_path:
            resolved_path = os.path.abspath(
                os.path.expanduser(checkpoint_path))
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(
                    f"Detector backbone preload checkpoint not found: {resolved_path}"
                )
            return resolved_path, checkpoint_url

        for candidate in self.DETECTOR_BACKBONE_DEFAULT_LOCAL_PATHS.get(self.backbone_pretrain_source, []):
            resolved_candidate = os.path.abspath(os.path.expanduser(candidate))
            if os.path.exists(resolved_candidate):
                return resolved_candidate, checkpoint_url

        if checkpoint_url:
            return None, checkpoint_url

        if self.backbone_pretrain_source == "dab_deformable_detr_backbone":
            suggested = os.path.abspath(
                os.path.expanduser(
                    "./Model_Weight/pretrained/dab_deformable_detr_r50_v2.pth")
            )
            raise FileNotFoundError(
                "DAB-Deformable-DETR backbone preload is enabled, but no official checkpoint was found. "
                f"Download the official `DAB-Deformable-DETR-R50-v2` checkpoint from the DAB-DETR model zoo "
                f"(README links it via Google Drive / Tsinghua Cloud) and place it at: {suggested}"
            )

        if self.backbone_pretrain_source == "dn_dab_deformable_detr_backbone":
            suggested = os.path.abspath(
                os.path.expanduser(
                    "./Model_Weight/pretrained/dn_dab_deformable_detr_r50_v2.pth")
            )
            raise FileNotFoundError(
                "DN-DAB-Deformable-DETR backbone preload is enabled, but no checkpoint was found. "
                f"Place the official checkpoint at: {suggested}"
            )

        if self.backbone_pretrain_source == "relation_detr_backbone":
            suggested = os.path.abspath(
                os.path.expanduser(
                    "./Model_Weight/pretrained/relation_detr_resnet50_12.pth")
            )
            raise FileNotFoundError(
                "Relation-DETR backbone preload is enabled, but no checkpoint was found. "
                "Download the official Relation-DETR ResNet-50 checkpoint from the Relation-DETR model zoo "
                f"and place it at: {suggested}"
            )

        raise ValueError(
            "Detector backbone preload requires either an existing local checkpoint path, "
            "a known default local checkpoint file, or a direct download URL."
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.body["conv1"](x)
        x = self.body["bn1"](x)
        x = self.body["relu"](x)
        x = self.body["maxpool"](x)
        layer1 = self.body["layer1"](x)
        layer2 = self.body["layer2"](layer1)
        layer3 = self.body["layer3"](layer2)
        layer4 = self.body["layer4"](layer3)
        if self.use_fpn_features:
            return [layer1, layer2, layer3, layer4]
        return [layer2, layer3, layer4]


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention with official CUDA op fallback."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        use_official_cuda_op: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads
        self.use_official_cuda_op = bool(use_official_cuda_op)
        self.msda_status = get_msda_status() if self.use_official_cuda_op else {
            "available": False, "error": "disabled"}
        self.official_impl = None
        if (
            self.use_official_cuda_op
            and not self.msda_status.get("available", False)
            and build_or_load_msda_extension is not None
        ):
            try:
                build_or_load_msda_extension(try_build=True, verbose=False)
                self.msda_status = get_msda_status()
            except Exception as exc:
                self.msda_status = {"available": False, "error": str(exc)}
        if self.use_official_cuda_op and OfficialMSDeformAttn is not None and self.msda_status.get("available", False):
            self.official_impl = OfficialMSDeformAttn(
                d_model=hidden_dim,
                n_levels=num_levels,
                n_heads=num_heads,
                n_points=num_points,
            )
            return

        self.sampling_offsets = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.official_impl is not None:
            self.official_impl._reset_parameters()
            return
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1)
        for point_idx in range(self.num_points):
            grid_init[:, :, point_idx, :] *= point_idx + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.official_impl is not None:
            return self.official_impl(
                query,
                reference_points,
                input_flatten,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        batch_size, num_queries, _ = query.shape
        _, total_tokens, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        value = value.view(batch_size, total_tokens,
                           self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)
        output = query.new_zeros(
            (batch_size, num_queries, self.num_heads, self.head_dim))
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
        elif reference_points.shape[-1] != 4:
            raise ValueError("reference_points last dim must be 2 or 4")

        for level_idx in range(self.num_levels):
            height, width = spatial_shapes[level_idx].tolist()
            start_index = int(level_start_index[level_idx].item())
            end_index = start_index + height * width
            value_level = value[:, start_index:end_index]
            value_level = value_level.reshape(
                batch_size, height, width, self.num_heads, self.head_dim)
            value_level = value_level.permute(0, 3, 4, 1, 2).reshape(
                batch_size * self.num_heads, self.head_dim, height, width)
            if reference_points.shape[-1] == 2:
                sampling_locations = reference_points[:,
                                                      :, None, level_idx, None, :]
                sampling_locations = sampling_locations + (
                    sampling_offsets[:, :, :, level_idx] /
                    offset_normalizer[level_idx].view(1, 1, 1, 1, 2)
                )
            else:
                sampling_locations = reference_points[:, :, None, level_idx, None, :2] + (
                    sampling_offsets[:, :, :, level_idx] / self.num_points
                ) * reference_points[:, :, None, level_idx, None, 2:] * 0.5
            sampling_grid = sampling_locations.mul(2.0).sub(1.0)
            sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).reshape(
                batch_size * self.num_heads, num_queries, self.num_points, 2)
            sampled_value = F.grid_sample(
                value_level, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled_value = sampled_value.view(
                batch_size, self.num_heads, self.head_dim, num_queries, self.num_points).permute(0, 3, 1, 4, 2)
            output = output + \
                (sampled_value *
                 attention_weights[:, :, :, level_idx].unsqueeze(-1)).sum(dim=3)

        return self.output_proj(output.reshape(batch_size, num_queries, self.hidden_dim))


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_feature_levels: int,
        nheads: int,
        num_points: int,
        use_official_cuda_op: bool = True,
    ):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(
            hidden_dim,
            nheads,
            num_feature_levels,
            num_points,
            use_official_cuda_op=use_official_cuda_op,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = _get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_feature_levels: int,
        nheads: int,
        num_points: int,
        use_official_cuda_op: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = MultiScaleDeformableAttention(
            hidden_dim,
            nheads,
            num_feature_levels,
            num_points,
            use_official_cuda_op=use_official_cuda_op,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.activation = _get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt, attn_mask=self_attn_mask, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: DeformableTransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
    exchange_xy: bool = False,
) -> torch.Tensor:
    dim_t = torch.arange(num_pos_feats // 2,
                         dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (dim_t * 2 / num_pos_feats)
    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    if exchange_xy and pos_res.shape[-2] >= 2:
        index = torch.cat(
            [
                torch.arange(1, -1, -1, device=pos_res.device),
                torch.arange(2, pos_res.shape[-2], device=pos_res.device),
            ]
        )
        pos_res = torch.index_select(pos_res, -2, index)
    return pos_res.flatten(-2)


def box_rel_encoding(src_boxes: torch.Tensor, tgt_boxes: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xy1, wh1 = src_boxes.split([2, 2], dim=-1)
    xy2, wh2 = tgt_boxes.split([2, 2], dim=-1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    return torch.cat([delta_xy, delta_wh], dim=-1)


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 8,
        temperature: int = 10000,
        scale: float = 100.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.temperature = int(temperature)
        self.scale = float(scale)
        self.pos_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim * 4, self.num_heads, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.pos_proj[0].weight)
        nn.init.constant_(self.pos_proj[0].bias, 0.0)

    def forward(self, src_boxes: torch.Tensor, tgt_boxes: torch.Tensor | None = None) -> torch.Tensor:
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = get_sine_pos_embed(
                pos_embed,
                num_pos_feats=self.embed_dim,
                temperature=self.temperature,
                scale=self.scale,
                exchange_xy=False,
            ).permute(0, 3, 1, 2)
        return self.pos_proj(pos_embed).clone()


class RelationTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.memory_fusion = nn.Sequential(
            nn.Linear((self.num_layers + 1) *
                      self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        for module in self.memory_fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        outputs = [output]
        for layer in self.layers:
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)
            outputs.append(output)
        return self.memory_fusion(torch.cat(outputs, dim=-1))


class RelationTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ref_point_head = MLP(
            self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, 2)
        self.query_scale = MLP(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.position_relation_embedding = PositionRelationEmbedding(
            embed_dim=16, num_heads=self.num_heads)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        class_embed: nn.ModuleList,
        bbox_embed: nn.ModuleList,
        objectness_embed: nn.ModuleList | None = None,
        attn_mask: torch.Tensor | None = None,
        skip_relation: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None], list[torch.Tensor]]:
        outputs_classes: list[torch.Tensor] = []
        outputs_objectness: list[torch.Tensor | None] = []
        outputs_coords: list[torch.Tensor] = []
        valid_ratio_scale = torch.cat(
            [valid_ratios, valid_ratios], dim=-1)[:, None]
        pos_relation = attn_mask
        prev_boxes = None
        output = query
        current_reference = reference_points

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = current_reference[:,
                                                       :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :],
                num_pos_feats=self.hidden_dim // 2,
                exchange_xy=False,
            )
            query_pos = self.ref_point_head(query_sine_embed)
            if layer_idx != 0:
                query_pos = query_pos * self.query_scale(output)

            output = layer(
                output,
                query_pos,
                reference_points_input,
                value,
                spatial_shapes,
                level_start_index,
                src_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )
            decoder_feature = self.norm(output)
            pred_logits = class_embed[layer_idx](decoder_feature)
            pred_objectness = objectness_embed[layer_idx](
                decoder_feature) if objectness_embed is not None else None
            pred_boxes = bbox_embed[layer_idx](
                decoder_feature) + inverse_sigmoid(current_reference)
            pred_boxes = pred_boxes.sigmoid()

            outputs_classes.append(pred_logits)
            outputs_objectness.append(pred_objectness)
            outputs_coords.append(pred_boxes)

            if layer_idx < self.num_layers - 1:
                if not skip_relation:
                    src_boxes = prev_boxes if prev_boxes is not None else current_reference
                    pos_relation = self.position_relation_embedding(
                        src_boxes, pred_boxes).flatten(0, 1)
                    if attn_mask is not None:
                        pos_relation = pos_relation.masked_fill(
                            attn_mask, float("-inf"))
                prev_boxes = pred_boxes
                current_reference = pred_boxes.detach()

        return outputs_classes, outputs_objectness, outputs_coords


class QueryRelationLayer(nn.Module):
    """Lightweight relation-aware refinement over two-stage top-k queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        geometry_hidden_dim: int = 128,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(5, geometry_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(geometry_hidden_dim, num_heads),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        for module in self.geometry_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _pairwise_geometry(reference_boxes: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        cx, cy, w, h = reference_boxes.unbind(dim=-1)
        dx = (cx.unsqueeze(2) - cx.unsqueeze(1)) / (w.unsqueeze(2) + eps)
        dy = (cy.unsqueeze(2) - cy.unsqueeze(1)) / (h.unsqueeze(2) + eps)
        dw = torch.log((w.unsqueeze(2) + eps) / (w.unsqueeze(1) + eps))
        dh = torch.log((h.unsqueeze(2) + eps) / (h.unsqueeze(1) + eps))
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        iou_maps = []
        for batch_boxes in boxes_xyxy:
            iou_maps.append(box_iou(batch_boxes, batch_boxes))
        iou = torch.stack(iou_maps, dim=0)
        return torch.stack([dx, dy, dw, dh, iou], dim=-1)

    def forward(self, query: torch.Tensor, reference_boxes: torch.Tensor) -> torch.Tensor:
        residual = query
        query_norm = self.norm1(query)
        batch_size, num_queries, _ = query_norm.shape
        q = self.q_proj(query_norm).view(batch_size, num_queries,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(query_norm).view(batch_size, num_queries,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(query_norm).view(batch_size, num_queries,
                                         self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(
            q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        geometry_bias = self.geometry_mlp(
            self._pairwise_geometry(reference_boxes)).permute(0, 3, 1, 2)
        attn = F.softmax(attn_logits + geometry_bias, dim=-1)
        attn = self.attn_dropout(attn)

        relation_context = torch.matmul(attn, v).transpose(
            1, 2).contiguous().view(batch_size, num_queries, self.hidden_dim)
        query = residual + self.out_dropout(self.out_proj(relation_context))
        query = query + self.ffn(self.norm2(query))
        return query


class QueryRelationEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        geometry_hidden_dim: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QueryRelationLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_dim=ffn_dim,
                    geometry_hidden_dim=geometry_hidden_dim,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, query: torch.Tensor, reference_boxes: torch.Tensor) -> torch.Tensor:
        output = query
        for layer in self.layers:
            output = layer(output, reference_boxes)
        return output


class DETRModel(nn.Module):
    """Two-stage multi-scale Deformable DETR-R50."""

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_queries: int = 20,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        freeze_stem_and_layer1: bool = False,
        num_feature_levels: int = 3,
        encoder_n_points: int = 4,
        decoder_n_points: int = 4,
        activation: str = "relu",
        two_stage: bool = True,
        use_official_cuda_msda: bool = True,
        use_fpn_features: bool = False,
        use_exp32: bool = False,
        exp32_num_groups: int = 4,
        use_backbone_dc5: bool = False,
        use_relation_detr_main: bool = False,
        align_relation_official_head_loss: bool = False,
        use_relation_hybrid: bool = False,
        relation_hybrid_num_proposals: int | None = None,
        relation_hybrid_assign: int = 6,
        use_relation_cdn: bool = False,
        relation_denoising_nums: int = 100,
        relation_label_noise_prob: float = 0.5,
        relation_box_noise_scale: float = 1.0,
        use_query_relation: bool = False,
        query_relation_layers: int = 1,
        query_relation_num_heads: int = 8,
        query_relation_ffn_dim: int = 1024,
        query_relation_geometry_hidden_dim: int = 128,
        backbone_pretrain_source: str = "imagenet",
        backbone_pretrain_checkpoint_path: str | None = None,
        backbone_pretrain_url: str | None = None,
        use_aux_digit_classifier: bool = False,
        aux_digit_pool_size: int = 5,
        aux_digit_hidden_dim: int = 256,
        aux_digit_classifier_fusion_weight: float = 0.0,
        use_aux_digit_classifier_gated_fusion: bool = False,
        aux_digit_gate_top1_threshold: float = 0.90,
        aux_digit_gate_margin_threshold: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_feature_levels = max(1, int(num_feature_levels))
        self.two_stage = two_stage
        self.use_fpn_features = bool(use_fpn_features)
        self.use_relation_detr_main = bool(
            use_relation_detr_main) and bool(two_stage)
        self.align_relation_official_head_loss = bool(
            align_relation_official_head_loss) and self.use_relation_detr_main
        self.use_relation_hybrid = bool(
            use_relation_hybrid) and self.use_relation_detr_main
        self.relation_hybrid_num_proposals = max(
            int(relation_hybrid_num_proposals or (num_queries * 2)), int(num_queries))
        self.relation_hybrid_assign = max(1, int(relation_hybrid_assign))
        self.use_relation_cdn = bool(
            use_relation_cdn) and self.use_relation_detr_main
        self.relation_denoising_nums = max(1, int(relation_denoising_nums))
        self.relation_label_noise_prob = float(relation_label_noise_prob)
        self.relation_box_noise_scale = float(relation_box_noise_scale)
        self.use_exp32 = bool(
            use_exp32) and not self.align_relation_official_head_loss
        self.exp32_num_groups = max(1, int(exp32_num_groups))
        self.use_query_relation = bool(use_query_relation) and bool(
            two_stage) and not self.use_relation_detr_main
        self.use_aux_digit_classifier = bool(use_aux_digit_classifier)
        self.aux_digit_pool_size = max(1, int(aux_digit_pool_size))
        self.aux_digit_classifier_fusion_weight = float(
            aux_digit_classifier_fusion_weight)
        self.use_aux_digit_classifier_gated_fusion = bool(
            use_aux_digit_classifier_gated_fusion)
        self.aux_digit_gate_top1_threshold = float(
            aux_digit_gate_top1_threshold)
        self.aux_digit_gate_margin_threshold = float(
            aux_digit_gate_margin_threshold)
        if bool(use_relation_detr_main) and not bool(two_stage):
            raise ValueError(
                "Relation-DETR main trunk currently requires two_stage=True.")
        self.backbone = Backbone(
            pretrained=pretrained_backbone,
            freeze_stem_and_layer1=freeze_stem_and_layer1,
            use_fpn_features=self.use_fpn_features,
            use_dc5=bool(use_backbone_dc5),
            backbone_pretrain_source=backbone_pretrain_source,
            backbone_pretrain_checkpoint_path=backbone_pretrain_checkpoint_path,
            backbone_pretrain_url=backbone_pretrain_url,
        )
        input_proj = []
        backbone_out_channels = self.backbone.num_channels
        if self.use_fpn_features:
            self.fpn_lateral = nn.ModuleList(
                [nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                 for in_channels in backbone_out_channels]
            )
            self.fpn_output = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim,
                                  kernel_size=3, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                    for _ in backbone_out_channels
                ]
            )
        else:
            for in_channels in backbone_out_channels:
                input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
        extra_levels = max(0, self.num_feature_levels -
                           len(backbone_out_channels))
        for level_idx in range(extra_levels):
            in_channels = backbone_out_channels[-1] if level_idx == 0 and not self.use_fpn_features else hidden_dim
            input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
        self.input_proj = nn.ModuleList(input_proj[: self.num_feature_levels])
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, hidden_dim))

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            activation,
            self.num_feature_levels,
            nheads,
            encoder_n_points,
            use_official_cuda_op=use_official_cuda_msda,
        )
        if self.use_relation_detr_main:
            self.encoder = RelationTransformerEncoder(
                encoder_layer, num_encoder_layers, hidden_dim)
        else:
            self.encoder = DeformableTransformerEncoder(
                encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            activation,
            self.num_feature_levels,
            nheads,
            decoder_n_points,
            use_official_cuda_op=use_official_cuda_msda,
        )
        if self.use_relation_detr_main:
            self.decoder = RelationTransformerDecoder(
                decoder_layer, num_decoder_layers, hidden_dim, nheads)
        else:
            self.decoder = DeformableTransformerDecoder(
                decoder_layer, num_decoder_layers)

        if self.two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            if self.use_exp32:
                self.enc_class_embed = GroupedPredictionHead(
                    hidden_dim, num_classes, num_groups=self.exp32_num_groups)
                self.enc_objectness_embed = nn.Linear(hidden_dim, 1)
            else:
                self.enc_class_embed = nn.Linear(hidden_dim, num_classes)
                self.enc_objectness_embed = None
            self.enc_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.pos_trans = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(hidden_dim * 2)
            self.relation_tgt_embed = nn.Embedding(
                num_queries, hidden_dim) if self.use_relation_detr_main else None
            self.hybrid_tgt_embed = (
                nn.Embedding(self.relation_hybrid_num_proposals, hidden_dim)
                if self.use_relation_hybrid
                else None
            )
            self.query_relation_encoder = (
                QueryRelationEncoder(
                    hidden_dim=hidden_dim,
                    num_heads=int(query_relation_num_heads),
                    num_layers=int(query_relation_layers),
                    dropout=dropout,
                    ffn_dim=int(query_relation_ffn_dim),
                    geometry_hidden_dim=int(
                        query_relation_geometry_hidden_dim),
                )
                if self.use_query_relation
                else None
            )
            self.hybrid_enc_class_embed = nn.Linear(
                hidden_dim, num_classes) if self.use_relation_hybrid else None
            self.hybrid_enc_bbox_embed = MLP(
                hidden_dim, hidden_dim, 4, 3) if self.use_relation_hybrid else None
            self.denoising_generator = (
                GenerateCDNQueries(
                    num_queries=num_queries,
                    num_classes=num_classes,
                    label_embed_dim=hidden_dim,
                    denoising_nums=self.relation_denoising_nums,
                    label_noise_prob=self.relation_label_noise_prob,
                    box_noise_scale=self.relation_box_noise_scale,
                )
                if self.use_relation_cdn
                else None
            )
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
            self.reference_points = nn.Linear(hidden_dim, 2)

        if self.use_exp32:
            self.class_embed = nn.ModuleList(
                [GroupedPredictionHead(
                    hidden_dim, num_classes, num_groups=self.exp32_num_groups) for _ in range(num_decoder_layers)]
            )
            self.objectness_embed = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in range(num_decoder_layers)])
        else:
            self.class_embed = nn.ModuleList(
                [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
            self.objectness_embed = None
        self.bbox_embed = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)])
        self.hybrid_class_embed = (
            nn.ModuleList([nn.Linear(hidden_dim, num_classes)
                          for _ in range(num_decoder_layers)])
            if self.use_relation_hybrid
            else None
        )
        self.hybrid_bbox_embed = (
            nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3)
                          for _ in range(num_decoder_layers)])
            if self.use_relation_hybrid
            else None
        )
        if self.use_aux_digit_classifier:
            aux_hidden_dim = max(32, int(aux_digit_hidden_dim))
            self.aux_digit_classifier_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, aux_hidden_dim,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(aux_hidden_dim, aux_hidden_dim,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.aux_digit_classifier_head = nn.Linear(
                aux_hidden_dim, num_classes)
        else:
            self.aux_digit_classifier_conv = None
            self.aux_digit_classifier_head = None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        if self.two_stage:
            nn.init.xavier_uniform_(self.enc_output.weight)
            nn.init.constant_(self.enc_output.bias, 0.0)
            if isinstance(self.enc_class_embed, nn.Linear):
                if self.use_relation_detr_main:
                    prior_prob = 0.01
                    bias_value = -math.log((1 - prior_prob) / prior_prob)
                    nn.init.constant_(self.enc_class_embed.bias, bias_value)
                else:
                    nn.init.constant_(self.enc_class_embed.bias, 0.0)
            else:
                self.enc_class_embed.reset_parameters()
            if self.enc_objectness_embed is not None:
                nn.init.xavier_uniform_(self.enc_objectness_embed.weight)
                nn.init.constant_(self.enc_objectness_embed.bias, 0.0)
            nn.init.xavier_uniform_(self.pos_trans.weight)
            nn.init.constant_(self.pos_trans.bias, 0.0)
            nn.init.constant_(self.enc_bbox_embed.layers[-1].weight, 0.0)
            nn.init.constant_(self.enc_bbox_embed.layers[-1].bias, 0.0)
            if self.relation_tgt_embed is not None:
                nn.init.normal_(self.relation_tgt_embed.weight)
            if self.hybrid_tgt_embed is not None:
                nn.init.normal_(self.hybrid_tgt_embed.weight)
            if self.hybrid_enc_class_embed is not None:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.hybrid_enc_class_embed.bias, bias_value)
            if self.hybrid_enc_bbox_embed is not None:
                nn.init.constant_(
                    self.hybrid_enc_bbox_embed.layers[-1].weight, 0.0)
                nn.init.constant_(
                    self.hybrid_enc_bbox_embed.layers[-1].bias, 0.0)
        else:
            nn.init.normal_(self.query_embed.weight, std=0.02)
            nn.init.normal_(self.query_pos_embed.weight, std=0.02)
            nn.init.xavier_uniform_(self.reference_points.weight)
            nn.init.constant_(self.reference_points.bias, 0.0)
        for module in self.input_proj:
            nn.init.xavier_uniform_(module[0].weight)
            nn.init.constant_(module[0].bias, 0.0)
        if self.use_fpn_features:
            for module in self.fpn_lateral:
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            for module in self.fpn_output:
                nn.init.xavier_uniform_(module[0].weight)
                nn.init.constant_(module[0].bias, 0.0)
        for class_embed in self.class_embed:
            if isinstance(class_embed, nn.Linear):
                if self.use_relation_detr_main:
                    prior_prob = 0.01
                    bias_value = -math.log((1 - prior_prob) / prior_prob)
                    nn.init.constant_(class_embed.bias, bias_value)
                else:
                    nn.init.constant_(class_embed.bias, 0.0)
            else:
                class_embed.reset_parameters()
        if self.objectness_embed is not None:
            for objectness_embed in self.objectness_embed:
                nn.init.xavier_uniform_(objectness_embed.weight)
                nn.init.constant_(objectness_embed.bias, 0.0)
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_embed.layers[-1].bias, 0.0)
        if self.hybrid_class_embed is not None:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for class_embed in self.hybrid_class_embed:
                nn.init.constant_(class_embed.bias, bias_value)
        if self.hybrid_bbox_embed is not None:
            for bbox_embed in self.hybrid_bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight, 0.0)
                nn.init.constant_(bbox_embed.layers[-1].bias, 0.0)
        if self.aux_digit_classifier_conv is not None:
            for module in self.aux_digit_classifier_conv.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
        if self.aux_digit_classifier_head is not None:
            nn.init.xavier_uniform_(self.aux_digit_classifier_head.weight)
            nn.init.constant_(self.aux_digit_classifier_head.bias, 0.0)

    def get_parameter_groups(self, lr_base: float, lr_backbone: float):
        backbone_params = [
            param for param in self.backbone.parameters() if param.requires_grad]
        head_params = []
        for module in [self.input_proj, self.encoder, self.decoder, self.class_embed, self.bbox_embed]:
            head_params.extend(
                [param for param in module.parameters() if param.requires_grad])
        if self.objectness_embed is not None:
            head_params.extend(
                [param for param in self.objectness_embed.parameters() if param.requires_grad])
        if self.hybrid_class_embed is not None:
            head_params.extend(
                [param for param in self.hybrid_class_embed.parameters() if param.requires_grad])
        if self.hybrid_bbox_embed is not None:
            head_params.extend(
                [param for param in self.hybrid_bbox_embed.parameters() if param.requires_grad])
        if self.aux_digit_classifier_conv is not None:
            head_params.extend(
                [param for param in self.aux_digit_classifier_conv.parameters() if param.requires_grad])
        if self.aux_digit_classifier_head is not None:
            head_params.extend(
                [param for param in self.aux_digit_classifier_head.parameters() if param.requires_grad])
        if self.use_fpn_features:
            for module in [self.fpn_lateral, self.fpn_output]:
                head_params.extend(
                    [param for param in module.parameters() if param.requires_grad])
        head_params.append(self.level_embed)
        if self.two_stage:
            modules = [self.enc_output, self.enc_output_norm, self.enc_class_embed,
                       self.enc_bbox_embed, self.pos_trans, self.pos_trans_norm]
            if self.enc_objectness_embed is not None:
                modules.append(self.enc_objectness_embed)
            if self.query_relation_encoder is not None:
                modules.append(self.query_relation_encoder)
            if self.relation_tgt_embed is not None:
                modules.append(self.relation_tgt_embed)
            if self.hybrid_tgt_embed is not None:
                modules.append(self.hybrid_tgt_embed)
            if self.hybrid_enc_class_embed is not None:
                modules.append(self.hybrid_enc_class_embed)
            if self.hybrid_enc_bbox_embed is not None:
                modules.append(self.hybrid_enc_bbox_embed)
            if self.denoising_generator is not None:
                modules.append(self.denoising_generator)
            for module in modules:
                head_params.extend(
                    [param for param in module.parameters() if param.requires_grad])
        else:
            for module in [self.query_embed, self.query_pos_embed, self.reference_points]:
                head_params.extend(
                    [param for param in module.parameters() if param.requires_grad])
        return [{"params": backbone_params, "lr": lr_backbone}, {"params": head_params, "lr": lr_base}]

    @staticmethod
    def _combine_class_and_objectness(
        class_logits: torch.Tensor,
        objectness_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if objectness_logits is None:
            return class_logits.sigmoid()
        return class_logits.softmax(dim=-1) * objectness_logits.sigmoid()

    @staticmethod
    def _get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        valid_height = (~mask[:, :, 0]).sum(dim=1)
        valid_width = (~mask[:, 0, :]).sum(dim=1)
        return torch.stack([valid_width.float() / mask.shape[2], valid_height.float() / mask.shape[1]], dim=-1)

    @staticmethod
    def _get_reference_points(spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor, device: torch.device) -> torch.Tensor:
        reference_points = []
        for level_idx, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height,
                               dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width,
                               dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, level_idx, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, level_idx, 0] * width)
            reference_points.append(torch.stack((ref_x, ref_y), dim=-1))
        reference_points = torch.cat(reference_points, dim=1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def _get_proposal_pos_embed(self, proposals_unact: torch.Tensor) -> torch.Tensor:
        num_pos_feats = self.hidden_dim // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals_unact.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals_unact.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def _build_fpn_features(self, backbone_features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [lateral(feature) for lateral, feature in zip(
            self.fpn_lateral, backbone_features)]
        for level_idx in range(len(laterals) - 1, 0, -1):
            laterals[level_idx - 1] = laterals[level_idx - 1] + F.interpolate(
                laterals[level_idx],
                size=laterals[level_idx - 1].shape[-2:],
                mode="nearest",
            )
        return [output(feature) for output, feature in zip(self.fpn_output, laterals)]

    def _prepare_multi_scale_features(self, images: torch.Tensor, masks: torch.Tensor):
        backbone_features = self.backbone(images)
        if self.use_fpn_features:
            backbone_features = self._build_fpn_features(backbone_features)
        selected_features = list(backbone_features[: min(
            len(backbone_features), self.num_feature_levels)])
        while len(selected_features) < self.num_feature_levels:
            if len(selected_features) == len(backbone_features) and not self.use_fpn_features:
                next_feature = self.input_proj[len(selected_features)](
                    backbone_features[-1])
            else:
                next_feature = self.input_proj[len(selected_features)](
                    selected_features[-1])
            selected_features.append(next_feature)

        src_flatten = []
        mask_flatten = []
        pos_flatten = []
        spatial_shapes = []
        masks_per_level = []
        projected_features = []
        for level_idx, feature in enumerate(selected_features):
            if level_idx < len(backbone_features) and not self.use_fpn_features:
                src = self.input_proj[level_idx](feature)
            else:
                src = feature
            mask = F.interpolate(
                masks[None].float(), size=src.shape[-2:], mode="nearest")[0].to(torch.bool)
            pos = self.position_embedding(
                mask) + self.level_embed[level_idx].view(1, -1, 1, 1)
            _, _, height, width = src.shape
            spatial_shapes.append((height, width))
            masks_per_level.append(mask)
            projected_features.append(src)
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_flatten.append(pos.flatten(2).transpose(1, 2))

        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        pos_flatten = torch.cat(pos_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=images.device)
        level_start_index = torch.cat(
            [spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]])
        valid_ratios = torch.stack(
            [self._get_valid_ratio(mask) for mask in masks_per_level], dim=1)
        return src_flatten, mask_flatten, pos_flatten, spatial_shapes, level_start_index, valid_ratios, projected_features, masks_per_level

    @staticmethod
    def _infer_image_sizes(images: torch.Tensor, masks: torch.Tensor, targets: list[dict] | None) -> torch.Tensor:
        if targets is not None and len(targets) == images.shape[0]:
            sizes = []
            for target in targets:
                size = target.get("size")
                if size is None:
                    break
                sizes.append(size.to(images.device))
            if len(sizes) == images.shape[0]:
                return torch.stack(sizes, dim=0).to(dtype=torch.float32)

        valid_rows = (~masks).any(dim=2).sum(dim=1)
        valid_cols = (~masks).any(dim=1).sum(dim=1)
        return torch.stack([valid_rows, valid_cols], dim=1).to(dtype=torch.float32)

    def _compute_aux_digit_logits(
        self,
        feature_map: torch.Tensor,
        query_boxes: torch.Tensor,
        image_sizes: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        if self.aux_digit_classifier_conv is None or self.aux_digit_classifier_head is None:
            raise RuntimeError(
                "Auxiliary digit classifier is not initialized.")

        batch_size, num_queries = query_boxes.shape[:2]
        pooled = self._pool_query_roi_features(
            feature_map=feature_map,
            query_boxes=query_boxes,
            image_sizes=image_sizes,
            image_shape=image_shape,
            pool_size=self.aux_digit_pool_size,
        )
        pooled = self.aux_digit_classifier_conv(pooled).flatten(1)
        aux_logits = self.aux_digit_classifier_head(pooled)
        return aux_logits.view(batch_size, num_queries, self.num_classes)

    def _compute_aux_digit_gate(
        self,
        detector_logits: torch.Tensor,
        detector_objectness_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        detector_probs = self._combine_class_and_objectness(
            detector_logits.detach(),
            None if detector_objectness_logits is None else detector_objectness_logits.detach(),
        )
        topk_k = 2 if detector_probs.shape[-1] > 1 else 1
        top_values = detector_probs.topk(k=topk_k, dim=-1).values
        top1 = top_values[..., 0]
        if top_values.shape[-1] > 1:
            margin = top1 - top_values[..., 1]
        else:
            margin = top1
        gate_mask = (top1 < self.aux_digit_gate_top1_threshold) | (
            margin < self.aux_digit_gate_margin_threshold)
        return gate_mask.to(dtype=detector_logits.dtype).unsqueeze(-1)

    @staticmethod
    def _build_query_rois(
        query_boxes: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_queries = query_boxes.shape[:2]
        boxes_xyxy = box_cxcywh_to_xyxy(query_boxes.float().detach())
        scale = torch.stack(
            [image_sizes[:, 1], image_sizes[:, 0],
                image_sizes[:, 1], image_sizes[:, 0]],
            dim=1,
        ).unsqueeze(1)
        boxes_xyxy = boxes_xyxy * scale
        width_limits = image_sizes[:, 1].view(-1, 1)
        height_limits = image_sizes[:, 0].view(-1, 1)
        zero = boxes_xyxy.new_zeros((batch_size, num_queries))
        x1 = torch.maximum(zero, torch.minimum(
            boxes_xyxy[..., 0], width_limits))
        y1 = torch.maximum(zero, torch.minimum(
            boxes_xyxy[..., 1], height_limits))
        x2 = torch.maximum(
            x1 + 1e-3, torch.minimum(boxes_xyxy[..., 2], width_limits))
        y2 = torch.maximum(
            y1 + 1e-3, torch.minimum(boxes_xyxy[..., 3], height_limits))
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        rois = []
        for batch_idx in range(batch_size):
            batch_boxes = boxes_xyxy[batch_idx]
            batch_column = batch_boxes.new_full(
                (num_queries, 1), float(batch_idx))
            rois.append(torch.cat([batch_column, batch_boxes], dim=1))
        return torch.cat(rois, dim=0)

    def _pool_query_roi_features(
        self,
        feature_map: torch.Tensor,
        query_boxes: torch.Tensor,
        image_sizes: torch.Tensor,
        image_shape: tuple[int, int],
        pool_size: int,
    ) -> torch.Tensor:
        rois = self._build_query_rois(
            query_boxes=query_boxes, image_sizes=image_sizes)
        spatial_scale = feature_map.shape[-1] / max(1.0, float(image_shape[1]))
        return roi_align(
            input=feature_map,
            boxes=rois,
            output_size=(pool_size, pool_size),
            spatial_scale=float(spatial_scale),
            aligned=True,
        )

    def _generate_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.shape[0]
        proposals = []
        current_index = 0
        for level_idx, (height, width) in enumerate(spatial_shapes.tolist()):
            mask_level = memory_padding_mask[:, current_index:current_index +
                                             height * width].view(batch_size, height, width)
            valid_height = (~mask_level[:, :, 0]).sum(dim=1)
            valid_width = (~mask_level[:, 0, :]).sum(dim=1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32,
                             device=memory.device),
                torch.arange(width, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)
            scale = torch.stack([valid_width, valid_height],
                                dim=1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0) + 0.5) / scale
            base_scale = 0.05 * (2.0 ** level_idx)
            wh = torch.ones_like(grid) * base_scale
            proposals.append(
                torch.cat([grid, wh], dim=-1).view(batch_size, -1, 4))
            current_index += height * width

        output_proposals = torch.cat(proposals, dim=1).clamp(1e-4, 1.0 - 1e-4)
        output_proposals_unact = inverse_sigmoid(output_proposals)
        output_memory = memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), 0.0)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals_unact

    @staticmethod
    def _prepare_relation_targets(targets: list[dict] | None) -> list[dict]:
        prepared_targets = []
        if targets is None:
            return prepared_targets
        for target in targets:
            boxes = target["boxes"]
            size = target["size"].to(boxes.device)
            height, width = size[0].float(), size[1].float()
            if boxes.numel() == 0:
                boxes_normalized = boxes.reshape(0, 4)
            else:
                scale = torch.tensor(
                    [width, height, width, height], device=boxes.device)
                boxes_normalized = box_xyxy_to_cxcywh(boxes / scale)
            prepared_target = dict(target)
            prepared_target["boxes_normalized"] = boxes_normalized
            prepared_targets.append(prepared_target)
        return prepared_targets

    def forward(self, images: torch.Tensor, masks: torch.Tensor, targets: list[dict] | None = None) -> dict:
        (
            src_flatten,
            mask_flatten,
            pos_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            projected_features,
            masks_per_level,
        ) = self._prepare_multi_scale_features(images, masks)
        prepared_targets = self._prepare_relation_targets(targets)
        encoder_reference_points = self._get_reference_points(
            spatial_shapes, valid_ratios, images.device)
        memory = self.encoder(
            src_flatten,
            pos_flatten,
            encoder_reference_points,
            spatial_shapes,
            level_start_index,
            mask_flatten,
        )

        batch_size = images.shape[0]
        noised_label_queries = None
        noised_box_queries = None
        relation_attn_mask = None
        denoising_groups = None
        max_gt_num_per_image = None
        if self.training and self.use_relation_cdn and self.denoising_generator is not None and prepared_targets:
            gt_labels_list = [target["labels"] for target in prepared_targets]
            gt_boxes_list = [target["boxes_normalized"]
                             for target in prepared_targets]
            (
                noised_label_queries,
                noised_box_queries,
                relation_attn_mask,
                denoising_groups,
                max_gt_num_per_image,
            ) = self.denoising_generator(gt_labels_list, gt_boxes_list)

        if self.two_stage:
            encoder_memory, output_proposals_unact = self._generate_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.enc_class_embed(encoder_memory)
            enc_outputs_objectness = self.enc_objectness_embed(
                encoder_memory) if self.enc_objectness_embed is not None else None
            enc_outputs_coord_unact = self.enc_bbox_embed(
                encoder_memory) + output_proposals_unact
            enc_scores = self._combine_class_and_objectness(
                enc_outputs_class, enc_outputs_objectness).max(dim=-1)[0]
            enc_scores = enc_scores.masked_fill(mask_flatten, float("-inf"))
            topk_indices = enc_scores.topk(
                min(self.num_queries, enc_scores.shape[1]), dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, 4),
            ).detach()
            reference_points = topk_coords_unact.sigmoid()
            if self.use_relation_detr_main:
                query_pos = None
                proposal_query_count = reference_points.shape[1]
                tgt = self.relation_tgt_embed.weight[:proposal_query_count].unsqueeze(
                    0).expand(batch_size, -1, -1)
                if noised_label_queries is not None and noised_box_queries is not None:
                    tgt = torch.cat([noised_label_queries, tgt], dim=1)
                    reference_points = torch.cat(
                        [noised_box_queries.sigmoid(), reference_points], dim=1)
                    if relation_attn_mask is not None and relation_attn_mask.shape[0] != tgt.shape[1]:
                        relation_attn_mask = relation_attn_mask[: tgt.shape[1],
                                                                : tgt.shape[1]]
            else:
                topk_memory = torch.gather(
                    encoder_memory,
                    1,
                    topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                )
                query_embed = self.pos_trans_norm(self.pos_trans(
                    self._get_proposal_pos_embed(topk_coords_unact)))
                query_pos, tgt = torch.split(
                    query_embed, self.hidden_dim, dim=-1)
                tgt = tgt + topk_memory
                if self.query_relation_encoder is not None:
                    tgt = self.query_relation_encoder(
                        tgt, topk_coords_unact.sigmoid())
        else:
            query_pos = self.query_pos_embed.weight.unsqueeze(
                0).repeat(batch_size, 1, 1)
            tgt = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points(query_pos).sigmoid()

        if self.use_relation_detr_main:
            outputs_classes, outputs_objectness, outputs_coords = self.decoder(
                query=tgt,
                reference_points=reference_points,
                value=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                class_embed=self.class_embed,
                bbox_embed=self.bbox_embed,
                objectness_embed=self.objectness_embed,
                attn_mask=relation_attn_mask,
            )
        else:
            outputs_classes = []
            outputs_objectness = []
            outputs_coords = []
            current_reference = reference_points
            current_tgt = tgt
            for layer_idx, decoder_layer in enumerate(self.decoder.layers):
                if current_reference.shape[-1] == 4:
                    reference_points_input = current_reference[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], dim=-1)[:, None]
                else:
                    reference_points_input = current_reference[:,
                                                               :, None] * valid_ratios[:, None]
                current_tgt = decoder_layer(
                    current_tgt, query_pos, reference_points_input, memory, spatial_shapes, level_start_index, mask_flatten)
                outputs_class = self.class_embed[layer_idx](current_tgt)
                outputs_objectness_layer = self.objectness_embed[layer_idx](
                    current_tgt) if self.objectness_embed is not None else None
                delta_box = self.bbox_embed[layer_idx](current_tgt)
                if current_reference.shape[-1] == 4:
                    delta_box = delta_box + inverse_sigmoid(current_reference)
                else:
                    delta_box[..., :2] = delta_box[..., :2] + \
                        inverse_sigmoid(current_reference)
                outputs_coord = delta_box.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_objectness.append(outputs_objectness_layer)
                outputs_coords.append(outputs_coord)
                current_reference = outputs_coord.detach()

        dn_outputs = None
        if noised_label_queries is not None and noised_box_queries is not None and max_gt_num_per_image is not None:
            padding_size = max_gt_num_per_image * \
                max(int(denoising_groups or 0), 0)
            if padding_size > 0:
                dn_classes = []
                dn_objectness = []
                dn_coords = []
                main_classes = []
                main_objectness = []
                main_coords = []
                for pred_class, pred_objectness, pred_boxes in zip(outputs_classes, outputs_objectness, outputs_coords):
                    dn_classes.append(pred_class[:, :padding_size, :])
                    dn_objectness.append(
                        pred_objectness[:, :padding_size, :] if pred_objectness is not None else None)
                    dn_coords.append(pred_boxes[:, :padding_size, :])
                    main_classes.append(pred_class[:, padding_size:, :])
                    main_objectness.append(
                        pred_objectness[:, padding_size:, :] if pred_objectness is not None else None)
                    main_coords.append(pred_boxes[:, padding_size:, :])
                outputs_classes = main_classes
                outputs_objectness = main_objectness
                outputs_coords = main_coords
                dn_outputs = {
                    "pred_logits": dn_classes[-1],
                    "pred_boxes": dn_coords[-1],
                    "aux_outputs": [
                        {"pred_logits": pred_class, "pred_boxes": pred_boxes}
                        for pred_class, pred_boxes in zip(dn_classes[:-1], dn_coords[:-1])
                    ],
                    "denoising_groups": int(denoising_groups or 0),
                    "max_gt_num_per_image": int(max_gt_num_per_image),
                }

        final_logits = outputs_classes[-1]
        final_boxes = outputs_coords[-1]
        image_sizes = None
        if projected_features and self.use_aux_digit_classifier:
            image_sizes = self._infer_image_sizes(images, masks, targets)

        aux_digit_logits = None
        aux_digit_gate = None
        if self.use_aux_digit_classifier and projected_features and image_sizes is not None:
            aux_digit_logits = self._compute_aux_digit_logits(
                feature_map=projected_features[0],
                query_boxes=final_boxes,
                image_sizes=image_sizes,
                image_shape=(images.shape[-2], images.shape[-1]),
            )
        refined_boxes = final_boxes
        refined_logits = final_logits
        fused_final_logits = refined_logits
        if (
            aux_digit_logits is not None
            and not self.training
            and abs(self.aux_digit_classifier_fusion_weight) > 1e-8
        ):
            if self.use_aux_digit_classifier_gated_fusion:
                aux_digit_gate = self._compute_aux_digit_gate(
                    detector_logits=refined_logits,
                    detector_objectness_logits=outputs_objectness[-1],
                )
            else:
                aux_digit_gate = 1.0
            fused_final_logits = refined_logits + \
                (aux_digit_gate * self.aux_digit_classifier_fusion_weight * aux_digit_logits)
        output = {"pred_logits": fused_final_logits if not self.training else refined_logits,
                  "pred_boxes": refined_boxes}
        if aux_digit_logits is not None:
            output["pred_aux_digit_logits"] = aux_digit_logits
            output["pred_detector_logits"] = final_logits
        if aux_digit_gate is not None:
            output["pred_aux_digit_gate"] = aux_digit_gate
        if outputs_objectness[-1] is not None:
            output["pred_class_logits"] = fused_final_logits if not self.training else refined_logits
            output["pred_objectness_logits"] = outputs_objectness[-1]
        if self.two_stage:
            enc_output = {"pred_logits": enc_outputs_class,
                          "pred_boxes": enc_outputs_coord_unact.sigmoid()}
            if enc_outputs_objectness is not None:
                enc_output["pred_class_logits"] = enc_outputs_class
                enc_output["pred_objectness_logits"] = enc_outputs_objectness
            output["enc_outputs"] = enc_output
        if len(outputs_classes) > 1:
            aux_outputs = []
            for pred_logits, pred_objectness, pred_boxes in zip(outputs_classes[:-1], outputs_objectness[:-1], outputs_coords[:-1]):
                aux_output = {"pred_logits": pred_logits,
                              "pred_boxes": pred_boxes}
                if pred_objectness is not None:
                    aux_output["pred_class_logits"] = pred_logits
                    aux_output["pred_objectness_logits"] = pred_objectness
                aux_outputs.append(aux_output)
            output["aux_outputs"] = aux_outputs
        if dn_outputs is not None:
            output["dn_outputs"] = dn_outputs

        if self.training and self.use_relation_hybrid and self.hybrid_tgt_embed is not None:
            hybrid_enc_class = self.hybrid_enc_class_embed(encoder_memory)
            hybrid_enc_coord_unact = self.hybrid_enc_bbox_embed(
                encoder_memory) + output_proposals_unact
            hybrid_scores = hybrid_enc_class.sigmoid().max(
                dim=-1)[0].masked_fill(mask_flatten, float("-inf"))
            hybrid_topk = min(
                self.relation_hybrid_num_proposals, hybrid_scores.shape[1])
            hybrid_topk_indices = hybrid_scores.topk(hybrid_topk, dim=1)[1]
            hybrid_reference_points = torch.gather(
                hybrid_enc_coord_unact,
                1,
                hybrid_topk_indices.unsqueeze(-1).expand(-1, -1, 4),
            ).detach().sigmoid()
            hybrid_enc_class = torch.gather(
                hybrid_enc_class,
                1,
                hybrid_topk_indices.unsqueeze(-1).expand(-1, -1,
                                                         self.num_classes),
            )
            hybrid_enc_coord = torch.gather(
                hybrid_enc_coord_unact.sigmoid(),
                1,
                hybrid_topk_indices.unsqueeze(-1).expand(-1, -1, 4),
            )
            hybrid_tgt = self.hybrid_tgt_embed.weight[:hybrid_topk].unsqueeze(
                0).expand(batch_size, -1, -1)
            hybrid_classes, _, hybrid_coords = self.decoder(
                query=hybrid_tgt,
                reference_points=hybrid_reference_points,
                value=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                class_embed=self.hybrid_class_embed,
                bbox_embed=self.hybrid_bbox_embed,
                objectness_embed=None,
                skip_relation=True,
            )
            hybrid_outputs = {
                "pred_logits": hybrid_classes[-1],
                "pred_boxes": hybrid_coords[-1],
                "aux_outputs": [
                    {"pred_logits": pred_class, "pred_boxes": pred_boxes}
                    for pred_class, pred_boxes in zip(hybrid_classes[:-1], hybrid_coords[:-1])
                ],
                "enc_outputs": {"pred_logits": hybrid_enc_class, "pred_boxes": hybrid_enc_coord},
                "hybrid_assign": self.relation_hybrid_assign,
            }
            output["hybrid_outputs"] = hybrid_outputs
        return output


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)), dim=-1)


def complete_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise CIoU for boxes in xyxy format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (bottom_right - top_left).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    wh1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=eps)
    wh2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=eps)
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    union = area1[:, None] + area2[None, :] - inter + eps
    iou = inter / union

    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    center_distance = (
        (centers1[:, None, :] - centers2[None, :, :]) ** 2).sum(dim=-1)

    enclose_top_left = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_bottom_right = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_bottom_right - enclose_top_left).clamp(min=eps)
    diagonal_distance = (enclose_wh ** 2).sum(dim=-1) + eps

    atan1 = torch.atan(wh1[:, 0] / wh1[:, 1])
    atan2 = torch.atan(wh2[:, 0] / wh2[:, 1])
    v = (4.0 / (math.pi ** 2)) * (atan1[:, None] - atan2[None, :]) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - (center_distance / diagonal_distance + alpha * v)


def rescale_boxes_to_pixels(boxes: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    height, width = size_hw.unbind(-1)
    scale = torch.stack([width, height, width, height],
                        dim=-1).to(boxes.device)
    return boxes * scale


def _lazy_import_hf_rtdetr_v2():
    try:
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RT-DETRv2 backend requires `transformers`. "
            "Install `transformers` (and likely `timm`) in the active environment."
        ) from exc
    return RTDetrV2Config, RTDetrV2ForObjectDetection


def _is_hf_rtdetr_v2_backend_name(model_backend: str | None) -> bool:
    return str(model_backend or "").lower() in {"hf_rtdetr_v2", "hf_rtdetr_v2_aux", "hf_rtdetr_v2_qs"}


class HFRTDetrV2Adapter(nn.Module):
    """Thin wrapper around Hugging Face RT-DETRv2 for local train/eval loops."""

    def __init__(
        self,
        num_classes: int = 10,
        num_queries: int = 300,
        hf_model_name_or_path: str = "PekingU/rtdetr_v2_r50vd",
        hf_load_strategy: str = "pretrained_reset_transformer",
        hf_ignore_mismatched_sizes: bool = True,
        hf_eos_coefficient: float | None = None,
        hf_backbone_name: str = "resnet50",
        hf_use_timm_backbone: bool = True,
        hf_use_pretrained_backbone: bool = True,
    ):
        super().__init__()
        RTDetrV2Config, RTDetrV2ForObjectDetection = _lazy_import_hf_rtdetr_v2()

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.model_name_or_path = str(hf_model_name_or_path)
        self.hf_load_strategy = str(hf_load_strategy).lower()

        config = RTDetrV2Config.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_classes,
            num_queries=self.num_queries,
        )
        config.id2label = {index: str(index) for index in range(self.num_classes)}
        config.label2id = {str(index): index for index in range(self.num_classes)}
        if hf_eos_coefficient is not None and hasattr(config, "eos_coefficient"):
            config.eos_coefficient = float(hf_eos_coefficient)

        if self.hf_load_strategy == "backbone_only_pretrained":
            if hasattr(config, "backbone"):
                config.backbone = str(hf_backbone_name)
            if hasattr(config, "use_timm_backbone"):
                config.use_timm_backbone = bool(hf_use_timm_backbone)
            if hasattr(config, "use_pretrained_backbone"):
                config.use_pretrained_backbone = bool(hf_use_pretrained_backbone)
            self.model = RTDetrV2ForObjectDetection(config)
        elif self.hf_load_strategy in {"pretrained", "pretrained_reset_transformer"}:
            self.model = RTDetrV2ForObjectDetection.from_pretrained(
                self.model_name_or_path,
                config=config,
                ignore_mismatched_sizes=bool(hf_ignore_mismatched_sizes),
            )
            if self.hf_load_strategy == "pretrained_reset_transformer":
                if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
                    self.model.model.encoder.apply(self.model._init_weights)
                if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
                    self.model.model.decoder.apply(self.model._init_weights)
        elif self.hf_load_strategy == "config_init":
            self.model = RTDetrV2ForObjectDetection(config)
        else:
            raise ValueError(
                "Unsupported hf_load_strategy. Expected one of "
                "{'backbone_only_pretrained', 'pretrained', 'pretrained_reset_transformer', 'config_init'}."
            )

    def get_parameter_groups(self, lr_base: float, lr_backbone: float):
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.model.backbone.") or ".backbone." in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [{"params": backbone_params, "lr": lr_backbone}, {"params": head_params, "lr": lr_base}]

    @staticmethod
    def adapt_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return state_dict

    def _build_hf_labels(self, targets: list[dict]) -> list[dict]:
        hf_labels = []
        for target in targets:
            boxes = target["boxes"]
            size = target["size"].to(boxes.device)
            height, width = size[0].float(), size[1].float()
            if boxes.numel() == 0:
                normalized_boxes = boxes.reshape(0, 4)
            else:
                scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
                normalized_boxes = box_xyxy_to_cxcywh(boxes / scale)
            hf_labels.append(
                {
                    "class_labels": target["labels"].long(),
                    "boxes": normalized_boxes,
                }
            )
        return hf_labels

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict:
        pixel_mask = None if masks is None else (~masks).to(dtype=torch.long)
        hf_labels = self._build_hf_labels(targets) if targets is not None else None
        hf_outputs = self.model(
            pixel_values=images,
            pixel_mask=pixel_mask,
            labels=hf_labels,
            return_dict=True,
            output_hidden_states=False,
        )

        output = {
            "pred_logits": hf_outputs.logits,
            "pred_boxes": hf_outputs.pred_boxes,
        }
        if getattr(hf_outputs, "loss", None) is not None:
            output["official_loss"] = hf_outputs.loss
        if getattr(hf_outputs, "loss_dict", None) is not None:
            output["official_loss_dict"] = hf_outputs.loss_dict
        return output


class HFRTDetrV2AuxAdapter(HFRTDetrV2Adapter):
    """HF RT-DETRv2 with a training-side auxiliary digit classification head."""

    _DEFAULT_CONFUSION_FAMILIES: tuple[tuple[int, ...], ...] = (
        (3, 5),
        (6, 8),
        (1, 4, 7),
    )

    def __init__(
        self,
        num_classes: int = 10,
        num_queries: int = 300,
        hf_model_name_or_path: str = "PekingU/rtdetr_v2_r50vd",
        hf_load_strategy: str = "pretrained_reset_transformer",
        hf_ignore_mismatched_sizes: bool = True,
        hf_eos_coefficient: float | None = None,
        hf_backbone_name: str = "resnet50",
        hf_use_timm_backbone: bool = True,
        hf_use_pretrained_backbone: bool = True,
        aux_digit_hidden_dim: int = 256,
        aux_digit_classifier_fusion_weight: float = 0.0,
        use_aux_digit_classifier_gated_fusion: bool = False,
        aux_digit_gate_top1_threshold: float = 0.90,
        aux_digit_gate_margin_threshold: float = 0.15,
        use_aux_digit_confusion_family_selective_fusion: bool = False,
        use_aux_digit_confusion_family_attenuation: bool = False,
        aux_digit_confusion_families: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
        aux_digit_family_fusion_weights: dict | None = None,
        aux_digit_family_attenuation_weights: dict | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            num_queries=num_queries,
            hf_model_name_or_path=hf_model_name_or_path,
            hf_load_strategy=hf_load_strategy,
            hf_ignore_mismatched_sizes=hf_ignore_mismatched_sizes,
            hf_eos_coefficient=hf_eos_coefficient,
            hf_backbone_name=hf_backbone_name,
            hf_use_timm_backbone=hf_use_timm_backbone,
            hf_use_pretrained_backbone=hf_use_pretrained_backbone,
        )
        hidden_dim = int(getattr(self.model.config, "d_model", 256))
        aux_hidden_dim = max(32, int(aux_digit_hidden_dim))
        self.aux_digit_classifier_norm = nn.LayerNorm(hidden_dim)
        self.aux_digit_classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, aux_hidden_dim),
            nn.GELU(),
            nn.Linear(aux_hidden_dim, self.num_classes),
        )
        self.aux_digit_classifier_fusion_weight = float(aux_digit_classifier_fusion_weight)
        self.use_aux_digit_classifier_gated_fusion = bool(use_aux_digit_classifier_gated_fusion)
        self.aux_digit_gate_top1_threshold = float(aux_digit_gate_top1_threshold)
        self.aux_digit_gate_margin_threshold = float(aux_digit_gate_margin_threshold)
        self.use_aux_digit_confusion_family_selective_fusion = bool(
            use_aux_digit_confusion_family_selective_fusion
        )
        self.use_aux_digit_confusion_family_attenuation = bool(
            use_aux_digit_confusion_family_attenuation
        )
        self.aux_digit_confusion_families = self._parse_confusion_families(
            aux_digit_confusion_families,
            num_classes=self.num_classes,
        )
        self.aux_digit_family_fusion_weights = self._parse_family_fusion_weights(
            aux_digit_family_fusion_weights,
            families=self.aux_digit_confusion_families,
            default_weight=self.aux_digit_classifier_fusion_weight,
        )
        self.aux_digit_family_attenuation_weights = self._parse_family_fusion_weights(
            aux_digit_family_attenuation_weights,
            families=self.aux_digit_confusion_families,
            default_weight=self.aux_digit_classifier_fusion_weight,
        )
        self._reset_aux_digit_classifier_parameters()

    def _reset_aux_digit_classifier_parameters(self) -> None:
        nn.init.constant_(self.aux_digit_classifier_norm.bias, 0.0)
        nn.init.constant_(self.aux_digit_classifier_norm.weight, 1.0)
        for module in self.aux_digit_classifier_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _extract_decoder_query_features(hf_outputs) -> torch.Tensor | None:
        intermediate_hidden_states = getattr(hf_outputs, "intermediate_hidden_states", None)
        if intermediate_hidden_states is not None and torch.is_tensor(intermediate_hidden_states):
            if intermediate_hidden_states.ndim == 4 and intermediate_hidden_states.shape[1] > 0:
                return intermediate_hidden_states[:, -1]
        last_hidden_state = getattr(hf_outputs, "last_hidden_state", None)
        if last_hidden_state is not None and torch.is_tensor(last_hidden_state):
            return last_hidden_state
        return None

    @classmethod
    def _canonical_family_key(cls, family: list[int] | tuple[int, ...]) -> str:
        return ",".join(str(int(value)) for value in sorted(set(int(v) for v in family)))

    @classmethod
    def _parse_confusion_families(
        cls,
        value,
        num_classes: int,
    ) -> tuple[tuple[int, ...], ...]:
        raw_families = value if value is not None else cls._DEFAULT_CONFUSION_FAMILIES
        parsed_families: list[tuple[int, ...]] = []
        if not isinstance(raw_families, (list, tuple)):
            return cls._DEFAULT_CONFUSION_FAMILIES
        for family in raw_families:
            if not isinstance(family, (list, tuple)):
                continue
            cleaned = tuple(
                sorted(
                    {
                        int(class_id)
                        for class_id in family
                        if 0 <= int(class_id) < int(num_classes)
                    }
                )
            )
            if len(cleaned) >= 2:
                parsed_families.append(cleaned)
        return tuple(parsed_families) if parsed_families else cls._DEFAULT_CONFUSION_FAMILIES

    @classmethod
    def _parse_family_fusion_weights(
        cls,
        value,
        families: tuple[tuple[int, ...], ...],
        default_weight: float,
    ) -> tuple[float, ...]:
        if not isinstance(value, dict):
            return tuple(float(default_weight) for _ in families)
        parsed_weights: list[float] = []
        for family in families:
            canonical_key = cls._canonical_family_key(family)
            raw_weight = value.get(canonical_key, default_weight)
            try:
                parsed_weights.append(float(raw_weight))
            except (TypeError, ValueError):
                parsed_weights.append(float(default_weight))
        return tuple(parsed_weights)

    def _compute_aux_digit_gate(self, detector_logits: torch.Tensor) -> torch.Tensor:
        detector_probs = detector_logits.detach().float().sigmoid()
        topk_k = 2 if detector_probs.shape[-1] > 1 else 1
        top_values = detector_probs.topk(k=topk_k, dim=-1).values
        top1 = top_values[..., 0]
        if top_values.shape[-1] > 1:
            margin = top1 - top_values[..., 1]
        else:
            margin = top1
        gate_mask = (top1 < self.aux_digit_gate_top1_threshold) | (
            margin < self.aux_digit_gate_margin_threshold
        )
        return gate_mask.to(dtype=detector_logits.dtype).unsqueeze(-1)

    def _compute_aux_digit_family_selective_masks(
        self,
        detector_logits: torch.Tensor,
        family_weights: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_family_weights = (
            family_weights
            if family_weights is not None
            else self.aux_digit_family_fusion_weights
        )
        detector_probs = detector_logits.detach().float().sigmoid()
        topk_k = 2 if detector_probs.shape[-1] > 1 else 1
        topk = detector_probs.topk(k=topk_k, dim=-1)
        top_values = topk.values
        top_indices = topk.indices

        top1 = top_values[..., 0]
        top1_labels = top_indices[..., 0]
        if top_values.shape[-1] > 1:
            top2_labels = top_indices[..., 1]
            margin = top1 - top_values[..., 1]
        else:
            top2_labels = top1_labels
            margin = top1

        batch_size, num_queries, num_classes = detector_logits.shape
        query_gate = detector_logits.new_zeros((batch_size, num_queries, 1))
        family_mask = detector_logits.new_zeros((batch_size, num_queries, num_classes))
        family_weight_mask = detector_logits.new_zeros((batch_size, num_queries, num_classes))

        for family, family_weight in zip(self.aux_digit_confusion_families, active_family_weights):
            class_ids = list(family)
            top1_in_family = torch.zeros_like(top1_labels, dtype=torch.bool)
            top2_in_family = torch.zeros_like(top2_labels, dtype=torch.bool)
            for class_id in class_ids:
                top1_in_family |= top1_labels == int(class_id)
                top2_in_family |= top2_labels == int(class_id)

            family_active = top1_in_family & (
                top2_in_family | (margin < self.aux_digit_gate_margin_threshold)
            )
            if not torch.any(family_active):
                continue

            expanded_active = family_active.unsqueeze(-1).to(dtype=detector_logits.dtype)
            query_gate = torch.maximum(query_gate, expanded_active)
            for class_id in class_ids:
                family_mask[..., int(class_id)] = torch.maximum(
                    family_mask[..., int(class_id)],
                    family_active.to(dtype=detector_logits.dtype),
                )
                family_weight_mask[..., int(class_id)] = torch.where(
                    family_active,
                    detector_logits.new_full(family_active.shape, float(family_weight)),
                    family_weight_mask[..., int(class_id)],
                )

        return query_gate, family_mask, family_weight_mask

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict:
        pixel_mask = None if masks is None else (~masks).to(dtype=torch.long)
        hf_labels = self._build_hf_labels(targets) if targets is not None else None
        hf_outputs = self.model(
            pixel_values=images,
            pixel_mask=pixel_mask,
            labels=hf_labels,
            return_dict=True,
            output_hidden_states=False,
        )

        detector_logits = hf_outputs.logits
        output = {
            "pred_logits": detector_logits,
            "pred_boxes": hf_outputs.pred_boxes,
        }
        if getattr(hf_outputs, "loss", None) is not None:
            output["official_loss"] = hf_outputs.loss
        if getattr(hf_outputs, "loss_dict", None) is not None:
            output["official_loss_dict"] = hf_outputs.loss_dict

        decoder_query_features = self._extract_decoder_query_features(hf_outputs)
        if decoder_query_features is None:
            raise RuntimeError(
                "HF RT-DETRv2 auxiliary backend could not extract decoder query features."
            )
        normalized_features = self.aux_digit_classifier_norm(decoder_query_features)
        aux_digit_logits = self.aux_digit_classifier_head(normalized_features)
        aux_digit_gate = None
        aux_digit_family_mask = None
        aux_digit_effective_weight_mask = None
        if not self.training and abs(self.aux_digit_classifier_fusion_weight) > 1e-8:
            if self.use_aux_digit_confusion_family_selective_fusion:
                aux_digit_gate, aux_digit_family_mask, aux_digit_family_weight_mask = (
                    self._compute_aux_digit_family_selective_masks(detector_logits)
                )
                if self.use_aux_digit_classifier_gated_fusion:
                    aux_digit_gate = aux_digit_gate * self._compute_aux_digit_gate(detector_logits)
                aux_digit_effective_weight_mask = aux_digit_family_mask * aux_digit_family_weight_mask
                output["pred_logits"] = detector_logits + (
                    aux_digit_gate * aux_digit_effective_weight_mask * aux_digit_logits
                )
            else:
                if self.use_aux_digit_classifier_gated_fusion:
                    aux_digit_gate = self._compute_aux_digit_gate(detector_logits)
                else:
                    aux_digit_gate = torch.ones_like(aux_digit_logits[..., :1], dtype=detector_logits.dtype)
                aux_digit_effective_weight_mask = torch.full_like(
                    aux_digit_logits,
                    fill_value=float(self.aux_digit_classifier_fusion_weight),
                )
                if self.use_aux_digit_confusion_family_attenuation:
                    (
                        _attenuation_query_gate,
                        aux_digit_family_mask,
                        aux_digit_family_attenuation_weight_mask,
                    ) = self._compute_aux_digit_family_selective_masks(
                        detector_logits,
                        family_weights=self.aux_digit_family_attenuation_weights,
                    )
                    aux_digit_effective_weight_mask = torch.where(
                        aux_digit_family_mask > 0,
                        aux_digit_family_attenuation_weight_mask,
                        aux_digit_effective_weight_mask,
                    )
                output["pred_logits"] = detector_logits + (
                    aux_digit_gate * aux_digit_effective_weight_mask * aux_digit_logits
                )
        output["pred_aux_digit_logits"] = aux_digit_logits
        output["pred_detector_logits"] = detector_logits
        if aux_digit_gate is not None:
            output["pred_aux_digit_gate"] = aux_digit_gate
        if aux_digit_family_mask is not None:
            output["pred_aux_digit_family_mask"] = aux_digit_family_mask
        if aux_digit_effective_weight_mask is not None:
            output["pred_aux_digit_effective_weight_mask"] = aux_digit_effective_weight_mask
        return output


class HFRTDetrV2QuerySelectionAdapter(HFRTDetrV2Adapter):
    """HF RT-DETRv2 with quality-aware encoder proposal selection."""

    def __init__(
        self,
        num_classes: int = 10,
        num_queries: int = 300,
        hf_model_name_or_path: str = "PekingU/rtdetr_v2_r50vd",
        hf_load_strategy: str = "pretrained_reset_transformer",
        hf_ignore_mismatched_sizes: bool = True,
        hf_eos_coefficient: float | None = None,
        hf_backbone_name: str = "resnet50",
        hf_use_timm_backbone: bool = True,
        hf_use_pretrained_backbone: bool = True,
        query_quality_hidden_dim: int = 256,
        query_selection_alpha: float = 1.0,
        query_selection_beta: float = 1.0,
    ):
        super().__init__(
            num_classes=num_classes,
            num_queries=num_queries,
            hf_model_name_or_path=hf_model_name_or_path,
            hf_load_strategy=hf_load_strategy,
            hf_ignore_mismatched_sizes=hf_ignore_mismatched_sizes,
            hf_eos_coefficient=hf_eos_coefficient,
            hf_backbone_name=hf_backbone_name,
            hf_use_timm_backbone=hf_use_timm_backbone,
            hf_use_pretrained_backbone=hf_use_pretrained_backbone,
        )
        hidden_dim = int(getattr(self.model.config, "d_model", 256))
        quality_hidden_dim = max(32, int(query_quality_hidden_dim))
        self.query_quality_norm = nn.LayerNorm(hidden_dim)
        self.query_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, quality_hidden_dim),
            nn.GELU(),
            nn.Linear(quality_hidden_dim, 1),
        )
        self.query_selection_alpha = max(0.0, float(query_selection_alpha))
        self.query_selection_beta = max(0.0, float(query_selection_beta))
        self._reset_query_quality_parameters()

    def _reset_query_quality_parameters(self) -> None:
        nn.init.constant_(self.query_quality_norm.bias, 0.0)
        nn.init.constant_(self.query_quality_norm.weight, 1.0)
        for module in self.query_quality_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _lazy_import_denoising_helper():
        try:
            from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import get_contrastive_denoising_training_group
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "HF RT-DETRv2 quality-selection backend requires the RT-DETRv2 modeling helpers from transformers."
            ) from exc
        return get_contrastive_denoising_training_group

    def _build_detection_outputs_with_quality_selection(
        self,
        images: torch.Tensor,
        pixel_mask: torch.Tensor | None,
        hf_labels: list[dict] | None,
    ) -> dict:
        rtdetr_model = self.model.model
        config = self.model.config

        if pixel_mask is None:
            batch_size, _, height, width = images.shape
            pixel_mask = torch.ones(
                (batch_size, height, width),
                device=images.device,
                dtype=torch.long,
            )

        features = rtdetr_model.backbone(images, pixel_mask)
        proj_feats = [
            rtdetr_model.encoder_input_proj[level](source)
            for level, (source, _mask) in enumerate(features)
        ]
        encoder_outputs = rtdetr_model.encoder(proj_feats)

        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(rtdetr_model.decoder_input_proj[level](source))

        if config.num_feature_levels > len(sources):
            source_count = len(sources)
            sources.append(rtdetr_model.decoder_input_proj[source_count](encoder_outputs.last_hidden_state)[-1])
            for level in range(source_count + 1, config.num_feature_levels):
                sources.append(rtdetr_model.decoder_input_proj[level](encoder_outputs.last_hidden_state[-1]))

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=images.device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, dim=1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        if self.training and config.num_denoising > 0 and hf_labels is not None:
            get_contrastive_denoising_training_group = self._lazy_import_denoising_helper()
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=hf_labels,
                num_classes=config.num_labels,
                num_queries=config.num_queries,
                class_embed=rtdetr_model.denoising_class_embed,
                num_denoising_queries=config.num_denoising,
                label_noise_ratio=config.label_noise_ratio,
                box_noise_scale=config.box_noise_scale,
            )
        else:
            denoising_class = None
            denoising_bbox_unact = None
            attention_mask = None
            denoising_meta_values = None

        dtype = source_flatten.dtype
        if self.training or config.anchor_image_size is None:
            anchors, valid_mask = rtdetr_model.generate_anchors(
                tuple(spatial_shapes_list),
                device=images.device,
                dtype=dtype,
            )
        else:
            anchors, valid_mask = rtdetr_model.anchors, rtdetr_model.valid_mask
            anchors = anchors.to(images.device, dtype)
            valid_mask = valid_mask.to(images.device, dtype)

        memory = valid_mask.to(dtype) * source_flatten
        output_memory = rtdetr_model.enc_output(memory)

        enc_outputs_class = rtdetr_model.enc_score_head(output_memory)
        enc_outputs_coord_logits = rtdetr_model.enc_bbox_head(output_memory) + anchors

        normalized_output_memory = self.query_quality_norm(output_memory)
        enc_outputs_quality_logits = self.query_quality_head(normalized_output_memory)
        class_confidence = enc_outputs_class.float().sigmoid().amax(dim=-1)
        quality_confidence = enc_outputs_quality_logits.float().sigmoid().squeeze(-1)
        selection_score = class_confidence.pow(self.query_selection_alpha) * quality_confidence.pow(self.query_selection_beta)
        valid_anchor_mask = valid_mask.squeeze(-1).to(dtype=torch.bool)
        selection_score = selection_score.masked_fill(~valid_anchor_mask, -1.0)

        _, topk_ind = torch.topk(selection_score, config.num_queries, dim=1)
        gather_index_4 = topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        reference_points_unact = enc_outputs_coord_logits.gather(dim=1, index=gather_index_4)
        enc_topk_bboxes = reference_points_unact.sigmoid()

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat([denoising_bbox_unact, reference_points_unact], dim=1)

        gather_index_logits = topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=gather_index_logits)
        enc_topk_quality_logits = enc_outputs_quality_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_quality_logits.shape[-1]),
        )

        if config.learn_initial_query:
            target = rtdetr_model.weight_embedding.tile([images.shape[0], 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
            ).detach()

        if denoising_class is not None:
            target = torch.cat([denoising_class, target], dim=1)

        init_reference_points = reference_points_unact.detach()
        decoder_outputs = rtdetr_model.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        outputs_class = decoder_outputs.intermediate_logits
        outputs_coord = decoder_outputs.intermediate_reference_points
        predicted_corners = decoder_outputs.intermediate_predicted_corners
        initial_reference_points = decoder_outputs.initial_reference_points
        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        official_loss = None
        official_loss_dict = None
        official_auxiliary_outputs = None
        if hf_labels is not None:
            official_loss, official_loss_dict, official_auxiliary_outputs = self.model.loss_function(
                logits,
                hf_labels,
                self.model.device,
                pred_boxes,
                config,
                outputs_class,
                outputs_coord,
                enc_topk_logits=enc_topk_logits,
                enc_topk_bboxes=enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values if self.training else None,
                predicted_corners=predicted_corners,
                initial_reference_points=initial_reference_points,
            )

        return {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
            "official_loss": official_loss,
            "official_loss_dict": official_loss_dict,
            "official_auxiliary_outputs": official_auxiliary_outputs,
            "decoder_last_hidden_state": decoder_outputs.last_hidden_state,
            "decoder_intermediate_hidden_states": decoder_outputs.intermediate_hidden_states,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord_logits": enc_outputs_coord_logits,
            "enc_outputs_quality_logits": enc_outputs_quality_logits,
            "enc_topk_logits": enc_topk_logits,
            "enc_topk_bboxes": enc_topk_bboxes,
            "enc_topk_quality_logits": enc_topk_quality_logits,
            "enc_query_selection_scores": selection_score,
            "enc_valid_mask": valid_anchor_mask,
        }

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict:
        pixel_mask = None if masks is None else (~masks).to(dtype=torch.long)
        hf_labels = self._build_hf_labels(targets) if targets is not None else None
        outputs = self._build_detection_outputs_with_quality_selection(
            images=images,
            pixel_mask=pixel_mask,
            hf_labels=hf_labels,
        )

        output = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
            "enc_outputs_class": outputs["enc_outputs_class"],
            "enc_outputs_coord_logits": outputs["enc_outputs_coord_logits"],
            "enc_outputs_quality_logits": outputs["enc_outputs_quality_logits"],
            "enc_topk_logits": outputs["enc_topk_logits"],
            "enc_topk_bboxes": outputs["enc_topk_bboxes"],
            "enc_topk_quality_logits": outputs["enc_topk_quality_logits"],
            "enc_query_selection_scores": outputs["enc_query_selection_scores"],
            "enc_valid_mask": outputs["enc_valid_mask"],
        }
        if outputs["official_loss"] is not None:
            output["official_loss"] = outputs["official_loss"]
        if outputs["official_loss_dict"] is not None:
            output["official_loss_dict"] = outputs["official_loss_dict"]
        return output


def _validate_hf_rtdetr_v2_official_mode(config: dict) -> None:
    unsupported_flags: list[str] = []

    def _nonzero(name: str, eps: float = 1e-8) -> bool:
        try:
            return abs(float(config.get(name, 0.0))) > eps
        except (TypeError, ValueError):
            return False

    if bool(config.get("use_aux_digit_classifier", False)):
        unsupported_flags.append("use_aux_digit_classifier")
    if _nonzero("aux_digit_classifier_loss_weight"):
        unsupported_flags.append("aux_digit_classifier_loss_weight")
    if _nonzero("aux_digit_classifier_fusion_weight"):
        unsupported_flags.append("aux_digit_classifier_fusion_weight")
    if bool(config.get("use_aux_digit_classifier_gated_fusion", False)):
        unsupported_flags.append("use_aux_digit_classifier_gated_fusion")
    if _nonzero("query_quality_loss_weight"):
        unsupported_flags.append("query_quality_loss_weight")

    if unsupported_flags:
        joined = ", ".join(unsupported_flags)
        raise ValueError(
            "HF RT-DETRv2 backend now runs in pure official mode. "
            f"Disable these custom settings for hf_rtdetr_v2: {joined}."
        )


def _validate_hf_rtdetr_v2_aux_mode(config: dict) -> None:
    """Keep the HF aux backend aligned to EXP56 and free of EXP59 query-quality logic."""

    unsupported_flags: list[str] = []

    def _nonzero(name: str, default: float = 0.0, eps: float = 1e-8) -> bool:
        try:
            return abs(float(config.get(name, default))) > eps
        except (TypeError, ValueError):
            return True

    if _nonzero("query_quality_loss_weight"):
        unsupported_flags.append("query_quality_loss_weight")
    if "query_quality_hidden_dim" in config:
        unsupported_flags.append("query_quality_hidden_dim")
    if "query_selection_alpha" in config:
        unsupported_flags.append("query_selection_alpha")
    if "query_selection_beta" in config:
        unsupported_flags.append("query_selection_beta")

    if unsupported_flags:
        joined = ", ".join(unsupported_flags)
        raise ValueError(
            "hf_rtdetr_v2_aux must stay on the EXP56-style architecture. "
            f"Remove query-quality settings: {joined}."
        )


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        iou_cost_type: str = "giou",
        class_cost_type: str = "auto",
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_cost_type = str(iou_cost_type).lower()
        self.class_cost_type = str(class_cost_type).lower()

    @staticmethod
    def _get_class_probabilities(outputs: dict) -> torch.Tensor:
        if "pred_class_logits" in outputs and "pred_objectness_logits" in outputs:
            class_prob = outputs["pred_class_logits"].float().softmax(dim=-1)
            objectness_prob = outputs["pred_objectness_logits"].float(
            ).sigmoid()
            return class_prob * objectness_prob
        return outputs["pred_logits"].float().sigmoid()

    def _calculate_class_cost(self, outputs: dict, batch_idx: int, tgt_ids: torch.Tensor) -> torch.Tensor:
        has_decoupled_outputs = "pred_class_logits" in outputs and "pred_objectness_logits" in outputs
        use_focal_cost = self.class_cost_type == "focal" or (
            self.class_cost_type == "auto" and not has_decoupled_outputs
        )

        if use_focal_cost:
            out_prob = outputs["pred_logits"][batch_idx].float().sigmoid()
            neg_cost = -(1 - self.focal_alpha) * (out_prob **
                                                  self.focal_gamma) * (1 - out_prob + 1e-8).log()
            pos_cost = -self.focal_alpha * \
                ((1 - out_prob) ** self.focal_gamma) * (out_prob + 1e-8).log()
            return pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

        out_prob = self._get_class_probabilities(outputs)[batch_idx]
        return -torch.log(out_prob[:, tgt_ids].clamp_min(1e-8))

    def _pairwise_iou_cost(self, pred_boxes_xyxy: torch.Tensor, tgt_boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if self.iou_cost_type == "ciou":
            return -complete_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
        return -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        pred_boxes = outputs["pred_boxes"].float()
        indices = []
        for batch_idx in range(pred_boxes.shape[0]):
            tgt_ids = targets[batch_idx]["labels"]
            tgt_boxes = targets[batch_idx]["boxes_normalized"]
            if tgt_ids.numel() == 0:
                indices.append(
                    (
                        torch.empty((0,), dtype=torch.int64,
                                    device=pred_boxes.device),
                        torch.empty((0,), dtype=torch.int64,
                                    device=pred_boxes.device),
                    )
                )
                continue
            cost_class = self._calculate_class_cost(
                outputs, batch_idx, tgt_ids)
            cost_bbox = torch.cdist(pred_boxes[batch_idx], tgt_boxes, p=1)
            cost_giou = self._pairwise_iou_cost(
                box_cxcywh_to_xyxy(pred_boxes[batch_idx]),
                box_cxcywh_to_xyxy(tgt_boxes),
            )
            cost_matrix = self.cost_class * cost_class + \
                self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64,
                                    device=pred_boxes.device),
                    torch.as_tensor(col_ind, dtype=torch.int64,
                                    device=pred_boxes.device),
                )
            )
        return indices


class SetCriterion(nn.Module):
    """Deformable DETR loss with decoupled objectness/class supervision."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        loss_ce_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        enc_loss_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_objectness_weight: float = 1.0,
        exp32_aux_positive_topk: int = 0,
        exp32_aux_positive_weight: float = 0.0,
        exp32_aux_min_iou: float = 0.15,
        box_iou_loss_type: str = "giou",
        targeted_confusion_margin_loss_weight: float = 0.0,
        targeted_confusion_margin_rules: list[dict] | None = None,
        aux_digit_classifier_loss_weight: float = 0.0,
        query_quality_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_ce_weight = loss_ce_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.enc_loss_weight = enc_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_objectness_weight = loss_objectness_weight
        self.exp32_aux_positive_topk = max(0, int(exp32_aux_positive_topk))
        self.exp32_aux_positive_weight = max(
            0.0, float(exp32_aux_positive_weight))
        self.exp32_aux_min_iou = max(0.0, float(exp32_aux_min_iou))
        self.box_iou_loss_type = str(box_iou_loss_type).lower()
        self.targeted_confusion_margin_loss_weight = max(
            0.0, float(targeted_confusion_margin_loss_weight))
        self.targeted_confusion_margin_rules = self._normalize_targeted_confusion_margin_rules(
            targeted_confusion_margin_rules)
        self.aux_digit_classifier_loss_weight = max(
            0.0, float(aux_digit_classifier_loss_weight))
        self.query_quality_loss_weight = max(
            0.0, float(query_quality_loss_weight))

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, idx)
                              for idx, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _has_decoupled_outputs(outputs: dict) -> bool:
        return "pred_class_logits" in outputs and "pred_objectness_logits" in outputs

    @staticmethod
    def _get_class_probabilities(outputs: dict) -> torch.Tensor:
        if SetCriterion._has_decoupled_outputs(outputs):
            class_prob = outputs["pred_class_logits"].float().softmax(dim=-1)
            objectness_prob = outputs["pred_objectness_logits"].float(
            ).sigmoid()
            return class_prob * objectness_prob
        return outputs["pred_logits"].float().sigmoid()

    @staticmethod
    def _normalize_targeted_confusion_margin_rules(rules: list[dict] | None) -> list[dict]:
        normalized_rules: list[dict] = []
        if not rules:
            return normalized_rules

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            try:
                penalized_class = int(rule["penalized_class"])
            except (KeyError, TypeError, ValueError):
                continue

            true_classes_raw = rule.get("true_classes", [])
            if not isinstance(true_classes_raw, (list, tuple)):
                continue
            true_classes = []
            for class_idx in true_classes_raw:
                try:
                    true_classes.append(int(class_idx))
                except (TypeError, ValueError):
                    continue
            if not true_classes:
                continue

            try:
                weight = float(rule.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            if weight <= 0.0:
                continue

            normalized_rules.append(
                {
                    "penalized_class": penalized_class,
                    "true_classes": tuple(sorted(set(true_classes))),
                    "weight": weight,
                }
            )
        return normalized_rules

    def _prepare_targets(self, targets: list[dict]) -> list[dict]:
        prepared_targets = []
        for target in targets:
            boxes = target["boxes"]
            size = target["size"].to(boxes.device)
            height, width = size[0].float(), size[1].float()
            if boxes.numel() == 0:
                boxes_normalized = boxes.reshape(0, 4)
            else:
                scale = torch.tensor(
                    [width, height, width, height], device=boxes.device)
                boxes_normalized = box_xyxy_to_cxcywh(boxes / scale)
            prepared_target = dict(target)
            prepared_target["boxes_normalized"] = boxes_normalized
            prepared_targets.append(prepared_target)
        return prepared_targets

    def loss_objectness(self, outputs: dict, indices, extra_indices, num_boxes: float):
        if not self._has_decoupled_outputs(outputs):
            return {}

        src_logits = outputs["pred_objectness_logits"].float()
        batch_size, num_queries = src_logits.shape[:2]
        target_objectness = torch.zeros(
            (batch_size, num_queries, 1), dtype=src_logits.dtype, device=src_logits.device)

        for batch_idx, (src_idx, _) in enumerate(indices):
            if src_idx.numel() > 0:
                target_objectness[batch_idx, src_idx, 0] = 1.0
        if extra_indices is not None:
            for batch_idx, (src_idx, _) in enumerate(extra_indices):
                if src_idx.numel() > 0:
                    target_objectness[batch_idx, src_idx, 0] = 1.0

        return {
            "loss_objectness": sigmoid_focal_loss(
                src_logits,
                target_objectness,
                num_boxes=num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
        }

    def loss_labels(self, outputs: dict, targets: list[dict], indices, num_boxes: float):
        if self._has_decoupled_outputs(outputs):
            idx = self._get_src_permutation_idx(indices)
            src_logits = outputs["pred_class_logits"].float()
            if idx[0].numel() == 0:
                return {"loss_ce": src_logits.sum() * 0.0}
            src_logits = src_logits[idx]
            target_classes = torch.cat(
                [target["labels"][tgt_idx]
                    for target, (_, tgt_idx) in zip(targets, indices)],
                dim=0,
            )
            return {"loss_ce": F.cross_entropy(src_logits, target_classes, reduction="sum") / max(1.0, num_boxes)}

        src_logits = outputs["pred_logits"].float()
        batch_size, num_queries = src_logits.shape[:2]
        target_classes = torch.full(
            (batch_size, num_queries), self.num_classes, dtype=torch.int64, device=src_logits.device)
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[batch_idx,
                               src_idx] = targets[batch_idx]["labels"][tgt_idx]
        target_classes_onehot = torch.zeros(
            (batch_size, num_queries, self.num_classes + 1),
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        return {
            "loss_ce": sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes=num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            * num_queries
        }

    def loss_labels_varifocal(self, outputs: dict, targets: list[dict], indices, num_boxes: float):
        if self._has_decoupled_outputs(outputs):
            return self.loss_labels(outputs, targets, indices, num_boxes)

        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {"loss_ce": src_logits.sum() * 0.0}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([target["boxes_normalized"][tgt_idx]
                                 for target, (_, tgt_idx) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(box_iou(box_cxcywh_to_xyxy(
            src_boxes), box_cxcywh_to_xyxy(target_boxes))).detach()

        target_classes_o = torch.cat(
            [target["labels"][tgt_idx] for target, (_, tgt_idx) in zip(targets, indices)], dim=0)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(
            target_classes, self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        target_score = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score[idx] = iou_score

        return {
            "loss_ce": vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score,
                num_boxes=num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            * src_logits.shape[1]
        }

    def loss_boxes(self, outputs: dict, targets: list[dict], indices, num_boxes: float):
        src_boxes = outputs["pred_boxes"]
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = src_boxes.sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = src_boxes[idx]
        target_boxes = torch.cat([target["boxes_normalized"][tgt_idx]
                                 for target, (_, tgt_idx) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes,
                              reduction="none").sum() / max(1.0, num_boxes)
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        if self.box_iou_loss_type == "ciou":
            loss_giou = 1 - \
                torch.diag(complete_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        else:
            loss_giou = 1 - \
                torch.diag(generalized_box_iou(
                    src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = loss_giou.sum() / max(1.0, num_boxes)
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_targeted_confusion_margin(self, outputs: dict, targets: list[dict], indices):
        if self.targeted_confusion_margin_loss_weight <= 0.0 or not self.targeted_confusion_margin_rules:
            logits = outputs.get("pred_class_logits",
                                 outputs.get("pred_logits"))
            zero = logits.float().sum() * 0.0
            return {"loss_targeted_confusion_margin": zero}

        class_logits = outputs.get(
            "pred_class_logits", outputs.get("pred_logits"))
        if class_logits is None:
            raise KeyError(
                "Outputs must contain `pred_logits` or `pred_class_logits` for targeted confusion loss.")
        class_logits = class_logits.float()
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {"loss_targeted_confusion_margin": class_logits.sum() * 0.0}

        matched_logits = class_logits[idx]
        matched_true_classes = torch.cat(
            [target["labels"][tgt_idx]
                for target, (_, tgt_idx) in zip(targets, indices)],
            dim=0,
        )
        if matched_true_classes.numel() == 0:
            return {"loss_targeted_confusion_margin": class_logits.sum() * 0.0}

        total_rule_loss = class_logits.sum() * 0.0
        total_rule_weight = 0.0
        for rule in self.targeted_confusion_margin_rules:
            penalized_class = int(rule["penalized_class"])
            if penalized_class < 0 or penalized_class >= class_logits.shape[-1]:
                continue
            true_classes_tensor = torch.tensor(
                rule["true_classes"],
                dtype=matched_true_classes.dtype,
                device=matched_true_classes.device,
            )
            matched_mask = (matched_true_classes.unsqueeze(
                1) == true_classes_tensor.unsqueeze(0)).any(dim=1)
            if not matched_mask.any():
                continue

            selected_logits = matched_logits[matched_mask]
            selected_true_classes = matched_true_classes[matched_mask]
            true_logits = selected_logits.gather(
                1, selected_true_classes.unsqueeze(1)).squeeze(1)
            penalized_logits = selected_logits[:, penalized_class]
            rule_loss = F.softplus(penalized_logits - true_logits).mean()
            rule_weight = float(rule["weight"])
            total_rule_loss = total_rule_loss + (rule_loss * rule_weight)
            total_rule_weight += rule_weight

        if total_rule_weight <= 0.0:
            return {"loss_targeted_confusion_margin": class_logits.sum() * 0.0}
        return {"loss_targeted_confusion_margin": total_rule_loss / total_rule_weight}

    def loss_aux_digit_classifier(self, outputs: dict, targets: list[dict], indices, num_boxes: float):
        aux_logits = outputs.get("pred_aux_digit_logits")
        if self.aux_digit_classifier_loss_weight <= 0.0 or aux_logits is None:
            base = outputs.get("pred_boxes")
            if base is not None:
                zero = base.float().sum() * 0.0
            else:
                zero = next(iter(outputs.values())).float().sum() * 0.0
            return {"loss_aux_digit_cls": zero}

        aux_logits = aux_logits.float()
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {"loss_aux_digit_cls": aux_logits.sum() * 0.0}

        selected_logits = aux_logits[idx]
        target_classes = torch.cat(
            [target["labels"][tgt_idx]
                for target, (_, tgt_idx) in zip(targets, indices)],
            dim=0,
        )
        if target_classes.numel() == 0:
            return {"loss_aux_digit_cls": aux_logits.sum() * 0.0}

        loss_aux_digit_cls = F.cross_entropy(
            selected_logits, target_classes, reduction="sum") / max(1.0, num_boxes)
        return {"loss_aux_digit_cls": loss_aux_digit_cls}

    def loss_query_quality(self, outputs: dict, targets: list[dict]):
        quality_logits = outputs.get("enc_outputs_quality_logits")
        coord_logits = outputs.get("enc_outputs_coord_logits")
        if self.query_quality_loss_weight <= 0.0 or quality_logits is None or coord_logits is None:
            base = outputs.get("pred_boxes")
            if base is not None:
                zero = base.float().sum() * 0.0
            else:
                zero = next(iter(outputs.values())).float().sum() * 0.0
            return {"loss_query_quality": zero}

        quality_logits = quality_logits.float().squeeze(-1)
        pred_boxes = coord_logits.float().sigmoid()
        valid_mask = outputs.get("enc_valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones_like(quality_logits, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(dtype=torch.bool, device=quality_logits.device)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.squeeze(-1)
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.unsqueeze(0)
            if valid_mask.shape[0] == 1 and quality_logits.shape[0] > 1:
                valid_mask = valid_mask.expand(quality_logits.shape[0], -1)
            if valid_mask.shape != quality_logits.shape:
                raise RuntimeError(
                    "enc_valid_mask shape does not match query-quality logits: "
                    f"mask={tuple(valid_mask.shape)} vs logits={tuple(quality_logits.shape)}"
                )

        target_quality = torch.zeros_like(quality_logits)
        for batch_idx, target in enumerate(targets):
            target_boxes = target.get("boxes_normalized")
            if target_boxes is None or target_boxes.numel() == 0:
                continue
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[batch_idx])
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes.float())
            max_iou = box_iou(pred_boxes_xyxy, target_boxes_xyxy).max(dim=1).values
            target_quality[batch_idx] = max_iou

        if not bool(valid_mask.any().item()):
            return {"loss_query_quality": quality_logits.sum() * 0.0}

        loss_query_quality = F.binary_cross_entropy_with_logits(
            quality_logits[valid_mask],
            target_quality[valid_mask].detach(),
            reduction="mean",
        )
        return {"loss_query_quality": loss_query_quality}

    def compute_additional_losses(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        prepared_targets = self._prepare_targets(targets)
        num_boxes = float(sum(target["labels"].shape[0] for target in prepared_targets))
        main_outputs = {
            key: value
            for key, value in outputs.items()
            if key in {
                "pred_logits",
                "pred_boxes",
                "pred_class_logits",
                "pred_objectness_logits",
                "pred_aux_digit_logits",
                "pred_detector_logits",
            }
        }
        main_indices = self.matcher(main_outputs, prepared_targets)

        targeted_confusion_loss_dict = self.loss_targeted_confusion_margin(
            main_outputs,
            prepared_targets,
            main_indices,
        )
        targeted_confusion_total = (
            self.targeted_confusion_margin_loss_weight
            * targeted_confusion_loss_dict["loss_targeted_confusion_margin"]
        )
        aux_digit_loss_dict = self.loss_aux_digit_classifier(
            main_outputs,
            prepared_targets,
            main_indices,
            num_boxes=num_boxes,
        )
        aux_digit_total = self.aux_digit_classifier_loss_weight * aux_digit_loss_dict["loss_aux_digit_cls"]
        query_quality_loss_dict = self.loss_query_quality(outputs, prepared_targets)
        query_quality_total = self.query_quality_loss_weight * query_quality_loss_dict["loss_query_quality"]
        extra_total = targeted_confusion_total + aux_digit_total + query_quality_total
        loss_dict = {}
        loss_dict.update(targeted_confusion_loss_dict)
        loss_dict.update(aux_digit_loss_dict)
        loss_dict.update(query_quality_loss_dict)
        loss_dict["loss_targeted_confusion"] = targeted_confusion_total
        loss_dict["loss_aux_digit_cls_weighted"] = aux_digit_total
        loss_dict["loss_query_quality_weighted"] = query_quality_total
        loss_dict["loss_extra_custom"] = extra_total
        return loss_dict

    def _compute_losses(
        self,
        outputs: dict,
        targets: list[dict],
        num_boxes: float,
        indices=None,
        extra_indices=None,
        use_varifocal_labels: bool = False,
    ) -> dict[str, torch.Tensor]:
        if indices is None:
            indices = self.matcher(outputs, targets)
        losses = {}
        if use_varifocal_labels:
            losses.update(self.loss_labels_varifocal(
                outputs, targets, indices, num_boxes=num_boxes))
        else:
            losses.update(self.loss_labels(
                outputs, targets, indices, num_boxes=num_boxes))
        losses.update(self.loss_boxes(
            outputs, targets, indices, num_boxes=num_boxes))
        losses.update(self.loss_objectness(outputs, indices,
                      extra_indices, num_boxes=num_boxes))
        return losses

    def _weighted_total(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        total = (
            self.loss_ce_weight * loss_dict["loss_ce"]
            + self.loss_bbox_weight * loss_dict["loss_bbox"]
            + self.loss_giou_weight * loss_dict["loss_giou"]
        )
        if "loss_objectness" in loss_dict:
            total = total + self.loss_objectness_weight * \
                loss_dict["loss_objectness"]
        return total

    def _build_extra_positive_indices(self, outputs: dict, targets: list[dict], main_indices):
        empty_indices = [
            (
                torch.empty((0,), dtype=torch.int64,
                            device=outputs["pred_boxes"].device),
                torch.empty((0,), dtype=torch.int64,
                            device=outputs["pred_boxes"].device),
            )
            for _ in range(outputs["pred_boxes"].shape[0])
        ]
        if (
            not self._has_decoupled_outputs(outputs)
            or self.exp32_aux_positive_topk <= 0
            or self.exp32_aux_positive_weight <= 0.0
        ):
            return empty_indices

        pred_boxes = outputs["pred_boxes"].float()
        class_prob = self._get_class_probabilities(outputs)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        extra_indices = []

        for batch_idx in range(pred_boxes.shape[0]):
            tgt_ids = targets[batch_idx]["labels"]
            tgt_boxes = targets[batch_idx]["boxes_normalized"]
            if tgt_ids.numel() == 0:
                extra_indices.append(empty_indices[batch_idx])
                continue

            used_queries = torch.zeros(
                pred_boxes.shape[1], dtype=torch.bool, device=pred_boxes.device)
            used_queries[main_indices[batch_idx][0]] = True
            batch_src = []
            batch_tgt = []
            batch_pred_boxes_xyxy = pred_boxes_xyxy[batch_idx]

            for tgt_local_idx, tgt_id in enumerate(tgt_ids):
                class_cost = - \
                    torch.log(class_prob[batch_idx][:, tgt_id].clamp_min(1e-8))
                bbox_cost = torch.cdist(
                    pred_boxes[batch_idx], tgt_boxes[tgt_local_idx].unsqueeze(0), p=1).squeeze(1)
                gt_xyxy = box_cxcywh_to_xyxy(
                    tgt_boxes[tgt_local_idx].unsqueeze(0))
                pair_iou = box_iou(batch_pred_boxes_xyxy, gt_xyxy).squeeze(1)
                giou_cost = self.matcher._pairwise_iou_cost(
                    batch_pred_boxes_xyxy, gt_xyxy).squeeze(1)
                total_cost = (
                    self.matcher.cost_class * class_cost
                    + self.matcher.cost_bbox * bbox_cost
                    + self.matcher.cost_giou * giou_cost
                )

                filtered_cost = total_cost.clone()
                filtered_cost[used_queries] = float("inf")
                if self.exp32_aux_min_iou > 0.0:
                    filtered_cost[pair_iou <
                                  self.exp32_aux_min_iou] = float("inf")

                candidate_k = min(self.exp32_aux_positive_topk,
                                  int((~used_queries).sum().item()))
                if candidate_k <= 0:
                    continue

                candidate_cost, candidate_query = torch.topk(
                    filtered_cost, k=candidate_k, largest=False)
                valid_mask = torch.isfinite(candidate_cost)
                if not valid_mask.any():
                    fallback_cost = total_cost.clone()
                    fallback_cost[used_queries] = float("inf")
                    candidate_cost, candidate_query = torch.topk(
                        fallback_cost, k=candidate_k, largest=False)
                    valid_mask = torch.isfinite(candidate_cost)
                candidate_query = candidate_query[valid_mask]
                if candidate_query.numel() == 0:
                    continue

                used_queries[candidate_query] = True
                batch_src.append(candidate_query)
                batch_tgt.append(torch.full_like(
                    candidate_query, tgt_local_idx))

            if batch_src:
                extra_indices.append(
                    (torch.cat(batch_src, dim=0), torch.cat(batch_tgt, dim=0)))
            else:
                extra_indices.append(empty_indices[batch_idx])

        return extra_indices

    def forward(self, outputs: dict, targets: list[dict]):
        prepared_targets = self._prepare_targets(targets)
        num_boxes = float(sum(target["labels"].shape[0]
                          for target in prepared_targets))
        main_outputs = {
            key: value
            for key, value in outputs.items()
            if key in {
                "pred_logits",
                "pred_boxes",
                "pred_class_logits",
                "pred_objectness_logits",
                "pred_aux_digit_logits",
                "pred_detector_logits",
            }
        }
        main_indices = self.matcher(main_outputs, prepared_targets)
        extra_indices = self._build_extra_positive_indices(
            main_outputs, prepared_targets, main_indices)
        main_loss_dict = self._compute_losses(
            main_outputs,
            prepared_targets,
            num_boxes=num_boxes,
            indices=main_indices,
            extra_indices=extra_indices,
        )
        main_total = self._weighted_total(main_loss_dict)
        targeted_confusion_loss_dict = self.loss_targeted_confusion_margin(
            main_outputs, prepared_targets, main_indices)
        targeted_confusion_total = (
            self.targeted_confusion_margin_loss_weight
            * targeted_confusion_loss_dict["loss_targeted_confusion_margin"]
        )
        aux_digit_loss_dict = self.loss_aux_digit_classifier(
            main_outputs,
            prepared_targets,
            main_indices,
            num_boxes=num_boxes,
        )
        aux_digit_total = self.aux_digit_classifier_loss_weight * \
            aux_digit_loss_dict["loss_aux_digit_cls"]
        query_quality_loss_dict = self.loss_query_quality(outputs, prepared_targets)
        query_quality_total = self.query_quality_loss_weight * \
            query_quality_loss_dict["loss_query_quality"]
        total_loss = main_total + targeted_confusion_total + aux_digit_total + query_quality_total
        aux_total = main_total.new_tensor(0.0)
        enc_total = main_total.new_tensor(0.0)
        group_total = main_total.new_tensor(0.0)
        loss_dict = dict(main_loss_dict)
        loss_dict.update(targeted_confusion_loss_dict)
        loss_dict.update(aux_digit_loss_dict)
        loss_dict.update(query_quality_loss_dict)

        if self.exp32_aux_positive_weight > 0.0 and self._has_decoupled_outputs(main_outputs):
            group_loss_dict = {}
            group_loss_dict.update(self.loss_labels(
                main_outputs, prepared_targets, extra_indices, num_boxes=num_boxes))
            group_loss_dict.update(self.loss_boxes(
                main_outputs, prepared_targets, extra_indices, num_boxes=num_boxes))
            group_total = self.exp32_aux_positive_weight * (
                self.loss_ce_weight * group_loss_dict["loss_ce"]
                + self.loss_bbox_weight * group_loss_dict["loss_bbox"]
                + self.loss_giou_weight * group_loss_dict["loss_giou"]
            )
            total_loss = total_loss + group_total
            for key, value in group_loss_dict.items():
                loss_dict[f"{key}_group"] = value

        aux_outputs = outputs.get("aux_outputs", [])
        for layer_idx, aux_output in enumerate(aux_outputs):
            aux_loss_dict = self._compute_losses(
                aux_output, prepared_targets, num_boxes=num_boxes)
            aux_weighted = self._weighted_total(aux_loss_dict)
            aux_total = aux_total + aux_weighted
            total_loss = total_loss + aux_weighted
            for key, value in aux_loss_dict.items():
                loss_dict[f"{key}_aux_{layer_idx}"] = value

        if "enc_outputs" in outputs:
            enc_loss_dict = self._compute_losses(
                outputs["enc_outputs"], prepared_targets, num_boxes=num_boxes)
            enc_total = self.enc_loss_weight * \
                self._weighted_total(enc_loss_dict)
            total_loss = total_loss + enc_total
            for key, value in enc_loss_dict.items():
                loss_dict[f"{key}_enc"] = value

        dn_total = main_total.new_tensor(0.0)
        dn_outputs = outputs.get("dn_outputs")
        if dn_outputs is not None:
            denoising_groups = int(dn_outputs.get("denoising_groups", 0))
            max_gt_num_per_image = int(
                dn_outputs.get("max_gt_num_per_image", 0))
            if denoising_groups > 0 and max_gt_num_per_image > 0:
                dn_indices = []
                for target in prepared_targets:
                    num_target_boxes = len(target["labels"])
                    if num_target_boxes > 0:
                        group_index, target_index = torch.meshgrid(
                            torch.arange(denoising_groups,
                                         device=target["labels"].device),
                            torch.arange(num_target_boxes,
                                         device=target["labels"].device),
                            indexing="ij",
                        )
                        output_idx = (
                            group_index * max_gt_num_per_image + target_index).flatten()
                        tgt_idx = target_index.flatten()
                    else:
                        output_idx = torch.empty(
                            (0,), dtype=torch.long, device=target["labels"].device)
                        tgt_idx = torch.empty(
                            (0,), dtype=torch.long, device=target["labels"].device)
                    dn_indices.append((output_idx, tgt_idx))

                dn_main = {
                    key: value
                    for key, value in dn_outputs.items()
                    if key in {"pred_logits", "pred_boxes", "pred_class_logits", "pred_objectness_logits"}
                }
                dn_num_boxes = num_boxes * denoising_groups
                dn_loss_dict = self._compute_losses(
                    dn_main, prepared_targets, num_boxes=dn_num_boxes, indices=dn_indices)
                dn_weighted = self._weighted_total(dn_loss_dict)
                dn_total = dn_total + dn_weighted
                total_loss = total_loss + dn_weighted
                for key, value in dn_loss_dict.items():
                    loss_dict[f"{key}_dn"] = value

                for layer_idx, aux_output in enumerate(dn_outputs.get("aux_outputs", [])):
                    dn_aux_loss_dict = self._compute_losses(
                        aux_output, prepared_targets, num_boxes=dn_num_boxes, indices=dn_indices)
                    dn_aux_weighted = self._weighted_total(dn_aux_loss_dict)
                    dn_total = dn_total + dn_aux_weighted
                    total_loss = total_loss + dn_aux_weighted
                    for key, value in dn_aux_loss_dict.items():
                        loss_dict[f"{key}_dn_aux_{layer_idx}"] = value

        hybrid_total = main_total.new_tensor(0.0)
        hybrid_outputs = outputs.get("hybrid_outputs")
        if hybrid_outputs is not None:
            hybrid_assign = max(1, int(hybrid_outputs.get("hybrid_assign", 1)))
            repeated_targets = []
            for target in prepared_targets:
                repeated_target = dict(target)
                repeated_target["labels"] = target["labels"].repeat(
                    hybrid_assign)
                repeated_target["boxes_normalized"] = target["boxes_normalized"].repeat(
                    hybrid_assign, 1)
                repeated_targets.append(repeated_target)
            hybrid_num_boxes = float(
                sum(target["labels"].shape[0] for target in repeated_targets))

            hybrid_main = {
                key: value
                for key, value in hybrid_outputs.items()
                if key in {"pred_logits", "pred_boxes", "pred_class_logits", "pred_objectness_logits"}
            }
            hybrid_loss_dict = self._compute_losses(
                hybrid_main,
                repeated_targets,
                num_boxes=hybrid_num_boxes,
                use_varifocal_labels=True,
            )
            hybrid_weighted = self._weighted_total(hybrid_loss_dict)
            hybrid_total = hybrid_total + hybrid_weighted
            total_loss = total_loss + hybrid_weighted
            for key, value in hybrid_loss_dict.items():
                loss_dict[f"{key}_hybrid"] = value

            for layer_idx, aux_output in enumerate(hybrid_outputs.get("aux_outputs", [])):
                hybrid_aux_loss_dict = self._compute_losses(
                    aux_output,
                    repeated_targets,
                    num_boxes=hybrid_num_boxes,
                    use_varifocal_labels=True,
                )
                hybrid_aux_weighted = self._weighted_total(
                    hybrid_aux_loss_dict)
                hybrid_total = hybrid_total + hybrid_aux_weighted
                total_loss = total_loss + hybrid_aux_weighted
                for key, value in hybrid_aux_loss_dict.items():
                    loss_dict[f"{key}_hybrid_aux_{layer_idx}"] = value

            if "enc_outputs" in hybrid_outputs:
                hybrid_enc_loss_dict = self._compute_losses(
                    hybrid_outputs["enc_outputs"],
                    repeated_targets,
                    num_boxes=hybrid_num_boxes,
                    use_varifocal_labels=True,
                )
                hybrid_enc_weighted = self.enc_loss_weight * \
                    self._weighted_total(hybrid_enc_loss_dict)
                hybrid_total = hybrid_total + hybrid_enc_weighted
                total_loss = total_loss + hybrid_enc_weighted
                for key, value in hybrid_enc_loss_dict.items():
                    loss_dict[f"{key}_hybrid_enc"] = value

        loss_dict["loss_main"] = main_total
        loss_dict["loss_aux"] = aux_total
        loss_dict["loss_group"] = group_total
        loss_dict["loss_enc"] = enc_total
        loss_dict["loss_dn"] = dn_total
        loss_dict["loss_hybrid"] = hybrid_total
        loss_dict["loss_targeted_confusion"] = targeted_confusion_total
        loss_dict["loss_aux_digit_cls_weighted"] = aux_digit_total
        loss_dict["loss"] = total_loss
        return loss_dict


def adapt_msda_checkpoint_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map MSDeformAttn parameter names between fallback and official CUDA module layouts."""
    model_keys = set(model.state_dict().keys())
    adapted_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        candidate_keys = [key]
        if ".official_impl." in key:
            candidate_keys.append(key.replace(".official_impl.", "."))
        else:
            if ".self_attn." in key:
                candidate_keys.append(key.replace(
                    ".self_attn.", ".self_attn.official_impl."))
            if ".cross_attn." in key:
                candidate_keys.append(key.replace(
                    ".cross_attn.", ".cross_attn.official_impl."))

        mapped_key = next(
            (candidate for candidate in candidate_keys if candidate in model_keys), key)
        adapted_state_dict[mapped_key] = value

    return adapted_state_dict


def adapt_checkpoint_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if hasattr(model, "adapt_checkpoint_state_dict"):
        return model.adapt_checkpoint_state_dict(state_dict)
    return adapt_msda_checkpoint_state_dict(model, state_dict)


def build_model_from_config(
    config: dict,
    pretrained_backbone_override: bool | None = None,
) -> nn.Module:
    model_backend = str(config.get("model_backend", "custom_relation_detr")).lower()
    if model_backend == "hf_rtdetr_v2":
        _validate_hf_rtdetr_v2_official_mode(config)
        return HFRTDetrV2Adapter(
            num_classes=int(config["num_classes"]),
            num_queries=int(config.get("num_queries", 300)),
            hf_model_name_or_path=str(config.get("hf_model_name_or_path", "PekingU/rtdetr_v2_r50vd")),
            hf_load_strategy=str(config.get("hf_load_strategy", "pretrained_reset_transformer")),
            hf_ignore_mismatched_sizes=bool(config.get("hf_ignore_mismatched_sizes", True)),
            hf_eos_coefficient=config.get("hf_eos_coefficient"),
            hf_backbone_name=str(config.get("hf_backbone_name", "resnet50")),
            hf_use_timm_backbone=bool(config.get("hf_use_timm_backbone", True)),
            hf_use_pretrained_backbone=bool(config.get("hf_use_pretrained_backbone", True)),
        )
    if model_backend == "hf_rtdetr_v2_aux":
        _validate_hf_rtdetr_v2_aux_mode(config)
        return HFRTDetrV2AuxAdapter(
            num_classes=int(config["num_classes"]),
            num_queries=int(config.get("num_queries", 300)),
            hf_model_name_or_path=str(config.get("hf_model_name_or_path", "PekingU/rtdetr_v2_r50vd")),
            hf_load_strategy=str(config.get("hf_load_strategy", "pretrained_reset_transformer")),
            hf_ignore_mismatched_sizes=bool(config.get("hf_ignore_mismatched_sizes", True)),
            hf_eos_coefficient=config.get("hf_eos_coefficient"),
            hf_backbone_name=str(config.get("hf_backbone_name", "resnet50")),
            hf_use_timm_backbone=bool(config.get("hf_use_timm_backbone", True)),
            hf_use_pretrained_backbone=bool(config.get("hf_use_pretrained_backbone", True)),
            aux_digit_hidden_dim=int(config.get("aux_digit_hidden_dim", 256)),
            aux_digit_classifier_fusion_weight=float(config.get("aux_digit_classifier_fusion_weight", 0.0)),
            use_aux_digit_classifier_gated_fusion=bool(config.get("use_aux_digit_classifier_gated_fusion", False)),
            aux_digit_gate_top1_threshold=float(config.get("aux_digit_gate_top1_threshold", 0.90)),
            aux_digit_gate_margin_threshold=float(config.get("aux_digit_gate_margin_threshold", 0.15)),
            use_aux_digit_confusion_family_selective_fusion=bool(
                config.get("use_aux_digit_confusion_family_selective_fusion", False)
            ),
            use_aux_digit_confusion_family_attenuation=bool(
                config.get("use_aux_digit_confusion_family_attenuation", False)
            ),
            aux_digit_confusion_families=config.get("aux_digit_confusion_families"),
            aux_digit_family_fusion_weights=config.get("aux_digit_family_fusion_weights"),
            aux_digit_family_attenuation_weights=config.get("aux_digit_family_attenuation_weights"),
        )
    if model_backend == "hf_rtdetr_v2_qs":
        return HFRTDetrV2QuerySelectionAdapter(
            num_classes=int(config["num_classes"]),
            num_queries=int(config.get("num_queries", 300)),
            hf_model_name_or_path=str(config.get("hf_model_name_or_path", "PekingU/rtdetr_v2_r50vd")),
            hf_load_strategy=str(config.get("hf_load_strategy", "pretrained_reset_transformer")),
            hf_ignore_mismatched_sizes=bool(config.get("hf_ignore_mismatched_sizes", True)),
            hf_eos_coefficient=config.get("hf_eos_coefficient"),
            hf_backbone_name=str(config.get("hf_backbone_name", "resnet50")),
            hf_use_timm_backbone=bool(config.get("hf_use_timm_backbone", True)),
            hf_use_pretrained_backbone=bool(config.get("hf_use_pretrained_backbone", True)),
            query_quality_hidden_dim=int(config.get("query_quality_hidden_dim", 256)),
            query_selection_alpha=float(config.get("query_selection_alpha", 1.0)),
            query_selection_beta=float(config.get("query_selection_beta", 1.0)),
        )

    if model_backend not in {"custom_relation_detr", "relation_detr_custom"}:
        raise ValueError(
            f"Unsupported model_backend '{model_backend}'. "
            "Supported backends are "
            "{'custom_relation_detr', 'relation_detr_custom', 'hf_rtdetr_v2', 'hf_rtdetr_v2_aux', 'hf_rtdetr_v2_qs'}."
        )

    pretrained_backbone = bool(config.get("pretrained_backbone", True))
    if pretrained_backbone_override is not None:
        pretrained_backbone = bool(pretrained_backbone_override)

    return DETRModel(
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_queries=int(config.get("num_queries", 20)),
        nheads=int(config.get("nheads", 8)),
        num_encoder_layers=int(config.get("num_encoder_layers", 6)),
        num_decoder_layers=int(config.get("num_decoder_layers", 6)),
        dim_feedforward=int(config.get("dim_feedforward", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        pretrained_backbone=pretrained_backbone,
        freeze_stem_and_layer1=bool(config.get("freeze_stem_and_layer1", False)),
        num_feature_levels=int(config.get("num_feature_levels", 3)),
        encoder_n_points=int(config.get("encoder_n_points", 4)),
        decoder_n_points=int(config.get("decoder_n_points", 4)),
        activation=str(config.get("transformer_activation", "relu")),
        two_stage=bool(config.get("two_stage", True)),
        use_official_cuda_msda=bool(config.get("use_official_cuda_msda", True)),
        use_fpn_features=bool(config.get("use_fpn_features", False)),
        use_exp32=bool(config.get("use_exp32", False)),
        exp32_num_groups=int(config.get("exp32_num_groups", 4)),
        use_backbone_dc5=bool(config.get("use_backbone_dc5", False)),
        use_relation_detr_main=bool(config.get("use_relation_detr_main", False)),
        align_relation_official_head_loss=bool(config.get("align_relation_official_head_loss", False)),
        use_relation_hybrid=bool(config.get("use_relation_hybrid", False)),
        relation_hybrid_num_proposals=int(config.get("relation_hybrid_num_proposals", max(1, int(config.get("num_queries", 20)) * 2))),
        relation_hybrid_assign=int(config.get("relation_hybrid_assign", 6)),
        use_relation_cdn=bool(config.get("use_relation_cdn", False)),
        relation_denoising_nums=int(config.get("relation_denoising_nums", 100)),
        relation_label_noise_prob=float(config.get("relation_label_noise_prob", 0.5)),
        relation_box_noise_scale=float(config.get("relation_box_noise_scale", 1.0)),
        use_query_relation=bool(config.get("use_query_relation", False)),
        query_relation_layers=int(config.get("query_relation_layers", 1)),
        query_relation_num_heads=int(config.get("query_relation_num_heads", config.get("nheads", 8))),
        query_relation_ffn_dim=int(config.get("query_relation_ffn_dim", 1024)),
        query_relation_geometry_hidden_dim=int(config.get("query_relation_geometry_hidden_dim", 128)),
        use_aux_digit_classifier=bool(config.get("use_aux_digit_classifier", False)),
        aux_digit_pool_size=int(config.get("aux_digit_pool_size", 5)),
        aux_digit_hidden_dim=int(config.get("aux_digit_hidden_dim", 256)),
        aux_digit_classifier_fusion_weight=float(config.get("aux_digit_classifier_fusion_weight", 0.0)),
        use_aux_digit_classifier_gated_fusion=bool(config.get("use_aux_digit_classifier_gated_fusion", False)),
        aux_digit_gate_top1_threshold=float(config.get("aux_digit_gate_top1_threshold", 0.90)),
        aux_digit_gate_margin_threshold=float(config.get("aux_digit_gate_margin_threshold", 0.15)),
        backbone_pretrain_source=str(config.get("backbone_pretrain_source", "imagenet")),
        backbone_pretrain_checkpoint_path=config.get("backbone_pretrain_checkpoint_path"),
        backbone_pretrain_url=config.get("backbone_pretrain_url"),
    )
