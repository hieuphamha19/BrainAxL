#!/usr/bin/env python3
"""Run asparagus finetune_seg with small-object train and validation crops.

This wrapper is intentionally outside the source package. It monkey-patches the
exported segmentation CPU transform presets before Hydra instantiates them.

Training:
  Normalize -> foreground-biased CropPad -> mild spatial aug -> mirror.

Validation:
  Default nnU-Net-like mode: Normalize only, then full-volume/sliding-window
  validation in the Lightning validation step. No label-guided validation crop.

Test/predict are not patched; full-volume sliding-window inference remains the
evaluation path.
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2_robust_transforms import install_task2_gpu_transforms
from task2_acquisition_consistency import install_acquisition_consistency_training


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _parse_triplet_env(name: str, default: Sequence[float]) -> tuple[float, float, float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return tuple(float(v) for v in default)
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three values, got: {value!r}")
    return tuple(float(v) for v in parts)


class MonaiSingleSampleCrop:
    def __init__(self, cropper):
        self.cropper = cropper

    def __call__(self, data_dict):
        out = self.cropper(data_dict)
        if isinstance(out, list):
            if len(out) != 1:
                raise RuntimeError(f"Expected MONAI crop num_samples=1, got {len(out)}")
            out = out[0]
        out.setdefault("transforms_applied", {})["monai_single_sample_crop"] = self.cropper.__class__.__name__
        return out


class PadToMinimumSpatialSize:
    def __init__(self, spatial_size: Sequence[int], keys=("image", "label")):
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.keys = tuple(keys)

    def __call__(self, data_dict):
        for key in self.keys:
            tensor = data_dict.get(key)
            if tensor is None:
                continue
            ndim = len(self.spatial_size)
            if tensor.ndim < ndim:
                raise RuntimeError(f"{key} tensor ndim={tensor.ndim} smaller than spatial ndim={ndim}")
            spatial_shape = tuple(int(v) for v in tensor.shape[-ndim:])
            pad = []
            needs_pad = False
            for current, target in reversed(list(zip(spatial_shape, self.spatial_size))):
                total = max(0, target - current)
                left = total // 2
                right = total - left
                pad.extend([left, right])
                needs_pad = needs_pad or total > 0
            if needs_pad:
                value = 0.0 if key == "image" else 0
                data_dict[key] = F.pad(tensor, pad, mode="constant", value=value)
        data_dict.setdefault("transforms_applied", {})["pad_to_minimum_spatial_size"] = self.spatial_size
        return data_dict


class Task2FlairAnchorAugment:
    """Make the auxiliary DWI channel optional and registration-tolerant.

    Task 2 labels are defined in the FLAIR frame. A model trained on only 23
    cases can nevertheless learn brittle cross-channel correspondences, so this
    opt-in transform either removes DWI or translates DWI relative to FLAIR.
    FLAIR and the label are never changed independently. Applying it after
    normalization makes zero a neutral missing-channel value.
    """

    def __init__(
        self,
        dropout_probability: float = 0.0,
        shift_probability: float = 0.0,
        max_shift: Sequence[int] = (0, 0, 0),
        dwi_channel: int = 1,
    ):
        self.dropout_probability = float(dropout_probability)
        self.shift_probability = float(shift_probability)
        self.max_shift = tuple(int(v) for v in max_shift)
        self.dwi_channel = int(dwi_channel)
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("TASK2_DWI_DROPOUT_PROB must be in [0, 1]")
        if not 0.0 <= self.shift_probability <= 1.0:
            raise ValueError("TASK2_DWI_SHIFT_PROB must be in [0, 1]")
        if len(self.max_shift) != 3 or any(v < 0 for v in self.max_shift):
            raise ValueError("TASK2_DWI_MAX_SHIFT must contain three non-negative integers")

    @staticmethod
    def _translate_without_wrap(channel: torch.Tensor, shifts: Sequence[int]) -> torch.Tensor:
        output = torch.zeros_like(channel)
        source_slices = []
        destination_slices = []
        for size, shift in zip(channel.shape[-3:], shifts):
            if abs(int(shift)) >= int(size):
                return output
            if shift > 0:
                source_slices.append(slice(0, size - shift))
                destination_slices.append(slice(shift, size))
            elif shift < 0:
                source_slices.append(slice(-shift, size))
                destination_slices.append(slice(0, size + shift))
            else:
                source_slices.append(slice(None))
                destination_slices.append(slice(None))
        output[tuple(destination_slices)] = channel[tuple(source_slices)]
        return output

    def __call__(self, data_dict):
        image = data_dict.get("image")
        if image is None or image.ndim < 4 or image.shape[0] <= self.dwi_channel:
            return data_dict

        dropped = bool(torch.rand(()) < self.dropout_probability)
        shifts = (0, 0, 0)
        if dropped:
            image = image.clone()
            image[self.dwi_channel].zero_()
        elif bool(torch.rand(()) < self.shift_probability):
            shifts = tuple(
                int(torch.randint(-limit, limit + 1, ()).item()) if limit else 0
                for limit in self.max_shift
            )
            if any(shifts):
                image = image.clone()
                image[self.dwi_channel] = self._translate_without_wrap(
                    image[self.dwi_channel], shifts
                )

        data_dict["image"] = image
        data_dict.setdefault("transforms_applied", {})["task2_flair_anchor"] = {
            "dwi_dropped": dropped,
            "dwi_shift": shifts,
        }
        return data_dict


def _parse_float_list_env(name: str, default: Sequence[float]) -> list[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return [float(v) for v in default]
    return [float(v) for v in value.replace(",", " ").split()]


def install_small_object_transforms() -> None:
    from torchvision import transforms

    from gardening_tools.functional.transforms.spatial import get_max_rotated_size
    from gardening_tools.modules.transforms.cropping_and_padding import Torch_CropPad
    from gardening_tools.modules.transforms.mirror import Torch_Mirror
    from gardening_tools.modules.transforms.normalize import Torch_Normalize
    from gardening_tools.modules.transforms.spatial import Torch_Spatial
    from monai.transforms import RandCropByLabelClassesd, RandCropByPosNegLabeld

    import asparagus.modules.transforms.presets as presets
    import asparagus.modules.transforms.presets.train as train_presets

    train_oversample_fg = _parse_float_env("SMALL_OBJECT_OVERSAMPLE_FG", 0.95)
    sampler = os.environ.get("SMALL_OBJECT_SAMPLER", "torch_croppad").lower()
    monai_pos = _parse_float_env("MONAI_POS", 2.0)
    monai_neg = _parse_float_env("MONAI_NEG", 1.0)
    monai_label_ratios = _parse_float_list_env("MONAI_LABEL_RATIOS", (1.0, 2.0, 2.0))
    monai_num_classes = int(_parse_float_env("MONAI_NUM_CLASSES", len(monai_label_ratios)))
    val_oversample_fg = _parse_float_env("SMALL_OBJECT_VAL_OVERSAMPLE_FG", 1.0)
    val_mode = os.environ.get("SMALL_OBJECT_VAL_MODE", "sliding_image").lower()
    p_rot = _parse_float_env("SMALL_OBJECT_ROT_PROB", 0.10)
    p_scale = _parse_float_env("SMALL_OBJECT_SCALE_PROB", 0.10)
    p_mirror = _parse_float_env("SMALL_OBJECT_MIRROR_PROB", 1.0)
    rot_degrees = _parse_triplet_env("SMALL_OBJECT_ROT_DEGREES", (10.0, 10.0, 10.0))
    scale_min = _parse_float_env("SMALL_OBJECT_SCALE_MIN", 0.90)
    scale_max = _parse_float_env("SMALL_OBJECT_SCALE_MAX", 1.10)
    dwi_dropout_probability = _parse_float_env("TASK2_DWI_DROPOUT_PROB", 0.0)
    dwi_shift_probability = _parse_float_env("TASK2_DWI_SHIFT_PROB", 0.0)
    dwi_max_shift = tuple(int(v) for v in _parse_triplet_env("TASK2_DWI_MAX_SHIFT", (0, 0, 0)))
    dwi_channel = int(os.environ.get("TASK2_DWI_CHANNEL", "1"))
    eval_dwi_mode = os.environ.get("TASK2_EVAL_DWI_MODE", "none").lower()
    eval_dwi_shift = tuple(int(v) for v in _parse_triplet_env("TASK2_EVAL_DWI_SHIFT", (0, 0, 0)))
    # Restrict mirroring to a subset of axes. Needed for side-canonicalised
    # datasets (Task 4 bilateral ROI), where the crops have already been flipped
    # into a common left frame: mirroring the left-right axis there would undo
    # that canonicalisation on half the samples.
    mirror_axes_env = os.environ.get("SMALL_OBJECT_MIRROR_AXES", "").strip()

    def CPU_seg_train_transforms_small_object_roi(patch_size, normalize=True):
        axes = (0, 1) if len(patch_size) == 2 else (0, 1, 2)
        if mirror_axes_env:
            axes = tuple(int(a) for a in mirror_axes_env.replace(",", " ").split())
            if not set(axes) <= set(range(len(patch_size))):
                raise ValueError(
                    f"SMALL_OBJECT_MIRROR_AXES={mirror_axes_env!r} is out of range "
                    f"for a {len(patch_size)}D patch"
                )

        if p_rot > 0 or p_scale > 0:
            pre_aug_patch_size = get_max_rotated_size(patch_size)
        else:
            pre_aug_patch_size = patch_size

        if sampler in {"monai_posneg", "posneg"}:
            crop_transform = MonaiSingleSampleCrop(
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=tuple(int(v) for v in pre_aug_patch_size),
                    pos=monai_pos,
                    neg=monai_neg,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0.0,
                    allow_smaller=True,
                )
            )
        elif sampler in {"monai_label_classes", "label_classes"}:
            crop_transform = MonaiSingleSampleCrop(
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=tuple(int(v) for v in pre_aug_patch_size),
                    ratios=monai_label_ratios,
                    num_classes=monai_num_classes,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0.0,
                    allow_smaller=True,
                )
            )
        else:
            crop_transform = Torch_CropPad(
                patch_size=pre_aug_patch_size,
                p_oversample_foreground=train_oversample_fg,
            )

        train_transforms = [Torch_Normalize(normalize=normalize)]
        if dwi_dropout_probability > 0 or dwi_shift_probability > 0:
            train_transforms.append(
                Task2FlairAnchorAugment(
                    dropout_probability=dwi_dropout_probability,
                    shift_probability=dwi_shift_probability,
                    max_shift=dwi_max_shift,
                    dwi_channel=dwi_channel,
                )
            )
        train_transforms.extend(
            [
                crop_transform,
                PadToMinimumSpatialSize(pre_aug_patch_size),
                Torch_Spatial(
                    patch_size=patch_size,
                    p_deform_all_channel=0.0,
                    p_rot_all_channel=p_rot,
                    p_rot_per_axis=0.3,
                    x_rot_in_degrees=(-rot_degrees[0], rot_degrees[0]),
                    y_rot_in_degrees=(-rot_degrees[1], rot_degrees[1]),
                    z_rot_in_degrees=(-rot_degrees[2], rot_degrees[2]),
                    p_scale_all_channel=p_scale,
                    scale_factor=(scale_min, scale_max),
                ),
                Torch_Mirror(
                    p_per_sample=p_mirror,
                    p_mirror_per_axis=0.5,
                    axes=axes,
                ),
            ]
        )
        return transforms.Compose(train_transforms)

    def CPU_seg_val_transforms_small_object_roi(patch_size, normalize=True):
        val_transforms = [Torch_Normalize(normalize=normalize)]
        if eval_dwi_mode != "none":
            val_transforms.append(Task2DwiStressTransform(eval_dwi_mode, eval_dwi_shift, dwi_channel))
        if val_mode in {"sliding_image", "full_image", "nnunet"}:
            return transforms.Compose(val_transforms)
        val_transforms.extend(
            [
                Torch_CropPad(
                    patch_size=patch_size,
                    p_oversample_foreground=val_oversample_fg,
                ),
            ]
        )
        return transforms.Compose(val_transforms)

    presets.CPU_seg_train_transforms = CPU_seg_train_transforms_small_object_roi
    presets.CPU_seg_val_transforms = CPU_seg_val_transforms_small_object_roi
    train_presets.CPU_seg_train_transforms = CPU_seg_train_transforms_small_object_roi
    train_presets.CPU_seg_val_transforms = CPU_seg_val_transforms_small_object_roi
    print(
        "[small-object-roi] patched CPU_seg_train_transforms and CPU_seg_val_transforms "
        f"train_oversample_fg={train_oversample_fg} sampler={sampler} "
        f"val_mode={val_mode} val_oversample_fg={val_oversample_fg} "
        f"p_rot={p_rot} p_scale={p_scale} "
        f"p_mirror={p_mirror} mirror_axes={mirror_axes_env or 'all'} "
        f"rot_degrees={rot_degrees} scale=({scale_min}, {scale_max}) "
        f"dwi_dropout={dwi_dropout_probability} dwi_shift={dwi_shift_probability} "
        f"dwi_max_shift={dwi_max_shift} dwi_channel={dwi_channel} "
        f"eval_dwi_mode={eval_dwi_mode} eval_dwi_shift={eval_dwi_shift} "
        f"monai_pos={monai_pos} monai_neg={monai_neg} "
        f"monai_label_ratios={monai_label_ratios} monai_num_classes={monai_num_classes}",
        flush=True,
    )


class SmallObjectTverskyCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        tversky_weight: float = 0.7,
        ce_weight: float = 0.3,
        foreground_ce_weight: float = 2.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tversky_weight = float(tversky_weight)
        self.ce_weight = float(ce_weight)
        self.foreground_ce_weight = float(foreground_ce_weight)
        self.smooth = float(smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == pred.ndim:
            target = target.squeeze(1)
        target = target.long()
        num_classes = pred.shape[1]
        probs = pred.softmax(dim=1)
        target_1h = F.one_hot(target.clamp_min(0), num_classes=num_classes).movedim(-1, 1).to(dtype=probs.dtype)

        spatial_dims = tuple(range(2, pred.ndim))
        fg = slice(1, None) if num_classes > 1 else slice(0, None)
        probs_fg = probs[:, fg]
        target_fg = target_1h[:, fg]
        tp = (probs_fg * target_fg).sum(dim=spatial_dims)
        fp = (probs_fg * (1.0 - target_fg)).sum(dim=spatial_dims)
        fn = ((1.0 - probs_fg) * target_fg).sum(dim=spatial_dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = (1.0 - tversky).clamp_min(0.0).pow(self.gamma).mean()

        class_weights = pred.new_ones(num_classes)
        if num_classes > 1:
            class_weights[1:] = self.foreground_ce_weight
        ce = F.cross_entropy(pred, target, weight=class_weights)
        return self.tversky_weight * tversky_loss + self.ce_weight * ce



def _get_small_object_loss(module):
    loss_name = os.environ.get("SMALL_OBJECT_LOSS", "dicece").lower()
    if loss_name in {"", "dicece", "default"}:
        return None
    valid_losses = {"tversky_ce", "focal_tversky_ce", "small_object_tversky_ce", "monai_gdfl", "generalized_dice_focal"}
    if loss_name not in valid_losses:
        raise ValueError(f"Unknown SMALL_OBJECT_LOSS={loss_name!r}")
    if not hasattr(module, "_small_object_loss"):
        if loss_name in {"monai_gdfl", "generalized_dice_focal"}:
            from monai.losses import GeneralizedDiceFocalLoss

            module._small_object_loss = GeneralizedDiceFocalLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                w_type=os.environ.get("MONAI_GDFL_W_TYPE", "square"),
                gamma=_parse_float_env("MONAI_FOCAL_GAMMA", 2.0),
                lambda_gdl=_parse_float_env("MONAI_LAMBDA_GDL", 1.0),
                lambda_focal=_parse_float_env("MONAI_LAMBDA_FOCAL", 1.0),
            )
            print(
                "[small-object-roi] using MONAI GeneralizedDiceFocalLoss "
                f"include_background=False to_onehot_y=True softmax=True "
                f"w_type={os.environ.get('MONAI_GDFL_W_TYPE', 'square')} "
                f"gamma={_parse_float_env('MONAI_FOCAL_GAMMA', 2.0)} "
                f"lambda_gdl={_parse_float_env('MONAI_LAMBDA_GDL', 1.0)} "
                f"lambda_focal={_parse_float_env('MONAI_LAMBDA_FOCAL', 1.0)}",
                flush=True,
            )
        else:
            module._small_object_loss = SmallObjectTverskyCELoss(
                alpha=_parse_float_env("TVERSKY_ALPHA", 0.3),
                beta=_parse_float_env("TVERSKY_BETA", 0.7),
                gamma=_parse_float_env("TVERSKY_GAMMA", 1.33),
                tversky_weight=_parse_float_env("TVERSKY_WEIGHT", 0.7),
                ce_weight=_parse_float_env("TVERSKY_CE_WEIGHT", 0.3),
                foreground_ce_weight=_parse_float_env("FOREGROUND_CE_WEIGHT", 2.0),
            )
            print(
                "[small-object-roi] using segmentation loss "
                f"loss={loss_name} alpha={module._small_object_loss.alpha} "
                f"beta={module._small_object_loss.beta} gamma={module._small_object_loss.gamma} "
                f"tversky_weight={module._small_object_loss.tversky_weight} "
                f"ce_weight={module._small_object_loss.ce_weight} "
                f"foreground_ce_weight={module._small_object_loss.foreground_ce_weight}",
                flush=True,
            )
    return module._small_object_loss


def patch_small_object_loss() -> None:
    loss_name = os.environ.get("SMALL_OBJECT_LOSS", "dicece").lower()
    if loss_name in {"", "dicece", "default"}:
        return

    from asparagus.functional.metrics.utils import format_multilabel_metrics
    from asparagus.modules.lightning_modules import segmentation_module
    from gardening_tools.modules.losses.deep_supervision import DeepSupervisionLoss

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        base_loss = _get_small_object_loss(self)
        if base_loss is None:
            loss = self.train_loss(pred, y)
        elif self.deep_supervision:
            max_outputs = max(1, int(os.environ.get("DEEP_SUPERVISION_MAX_OUTPUTS", "3")))
            pred_for_loss = pred[:max_outputs]
            # The transform's legacy label pyramid assumes every decoder level
            # halves all three axes.  That breaks anisotropic decoders that
            # preserve Z in shallow levels.  Derive every target from the full
            # label and match the actual logit shape instead; this is identical
            # to the old pyramid for isotropic 2x stages.
            full_target = y[0] if isinstance(y, (list, tuple)) else y
            y_for_loss = []
            for logits in pred_for_loss:
                if tuple(full_target.shape[2:]) == tuple(logits.shape[2:]):
                    target = full_target
                else:
                    target = F.interpolate(full_target.float(), size=logits.shape[2:], mode="nearest").to(dtype=full_target.dtype)
                y_for_loss.append(target)
            configured_weights = _parse_float_list_env("DEEP_SUPERVISION_WEIGHTS", (0.7, 0.2, 0.1))
            if len(configured_weights) < len(pred_for_loss):
                raise ValueError("DEEP_SUPERVISION_WEIGHTS has fewer values than selected outputs")
            weights = configured_weights[: len(pred_for_loss)]
            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("DEEP_SUPERVISION_WEIGHTS must sum to a positive value")
            weights = [weight / weight_sum for weight in weights]
            loss = DeepSupervisionLoss(loss=base_loss, weights=weights)(pred_for_loss, y_for_loss)
        else:
            loss = base_loss(pred, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if self.deep_supervision:
            pred = pred[0]
            y = y[0]
        metrics = self.train_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        return loss

    segmentation_module.SegmentationModule.training_step = training_step
    print(f"[small-object-roi] patched SegmentationModule.training_step loss={loss_name}", flush=True)

def _postprocess_logits(logits: torch.Tensor) -> torch.Tensor:
    labels = _postprocess_labels(logits)
    num_classes = logits.shape[1]
    one_hot = F.one_hot(labels.clamp_min(0), num_classes=num_classes).movedim(-1, 1).to(dtype=logits.dtype)
    return one_hot * 20.0 - (1.0 - one_hot) * 20.0


def _keep_score_ranked_components(
    labels: torch.Tensor,
    probs: torch.Tensor,
    min_size: int,
    keep_components: int,
    score_mode: str,
) -> torch.Tensor:
    """Keep foreground components by confidence instead of physical size.

    A largest-component rule is brittle for Task 2: a diffuse false positive can
    be larger than a small meningioma, and a single remote blob makes the global
    foreground bounding box unusable.  Ranking components by their probability
    statistics lets validation choose whether the expected lesion is the most
    confident compact region rather than the largest foreground region.
    """
    import numpy as np
    from scipy.ndimage import generate_binary_structure, label as cc_label

    if score_mode not in {"mean", "peak", "mean_log_volume", "mean_spatial", "mean_log_volume_spatial"}:
        raise ValueError(f"Unknown SEG_POST_COMPONENT_SCORE={score_mode!r}")
    use_spatial_prior = score_mode.endswith("_spatial")
    if use_spatial_prior:
        prior_mean = _parse_float_list_env("SEG_POST_SPATIAL_MEAN_XY", ())
        prior_std = _parse_float_list_env("SEG_POST_SPATIAL_STD_XY", ())
        if len(prior_mean) != 2 or len(prior_std) != 2 or any(value <= 0 for value in prior_std):
            raise ValueError(
                "Spatial component ranking requires SEG_POST_SPATIAL_MEAN_XY and "
                "SEG_POST_SPATIAL_STD_XY as two positive normalized XY values."
            )
        spatial_weight = _parse_float_env("SEG_POST_SPATIAL_WEIGHT", 0.5)
        if spatial_weight < 0:
            raise ValueError("SEG_POST_SPATIAL_WEIGHT must be non-negative")
    connectivity = generate_binary_structure(3, 1)
    processed = []
    for label_item, prob_item in zip(labels.detach().cpu(), probs.detach().cpu()):
        label_np = label_item.numpy()
        prob_np = prob_item.numpy()
        output = np.zeros_like(label_np)
        for class_idx in range(1, probs.shape[1]):
            components, count = cc_label(label_np == class_idx, structure=connectivity)
            ranked: list[tuple[float, int, np.ndarray]] = []
            for component_idx in range(1, count + 1):
                component = components == component_idx
                volume = int(component.sum())
                if volume < min_size:
                    continue
                values = prob_np[class_idx][component]
                if score_mode in {"mean", "mean_spatial"}:
                    score = float(values.mean())
                elif score_mode == "peak":
                    score = float(values.max())
                else:
                    score = float(values.mean() * np.log1p(volume))
                if use_spatial_prior:
                    # Soft ranker: remote blobs need stronger confidence, but are never hard-filtered.
                    center_xy = np.argwhere(component).mean(axis=0)[:2]
                    denom_xy = np.maximum(np.asarray(component.shape[:2], dtype=np.float64) - 1.0, 1.0)
                    normalized_xy = center_xy / denom_xy
                    z_score_sq = np.square((normalized_xy - np.asarray(prior_mean)) / np.asarray(prior_std)).sum()
                    spatial_score = float(np.exp(-0.5 * z_score_sq))
                    score *= spatial_score ** spatial_weight
                ranked.append((score, volume, component))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected = ranked if keep_components <= 0 else ranked[:keep_components]
            for _, _, component in selected:
                output[component] = class_idx
        processed.append(torch.from_numpy(output).long())
    return torch.stack(processed, dim=0).to(device=labels.device)


def _postprocess_labels(logits: torch.Tensor) -> torch.Tensor:
    mode = os.environ.get("SEG_POSTPROCESS", "none").lower()
    threshold = _parse_float_env("SEG_POST_THRESHOLD", 0.3)
    min_size = int(_parse_float_env("SEG_POST_MIN_SIZE", 0))
    keep_components = int(_parse_float_env("SEG_POST_KEEP_COMPONENTS", 0))
    fill_holes = os.environ.get("SEG_POST_FILL_HOLES", "false").lower() in {"1", "true", "yes"}
    closing_radius = int(_parse_float_env("SEG_POST_CLOSING_RADIUS", 0))
    component_score = os.environ.get("SEG_POST_COMPONENT_SCORE", "mean").lower()
    # Per-class thresholds (override global threshold if set)
    thr_c1 = _parse_float_env("SEG_POST_THR_C1", threshold)
    thr_c2 = _parse_float_env("SEG_POST_THR_C2", threshold)
    num_classes = logits.shape[1]
    probs = logits.softmax(dim=1)

    if mode.startswith("perclass_threshold"):
        # Independent per-class thresholding; background wins ties
        labels = torch.zeros(probs.shape[0], *probs.shape[2:], dtype=torch.long, device=probs.device)
        thresholds = [None, thr_c1, thr_c2] + [threshold] * (num_classes - 3)
        for cls_i in range(1, num_classes):
            thr = thresholds[cls_i] if cls_i < len(thresholds) else threshold
            mask = probs[:, cls_i] >= thr
            labels = torch.where(mask, torch.full_like(labels, cls_i), labels)
    elif mode.startswith("threshold"):
        if num_classes == 2:
            labels = (probs[:, 1] >= threshold).long()
        else:
            fg_prob, fg_idx = probs[:, 1:].max(dim=1)
            labels = torch.where(fg_prob >= threshold, fg_idx.long() + 1, torch.zeros_like(fg_idx).long())
    else:
        labels = probs.argmax(dim=1).long()

    if "score_cc" in mode:
        labels = _keep_score_ranked_components(
            labels=labels,
            probs=probs,
            min_size=min_size,
            keep_components=keep_components,
            score_mode=component_score,
        )
        # The confidence-ranked helper has already applied these two operations.
        min_size = 0
        keep_components = 0

    if "cc" in mode or min_size > 0 or keep_components > 0 or fill_holes or closing_radius > 0:
        from monai.transforms import FillHoles, KeepLargestConnectedComponent, RemoveSmallObjects

        processed = []
        applied = list(range(1, num_classes))
        for item in labels.detach().cpu():
            lab = item
            if min_size > 0:
                lab = RemoveSmallObjects(min_size=min_size, connectivity=1)(lab)
            if fill_holes:
                lab = FillHoles(applied_labels=applied, connectivity=1)(lab)
            if closing_radius > 0:
                import numpy as np
                from scipy.ndimage import binary_closing
                from scipy.ndimage import generate_binary_structure
                lab_np = lab.numpy()
                result = np.zeros_like(lab_np)
                struct = generate_binary_structure(3, 1)
                # Dilate struct to closing_radius iterations
                for cls_i in range(1, num_classes):
                    mask = lab_np == cls_i
                    if mask.any():
                        closed = binary_closing(mask, structure=struct, iterations=closing_radius)
                        result = np.where(closed & (result == 0), cls_i, result)
                # Preserve background where no fg class closed
                result = np.where(result == 0, lab_np, result)
                lab = torch.from_numpy(result).long()
            if keep_components > 0:
                lab = KeepLargestConnectedComponent(
                    applied_labels=applied,
                    is_onehot=False,
                    independent=True,
                    connectivity=1,
                    num_components=keep_components,
                )(lab)
            processed.append(torch.as_tensor(lab, dtype=torch.long))
        labels = torch.stack(processed, dim=0).to(device=logits.device)

    return labels.long()


def _foreground_dice_from_labels(pred_labels: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
    if target.ndim == pred_labels.ndim + 1:
        target = target.squeeze(1)
    target = target.long()
    pred_labels = pred_labels.long()
    eps = pred_labels.new_tensor(1e-6, dtype=torch.float32)
    per_class = []
    for cls_idx in range(1, num_classes):
        pred_fg = pred_labels == cls_idx
        target_fg = target == cls_idx
        intersection = (pred_fg & target_fg).sum(dtype=torch.float32)
        denom = pred_fg.sum(dtype=torch.float32) + target_fg.sum(dtype=torch.float32)
        dice = (2.0 * intersection + eps) / (denom + eps)
        per_class.append(dice)
    mean_dice = torch.stack(per_class).mean() if per_class else pred_labels.new_tensor(0.0, dtype=torch.float32)
    return {"mean": mean_dice, "per_class": per_class}


def _foreground_nsd_from_labels(
    pred_labels: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    spacing: Sequence[float],
    tolerance: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute foreground NSD in physical units using per-subject spacing.

    The challenge runner historically computed NSD only in ``test_step``.  Using
    the identical SurfaceDiceMetric here keeps checkpoint selection and
    validation ranking aligned, while preserving test-split quarantine.
    """
    from monai.metrics import SurfaceDiceMetric
    import torch.nn.functional as F

    if target.ndim == pred_labels.ndim + 1:
        target = target.squeeze(1)
    pred_labels = pred_labels.long()
    target = target.long()
    pred_onehot = F.one_hot(pred_labels, num_classes=num_classes).movedim(-1, 1).float().cpu()
    target_onehot = F.one_hot(target, num_classes=num_classes).movedim(-1, 1).float().cpu()
    nsd_metric = SurfaceDiceMetric(
        include_background=False,
        class_thresholds=[tolerance] * (num_classes - 1),
    )
    spacing = [float(value) for value in spacing]
    if len(spacing) != pred_labels.ndim - 1 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing {spacing} for prediction shape {tuple(pred_labels.shape)}")
    nsd_metric(y_pred=pred_onehot, y=target_onehot, spacing=[spacing] * pred_onehot.shape[0])
    nsd_values = nsd_metric.aggregate(reduction="none")
    # A fold may contain an empty class. Treat it as zero contribution rather
    # than allowing NaN to make checkpoint selection non-deterministic.
    per_class_cpu = torch.nan_to_num(nsd_values, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
    per_class = [pred_labels.new_tensor(value.item(), dtype=torch.float32) for value in per_class_cpu]
    mean_nsd = torch.stack(per_class).mean() if per_class else pred_labels.new_tensor(0.0, dtype=torch.float32)
    return {"mean": mean_nsd, "per_class": per_class}


def _foreground_alignment_from_labels(pred_labels: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if target.ndim == pred_labels.ndim + 1:
        target = target.squeeze(1)
    pred_fg = pred_labels.long() > 0
    target_fg = target.long() > 0
    if pred_fg.ndim == 4:
        pred_fg = pred_fg[0]
    if target_fg.ndim == 4:
        target_fg = target_fg[0]
    out = {
        "pred_fg_voxels": float(pred_fg.sum().detach().cpu()),
        "gt_fg_voxels": float(target_fg.sum().detach().cpu()),
        "centroid_distance_vox": float("nan"),
        "bbox_iou": 0.0,
    }
    if not bool(pred_fg.any()) or not bool(target_fg.any()):
        return out
    pred_idx = pred_fg.nonzero().float()
    target_idx = target_fg.nonzero().float()
    pred_centroid = pred_idx.mean(dim=0)
    target_centroid = target_idx.mean(dim=0)
    out["centroid_distance_vox"] = float(torch.linalg.vector_norm(pred_centroid - target_centroid).detach().cpu())
    pred_min, pred_max = pred_idx.min(dim=0).values, pred_idx.max(dim=0).values
    target_min, target_max = target_idx.min(dim=0).values, target_idx.max(dim=0).values
    inter_min = torch.maximum(pred_min, target_min)
    inter_max = torch.minimum(pred_max, target_max)
    inter_side = torch.clamp(inter_max - inter_min + 1.0, min=0.0)
    inter_vol = inter_side.prod()
    pred_vol = (pred_max - pred_min + 1.0).prod()
    target_vol = (target_max - target_min + 1.0).prod()
    union = pred_vol + target_vol - inter_vol
    if float(union.detach().cpu()) > 0:
        out["bbox_iou"] = float((inter_vol / union).detach().cpu())
    return out


def patch_deterministic_seg_validation_loader() -> None:
    from torch.utils.data import DataLoader
    from asparagus.functional.collate import collate_return
    from asparagus.modules.data_modules import training as training_data_modules

    def val_dataloader(self):
        # Evaluate every validation subject exactly once. The upstream loader uses
        # RandomSampler with replacement, which biases small validation sets.
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_return,
        )

    training_data_modules.SegDataModule.val_dataloader = val_dataloader
    print("[small-object-roi] patched SegDataModule.val_dataloader deterministic_each_subject_once", flush=True)


def patch_sliding_validation() -> None:
    val_mode = os.environ.get("SMALL_OBJECT_VAL_MODE", "sliding_image").lower()
    if val_mode not in {"sliding_image", "full_image", "nnunet"}:
        return

    from asparagus.functional.metrics.utils import format_multilabel_metrics
    from asparagus.functional.utils import fit_patch_size_to_image_size
    from asparagus.modules.lightning_modules import segmentation_module
    from monai.inferers import sliding_window_inference

    def _val_sw_predict(self, x):
        sw_batch_size = int(os.environ.get("SW_VAL_BATCH_SIZE", os.environ.get("SW_BATCH_SIZE", "1")))
        overlap = float(os.environ.get("SW_VAL_OVERLAP", os.environ.get("SW_OVERLAP", "0.5")))
        sw_mode = os.environ.get("SW_VAL_MODE", os.environ.get("SW_MODE", "gaussian"))
        x_pad, original_spatial_shape = self._pad_for_sliding_window_inference(x)
        patch_size = fit_patch_size_to_image_size(self.inference_patch_size, list(x_pad.shape[2:]))

        def predictor(patch_data):
            out = self.model(patch_data)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

        logits = sliding_window_inference(
            inputs=x_pad,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
            mode=sw_mode,
        )
        return self._crop_to_spatial_shape(logits, original_spatial_shape)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        if "voxel_spacing" not in batch:
            raise KeyError("Validation batch is missing voxel_spacing; physical NSD cannot be computed safely")
        with torch.no_grad():
            pred = _val_sw_predict(self, x)
            base_loss = _get_small_object_loss(self)
            loss = self.val_loss(pred, y) if base_loss is None else base_loss(pred, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if os.environ.get("SMALL_OBJECT_LIGHT_VAL_METRICS", "true").lower() in {"1", "true", "yes"}:
            pred_labels = _postprocess_labels(pred)
            dice = _foreground_dice_from_labels(pred_labels, y, pred.shape[1])
            nsd_tolerance = _parse_float_env("SEG_NSD_TOLERANCE", 1.0)
            spacing = torch.as_tensor(batch["voxel_spacing"]).reshape(-1, pred_labels.ndim - 1)[0].tolist()
            nsd = _foreground_nsd_from_labels(
                pred_labels, y, pred.shape[1], spacing=spacing, tolerance=nsd_tolerance
            )
            dsc_weight = _parse_float_env("SEG_DSC_SELECTION_WEIGHT", 0.5)
            if not 0.0 <= dsc_weight <= 1.0:
                raise ValueError("SEG_DSC_SELECTION_WEIGHT must be in [0, 1]")
            selection_score = dsc_weight * dice["mean"] + (1.0 - dsc_weight) * nsd["mean"]
            self.log(
                "val/foreground_dice_mean",
                dice["mean"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
            self.log(
                "val/foreground_F1_mean",
                dice["mean"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,
                )
            self.log(
                "val/foreground_nsd_mean",
                nsd["mean"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
            self.log(
                "val/foreground_dsc_nsd_mean",
                selection_score,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
            for idx, dice_value in enumerate(dice["per_class"], start=1):
                self.log(
                    f"val/foreground_dice_class_{idx}",
                    dice_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,
                )
            for idx, nsd_value in enumerate(nsd["per_class"], start=1):
                self.log(
                    f"val/foreground_nsd_class_{idx}",
                    nsd_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,
                )
            align = _foreground_alignment_from_labels(pred_labels, y)
            for name, value in align.items():
                if value == value:
                    self.log(
                        f"val/alignment_{name}",
                        pred.new_tensor(value, dtype=torch.float32),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.trainer.datamodule.batch_size,
                    )
            return

        pred_for_metrics = _postprocess_logits(pred)
        metrics = self.val_metrics(pred_for_metrics, y.squeeze(1))
        dice_vec = metrics.get("val/dice")
        f1_vec = metrics.get("val/F1")
        if dice_vec is not None and dice_vec.numel() > 1:
            self.log(
                "val/foreground_dice_mean",
                dice_vec[1:].float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
        if f1_vec is not None and f1_vec.numel() > 1:
            self.log(
                "val/foreground_F1_mean",
                f1_vec[1:].float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    segmentation_module.SegmentationModule.validation_step = validation_step
    print(f"[small-object-roi] patched SegmentationModule.validation_step val_mode={val_mode} sw_val_overlap={os.environ.get('SW_VAL_OVERLAP', os.environ.get('SW_OVERLAP', '0.5'))}", flush=True)


def patch_inference_for_fold_based_split() -> None:
    """
    Patch prepare_inference to support fold-based split files (e.g. split_80_10_10.json)
    which contain a list of folds [{train: [...], val: [...]}] instead of a flat list of paths.
    Controlled by SW_EVAL_FOLD env var (default: 0).
    Only activates if test_split file contains fold dicts rather than string paths.
    """
    from asparagus.pipeline.auto_configuration import experiment_setup
    from gardening_tools.functional.paths.read import load_json

    original_prepare_inference = experiment_setup.prepare_inference

    def patched_prepare_inference(cfg):
        from asparagus.pipeline.auto_configuration.versioning import pathing
        from asparagus.modules.dataclasses import DataFiles
        pathingcfg = pathing(cfg, train=False)
        raw = load_json(cfg.data.test_split_path)
        # Detect fold-based split: list of dicts with 'val' key
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            fold = int(os.environ.get("SW_EVAL_FOLD", "0"))
            test_files = raw[fold]["val"]
            print(f"[small-object-roi] fold-based split detected, using fold={fold} val set ({len(test_files)} files)", flush=True)
        else:
            test_files = raw  # plain list of paths (TEST_80_10_10)
        filecfg = DataFiles(
            dataset_json=load_json(pathingcfg.dataset_json_path),
            splits=None,
            test=test_files,
        )
        return filecfg, pathingcfg

    experiment_setup.prepare_inference = patched_prepare_inference
    print("[small-object-roi] patched prepare_inference to support fold-based val splits", flush=True)


def patch_postprocessed_test_predict() -> None:

    mode = os.environ.get("SEG_POSTPROCESS", "none").lower()

    from asparagus.functional.reverse_preprocessing import reverse_preprocessing
    from asparagus.modules.lightning_modules import segmentation_module
    from asparagus.functional.utils import fit_patch_size_to_image_size
    from monai.inferers import sliding_window_inference

    def _sw_predict(self, x):
        import torch
        import torch.nn.functional as F
        sw_batch_size = int(os.environ.get("SW_BATCH_SIZE", "1"))
        overlap = float(os.environ.get("SW_OVERLAP", "0.5"))
        sw_mode = os.environ.get("SW_MODE", "gaussian")
        tta = os.environ.get("SW_TTA", "0") not in ("0", "false", "")

        x_pad, original_spatial_shape = self._pad_for_sliding_window_inference(x)
        patch_size = fit_patch_size_to_image_size(self.inference_patch_size, list(x_pad.shape[2:]))

        def predictor(patch_data):
            out = self.model(patch_data)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

        def _infer_one(inp):
            return sliding_window_inference(
                inputs=inp,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=overlap,
                mode=sw_mode,
            )

        if not tta:
            logits = _infer_one(x_pad)
        else:
            # 8-flip TTA: all combinations of flips along D, H, W (dims 2,3,4)
            spatial_dims = [2, 3, 4]
            from itertools import combinations
            flip_combos = [[]]  # no flip
            for r in range(1, len(spatial_dims) + 1):
                for combo in combinations(spatial_dims, r):
                    flip_combos.append(list(combo))

            prob_sum = None
            for dims in flip_combos:
                inp = torch.flip(x_pad, dims) if dims else x_pad
                out = _infer_one(inp)
                prob = F.softmax(out, dim=1)
                if dims:
                    prob = torch.flip(prob, dims)
                if prob_sum is None:
                    prob_sum = prob
                else:
                    prob_sum = prob_sum + prob

            # Convert averaged probs back to log-like logits (log so downstream argmax still works)
            avg_prob = prob_sum / len(flip_combos)
            logits = torch.log(avg_prob.clamp(min=1e-7))

        return self._crop_to_spatial_shape(logits, original_spatial_shape)

    def test_step(self, batch, batch_idx):
        import time
        import torch
        from monai.metrics import SurfaceDiceMetric
        x = batch["image"]

        # ---- measure inference time ----
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = _sw_predict(self, x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        # --------------------------------

        if os.environ.get("SEG_POSTPROCESS", "none").lower() not in {"", "none", "raw"}:
            logits = _postprocess_logits(logits)
        src_logits = reverse_preprocessing(logits, batch["properties"])
        src_label = batch["src_label"]

        metrics = self.compute_metrics_from_confusion_matrix(src_logits, src_label)
        metrics["infer_time_s"] = round(infer_time, 4)
        metrics.update(_foreground_alignment_from_labels(src_logits.argmax(dim=1), src_label))

        # ---- NSD per class ----
        try:
            n_classes = src_logits.shape[1]
            pred_onehot = torch.zeros_like(src_logits)
            pred_onehot.scatter_(1, src_logits.argmax(dim=1, keepdim=True), 1)
            label_long = src_label.squeeze(1).long()  # (B,H,W,D)
            gt_onehot = torch.zeros(label_long.shape[0], n_classes, *label_long.shape[1:],
                                    dtype=torch.float32, device=label_long.device)
            gt_onehot.scatter_(1, label_long.unsqueeze(1), 1)
            # Compute source-space NSD in millimetres with the original NIfTI spacing.
            nsd_tolerance = _parse_float_env("SEG_NSD_TOLERANCE", 1.0)
            nsd_metric = SurfaceDiceMetric(
                include_background=False, class_thresholds=[nsd_tolerance] * (n_classes - 1)
            )
            spacing = batch["properties"].get("original_spacing")
            if spacing is None:
                raise KeyError("Test metadata is missing original_spacing; physical NSD is undefined")
            spacing = [float(value) for value in spacing]
            nsd_metric(y_pred=pred_onehot.cpu(), y=gt_onehot.cpu(), spacing=[spacing])
            nsd_vals = nsd_metric.aggregate(reduction="none")  # (B, C-1)
            for cls_i in range(1, n_classes):
                metrics[str(cls_i)]["nsd"] = float(nsd_vals[0, cls_i - 1])
            metrics["mean_nsd"] = float(nsd_vals[0].mean())
        except Exception as e:
            metrics["nsd_error"] = str(e)
        # -----------------------

        self.results[batch["file_path"]] = metrics

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        logits = _sw_predict(self, x)
        if os.environ.get("SEG_POSTPROCESS", "none").lower() not in {"", "none", "raw"}:
            logits = _postprocess_logits(logits)
        logits = reverse_preprocessing(array=logits, image_properties=batch["properties"])
        batch["logits"] = logits
        return batch

    def on_test_epoch_end(self):
        import numpy as np
        from gardening_tools.functional.paths.write import save_json
        avg_results = {}
        first_file = list(self.results.keys())[0]
        # Separate per-class dict keys from flat scalar keys
        scalar_keys = {k for k, v in self.results[first_file].items() if not isinstance(v, dict)}
        dict_keys = {k for k, v in self.results[first_file].items() if isinstance(v, dict)}

        for label in dict_keys:
            avg_results[label] = {}
            for metric in self.results[first_file][label].keys():
                avg_results[label][metric] = round(
                    np.nanmean([self.results[path][label][metric] for path in self.results]),
                    4,
                )
        # Aggregate scalar keys (infer_time_s, mean_nsd, nsd_error, etc.)
        for sk in scalar_keys:
            vals = [self.results[path][sk] for path in self.results
                    if isinstance(self.results[path].get(sk), (int, float))]
            if vals:
                avg_results[sk] = round(float(np.nanmean(vals)), 4)

        self.results["mean"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)

    segmentation_module.SegmentationModule.test_step = test_step
    segmentation_module.SegmentationModule.predict_step = predict_step
    segmentation_module.SegmentationModule.on_test_epoch_end = on_test_epoch_end
    print(
        "[small-object-roi] patched test/predict postprocess "
        f"mode={mode} threshold={_parse_float_env('SEG_POST_THRESHOLD', 0.3)} "
        f"min_size={int(_parse_float_env('SEG_POST_MIN_SIZE', 0))} "
        f"keep_components={int(_parse_float_env('SEG_POST_KEEP_COMPONENTS', 0))} "
        f"fill_holes={os.environ.get('SEG_POST_FILL_HOLES', 'false')}",
        flush=True,
    )


def patch_checkpoint_monitor() -> None:
    monitor = os.environ.get("CKPT_MONITOR", "").strip()
    mode = os.environ.get("CKPT_MONITOR_MODE", "max").strip() or "max"
    if not monitor:
        return

    import lightning.pytorch.callbacks as callbacks

    original_model_checkpoint = callbacks.ModelCheckpoint

    class ForegroundModelCheckpoint(original_model_checkpoint):
        def __init__(self, *args, **kwargs):
            if kwargs.get("monitor") == "val/loss" and kwargs.get("filename") == "best":
                kwargs["monitor"] = monitor
                kwargs["mode"] = mode
                print(
                    f"[small-object-roi] best checkpoint monitor patched to {monitor} mode={mode}",
                    flush=True,
                )
            super().__init__(*args, **kwargs)

    callbacks.ModelCheckpoint = ForegroundModelCheckpoint


def main() -> int:
    install_small_object_transforms()
    install_task2_gpu_transforms()
    patch_small_object_loss()
    install_acquisition_consistency_training()
    patch_deterministic_seg_validation_loader()
    patch_sliding_validation()
    patch_inference_for_fold_based_split()
    patch_postprocessed_test_predict()
    patch_checkpoint_monitor()
    if os.environ.get("TEST_ONLY_CKPT"):
        from asparagus.pipeline.run.test_seg import main as test_main

        test_main()
        return 0
    from asparagus.pipeline.run.finetune_seg import main as finetune_main

    finetune_main()
    return 0


class Task2DwiStressTransform:
    """Deterministic validation-only DWI corruption for robustness audits."""

    def __init__(self, mode: str, shift: Sequence[int], dwi_channel: int = 1):
        self.mode = str(mode).lower()
        self.shift = tuple(int(v) for v in shift)
        self.dwi_channel = int(dwi_channel)
        if self.mode not in {"none", "drop", "shift"}:
            raise ValueError("TASK2_EVAL_DWI_MODE must be one of: none, drop, shift")
        if len(self.shift) != 3:
            raise ValueError("TASK2_EVAL_DWI_SHIFT must contain three integers")

    def __call__(self, data_dict):
        image = data_dict.get("image")
        if self.mode == "none" or image is None or image.ndim < 4:
            return data_dict
        if image.shape[0] <= self.dwi_channel:
            raise RuntimeError(
                f"DWI stress requested for channel {self.dwi_channel}, "
                f"but image has only {image.shape[0]} channel(s)"
            )
        image = image.clone()
        if self.mode == "drop":
            image[self.dwi_channel].zero_()
        else:
            image[self.dwi_channel] = Task2FlairAnchorAugment._translate_without_wrap(
                image[self.dwi_channel], self.shift
            )
        data_dict["image"] = image
        data_dict.setdefault("transforms_applied", {})["task2_dwi_stress"] = {
            "mode": self.mode,
            "shift": self.shift,
        }
        return data_dict


if __name__ == "__main__":
    raise SystemExit(main())


