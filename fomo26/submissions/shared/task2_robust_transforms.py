"""Opt-in Task 2 acquisition-shift transforms."""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn.functional as F


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return default if not value else float(value)


def _float_pair_env(name: str, default: Sequence[float]) -> tuple[float, float]:
    value = os.environ.get(name, "").strip()
    values = list(default) if not value else [float(v) for v in value.replace(",", " ").split()]
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return float(values[0]), float(values[1])


class Task2ZAcquisitionAugment:
    """Simulate thick/irregular Z sampling while preserving the label grid."""

    def __init__(self, probability: float, zoom_range: Sequence[float]):
        self.probability = float(probability)
        self.zoom_range = tuple(float(v) for v in zoom_range)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("TASK2_Z_ACQ_PROB must be in [0, 1]")
        if len(self.zoom_range) != 2 or not 0.0 < self.zoom_range[0] <= self.zoom_range[1] <= 1.0:
            raise ValueError("TASK2_Z_ACQ_ZOOM_RANGE must satisfy 0 < min <= max <= 1")

    def __call__(self, data_dict):
        image = data_dict.get("image")
        if image is None or image.ndim not in {4, 5}:
            return data_dict
        unbatched = image.ndim == 4
        batched = image.unsqueeze(0) if unbatched else image
        output = batched.clone()
        applied = []
        z_size = int(output.shape[-1])
        for batch_index in range(output.shape[0]):
            for channel_index in range(output.shape[1]):
                if float(torch.rand(())) >= self.probability:
                    continue
                factor = float(torch.empty(()).uniform_(*self.zoom_range))
                target_z = max(2, min(z_size - 1, int(round(z_size * factor))))
                if target_z >= z_size:
                    continue
                sample = output[batch_index : batch_index + 1, channel_index : channel_index + 1]
                spatial_xy = tuple(int(v) for v in sample.shape[-3:-1])
                reduced = F.interpolate(
                    sample.float(),
                    size=(*spatial_xy, target_z),
                    mode="trilinear",
                    align_corners=False,
                )
                restored = F.interpolate(
                    reduced,
                    size=(*spatial_xy, z_size),
                    mode="trilinear",
                    align_corners=False,
                ).to(dtype=sample.dtype)
                output[batch_index, channel_index] = restored[0, 0]
                applied.append(
                    {
                        "batch": int(batch_index),
                        "channel": int(channel_index),
                        "factor": factor,
                        "target_z": target_z,
                    }
                )
        data_dict["image"] = output[0] if unbatched else output
        data_dict.setdefault("transforms_applied", {})["task2_z_acquisition"] = applied
        return data_dict


def install_task2_gpu_transforms() -> None:
    probability = _float_env("TASK2_Z_ACQ_PROB", 0.0)
    if probability <= 0:
        return
    zoom_range = _float_pair_env("TASK2_Z_ACQ_ZOOM_RANGE", (0.25, 0.75))

    import asparagus.modules.transforms.presets as presets
    import asparagus.modules.transforms.presets.train as train_presets

    original = train_presets.GPU_all_train_transforms

    def GPU_all_train_transforms_task2(ndim=3, deep_supervision=False):
        composed = original(ndim=ndim, deep_supervision=deep_supervision)
        insert_at = len(composed.transforms) - 1 if deep_supervision else len(composed.transforms)
        composed.transforms.insert(
            insert_at,
            Task2ZAcquisitionAugment(probability=probability, zoom_range=zoom_range),
        )
        return composed

    presets.GPU_all_train_transforms = GPU_all_train_transforms_task2
    train_presets.GPU_all_train_transforms = GPU_all_train_transforms_task2
    print(
        "[small-object-roi] patched GPU_all_train_transforms "
        f"task2_z_acq_prob={probability} task2_z_acq_zoom_range={zoom_range}",
        flush=True,
    )
