"""Acquisition-invariant consistency training for Task 2.

The clean view keeps the ordinary supervised segmentation objective. A second
view changes only acquisition appearance (per-channel contrast/offset, smooth
bias field, noise, and optional through-plane degradation), so its spatial
coordinates and label remain valid. The model learns to reproduce the detached
clean prediction on that shifted view. Nothing changes at inference.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def acquisition_shift(image: torch.Tensor) -> torch.Tensor:
    """Simulate scanner/style and slice-profile shifts without moving anatomy."""
    if image.ndim != 5:
        raise ValueError(f"Expected BCHWZ image, got shape={tuple(image.shape)}")

    batch, channels = image.shape[:2]
    dtype = image.dtype
    device = image.device
    scale = torch.empty(batch, channels, 1, 1, 1, device=device, dtype=dtype).uniform_(0.70, 1.30)
    offset = torch.empty(batch, channels, 1, 1, 1, device=device, dtype=dtype).uniform_(-0.20, 0.20)
    shifted = image * scale + offset

    field = torch.randn(batch, channels, 4, 4, 4, device=device, dtype=torch.float32)
    field = F.interpolate(field, size=image.shape[2:], mode="trilinear", align_corners=False)
    shifted = shifted * (1.0 + 0.20 * torch.tanh(field)).to(dtype=dtype)

    if image.shape[-1] >= 4 and bool(torch.rand((), device=device) < 0.5):
        reduced_z = max(2, image.shape[-1] // 2)
        shifted = F.interpolate(
            shifted,
            size=(*image.shape[2:-1], reduced_z),
            mode="trilinear",
            align_corners=False,
        )
        shifted = F.interpolate(
            shifted,
            size=image.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    noise_scale = shifted.detach().float().std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-3)
    noise = torch.randn_like(shifted, dtype=torch.float32) * (0.03 * noise_scale)
    return shifted + noise.to(dtype=dtype)


def foreground_consistency_loss(
    clean_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """Foreground-aware consistency, with the supervised clean view as teacher."""
    if clean_logits.shape != shifted_logits.shape:
        raise ValueError(
            "Clean and shifted logits must have identical shapes, got "
            f"{tuple(clean_logits.shape)} and {tuple(shifted_logits.shape)}"
        )
    clean_probability = clean_logits.softmax(1)[:, 1].detach().float()
    shifted_probability = shifted_logits.softmax(1)[:, 1].float()

    target = label[0] if isinstance(label, (list, tuple)) else label
    if target.ndim == clean_logits.ndim:
        target = target[:, 0]
    target = target.float()
    if tuple(target.shape[1:]) != tuple(clean_probability.shape[1:]):
        target = F.interpolate(
            target[:, None], size=clean_probability.shape[1:], mode="nearest"
        )[:, 0]

    # Counter the huge background volume so all-background is not rewarded.
    support = F.max_pool3d(target[:, None], kernel_size=9, stride=1, padding=4)[:, 0]
    voxel_weight = 1.0 + 4.0 * support
    mse = ((shifted_probability - clean_probability).square() * voxel_weight).sum()
    mse = mse / voxel_weight.sum().clamp_min(1.0)

    eps = 1e-4
    teacher = clean_probability.clamp(eps, 1.0 - eps)
    student = shifted_probability.clamp(eps, 1.0 - eps)
    kl = teacher * (teacher.log() - student.log())
    kl += (1.0 - teacher) * ((1.0 - teacher).log() - (1.0 - student).log())
    kl = (kl * voxel_weight).sum() / voxel_weight.sum().clamp_min(1.0)
    return 0.5 * mse + 0.5 * kl


def install_acquisition_consistency_training() -> None:
    """Wrap the installed training step when the registered method is enabled."""
    enabled = os.environ.get("TASK2_ACQ_CONSISTENCY", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if not enabled:
        return

    from asparagus.modules.lightning_modules import segmentation_module

    original_training_step = segmentation_module.SegmentationModule.training_step
    consistency_weight = 0.30

    def training_step(self, batch, batch_idx):
        captured = []

        def capture_output(_module, _inputs, output):
            captured.append(output)

        handle = self.model.register_forward_hook(capture_output)
        try:
            supervised_loss = original_training_step(self, batch, batch_idx)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Could not capture clean-view logits for acquisition consistency")

        clean_logits = captured[-1]
        if isinstance(clean_logits, (list, tuple)):
            clean_logits = clean_logits[0]
        shifted_logits = self.model(acquisition_shift(batch["image"]))
        if isinstance(shifted_logits, (list, tuple)):
            shifted_logits = shifted_logits[0]

        consistency = foreground_consistency_loss(clean_logits, shifted_logits, batch["label"])
        total_loss = supervised_loss + consistency_weight * consistency
        batch_size = self.trainer.datamodule.batch_size
        self.log(
            "train/acquisition_consistency",
            consistency,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return total_loss

    segmentation_module.SegmentationModule.training_step = training_step
    print(
        "[task2-acq-consistency] enabled clean-supervised + acquisition-shift "
        f"consistency weight={consistency_weight}",
        flush=True,
    )
