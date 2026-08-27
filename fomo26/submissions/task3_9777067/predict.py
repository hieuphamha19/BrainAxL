#!/usr/bin/env python3
"""FOMO26 Task 3 inference: 5-fold brain-age regression ensemble."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_to_output

from asparagus.modules.networks.dolphins_xlstm_unet import dolphins_xlstm_unet_b_clsreg

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "model"
PATCH_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)
SPACING_ATOL = 1e-3
DIRECTION_ATOL = 1e-3
FOLD_CHECKPOINTS = tuple(DEFAULT_MODEL_DIR / f"fold{idx}_best.ckpt" for idx in range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 3 brain-age regression")
    parser.add_argument("--t1", required=True, help="Input T1-weighted NIfTI")
    parser.add_argument("--output", required=True, help="Output text file")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("FOMO26_TASK3_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def checkpoint_paths(model_dir: str | os.PathLike[str]) -> list[Path]:
    base = Path(model_dir)
    paths = [base / f"fold{idx}_best.ckpt" for idx in range(5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing task3 ensemble checkpoint(s): {missing}")
    return paths


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = dolphins_xlstm_unet_b_clsreg(
        input_channels=1,
        output_channels=1,
        dimensions="3D",
        starting_filters=40,
        xlstm_stages=(3, 4),
        dropout_op_kwargs={
            "encoder_dropout_rate": 0.0,
            "decoder_dropout_rate": 0.0,
            "inplace": True,
        },
        late_fusion=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    state = {}
    for key, value in source.items():
        if not key.startswith("model."):
            continue
        key = key[len("model.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        state[key] = value
    if not state:
        raise RuntimeError(f"Checkpoint contains no model.* weights: {checkpoint_path}")

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def nonzero_crop_with_margin(x: torch.Tensor, margin: int = 16) -> torch.Tensor:
    """Crop to the nonzero foreground bbox +/- margin voxels, clipped to bounds.

    Mirrors asparagus.modules.transforms.nonzero_crop.Torch_NonzeroCropWithMargin,
    which the training/validation pipeline (CPU_clsreg_val_test_transforms_nonzero_crop)
    applies before z-normalization. Skipping this step lets background air dominate the
    normalization statistics and the fixed-size crop window.
    """
    spatial_shape = list(x.shape[1:])
    foreground = x.abs().amax(dim=0) > 0.0
    coords = foreground.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return x
    starts = coords.min(dim=0).values.tolist()
    ends = (coords.max(dim=0).values + 1).tolist()
    starts = [max(0, int(s) - margin) for s in starts]
    ends = [min(size, int(e) + margin) for e, size in zip(ends, spatial_shape)]
    if any(s >= e for s, e in zip(starts, ends)):
        return x
    slices = (slice(None), *[slice(s, e) for s, e in zip(starts, ends)])
    return x[slices]


def normalize_volume(volume: torch.Tensor) -> torch.Tensor:
    mean = volume.mean()
    std = volume.std(unbiased=False).clamp_min(1e-8)
    return (volume - mean) / std


def image_needs_resampling(
    image: nib.spatialimages.SpatialImage,
    target_spacing: Iterable[float] = TARGET_SPACING,
) -> bool:
    """Return whether an RAS image differs from the 1 mm axis-aligned train grid."""
    target = np.asarray(tuple(float(v) for v in target_spacing), dtype=np.float64)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid NIfTI voxel spacing: {spacing.tolist()}")
    if not np.allclose(spacing, target, rtol=0.0, atol=SPACING_ATOL):
        return True

    # as_closest_canonical permutes/flips axes but intentionally preserves
    # obliquity. The SALD finetuning grid is axis-aligned, so remove residual
    # rotations/shears as well as normalizing voxel size.
    direction = np.asarray(image.affine[:3, :3], dtype=np.float64) / spacing[np.newaxis, :]
    return not np.allclose(direction, np.eye(3), rtol=0.0, atol=DIRECTION_ATOL)


def canonicalize_and_resample(
    image: nib.spatialimages.SpatialImage,
    target_spacing: Iterable[float] = TARGET_SPACING,
    *,
    resample: bool = True,
) -> nib.spatialimages.SpatialImage:
    """Map an input image to the physical grid used by Task 3 finetuning.

    Already aligned 1 mm inputs take an exact no-interpolation path. Clinical
    inputs with anisotropic spacing or oblique direction cosines are resampled
    with trilinear interpolation and zero background before voxel-space crop.
    """
    image = nib.as_closest_canonical(image)
    target = tuple(float(v) for v in target_spacing)
    if len(target) != 3 or any(not np.isfinite(v) or v <= 0 for v in target):
        raise ValueError(f"target_spacing must contain three positive values, got {target}")
    if resample and image_needs_resampling(image, target):
        image = resample_to_output(
            image,
            voxel_sizes=target,
            order=1,
            mode="constant",
            cval=0.0,
        )
    return image


def center_crop_or_pad(x: torch.Tensor, target_size: Iterable[int] = PATCH_SIZE) -> torch.Tensor:
    target = tuple(int(v) for v in target_size)
    if x.ndim != 4:
        raise ValueError(f"Expected C,D,H,W tensor, got shape {tuple(x.shape)}")

    # Torch_Pad pads with the tensor's own min (post-normalization), not zero.
    pad_value = float(x.min())
    spatial = tuple(int(v) for v in x.shape[1:])
    pad_spec = []
    for size, wanted in reversed(tuple(zip(spatial, target))):
        total = max(wanted - size, 0)
        before = total // 2
        after = total - before
        pad_spec.extend((before, after))
    if any(pad_spec):
        x = F.pad(x, pad_spec, mode="constant", value=pad_value)

    slices = [slice(None)]
    for size, wanted in zip(tuple(int(v) for v in x.shape[1:]), target):
        start = max((size - wanted) // 2, 0)
        slices.append(slice(start, start + wanted))
    return x[tuple(slices)]


def preprocess_image(
    image: nib.spatialimages.SpatialImage,
    *,
    target_spacing: Iterable[float] = TARGET_SPACING,
    resample: bool = True,
) -> torch.Tensor:
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D T1 volume, got shape {image.shape}")
    image = canonicalize_and_resample(image, target_spacing, resample=resample)
    array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(array).unsqueeze(0)
    x = nonzero_crop_with_margin(x, margin=16)
    x = normalize_volume(x)
    x = center_crop_or_pad(x)
    return x.unsqueeze(0).contiguous()


def load_input(
    path: str | os.PathLike[str],
    *,
    target_spacing: Iterable[float] = TARGET_SPACING,
    resample: bool = True,
) -> torch.Tensor:
    return preprocess_image(
        nib.load(str(path)),
        target_spacing=target_spacing,
        resample=resample,
    )


@torch.inference_mode()
def predict_age(models: list[torch.nn.Module], x: torch.Tensor) -> tuple[float, list[float]]:
    predictions = []
    for model in models:
        pred = model(x)
        predictions.append(float(pred.squeeze().detach().cpu().item()))
    return float(np.mean(predictions)), predictions


def main() -> int:
    args = parse_args()
    input_path = Path(args.t1)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    source_image = nib.load(str(input_path))
    canonical_image = nib.as_closest_canonical(source_image)
    source_spacing = tuple(float(value) for value in source_image.header.get_zooms()[:3])
    spacing_resampled = image_needs_resampling(canonical_image)
    x = preprocess_image(source_image).to(device)
    models = [load_model(path, device) for path in checkpoint_paths(args.model_dir)]
    age, fold_predictions = predict_age(models, x)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{age:.4f}\n")
    print(
        f"saved={output_path} prediction={age:.4f} "
        f"fold_predictions={[round(v, 4) for v in fold_predictions]} device={device} "
        f"source_shape={source_image.shape} source_spacing={source_spacing} "
        f"spacing_resampled={spacing_resampled}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
