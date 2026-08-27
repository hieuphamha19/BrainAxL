#!/usr/bin/env python3
"""FOMO26 Task 2: 2-modality meningioma segmentation ensemble inference."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
import torch.nn.functional as F

from asparagus.modules.networks.dolphins_xlstm_unet import dolphins_xlstm_unet_b


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "model"
DEFAULT_CONFIG: dict[str, Any] = {
    "patch_size": [128, 128, 48],
    "postprocess": "threshold_cc",
    "threshold": 0.3,
    "min_size": 8,
    "keep_components": 1,
    "component_score": "mean",
    "fill_holes": False,
    "tta": True,
    "overlap": 0.5,
    "sw_batch_size": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 2 meningioma segmentation")
    parser.add_argument("--flair", required=True, help="Input FLAIR NIfTI")
    parser.add_argument("--dwi", required=True, help="Input DWI NIfTI")
    parser.add_argument("--t2s", help="Optional T2* NIfTI (not used by the 2-channel model)")
    parser.add_argument("--swi", help="Optional SWI NIfTI (not used by the 2-channel model)")
    parser.add_argument("--output", required=True, help="Output binary NIfTI mask")
    parser.add_argument("--model-dir", default=os.environ.get("FOMO26_TASK2_MODEL_DIR", str(DEFAULT_MODEL_DIR)), help=argparse.SUPPRESS)
    parser.add_argument("--postprocess", choices=("none", "threshold_cc", "threshold_score_cc"), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--keep-components", type=int, default=None)
    parser.add_argument("--component-score", choices=("mean", "peak", "mean_log_volume"), default=None)
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(args.model_dir) / "config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text())
        config.update({key: value for key, value in saved.items() if key in config})
    for key in ("postprocess", "threshold", "min_size", "keep_components", "component_score", "fill_holes", "tta", "overlap", "sw_batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    patch_size = tuple(int(value) for value in config["patch_size"])
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError(f"Invalid patch_size={config['patch_size']!r}")
    if not 0.0 < float(config["threshold"]) < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {config['threshold']}")
    if not 0.0 <= float(config["overlap"]) < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {config['overlap']}")
    config["patch_size"] = patch_size
    return config


def checkpoint_paths(model_dir: str | os.PathLike[str]) -> list[Path]:
    base = Path(model_dir)
    paths = [base / f"fold{fold}_best.ckpt" for fold in range(5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Task 2 ensemble checkpoint(s): {missing}")
    return paths


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = dolphins_xlstm_unet_b(
        input_channels=2,
        output_channels=2,
        dimensions="3D",
        starting_filters=40,
        deep_supervision=True,
        use_skip_connections=True,
        xlstm_stages=(3, 4),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    state: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if not key.startswith("model."):
            continue
        key = key[len("model.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        state[key] = value
    if not state:
        raise RuntimeError(f"Checkpoint contains no model.* weights: {checkpoint_path}")

    model_keys = model.state_dict()
    missing = [key for key in model_keys if key not in state and "ds_out_conv" not in key]
    unexpected = [key for key in state if key not in model_keys and "ds_out_conv" not in key]
    shape_mismatch = [
        key
        for key in state
        if key in model_keys and model_keys[key].shape != state[key].shape and "ds_out_conv" not in key
    ]
    model.load_state_dict(state, strict=False)
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )
    return model.to(device).eval()


def canonical_pair(flair_path: Path, dwi_path: Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    reference_original = nib.load(str(flair_path))
    flair = nib.as_closest_canonical(reference_original)
    dwi = nib.as_closest_canonical(nib.load(str(dwi_path)))
    if dwi.shape != flair.shape or not np.allclose(dwi.affine, flair.affine, rtol=1e-5, atol=1e-3):
        dwi = resample_from_to(dwi, (flair.shape, flair.affine), order=1)
    if len(flair.shape) != 3 or len(dwi.shape) != 3:
        raise ValueError(f"Expected 3D FLAIR/DWI, got {flair.shape=} and {dwi.shape=}")
    return reference_original, flair, dwi


def normalize_channels(flair: nib.Nifti1Image, dwi: nib.Nifti1Image) -> torch.Tensor:
    array = np.stack(
        [flair.get_fdata(dtype=np.float32), dwi.get_fdata(dtype=np.float32)], axis=0
    )
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(array).unsqueeze(0)
    mean = tensor.mean(dim=(2, 3, 4), keepdim=True)
    std = tensor.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1e-8)
    return (tensor - mean) / std


def pad_to_patch(x: torch.Tensor, patch_size: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    original_shape = tuple(int(value) for value in x.shape[2:])
    target_shape = tuple(max(size, patch) for size, patch in zip(original_shape, patch_size))
    pad_after = tuple(target - size for size, target in zip(original_shape, target_shape))
    pad = []
    for amount in reversed(pad_after):
        pad.extend((0, amount))
    return F.pad(x, pad, mode="constant", value=0), original_shape


def logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    return output[0] if isinstance(output, (list, tuple)) else output


@torch.inference_mode()
def predict_probabilities(
    checkpoint_paths: list[Path], device: torch.device, x: torch.Tensor, config: dict[str, Any]
) -> torch.Tensor:
    from monai.inferers import sliding_window_inference

    padded, original_shape = pad_to_patch(x, config["patch_size"])

    def infer_one(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "inputs": value,
            "roi_size": config["patch_size"],
            "sw_batch_size": int(config["sw_batch_size"]),
            "predictor": lambda patch: logits(model, patch),
            "overlap": float(config["overlap"]),
            "mode": "gaussian",
        }
        if device.type != "cuda":
            return sliding_window_inference(**kwargs)

        # Match the evaluator's bf16-mixed inference and substantially reduce
        # activation memory in the validator. Older CUDA devices fall back to fp16.
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            return sliding_window_inference(**kwargs)

    spatial_dims = (2, 3, 4)
    flip_sets = [()] if not config["tta"] else [
        combo
        for length in range(len(spatial_dims) + 1)
        for combo in itertools.combinations(spatial_dims, length)
    ]
    probability_sum: torch.Tensor | None = None
    # Keep only one model resident at a time. The five CV checkpoints have
    # identical architecture and their probability average is unchanged, while
    # this avoids validator/runtime failures on smaller GPU or RAM allocations.
    for checkpoint_path in checkpoint_paths:
        model = load_model(checkpoint_path, device)
        for flip_dims in flip_sets:
            value = torch.flip(padded, flip_dims) if flip_dims else padded
            probability = F.softmax(infer_one(model, value), dim=1, dtype=torch.float32)
            if flip_dims:
                probability = torch.flip(probability, flip_dims)
            probability_sum = probability if probability_sum is None else probability_sum + probability
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    assert probability_sum is not None
    probabilities = probability_sum / (len(checkpoint_paths) * len(flip_sets))
    slices = (slice(None), slice(None)) + tuple(slice(0, size) for size in original_shape)
    return probabilities[slices]


def postprocess(probabilities: torch.Tensor, config: dict[str, Any]) -> np.ndarray:
    mask = (probabilities[:, 1] >= float(config["threshold"])).to(torch.uint8)[0]
    if config["postprocess"] == "none":
        return mask.cpu().numpy()

    from monai.transforms import FillHoles, KeepLargestConnectedComponent, RemoveSmallObjects

    if config["postprocess"] == "threshold_score_cc":
        from scipy.ndimage import generate_binary_structure, label as cc_label

        labels, count = cc_label(mask.cpu().numpy() > 0, structure=generate_binary_structure(3, 1))
        probability = probabilities[0, 1].detach().cpu().numpy()
        ranked: list[tuple[float, int, np.ndarray]] = []
        for component_idx in range(1, count + 1):
            component = labels == component_idx
            volume = int(component.sum())
            if volume < int(config["min_size"]):
                continue
            values = probability[component]
            if config["component_score"] == "mean":
                score = float(values.mean())
            elif config["component_score"] == "peak":
                score = float(values.max())
            elif config["component_score"] == "mean_log_volume":
                score = float(values.mean() * np.log1p(volume))
            else:
                raise ValueError(f"Unknown component_score={config['component_score']!r}")
            ranked.append((score, volume, component))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked if int(config["keep_components"]) <= 0 else ranked[: int(config["keep_components"])]
        result = np.zeros_like(labels, dtype=np.uint8)
        for _, _, component in selected:
            result[component] = 1
        mask = torch.from_numpy(result)
        min_size = 0
        keep_components = 0
    else:
        min_size = int(config["min_size"])
        keep_components = int(config["keep_components"])

    result: torch.Tensor | np.ndarray = mask.cpu()
    if min_size > 0:
        result = RemoveSmallObjects(min_size=min_size, connectivity=1)(result)
    if bool(config["fill_holes"]):
        result = FillHoles(applied_labels=[1], connectivity=1)(result)
    if keep_components > 0:
        result = KeepLargestConnectedComponent(
            applied_labels=[1],
            is_onehot=False,
            independent=True,
            connectivity=1,
            num_components=keep_components,
        )(result)
    return np.asarray(result, dtype=np.uint8)


def restore_reference_orientation(
    mask_ras: np.ndarray, reference_original: nib.Nifti1Image, reference_ras: nib.Nifti1Image
) -> np.ndarray:
    original_ornt = nib.orientations.io_orientation(reference_original.affine)
    ras_ornt = nib.orientations.io_orientation(reference_ras.affine)
    transform = nib.orientations.ornt_transform(ras_ornt, original_ornt)
    restored = nib.orientations.apply_orientation(mask_ras, transform)
    if restored.shape != reference_original.shape:
        raise RuntimeError(f"Restored mask shape {restored.shape} != input shape {reference_original.shape}")
    return np.asarray(restored, dtype=np.uint8)


def main() -> int:
    args = parse_args()
    flair_path, dwi_path = Path(args.flair), Path(args.dwi)
    output_path = Path(args.output)
    for path in (flair_path, dwi_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input does not exist: {path}")

    config = load_runtime_config(args)
    reference_original, flair, dwi = canonical_pair(flair_path, dwi_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    paths = checkpoint_paths(args.model_dir)
    probabilities = predict_probabilities(
        paths, device, normalize_channels(flair, dwi).to(device), config
    )
    segmentation = restore_reference_orientation(
        postprocess(probabilities, config), reference_original, flair
    )
    if not np.isin(segmentation, (0, 1)).all():
        raise RuntimeError("Output contains labels outside {0, 1}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_original.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(segmentation, reference_original.affine, header), str(output_path))
    print(
        f"saved={output_path} shape={segmentation.shape} fg_voxels={int(segmentation.sum())} "
        f"models={len(paths)} tta={config['tta']} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
