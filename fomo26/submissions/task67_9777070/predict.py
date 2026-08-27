#!/usr/bin/env python3
"""FOMO26 frozen embedding extractor for Tasks 6/7."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

APP_ROOT = Path(__file__).resolve().parent
_REPO_FALLBACK = APP_ROOT.parents[2] if len(APP_ROOT.parents) >= 3 else Path.cwd()
for candidate in (APP_ROOT / "vendor", APP_ROOT / "repo" / "asparagus", _REPO_FALLBACK / "asparagus"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CHECKPOINT = Path(os.environ.get("FOMO26_EMBED_CHECKPOINT", APP_ROOT / "models" / "lstm_s512_19726.ckpt"))
PATCH_SIZE = (64, 64, 64)
STRIDE = (32, 32, 32)
EXTRACT_BATCH_SIZE = int(os.environ.get("FOMO26_EXTRACT_BATCH_SIZE", "8"))
MODEL_INPUT_CHANNELS = 1
STARTING_FILTERS = 40
XLSTM_STAGES = (3, 4)
SEMANTIC_PROJECTOR_DIM = 512

CHECKPOINT_PREFIXES = ("model._orig_mod.", "model.", "module.", "net.", "_forward_module.", "_orig_mod.")
MODEL = None
PROJECTOR = None
DEVICE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 frozen embedding extractor")
    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI")
    parser.add_argument("--output", type=str, required=True, help="Path to save embeddings .npy")
    return parser.parse_args()


def torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode

        safe = [ListConfig, ContainerMetadata, Metadata, AnyNode, defaultdict, Any, list, int, dict]
        with torch.serialization.safe_globals(safe):
            return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_checkpoint_key(raw_key: str) -> str:
    key = raw_key
    changed = True
    while changed:
        changed = False
        for prefix in CHECKPOINT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def build_backbone() -> nn.Module:
    from task67_backbone import create_s512_backbone

    return create_s512_backbone(
        input_channels=MODEL_INPUT_CHANNELS,
        output_channels=1,
        dimensions="3D",
        starting_filters=STARTING_FILTERS,
        use_skip_connections=True,
        deep_supervision=False,
        xlstm_stages=XLSTM_STAGES,
    )


def load_matching_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Embedding checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Could not find state_dict in checkpoint: {checkpoint_path}")

    target = model.state_dict()
    matched = {}
    for raw_key, value in state_dict.items():
        for key in (raw_key, normalize_checkpoint_key(raw_key)):
            if key in target and tuple(target[key].shape) == tuple(value.shape):
                matched[key] = value
                break
    if not any(key.startswith("encoder.") for key in matched):
        raise RuntimeError("Checkpoint loaded no encoder weights; refusing to extract embeddings.")
    new_state = dict(target)
    new_state.update(matched)
    model.load_state_dict(new_state, strict=True)


def load_semantic_projector(checkpoint_path: Path) -> nn.Module:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    projector_keys = {}
    for raw_key, value in state_dict.items():
        key = normalize_checkpoint_key(raw_key)
        if key.startswith("semantic_projector."):
            projector_keys[key.replace("semantic_projector.", "")] = value
    linear_weight = projector_keys.get("1.weight")
    if linear_weight is None or linear_weight.ndim != 2:
        raise RuntimeError("Checkpoint has no compatible semantic projector weights.")
    out_dim, pooled_dim = int(linear_weight.shape[0]), int(linear_weight.shape[1])
    if out_dim != SEMANTIC_PROJECTOR_DIM:
        raise RuntimeError(f"Expected projector dim {SEMANTIC_PROJECTOR_DIM}, found {out_dim}.")
    projector = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, out_dim, bias=False))
    missing, unexpected = projector.load_state_dict(projector_keys, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Semantic projector state mismatch: missing={missing}, unexpected={unexpected}")
    return projector


def get_model(device: torch.device):
    global MODEL, PROJECTOR, DEVICE
    if MODEL is None or PROJECTOR is None or DEVICE != device:
        model = build_backbone()
        load_matching_checkpoint(model, CHECKPOINT)
        projector = load_semantic_projector(CHECKPOINT)
        model.eval().to(device)
        projector.eval().to(device)
        for module in (model, projector):
            for param in module.parameters():
                param.requires_grad = False
        MODEL, PROJECTOR, DEVICE = model, projector, device
    return MODEL, PROJECTOR


def load_image(path: str) -> torch.Tensor:
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim == 4:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr))


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    out = image.float().clone()
    for c in range(out.shape[0]):
        channel = out[c]
        mask = channel != 0
        values = channel[mask] if bool(mask.any()) else channel.reshape(-1)
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1e-8)
        out[c] = (channel - mean) / std
    return out


def pad_to_patch(image: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    pads = []
    for size, target in reversed(list(zip(image.shape[-3:], patch_size))):
        total = max(0, int(target) - int(size))
        before = total // 2
        pads.extend([before, total - before])
    return F.pad(image, pads) if any(pads) else image


def compute_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    last = size - patch
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_patches(image: torch.Tensor) -> Iterable[torch.Tensor]:
    image = pad_to_patch(image, PATCH_SIZE)
    spatial = tuple(int(v) for v in image.shape[-3:])
    starts = [compute_starts(spatial[i], PATCH_SIZE[i], STRIDE[i]) for i in range(3)]
    for d0 in starts[0]:
        for d1 in starts[1]:
            for d2 in starts[2]:
                yield image[:, d0:d0 + PATCH_SIZE[0], d1:d1 + PATCH_SIZE[1], d2:d2 + PATCH_SIZE[2]]


def pool_skips(skips: Sequence[torch.Tensor]) -> torch.Tensor:
    pooled = []
    for skip in skips:
        dims = tuple(range(2, skip.ndim))
        pooled.append(skip.mean(dim=dims))
        pooled.append(skip.amax(dim=dims))
    return torch.cat(pooled, dim=1)


@torch.inference_mode()
def extract_patch_features(model: nn.Module, projector: nn.Module, patches: torch.Tensor, device: torch.device) -> torch.Tensor:
    patches = patches.to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        skips = model.encoder(patches)
        features = pool_skips(skips)
        features = projector(features.float())
    return features.float().cpu()


def extract_embedding(image: torch.Tensor, model: nn.Module, projector: nn.Module, device: torch.device) -> np.ndarray:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [C,D,H,W], got {tuple(image.shape)}")
    image = normalize_image(image)
    batches = []
    features = []
    for patch in iter_patches(image):
        batches.append(patch)
        if len(batches) >= EXTRACT_BATCH_SIZE:
            features.append(extract_patch_features(model, projector, torch.stack(batches, dim=0), device))
            batches = []
    if batches:
        features.append(extract_patch_features(model, projector, torch.stack(batches, dim=0), device))
    crop_features = torch.cat(features, dim=0)
    embedding = torch.cat([crop_features.mean(dim=0), crop_features.max(dim=0).values], dim=0)
    if not torch.isfinite(embedding).all():
        raise RuntimeError("Extracted embedding contains NaN/Inf.")
    return embedding.numpy().astype(np.float32, copy=False)


def predict(args: argparse.Namespace) -> np.ndarray:
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, projector = get_model(device)
    embedding = extract_embedding(load_image(args.input), model, projector, device)
    if embedding.shape != (1024,) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid embedding shape/value: {embedding.shape}")
    return embedding


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, predict(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
