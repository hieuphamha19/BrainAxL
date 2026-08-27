#!/usr/bin/env python3
"""FOMO26 Task 1 - Infarct Classification (frozen xLSTM encoder, two-view linear head).

Method: the xLSTM SSL encoder (run 19726) is FROZEN and a linear head on its pooled
multi-scale features classifies infarct. Full finetuning collapses to ~0.55 on this
21-subject task; the frozen probe does not. Metric is AUROC -> continuous P(infarct).

Why TWO views, and why THESE two. The wide member is the board-calibrated model:
mean+max over all 5 stages, 7440-d = task1.sif = AUROC 0.774. The second member is
mean+max over STAGE 3 ONLY (960-d).

Averaging only pays when the members make DIFFERENT errors. The obvious second view --
the mean-only 3720-d head (task1_v1.sif, board 0.769) -- turned out to be useless for
this: rank correlation 0.988 with mm_all, 0/21 subjects where they disagree, and the
average is identical to mm_all in 99% of bootstrap resamples, because mean-only is a
strict feature SUBSET of mean+max. Stage 3 alone is decorrelated (rank corr 0.757, 2/21
disagreements) while still strong on its own (LOO 0.875).

Scored fold-honestly (member scales from TRAIN folds only, exactly as done below):
nested-outer 0.8846 -> 0.9038, LOO 0.9038 -> 0.9327, repeated-5-fold over 20 seeds
0.8692+-0.0367 -> 0.9139+-0.0205, winning 19/20 seeds and losing 0/20. The halved
seed-to-seed spread is the variance reduction the redundant pair could not deliver.
One encoder pass feeds both views, so runtime is unchanged.

Scale handling: the challenge calls predict.py with ONE subject at a time, so the two
members' logits cannot be z-scored across a batch at inference. Each member's logit is
instead standardised by the mean/std measured over the 21 training subjects and frozen
into the npz (see downstream/task1_infarct_v2/fit_ensemble_head.py). Averaging raw
logits would let the wider head dominate the ranking for no good reason.

Preprocessing reproduces downstream/task1_infarct_v2/frozen_probe_oof.py EXACTLY
(per-modality 1-channel z-norm -> nonzero-crop -> trilinear resize to 128x128x32 ->
encode). Heads are plain numpy, so there is no sklearn dependency at inference.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

APP_ROOT = Path(__file__).resolve().parent
for candidate in (APP_ROOT / "vendor", APP_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CHECKPOINT = Path(os.environ.get("FOMO26_T1_CHECKPOINT", APP_ROOT / "models" / "lstm_s512_19726.ckpt"))
HEAD_PARAMS = Path(os.environ.get("FOMO26_T1_HEAD", APP_ROOT / "models" / "task1_head_params_3ch_ens_mm_s3.npz"))
SIZE = (128, 128, 32)
MODEL_INPUT_CHANNELS = 1
STARTING_FILTERS = 40
XLSTM_STAGES = (3, 4)
MEMBERS = ("mm", "s3")
STAGE3 = 2  # second member reads this encoder stage only
CHECKPOINT_PREFIXES = ("model._orig_mod.", "model.", "module.", "net.", "_forward_module.", "_orig_mod.")
_MODEL = None
_HEAD = None
_DEVICE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 1 - Infarct Classification")
    parser.add_argument("--flair", type=str, required=True, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, required=True, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, required=True, help="Path to DWI (b1000) image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional, unused by the 3ch heads)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional, unused by the 3ch heads)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt file")
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode

        safe = [ListConfig, ContainerMetadata, Metadata, AnyNode, defaultdict, Any, list, int, dict]
        with torch.serialization.safe_globals(safe):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _norm_key(raw_key: str) -> str:
    key, changed = raw_key, True
    while changed:
        changed = False
        for prefix in CHECKPOINT_PREFIXES:
            if key.startswith(prefix):
                key, changed = key[len(prefix):], True
    return key


def build_encoder(device: torch.device) -> nn.Module:
    from task67_backbone import create_s512_backbone

    model = create_s512_backbone(
        input_channels=MODEL_INPUT_CHANNELS, output_channels=1, dimensions="3D",
        starting_filters=STARTING_FILTERS, use_skip_connections=True,
        deep_supervision=False, xlstm_stages=XLSTM_STAGES,
    )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    ckpt = torch_load(CHECKPOINT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    target = model.state_dict()
    matched = {}
    for raw_key, value in state.items():
        for key in (raw_key, _norm_key(raw_key)):
            if key in target and tuple(target[key].shape) == tuple(value.shape):
                matched[key] = value
                break
    if not any(k.startswith("encoder.") for k in matched):
        raise RuntimeError("Checkpoint loaded no encoder weights.")
    merged = dict(target)
    merged.update(matched)
    model.load_state_dict(merged, strict=True)
    encoder = model.encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_head():
    if not HEAD_PARAMS.exists():
        raise FileNotFoundError(f"Missing head params: {HEAD_PARAMS}")
    z = np.load(HEAD_PARAMS)
    head = {}
    for tag in MEMBERS:
        head[tag] = {k: z[f"{tag}_{k}"] for k in
                     ("scaler_mean", "scaler_scale", "coef", "intercept", "logit_mean", "logit_std")}
    return head


def get_model(device: torch.device):
    global _MODEL, _HEAD, _DEVICE
    if _MODEL is None or _DEVICE != device:
        _MODEL, _HEAD, _DEVICE = build_encoder(device), load_head(), device
    return _MODEL, _HEAD


def znorm_crop_resize(vol: torch.Tensor) -> torch.Tensor:
    """3D volume -> (1,1,128,128,32). Matches frozen_probe_oof.znorm_crop exactly."""
    finite = torch.isfinite(vol)
    fg = finite & (vol != 0)
    v = vol[fg]
    if v.numel() > 1:
        vol = torch.where(finite, (vol - v.mean()) / v.std().clamp_min(1e-6), torch.zeros_like(vol))
        vol = torch.where(fg, vol, torch.zeros_like(vol))
    else:
        vol = torch.nan_to_num(vol)
    nz = torch.nonzero(vol != 0)
    if nz.numel():
        lo = nz.min(0).values
        hi = nz.max(0).values + 1
        vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return F.interpolate(vol[None, None].float(), size=SIZE, mode="trilinear", align_corners=False)


def pool_skips(skips) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (all-stage mean+max, stage-3-only mean+max) for one modality.

    Layout matches how the two heads were fitted (see fit_ensemble_head.py):
      mm -> concat over ALL stages of cat([stage_mean, stage_amax])  (2480 per modality)
      s3 -> cat([stage3_mean, stage3_amax])                          ( 320 per modality)
    """
    mm, s3 = [], None
    for i, f in enumerate(skips):
        dims = tuple(range(2, f.ndim))
        block = torch.cat([f.mean(dim=dims), f.amax(dim=dims)], dim=1)
        mm.append(block)
        if i == STAGE3:
            s3 = block
    if s3 is None:
        raise RuntimeError(f"encoder returned {len(skips)} stages, no index {STAGE3}")
    return torch.cat(mm, dim=1), s3


def load_modality(path: str) -> torch.Tensor:
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # NO reorient/norm (matches prepare_stage1.load_case)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D modality, got {arr.shape} for {path}")
    return torch.from_numpy(np.ascontiguousarray(arr))


def _member_z(x: np.ndarray, p: dict) -> float:
    """Standardised logit of one member, using the train-set logit scale frozen at fit time."""
    logit = ((x - p["scaler_mean"]) / p["scaler_scale"]) @ p["coef"] + float(p["intercept"])
    return float((logit - float(p["logit_mean"])) / max(float(p["logit_std"]), 1e-9))


@torch.inference_mode()
def predict(args: argparse.Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, head = get_model(device)
    mm_parts, s3_parts = [], []
    for path in (args.flair, args.adc, args.dwi):  # 3ch order must match training
        x = znorm_crop_resize(load_modality(path)).to(device)
        mm, s3 = pool_skips(encoder(x))
        mm_parts.append(mm.squeeze(0).float().cpu().numpy())
        s3_parts.append(s3.squeeze(0).float().cpu().numpy())
    vecs = {"mm": np.concatenate(mm_parts).astype(np.float64),    # (7440,)
            "s3": np.concatenate(s3_parts).astype(np.float64)}    # ( 960,)
    for tag, v in vecs.items():
        expected = head[tag]["coef"].shape[0]
        if v.shape[0] != expected:
            raise RuntimeError(f"{tag}: feature dim {v.shape[0]} != head dim {expected}")
    z = float(np.mean([_member_z(vecs[tag], head[tag]) for tag in MEMBERS]))
    prob = 1.0 / (1.0 + np.exp(-z))
    if not np.isfinite(prob):
        raise RuntimeError("Non-finite probability.")
    return float(prob)


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    probability = predict(args)
    subject_id = out.stem
    (out.parent / f"{subject_id}.txt").write_text(f"{probability:.6f}")
    print(f"{subject_id}: P(infarct)={probability:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
