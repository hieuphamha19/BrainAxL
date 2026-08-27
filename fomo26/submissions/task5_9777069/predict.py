#!/usr/bin/env python3
"""FOMO26 Task 5 - Polymicrogyria Classification, FOV-de-confounded sampling box.

Identical to the shipped task5_s3mean_v2 recipe (frozen xLSTM encoder 19726, world-RAS
2 mm grid centred on the nonzero-foreground centroid, stage-3 mean 320-d, numpy logistic
head) with EXACTLY ONE change: the sampled extent.

Why: the shipped grid samples 128^3 @ 2 mm = a 254 mm cube with padding_mode="zeros".
26-63% of that cube falls outside the acquired FOV, and the thickness of that zero shell
alone separates the classes at AUROC 0.910 -- the cases were simply acquired with a much
larger A-P field of view (up to 250 mm, covering face and neck) than the controls
(133-162 mm). The encoder was reading acquisition protocol, not anatomy.

Fix: sample a 200 x 140 x 230 mm box (world R,A,S) at the SAME 2 mm spacing, then pad the
resulting 100 x 70 x 115 voxels to 128^3 with a CONSTANT margin. 140 mm A-P is the widest
window that fits inside every subject's field of view; past it the leak returns. Measured
on the 48 training subjects (scripts/task5_deconf_grid_search.py): padding-fraction-alone
AUROC 0.910 -> 0.488, max out-of-FOV fraction 0.628 -> 0.057, with no L-R tissue lost.
Voxel spacing and absolute head scale are untouched.

v3 (2026-08-19) -- one further change, and it is ONLY the head .npz. Geometry, encoder,
BOX_MM, centring, spacing and C are byte-identical to task5_deconf_ap140_v2.

The challenge spec (FOMO26_Foundation_Model_Challenge_for_Brain_MRI.pdf, Task 5, "Data
pre-processing method") states verbatim: "Finetuning data: None. Validation and Test data:
defacing, using pydeface (v 2.0.2)." The graded input has its face removed; the finetuning
input does not. That mismatch bites this box harder than the shipped 254 mm cube, because
140 mm of A-P is centred on the head-mask centroid and deleting face voxels moves that
centroid -- so the window lands on different anatomy than the head was fitted on.

v2's head was fitted on the ORIGINAL volumes only. v3's is fitted on the original AND the
pydeface-v202 copies of the same 48 subjects. Subject-wise leave-one-out, scored on the
DEFACED domain because that is what gets graded
(scripts/task5_ap140_domain_match.py, asparagus_models/task5_frozen_probe/ap140_domain_match.json):

    head fitted on        AUROC(defaced)  AUROC(original)  partial|FOV
    original  C=0.05      0.6997          0.9601           -0.321   <- v2
    defaced   C=0.05      0.7674          0.8316           -0.034
    both      C=0.05      0.7674          0.9462           +0.008   <- v3

Augmenting keeps the original-domain performance the defaced-only fit gives up, and lands
partial|FOV nearest zero. C stays 0.05 -- the ap140 work established C in [0.03, 0.1] as a
flat plateau. Note the previously published ap140 partial|FOV of 0.024 was same-domain
(train original / test original); on the domain actually graded, v2 sits at -0.321.

Not changed, and deliberately so: the feature block stays stage-3 mean. A defaced-domain
block sweep found every amax-containing block scoring higher (best 0.8333 vs 0.7674), but
the proposed mechanism was falsified (amax features shift MORE under defacing than mean
features at every stage) and all four paired-bootstrap CIs include zero over 13 swept
configs -- see scripts/task5_ap140_block_sweep_defaced.py.
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

CHECKPOINT = Path(os.environ.get("FOMO26_T5_CHECKPOINT", APP_ROOT / "models" / "lstm_s512_19726.ckpt"))
HEAD_PARAMS = Path(os.environ.get("FOMO26_T5_HEAD", APP_ROOT / "models" / "task5_head_params_t1_ap140_s3mean_domainaug_C0.05.npz"))
GRID_N = 128
SPACING_MM = 2.0
# Sampled extent in world RAS mm (R, A, S). A-P is the confounded axis; 140 mm is the
# widest window that fits inside every training subject's acquired FOV.
BOX_MM = (200.0, 140.0, 230.0)
STARTING_FILTERS = 40
XLSTM_STAGES = (3, 4)
CHECKPOINT_PREFIXES = ("model._orig_mod.", "model.", "module.", "net.", "_forward_module.", "_orig_mod.")
_ENC = None
_HEAD = None
_DEVICE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 5 - Polymicrogyria Classification")
    parser.add_argument("--t1", type=str, required=True, help="Path to T1-weighted image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt")
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
        input_channels=1, output_channels=1, dimensions="3D",
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
    return {k: z[k] for k in ("scaler_mean", "scaler_scale", "coef", "intercept")}


def get_model(device: torch.device):
    global _ENC, _HEAD, _DEVICE
    if _ENC is None or _DEVICE != device:
        _ENC, _HEAD, _DEVICE = build_encoder(device), load_head(), device
    return _ENC, _HEAD


def head_mask(vol: np.ndarray) -> np.ndarray:
    """Largest connected component of the nonzero foreground.

    MUST match scripts/task5_deconf_probe.py::head_mask -- the ap140 head was fitted on
    features centred on THIS mask's centroid, not on the raw nonzero centroid. Using the
    raw centroid lets background speckle drag the box and the container then stops
    reproducing the measured model (worst |dp| 1.9e-2 in the 42538 parity gate).
    """
    from scipy import ndimage

    fg = np.isfinite(vol) & (vol != 0)
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (int(np.argmax(sizes)) + 1)
    return fg


def resample_to_physical_grid(vol: np.ndarray, affine: np.ndarray) -> torch.Tensor:
    """World-RAS aligned BOX_MM sampled at 2 mm, constant-padded to GRID_N^3.

    Centring is unchanged from the shipped container (nonzero-foreground centroid); only
    the sampled extent shrinks, so every subject receives an identical zero margin
    instead of a margin whose thickness encodes the acquisition FOV.
    """
    nz = np.argwhere(head_mask(vol))
    center_vox = nz.mean(axis=0) if nz.size else np.asarray(vol.shape, dtype=np.float64) / 2.0
    center_world = affine[:3, :3] @ center_vox + affine[:3, 3]

    n_ax = np.minimum(np.round(np.asarray(BOX_MM) / SPACING_MM).astype(int), GRID_N)
    axes = [(np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * SPACING_MM for n in n_ax]
    gi, gj, gk = np.meshgrid(*axes, indexing="ij")
    world = np.stack([gi, gj, gk], axis=-1) + center_world
    inverse = np.linalg.inv(affine)
    homogeneous = np.concatenate([world, np.ones(world.shape[:-1] + (1,))], axis=-1)
    voxels = homogeneous @ inverse.T[:, :3]
    norm = (voxels + 0.5) / np.asarray(vol.shape, dtype=np.float64) * 2.0 - 1.0
    grid = torch.from_numpy(norm[..., ::-1].copy()).float()[None]
    out = F.grid_sample(torch.from_numpy(vol)[None, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    finite = torch.isfinite(out)
    foreground = finite & (out != 0)
    values = out[foreground]
    if values.numel() > 1:
        out = torch.where(finite, (out - values.mean()) / values.std().clamp_min(1e-6), torch.zeros_like(out))
        out = torch.where(foreground, out, torch.zeros_like(out))
    else:
        out = torch.nan_to_num(out)

    # Constant symmetric margin up to the encoder's 128^3 input. F.pad takes the last
    # spatial dim first, hence the reversal.
    pads = []
    for n in n_ax[::-1]:
        total = GRID_N - int(n)
        pads.extend([total // 2, total - total // 2])
    return F.pad(out, pads, mode="constant", value=0.0)


def pool_skips(skips) -> torch.Tensor:
    """Scanner-robust 320-D mean pooling from encoder stage 3."""
    feature = skips[3]
    dims = tuple(range(2, feature.ndim))
    return feature.mean(dim=dims)


def load_modality(path: str) -> torch.Tensor:
    img = nib.load(path)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D T1, got {arr.shape} for {path}")
    return resample_to_physical_grid(np.ascontiguousarray(arr), np.asarray(img.affine, dtype=np.float64))


@torch.inference_mode()
def predict(args: argparse.Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, head = get_model(device)
    x = load_modality(args.t1).to(device)
    feat = pool_skips(encoder(x)).squeeze(0).float().cpu().numpy().astype(np.float64)  # (320,)
    z = ((feat - head["scaler_mean"]) / head["scaler_scale"]) @ head["coef"] + float(head["intercept"])
    prob = 1.0 / (1.0 + np.exp(-z))
    if not np.isfinite(prob):
        raise RuntimeError("Non-finite probability.")
    return float(prob)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability = predict(args)
    out_file = output_path.parent / f"{output_path.stem}.txt"
    out_file.write_text(f"{probability:.3f}")
    print(f"{output_path.stem}: P(polymicrogyria)={probability:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
