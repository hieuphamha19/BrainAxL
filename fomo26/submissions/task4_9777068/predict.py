#!/usr/bin/env python3
"""FOMO26 Task 4 inference: T2 MRI -> labels {0, 1, 2}."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from asparagus.modules.networks.dolphins_xlstm_unet import dolphins_xlstm_unet_b

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_DIR / "model" / "best.ckpt"
PATCH_SIZE = (128, 96, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 4 segmentation")
    parser.add_argument("--t2", required=True, help="Input T2-weighted NIfTI")
    parser.add_argument("--output", required=True, help="Output segmentation NIfTI")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FOMO26_TASK4_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--postprocess", default="threshold_cc", help="Post-processing mode (e.g., argmax_cc or threshold_cc)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for foreground probabilities")
    parser.add_argument("--min-size", type=int, default=4, help="Minimum component size to keep")
    parser.add_argument("--keep-components", type=int, default=2, help="Number of largest components to keep per class")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True, help="Fill holes in components")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="Use Test-Time Augmentation (8-flip)")
    parser.add_argument("--overlap", type=float, default=0.625, help="Sliding window overlap")
    parser.add_argument("--sw-batch-size", type=int, default=4, help="Sliding window batch size")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = dolphins_xlstm_unet_b(
        dimensions="3D",
        input_channels=1,
        output_channels=3,
        deep_supervision=True,
        use_skip_connections=True,
        starting_filters=40,
        xlstm_stages=(3, 4),
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
        raise RuntimeError("Checkpoint contains no model.* weights")
    model_keys = model.state_dict()
    missing = [
        key
        for key in model_keys
        if key not in state and "ds_out_conv" not in key
    ]
    unexpected = [
        key
        for key in state
        if key not in model_keys and "ds_out_conv" not in key
    ]
    shape_mismatch = [
        key
        for key in state
        if key in model_keys
        and model_keys[key].shape != state[key].shape
        and "ds_out_conv" not in key
    ]
    model.load_state_dict(state, strict=False)
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={shape_mismatch[:8]}"
        )
    return model.to(device).eval()


def normalize_volume(volume: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32))
    mean = tensor.mean()
    std = tensor.std(unbiased=False).clamp_min(1e-8)
    return ((tensor - mean) / std).unsqueeze(0).unsqueeze(0)


def pad_to_patch(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    original_shape = tuple(int(v) for v in x.shape[2:])
    target = tuple(max(size, patch) for size, patch in zip(original_shape, PATCH_SIZE))
    pad_after = tuple(t - s for s, t in zip(original_shape, target))
    pad = []
    for amount in reversed(pad_after):
        pad.extend((0, amount))
    return F.pad(x, pad, mode="constant", value=0), original_shape


@torch.inference_mode()
def predict(model: torch.nn.Module, x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    from monai.inferers import sliding_window_inference

    padded, original_shape = pad_to_patch(x)

    def predictor(patch_data):
        out = model(patch_data)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def _infer_one(inp):
        return sliding_window_inference(
            inputs=inp,
            roi_size=PATCH_SIZE,
            sw_batch_size=args.sw_batch_size,
            predictor=predictor,
            overlap=args.overlap,
            mode="gaussian",
        )

    if args.tta:
        # 8-flip TTA ensembling
        spatial_dims = [2, 3, 4]  # D, H, W (for shape 1, 1, D, H, W)
        from itertools import combinations
        flip_combos = [[]]  # no flip
        for r in range(1, len(spatial_dims) + 1):
            for combo in combinations(spatial_dims, r):
                flip_combos.append(list(combo))

        prob_sum = None
        for dims in flip_combos:
            inp = torch.flip(padded, dims) if dims else padded
            out = _infer_one(inp)
            prob = F.softmax(out, dim=1)
            if dims:
                prob = torch.flip(prob, dims)
            if prob_sum is None:
                prob_sum = prob
            else:
                prob_sum = prob_sum + prob

        avg_prob = prob_sum / len(flip_combos)
    else:
        out = _infer_one(padded)
        avg_prob = F.softmax(out, dim=1)

    slices = (slice(None), slice(None)) + tuple(slice(0, n) for n in original_shape)
    probs = avg_prob[slices]

    num_classes = probs.shape[1]

    if args.postprocess.startswith("threshold"):
        if num_classes == 2:
            labels = (probs[:, 1] >= args.threshold).long()
        else:
            fg_prob, fg_idx = probs[:, 1:].max(dim=1)
            labels = torch.where(fg_prob >= args.threshold, fg_idx.long() + 1, torch.zeros_like(fg_idx).long())
    else:
        labels = probs.argmax(dim=1).long()

    if "cc" in args.postprocess or args.min_size > 0 or args.keep_components > 0 or args.fill_holes:
        from monai.transforms import FillHoles, KeepLargestConnectedComponent, RemoveSmallObjects

        applied = list(range(1, num_classes))
        for item in labels.detach().cpu():
            lab = item
            if args.min_size > 0:
                lab = RemoveSmallObjects(min_size=args.min_size, connectivity=1)(lab)
            if args.fill_holes:
                lab = FillHoles(applied_labels=applied, connectivity=1)(lab)
            if args.keep_components > 0:
                lab = KeepLargestConnectedComponent(
                    applied_labels=applied,
                    is_onehot=False,
                    independent=True,
                    connectivity=1,
                    num_components=args.keep_components,
                )(lab)
            return torch.as_tensor(lab, dtype=torch.uint8)

    return labels[0].to(torch.uint8)


def main() -> int:
    args = parse_args()
    input_path = Path(args.t2)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    image = nib.load(str(input_path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D T2 volume, got shape {image.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = load_model(args.checkpoint, device)
    x = normalize_volume(image.get_fdata(dtype=np.float32)).to(device)
    segmentation = predict(model, x, args).cpu().numpy()

    if segmentation.shape != image.shape:
        raise RuntimeError(f"Output shape {segmentation.shape} != input shape {image.shape}")
    if not np.isin(segmentation, (0, 1, 2)).all():
        raise RuntimeError("Output contains labels outside {0, 1, 2}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(segmentation, image.affine, header), str(output_path))
    print(
        f"saved={output_path} shape={segmentation.shape} "
        f"labels={np.unique(segmentation).tolist()} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
