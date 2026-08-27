#!/usr/bin/env python3
"""Extract the two frozen BrainAxL views used by submission 9777066."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import predict as submitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    required = {"subject_id", "flair", "adc", "dwi_b1000", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest columns must include {sorted(required)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = submitted.build_encoder(device)
    all_mm, all_s3 = [], []
    for row in rows:
        mm_parts, s3_parts = [], []
        for key in ("flair", "adc", "dwi_b1000"):
            image = submitted.znorm_crop_resize(
                submitted.load_modality(row[key])
            ).to(device)
            mm, s3 = submitted.pool_skips(encoder(image))
            mm_parts.append(mm.squeeze(0).float().cpu().numpy())
            s3_parts.append(s3.squeeze(0).float().cpu().numpy())
        all_mm.append(np.concatenate(mm_parts))
        all_s3.append(np.concatenate(s3_parts))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ids=np.asarray([row["subject_id"] for row in rows]),
        y=np.asarray([int(row["label"]) for row in rows]),
        mm=np.asarray(all_mm, dtype=np.float64),
        s3=np.asarray(all_s3, dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
