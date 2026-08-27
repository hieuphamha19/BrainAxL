#!/usr/bin/env python3
"""Extract stage-3 mean features with the submitted ap140 geometry."""

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
    required = {"subject_id", "t1", "label", "domain"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest columns must include {sorted(required)}")
    if {row["domain"] for row in rows} - {"original", "defaced"}:
        raise ValueError("domain values must be 'original' or 'defaced'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = submitted.build_encoder(device)
    features = []
    for row in rows:
        image = submitted.load_modality(row["t1"]).to(device)
        vector = submitted.pool_skips(encoder(image))
        features.append(vector.squeeze(0).float().cpu().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ids=np.asarray([row["subject_id"] for row in rows]),
        domains=np.asarray([row["domain"] for row in rows]),
        y=np.asarray([int(row["label"]) for row in rows]),
        X=np.asarray(features, dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
