#!/usr/bin/env python3
"""Fit the two-view balanced logistic ensemble used by task1_ens.sif."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.features)
    labels = data["y"].astype(int)
    params: dict[str, np.ndarray] = {}
    for name in ("mm", "s3"):
        features = data[name].astype(np.float64)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.c, max_iter=8000, class_weight="balanced"
            ),
        ).fit(features, labels)
        scaler = model.named_steps["standardscaler"]
        head = model.named_steps["logisticregression"]
        logits = (
            (features - scaler.mean_) / scaler.scale_ @ head.coef_[0]
            + head.intercept_[0]
        )
        params[f"{name}_scaler_mean"] = scaler.mean_.astype(np.float64)
        params[f"{name}_scaler_scale"] = scaler.scale_.astype(np.float64)
        params[f"{name}_coef"] = head.coef_[0].astype(np.float64)
        params[f"{name}_intercept"] = np.asarray(
            head.intercept_[0], dtype=np.float64
        )
        params[f"{name}_logit_mean"] = np.asarray(
            logits.mean(), dtype=np.float64
        )
        params[f"{name}_logit_std"] = np.asarray(
            logits.std(), dtype=np.float64
        )

    params["C"] = np.asarray(args.c)
    params["members"] = np.asarray(["mm", "s3"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
