#!/usr/bin/env python3
"""Fit the original+pydeface domain-augmented Task 5 logistic head."""

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
    ids = data["ids"].astype(str)
    domains = data["domains"].astype(str)
    labels = data["y"].astype(int)
    features = data["X"].astype(np.float64)

    ordered_ids = sorted(set(ids))
    original, defaced, targets = [], [], []
    for subject_id in ordered_ids:
        indices = np.flatnonzero(ids == subject_id)
        by_domain = {domains[i]: i for i in indices}
        if set(by_domain) != {"original", "defaced"}:
            raise ValueError(f"{subject_id} must have one row per domain")
        subject_labels = set(labels[indices].tolist())
        if len(subject_labels) != 1:
            raise ValueError(f"inconsistent labels for {subject_id}")
        original.append(features[by_domain["original"]])
        defaced.append(features[by_domain["defaced"]])
        targets.append(subject_labels.pop())

    fit_x = np.concatenate([original, defaced])
    fit_y = np.concatenate([targets, targets])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c, max_iter=5000, class_weight="balanced"
        ),
    ).fit(fit_x, fit_y)
    scaler = model.named_steps["standardscaler"]
    head = model.named_steps["logisticregression"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        coef=head.coef_[0].astype(np.float64),
        intercept=np.asarray(head.intercept_[0], dtype=np.float64),
        box_mm=np.asarray([200.0, 140.0, 230.0]),
        center_mode=np.asarray("foreground"),
        spacing_mm=np.asarray(2.0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
