#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def read_paths(manifest_csv: Path, data_root: Path, path_column: str) -> list[str]:
    paths: list[str] = []
    with manifest_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if path_column not in (reader.fieldnames or []):
            raise ValueError(f"Column {path_column!r} not found in {manifest_csv}; columns={reader.fieldnames}")
        for row in reader:
            raw = (row.get(path_column) or "").strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = data_root / p
            paths.append(str(p))
    if not paths:
        raise ValueError(f"No paths found in {manifest_csv}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an Asparagus NIfTI-backed FOMO300K pretraining task.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asparagus-data-root", type=Path, required=True)
    parser.add_argument("--task-name", default="PT902_FOMO300K_FULL")
    parser.add_argument("--split-name", default="split_95_5_0")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    paths = read_paths(args.manifest_csv, args.data_root, args.path_column)
    rng = random.Random(args.seed)
    paths = list(dict.fromkeys(paths))
    rng.shuffle(paths)

    n_val = max(1, int(round(len(paths) * args.val_fraction)))
    val = sorted(paths[:n_val])
    train = sorted(paths[n_val:])

    missing: list[str] = []
    if args.check_paths:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing paths; first examples: {missing[:5]}")

    out_dir = args.asparagus_data_root / args.task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_json = {
        "name": args.task_name,
        "dataset_config": {
            "task_name": args.task_name,
            "n_modalities": 1,
            "n_classes": 1,
            "split": args.split_name,
        },
        "preprocessing_config": {
            "normalization_operation": ["runtime_pretrain_normalize"],
            "target_spacing": None,
            "target_orientation": "native",
            "crop_to_nonzero": False,
        },
        "saving_config": {
            "file_format": "nii.gz",
            "save_as_tensor": False,
        },
        "metadata": {
            "source": "official_fomo300k_manifest",
            "manifest_csv": str(args.manifest_csv),
            "total_rows": len(paths),
            "train_rows": len(train),
            "val_rows": len(val),
            "path_check": {
                "checked": len(paths) if args.check_paths else 0,
                "missing": len(missing),
                "all_exist": len(missing) == 0 if args.check_paths else None,
            },
        },
    }
    summary = {
        "dataset_name": args.task_name,
        "manifest_csv": str(args.manifest_csv),
        "data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "split_name": args.split_name,
        "total": len(paths),
        "train": len(train),
        "val": len(val),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "nifti_backed": True,
        "path_check": dataset_json["metadata"]["path_check"],
    }

    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    (out_dir / "paths.json").write_text(json.dumps(sorted(paths), indent=2) + "\n")
    (out_dir / f"{args.split_name}.json").write_text(json.dumps([{"train": train, "val": val}], indent=2) + "\n")
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
