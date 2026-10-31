#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _shape_for_path(path: str) -> tuple[str, tuple[int, ...] | None, str | None]:
    p = Path(path)
    try:
        if path.endswith((".nii", ".nii.gz")):
            import nibabel as nib

            img = nib.load(str(p))
            return path, tuple(int(v) for v in img.shape[:3]), None
        if path.endswith(".pt"):
            import torch

            x = torch.load(str(p), map_location="cpu")
            shape = tuple(int(v) for v in x.shape[-3:])
            return path, shape, None
        return path, None, f"unsupported extension: {p.suffix}"
    except Exception as exc:  # noqa: BLE001 - report and fall back to repeat 1.
        return path, None, f"{type(exc).__name__}: {exc}"


def _grid_patches(shape: tuple[int, ...], patch_size: tuple[int, int, int]) -> int:
    return int(math.prod(max(1, math.ceil(dim / patch)) for dim, patch in zip(shape, patch_size)))


def _repeat_for_shape(
    shape: tuple[int, ...] | None,
    patch_size: tuple[int, int, int],
    reference_grid_patches: int,
    min_repeat: int,
    max_repeat: int,
) -> tuple[int, int | None]:
    if shape is None:
        return min_repeat, None
    grid = _grid_patches(shape, patch_size)
    repeat = int(math.ceil(grid / max(1, reference_grid_patches)))
    repeat = max(min_repeat, min(max_repeat, repeat))
    return repeat, grid


def _load_split(split_path: Path) -> dict[str, list[str]]:
    payload = json.loads(split_path.read_text())
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError(f"Expected split JSON format [{{train: [...], val: [...]}}], got {type(payload)}")
    return payload[0]


def _copy_json_with_metadata(src: Path, dst: Path, task_name: str, metadata: dict[str, Any]) -> None:
    payload = json.loads(src.read_text())
    if isinstance(payload, dict):
        payload["name"] = task_name
        payload.setdefault("dataset_config", {})["task_name"] = task_name
        payload.setdefault("metadata", {}).update(metadata)
    dst.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a coverage-weighted Asparagus task by repeating large scans in train split.")
    parser.add_argument("--source-task-dir", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split-name", default="split_95_5_0")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--reference-grid-patches", type=int, default=24)
    parser.add_argument("--min-repeat", type=int, default=1)
    parser.add_argument("--max-repeat", type=int, default=2)
    parser.add_argument("--val-repeat", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process first N train paths.")
    args = parser.parse_args()

    source_task_dir = args.source_task_dir
    split = _load_split(source_task_dir / f"{args.split_name}.json")
    train = list(dict.fromkeys(split["train"]))
    val = list(dict.fromkeys(split["val"]))
    if args.limit and args.limit > 0:
        train = train[: args.limit]

    patch_size = tuple(int(v) for v in args.patch_size)
    unique_paths = sorted(set(train) | set(val))

    shape_by_path: dict[str, tuple[int, ...] | None] = {}
    error_by_path: dict[str, str] = {}
    workers = max(1, int(args.num_workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, shape, error in ex.map(_shape_for_path, train, chunksize=128):
            shape_by_path[path] = shape
            if error:
                error_by_path[path] = error

    repeated_train: list[str] = []
    per_path: list[dict[str, Any]] = []
    repeat_counts: Counter[int] = Counter()
    grid_counts: Counter[str] = Counter()
    for path in train:
        shape = shape_by_path.get(path)
        repeat, grid = _repeat_for_shape(
            shape,
            patch_size,
            int(args.reference_grid_patches),
            int(args.min_repeat),
            int(args.max_repeat),
        )
        repeated_train.extend([path] * repeat)
        repeat_counts[repeat] += 1
        if grid is None:
            grid_counts["unknown"] += 1
        elif grid <= 8:
            grid_counts["<=8"] += 1
        elif grid <= 16:
            grid_counts["9-16"] += 1
        elif grid <= 32:
            grid_counts["17-32"] += 1
        elif grid <= 64:
            grid_counts["33-64"] += 1
        else:
            grid_counts[">64"] += 1
        per_path.append(
            {
                "path": path,
                "shape": list(shape) if shape is not None else None,
                "grid_patches": grid,
                "repeat": repeat,
                "error": error_by_path.get(path),
            }
        )

    repeated_val = []
    for path in val:
        repeated_val.extend([path] * max(1, int(args.val_repeat)))

    out_dir = args.output_data_root / args.task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_task_dir": str(source_task_dir),
        "coverage_weighted": True,
        "coverage_weighting": {
            "patch_size": list(patch_size),
            "reference_grid_patches": int(args.reference_grid_patches),
            "min_repeat": int(args.min_repeat),
            "max_repeat": int(args.max_repeat),
            "val_repeat": int(args.val_repeat),
            "formula": "repeat = clamp(ceil(prod(ceil(shape_i / patch_i)) / reference_grid_patches), min_repeat, max_repeat)",
        },
        "unique_train_rows": len(train),
        "weighted_train_rows": len(repeated_train),
        "unique_val_rows": len(val),
        "weighted_val_rows": len(repeated_val),
    }

    _copy_json_with_metadata(source_task_dir / "dataset.json", out_dir / "dataset.json", args.task_name, metadata)
    (out_dir / "paths.json").write_text(json.dumps(unique_paths, indent=2) + "\n")
    (out_dir / f"{args.split_name}.json").write_text(json.dumps([{"train": repeated_train, "val": repeated_val}], indent=2) + "\n")
    (out_dir / "coverage_index.jsonl").write_text("\n".join(json.dumps(row) for row in per_path) + "\n")

    summary = {
        "task_name": args.task_name,
        "source_task_dir": str(source_task_dir),
        "out_dir": str(out_dir),
        "split_name": args.split_name,
        "patch_size": list(patch_size),
        "reference_grid_patches": int(args.reference_grid_patches),
        "min_repeat": int(args.min_repeat),
        "max_repeat": int(args.max_repeat),
        "unique_train_rows": len(train),
        "weighted_train_rows": len(repeated_train),
        "mean_train_repeat": len(repeated_train) / max(1, len(train)),
        "unique_val_rows": len(val),
        "weighted_val_rows": len(repeated_val),
        "repeat_counts": dict(sorted(repeat_counts.items())),
        "grid_count_bins": dict(grid_counts),
        "shape_errors": len(error_by_path),
        "shape_error_examples": list(error_by_path.items())[:10],
    }
    (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
