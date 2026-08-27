from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class Torch_NonzeroCropWithMargin:
    """Crop image/label to the non-zero image bounding box plus a fixed margin."""

    def __init__(
        self,
        data_key: str = "image",
        label_key: str = "label",
        margin: int | Sequence[int] = 16,
        threshold: float = 0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.margin = margin
        self.threshold = float(threshold)

    @staticmethod
    def _as_margin(margin: int | Sequence[int], ndim: int) -> list[int]:
        if isinstance(margin, int):
            return [int(margin)] * ndim
        values = [int(v) for v in margin]
        if len(values) != ndim:
            raise ValueError(f"Expected {ndim} margin values, got {len(values)}.")
        return values

    def _find_bbox(self, image: torch.Tensor) -> tuple[list[int], list[int]] | None:
        if image.ndim < 3:
            raise ValueError(f"Expected image tensor [C, ...], got shape {tuple(image.shape)}.")

        spatial_shape = list(image.shape[1:])
        margin = self._as_margin(self.margin, len(spatial_shape))
        finite_image = torch.nan_to_num(image.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        foreground = finite_image.abs().amax(dim=0) > self.threshold
        coords = foreground.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return None

        starts = coords.min(dim=0).values.tolist()
        ends = (coords.max(dim=0).values + 1).tolist()
        starts = [max(0, int(s) - m) for s, m in zip(starts, margin)]
        ends = [min(size, int(e) + m) for e, m, size in zip(ends, margin, spatial_shape)]
        if any(s >= e for s, e in zip(starts, ends)):
            return None
        return starts, ends

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        image = data_dict[self.data_key]
        bbox = self._find_bbox(image)
        if bbox is None:
            return data_dict

        starts, ends = bbox
        slices = (slice(None), *[slice(s, e) for s, e in zip(starts, ends)])
        data_dict[self.data_key] = image[slices]

        label = data_dict.get(self.label_key)
        if label is not None and tuple(label.shape[1:]) == tuple(image.shape[1:]):
            data_dict[self.label_key] = label[slices]

        transforms_applied = data_dict.setdefault("transforms_applied", {})
        transforms_applied["nonzero_crop_with_margin"] = {
            "starts": starts,
            "ends": ends,
            "input_shape": list(image.shape),
            "output_shape": list(data_dict[self.data_key].shape),
            "margin": self._as_margin(self.margin, len(starts)),
            "threshold": self.threshold,
        }
        return data_dict
