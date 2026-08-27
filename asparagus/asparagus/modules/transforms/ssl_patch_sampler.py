from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


class Torch_SSLPatchSampler:
    """Image-only SSL crop sampler for large 3D scans.

    The sampler never uses downstream labels. It samples fixed-size patches from
    image foreground, image boundary/ring, and low-foreground context regions.
    """

    def __init__(
        self,
        patch_size: Sequence[int],
        tissue_prob: float = 0.65,
        boundary_prob: float = 0.25,
        context_prob: float = 0.10,
        threshold: float = 0.0,
        margin: int | Sequence[int] = 16,
        deterministic: bool = False,
        max_attempts: int = 24,
        data_key: str = "image",
        label_key: str = "label",
    ):
        self.patch_size = [int(v) for v in patch_size]
        self.tissue_prob = float(tissue_prob)
        self.boundary_prob = float(boundary_prob)
        self.context_prob = float(context_prob)
        self.threshold = float(threshold)
        self.margin = margin
        self.deterministic = bool(deterministic)
        self.max_attempts = int(max_attempts)
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def _as_margin(margin: int | Sequence[int], ndim: int) -> list[int]:
        if isinstance(margin, int):
            return [int(margin)] * ndim
        values = [int(v) for v in margin]
        if len(values) != ndim:
            raise ValueError(f"Expected {ndim} margin values, got {len(values)}.")
        return values

    @staticmethod
    def _seed_from_path(path: str) -> int:
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _randint(self, low: int, high: int, generator: torch.Generator) -> int:
        if high <= low:
            return int(low)
        return int(torch.randint(low, high, (1,), generator=generator).item())

    def _choice_mode(self, generator: torch.Generator) -> str:
        if self.deterministic:
            r = float(torch.rand((), generator=generator).item())
        else:
            r = float(torch.rand((), generator=generator).item())
        if r < self.tissue_prob:
            return "tissue"
        if r < self.tissue_prob + self.boundary_prob:
            return "boundary"
        return "context"

    def _foreground_mask(self, image: torch.Tensor) -> torch.Tensor:
        finite_image = torch.nan_to_num(image.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        return finite_image.abs().amax(dim=0) > self.threshold

    def _bbox(self, foreground: torch.Tensor) -> tuple[list[int], list[int]] | None:
        coords = foreground.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return None
        starts = coords.min(dim=0).values.tolist()
        ends = (coords.max(dim=0).values + 1).tolist()
        return [int(v) for v in starts], [int(v) for v in ends]

    def _clamp_start(self, start: list[int], spatial_shape: Sequence[int]) -> list[int]:
        out = []
        for s, size, patch in zip(start, spatial_shape, self.patch_size):
            out.append(max(0, min(int(s), int(size) - int(patch))))
        return out

    def _sample_tissue_start(
        self,
        bbox_starts: list[int],
        bbox_ends: list[int],
        spatial_shape: Sequence[int],
        generator: torch.Generator,
    ) -> list[int]:
        center = [self._randint(s, max(s + 1, e), generator) for s, e in zip(bbox_starts, bbox_ends)]
        start = [
            c - self._randint(0, max(1, p), generator)
            for c, p in zip(center, self.patch_size)
        ]
        return self._clamp_start(start, spatial_shape)

    def _sample_boundary_start(
        self,
        bbox_starts: list[int],
        bbox_ends: list[int],
        spatial_shape: Sequence[int],
        margin: list[int],
        generator: torch.Generator,
    ) -> list[int]:
        ndim = len(self.patch_size)
        dim = self._randint(0, ndim, generator)
        low_side = bool(self._randint(0, 2, generator))
        start = []
        for i, (box_s, box_e, size, patch, m) in enumerate(zip(bbox_starts, bbox_ends, spatial_shape, self.patch_size, margin)):
            if i == dim:
                edge = box_s if low_side else box_e
                band_lo = edge - patch + max(1, m // 2)
                band_hi = edge - max(1, m // 2)
                start.append(self._randint(band_lo, band_hi + 1, generator))
            else:
                lo = max(0, box_s - m)
                hi = min(int(size) - int(patch), box_e + m - patch)
                start.append(self._randint(lo, hi + 1, generator))
        return self._clamp_start(start, spatial_shape)

    def _sample_context_start(
        self,
        bbox_starts: list[int],
        bbox_ends: list[int],
        spatial_shape: Sequence[int],
        margin: list[int],
        generator: torch.Generator,
    ) -> list[int]:
        ndim = len(self.patch_size)
        dim = self._randint(0, ndim, generator)
        low_side = bool(self._randint(0, 2, generator))
        start = []
        for i, (box_s, box_e, size, patch, m) in enumerate(zip(bbox_starts, bbox_ends, spatial_shape, self.patch_size, margin)):
            if i == dim:
                if low_side:
                    lo = box_s - patch - m
                    hi = box_s - max(1, patch // 4)
                else:
                    lo = box_e - int(0.75 * patch)
                    hi = box_e + m
                start.append(self._randint(lo, hi + 1, generator))
            else:
                lo = max(0, box_s - m)
                hi = min(int(size) - int(patch), box_e + m - patch)
                start.append(self._randint(lo, hi + 1, generator))
        return self._clamp_start(start, spatial_shape)

    def _sample_start(
        self,
        mode: str,
        bbox_starts: list[int],
        bbox_ends: list[int],
        spatial_shape: Sequence[int],
        foreground: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[list[int], float]:
        margin = self._as_margin(self.margin, len(self.patch_size))
        best_start = None
        best_frac = -1.0
        for _ in range(max(1, self.max_attempts)):
            if mode == "boundary":
                start = self._sample_boundary_start(bbox_starts, bbox_ends, spatial_shape, margin, generator)
            elif mode == "context":
                start = self._sample_context_start(bbox_starts, bbox_ends, spatial_shape, margin, generator)
            else:
                start = self._sample_tissue_start(bbox_starts, bbox_ends, spatial_shape, generator)
            slices = tuple(slice(s, s + p) for s, p in zip(start, self.patch_size))
            frac = float(foreground[slices].float().mean().item())
            if best_start is None or abs(frac - 0.35) < abs(best_frac - 0.35):
                best_start, best_frac = start, frac
            if mode == "tissue" and frac >= 0.10:
                return start, frac
            if mode == "boundary" and 0.05 <= frac <= 0.95:
                return start, frac
            if mode == "context" and 0.005 <= frac <= 0.50:
                return start, frac
        return best_start or [0] * len(self.patch_size), float(best_frac)

    def _pad_to_patch(self, image: torch.Tensor, label: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        spatial_shape = list(image.shape[1:])
        pad_pairs = []
        needs_pad = False
        for size, patch in zip(reversed(spatial_shape), reversed(self.patch_size)):
            total = max(0, patch - int(size))
            needs_pad = needs_pad or total > 0
            pad_pairs.extend([total // 2, total - total // 2])
        if not needs_pad:
            return image, label
        image = F.pad(image, tuple(pad_pairs), mode="constant", value=0.0)
        if label is not None:
            label = F.pad(label, tuple(pad_pairs), mode="constant", value=0.0)
        return image, label

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)
        image, label = self._pad_to_patch(image, label)
        spatial_shape = list(image.shape[1:])
        foreground = self._foreground_mask(image)
        bbox = self._bbox(foreground)

        seed_source = str(data_dict.get("file_path", ""))
        if self.deterministic:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed_from_path(seed_source))
        else:
            generator = torch.default_generator

        if bbox is None:
            start = [
                self._randint(0, max(1, int(size) - int(patch) + 1), generator)
                for size, patch in zip(spatial_shape, self.patch_size)
            ]
            mode = "random_empty"
            fg_frac = 0.0
        else:
            mode = self._choice_mode(generator)
            start, fg_frac = self._sample_start(mode, bbox[0], bbox[1], spatial_shape, foreground, generator)

        slices = (slice(None), *[slice(s, s + p) for s, p in zip(start, self.patch_size)])
        data_dict[self.data_key] = image[slices]
        if label is not None:
            data_dict[self.label_key] = label[slices]

        transforms_applied = data_dict.setdefault("transforms_applied", {})
        transforms_applied["ssl_patch_sampler"] = {
            "mode": mode,
            "start": [int(v) for v in start],
            "patch_size": list(self.patch_size),
            "input_shape": list(image.shape),
            "output_shape": list(data_dict[self.data_key].shape),
            "foreground_fraction": float(fg_frac),
            "deterministic": self.deterministic,
        }
        return data_dict
