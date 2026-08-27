import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from asparagus.functional.metrics import (
    distribution as dist_metrics,
    features as feat_metrics,
    loss as loss_metrics,
    masking as masking,
    performance as perf_metrics,
    reconstruction as recon_metrics,
    stability as stability_metrics,
    visualization,
)
from asparagus.functional.visualization import log_images_to_logger
from asparagus.modules.lightning_modules.base_module import BaseModule
from torchvision import transforms
from typing import Optional
from gardening_tools.modules.transforms.masking import Torch_Mask


class SelfSupervisedModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        log_images_every_n_epoch: int = 5,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        rec_loss_masked_only: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "AdamW",
        mlflow_logging: bool = False,
        log_every_n_steps: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        global_weight: float = 0.0,
        vicreg_sim_weight: float = 0.0,
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 0.04,
        vicreg_eps: float = 1e-4,
        vicreg_warmup_steps: int = 0,
        semantic_projection_dim: Optional[int] = None,
        reconstruction_loss: str = "mse",
        global_distill_weight: float = 0.0,
        dense_consistency_weight: float = 0.0,
        dense_consistency_scales: Optional[list] = None,
        physical_pair_mode: str = "off",
        physical_l0_weight: float = 0.0,
        physical_l1_weight: float = 0.0,
        physical_l1_magnitude_beta: float = 1.0,
        physical_stage: int = 3,
        physical_spacings: Optional[list] = None,
        physical_mask_ratio: float = 0.6,
        physical_tau: float = 0.01,
        physical_eta: float = 1e-4,
        initial_checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
        )

        self.model = model
        self.reconstruction_loss = str(reconstruction_loss).lower()
        if self.reconstruction_loss not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"Unknown reconstruction_loss={reconstruction_loss!r}; expected mse, l1, or smooth_l1")
        self.rec_loss_masked_only = rec_loss_masked_only
        self.log_images_every_n_epoch = log_images_every_n_epoch
        self.mlflow_logging = mlflow_logging
        self.log_every_n_steps = log_every_n_steps
        self.global_weight = float(global_weight)
        self.vicreg_sim_weight = float(vicreg_sim_weight)
        self.vicreg_var_weight = float(vicreg_var_weight)
        self.vicreg_cov_weight = float(vicreg_cov_weight)
        self.vicreg_eps = float(vicreg_eps)
        self.vicreg_warmup_steps = max(0, int(vicreg_warmup_steps or 0))
        self.global_distill_weight = float(global_distill_weight)
        self.dense_consistency_weight = float(dense_consistency_weight)
        self.dense_consistency_scales = list(dense_consistency_scales or [2, 3, 4])
        self.physical_pair_mode = str(physical_pair_mode).lower()
        if self.physical_pair_mode not in {"off", "identity", "regrid"}:
            raise ValueError("physical_pair_mode must be off, identity, or regrid")
        self.physical_l0_weight = float(physical_l0_weight)
        self.physical_l1_weight = float(physical_l1_weight)
        self.physical_l1_magnitude_beta = float(physical_l1_magnitude_beta)
        self.physical_stage = int(physical_stage)
        self.physical_spacings = [
            tuple(float(v) for v in spacing)
            for spacing in (physical_spacings or [[1.0, 1.0, 2.0], [1.0, 1.0, 4.0], [2.0, 2.0, 2.0]])
        ]
        self.physical_mask_ratio = float(physical_mask_ratio)
        self.physical_tau = float(physical_tau)
        self.physical_eta = float(physical_eta)
        self.physical_masker = Torch_Mask(ratio=self.physical_mask_ratio)
        self.semantic_projection_dim = None if semantic_projection_dim in (None, 0) else int(semantic_projection_dim)
        self.semantic_projector = None
        if self.semantic_projection_dim is not None:
            semantic_feature_dim = self._infer_semantic_feature_dim(model)
            if semantic_feature_dim is None:
                raise ValueError(
                    "training.semantic_projection_dim was set, but the SSL semantic feature "
                    "dimension could not be inferred for this model."
                )
            self.semantic_projector = nn.Sequential(
                nn.LayerNorm(semantic_feature_dim),
                nn.Linear(semantic_feature_dim, self.semantic_projection_dim, bias=False),
            )
        if initial_checkpoint_path:
            checkpoint = torch.load(
                initial_checkpoint_path, map_location="cpu", weights_only=False, mmap=True
            )
            state = checkpoint.get("state_dict", checkpoint)
            incompatibility = nn.Module.load_state_dict(self, state, strict=False)
            if incompatibility.missing_keys or incompatibility.unexpected_keys:
                raise RuntimeError(
                    "Initial checkpoint mismatch: "
                    f"missing={incompatibility.missing_keys[:10]}, "
                    f"unexpected={incompatibility.unexpected_keys[:10]}"
                )
            print(f"Initialized physical pilot from {initial_checkpoint_path}")

    def training_step(self, batch, batch_idx):
        if self.physical_pair_mode != "off":
            return self._physical_pair_step(batch, batch_idx, "train")
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return None

        mask = batch.get("mask", None)
        pred, encoder_features, semantic_features, dense_features = self._forward_ssl(x)
        teacher_semantic_features = None
        teacher_dense_features = None
        if self._needs_teacher_forward():
            with torch.no_grad():
                teacher_dense_features = self._encode_ssl_features(y)
                teacher_semantic_features = self._pool_features(teacher_dense_features)

        rec_loss = self._rec_loss(pred, y, mask if self.rec_loss_masked_only else None)
        global_loss, global_metrics = self._global_loss(semantic_features, teacher_semantic_features)
        global_distill_loss, distill_metrics = self._global_distill_loss(semantic_features, teacher_semantic_features)
        dense_loss, dense_metrics = self._dense_consistency_loss(dense_features, teacher_dense_features)
        vicreg_weight = self._vicreg_weight()
        loss = rec_loss + vicreg_weight * global_loss + self.global_distill_weight * global_distill_loss + self.dense_consistency_weight * dense_loss
        assert not torch.isnan(loss), "SSL loss is NaN"

        # Logging
        with torch.no_grad():
            metrics = {}
            transforms_applied = batch.get("transforms_applied", None)
            if self.global_step % self.log_every_n_steps == 0:  # dont compute if not being logged...
                metrics = {
                    "loss": loss_metrics.compute_train(loss, pred, y, mask, self._rec_loss),
                    "features": feat_metrics.compute_train(encoder_features),
                    "masking": masking.compute(mask, x),
                    "performance": perf_metrics.compute(transforms_applied, x.shape[0]),
                    "stability": stability_metrics.compute_nan_inf_metrics(loss=loss, pred=pred, activations=encoder_features),
                }
                metrics["loss"]["reconstruction_loss"] = rec_loss.item()
                metrics["loss"]["weighted_vicreg_loss"] = (vicreg_weight * global_loss).detach().item()
                metrics["loss"]["weighted_global_distill_loss"] = (self.global_distill_weight * global_distill_loss).detach().item()
                metrics["loss"]["weighted_dense_consistency_loss"] = (self.dense_consistency_weight * dense_loss).detach().item()
                total_detached = loss.detach().abs().clamp_min(torch.finfo(loss.dtype).eps)
                metrics["loss"]["loss_share_recon"] = (rec_loss.detach().abs() / total_detached).item()
                metrics["loss"]["loss_share_vicreg"] = ((vicreg_weight * global_loss).detach().abs() / total_detached).item()
                metrics["loss"]["loss_share_global_distill"] = ((self.global_distill_weight * global_distill_loss).detach().abs() / total_detached).item()
                metrics["loss"]["loss_share_dense_consistency"] = ((self.dense_consistency_weight * dense_loss).detach().abs() / total_detached).item()
                if global_metrics or distill_metrics or dense_metrics:
                    metrics["semantic"] = {**global_metrics, **distill_metrics, **dense_metrics}
                self.log_dict(
                    self._format_metrics("train", metrics),
                    sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,
                )

            if self.current_epoch % 10 == 0 and batch_idx == 0 and self.trainer.is_global_zero:
                images, error_images = visualization.create_visualizations(x, y, pred, mask, self.current_epoch)
                log_images_to_logger(
                    self.trainer.loggers,
                    images,
                    step=self.global_step,
                    prefix="images/train",
                )
                log_images_to_logger(
                    self.trainer.loggers,
                    error_images,
                    step=self.global_step,
                    prefix="images/train_error",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        if self.physical_pair_mode != "off":
            return self._physical_pair_step(batch, batch_idx, "val")
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return None

        mask = batch.get("mask", None)

        pred, encoder_features, semantic_features, dense_features = self._forward_ssl(x)
        teacher_semantic_features = None
        teacher_dense_features = None
        if self._needs_teacher_forward():
            with torch.no_grad():
                teacher_dense_features = self._encode_ssl_features(y)
                teacher_semantic_features = self._pool_features(teacher_dense_features)

        rec_loss = self._rec_loss(pred, y, mask if self.rec_loss_masked_only else None)
        global_loss, global_metrics = self._global_loss(semantic_features, teacher_semantic_features)
        global_distill_loss, distill_metrics = self._global_distill_loss(semantic_features, teacher_semantic_features)
        dense_loss, dense_metrics = self._dense_consistency_loss(dense_features, teacher_dense_features)
        vicreg_weight = self._vicreg_weight()
        loss = rec_loss + vicreg_weight * global_loss + self.global_distill_weight * global_distill_loss + self.dense_consistency_weight * dense_loss
        assert not torch.isnan(loss), "SSL loss is NaN"

        # Logging
        metrics = {
            "loss": loss_metrics.compute_val(loss, pred, y, mask, self._rec_loss),
            "features": feat_metrics.compute_val(encoder_features, self.model),
            "distribution": dist_metrics.compute(x, pred, y, encoder_features),
            "reconstruction": recon_metrics.compute(pred, y, mask),
        }
        metrics["loss"]["reconstruction_loss"] = rec_loss.item()
        metrics["loss"]["weighted_vicreg_loss"] = (vicreg_weight * global_loss).detach().item()
        metrics["loss"]["weighted_global_distill_loss"] = (self.global_distill_weight * global_distill_loss).detach().item()
        metrics["loss"]["weighted_dense_consistency_loss"] = (self.dense_consistency_weight * dense_loss).detach().item()
        if global_metrics or distill_metrics or dense_metrics:
            metrics["semantic"] = {**global_metrics, **distill_metrics, **dense_metrics}
        self.log_dict(
            self._format_metrics("val", metrics),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        # rank zero only
        if self.trainer.is_global_zero:
            images, error_images = visualization.create_visualizations(x, y, pred, mask, self.current_epoch)
            log_images_to_logger(self.trainer.loggers, images, step=self.global_step, prefix="images/val")
            log_images_to_logger(
                self.trainer.loggers,
                error_images,
                step=self.global_step,
                prefix="images/val_error_map",
            )

    @staticmethod
    def _physical_gaussian_kernel1d(sigma, device, dtype):
        if sigma <= 1e-6:
            return torch.ones(1, device=device, dtype=dtype)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coordinate / sigma).square())
        return kernel / kernel.sum()

    @classmethod
    def _physical_blur3d(cls, x, fwhm_voxels):
        out = x
        channels = int(x.shape[1])
        for axis, fwhm in enumerate(fwhm_voxels):
            sigma = float(fwhm) / math.sqrt(8.0 * math.log(2.0))
            kernel = cls._physical_gaussian_kernel1d(sigma, out.device, out.dtype)
            if kernel.numel() == 1:
                continue
            radius = kernel.numel() // 2
            if axis == 0:
                weight = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
                padding = (0, 0, 0, 0, radius, radius)
            elif axis == 1:
                weight = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
                padding = (0, 0, radius, radius, 0, 0)
            else:
                weight = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
                padding = (radius, radius, 0, 0, 0, 0)
            out = F.conv3d(F.pad(out, padding, mode="replicate"), weight, groups=channels)
        return out

    @classmethod
    def _physical_regrid(cls, x, spacing):
        if all(abs(float(v) - 1.0) <= 1e-8 for v in spacing):
            return x.clone()
        fwhm = [math.sqrt(max(float(v) ** 2 - 1.0, 0.0)) for v in spacing]
        blurred = cls._physical_blur3d(x, fwhm)
        acquired_shape = tuple(
            max(2, int(round(length / float(step))))
            for length, step in zip(x.shape[2:], spacing)
        )
        acquired = F.interpolate(
            blurred, size=acquired_shape, mode="trilinear", align_corners=False
        )
        return F.interpolate(
            acquired, size=x.shape[2:], mode="trilinear", align_corners=False
        )

    @staticmethod
    def _physical_channel_norm(x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5)

    def _physical_bandlimit(self, feature, spacing, stride_mm=8.0):
        common_rho = [max(float(value), stride_mm) for value in spacing]
        fwhm = [
            math.sqrt(max(rho * rho - stride_mm * stride_mm, 0.0)) / stride_mm
            for rho in common_rho
        ]
        return self._physical_blur3d(feature, fwhm)

    def _physical_l0(self, first, second):
        first = self._physical_channel_norm(first).movedim(1, -1).flatten(0, -2)
        second = self._physical_channel_norm(second).movedim(1, -1).flatten(0, -2)
        return (1.0 - F.cosine_similarity(first.float(), second.float(), dim=1)).mean()

    def _physical_l1(self, first, second, spacing):
        first = self._physical_channel_norm(self._physical_bandlimit(first, spacing))
        second = self._physical_channel_norm(self._physical_bandlimit(second, spacing))
        direction_num = first.new_zeros((), dtype=torch.float32)
        direction_den = first.new_zeros((), dtype=torch.float32)
        magnitude_losses = []
        for axis in range(3):
            dim = axis + 2
            extent = first.shape[dim] - 2
            da = (first.narrow(dim, 2, extent) - first.narrow(dim, 0, extent)) / 16.0
            db = (second.narrow(dim, 2, extent) - second.narrow(dim, 0, extent)) / 16.0
            a = da.movedim(1, -1).flatten(0, -2).float()
            b = db.movedim(1, -1).flatten(0, -2).float()
            channel_scale = math.sqrt(float(a.shape[1]))
            mag_a = torch.linalg.vector_norm(a, dim=1) / channel_scale
            mag_b = torch.linalg.vector_norm(b, dim=1) / channel_scale
            minimum = torch.minimum(mag_a, mag_b)
            weight = (minimum / (minimum + self.physical_tau)).detach()
            direction_num = direction_num + (
                weight * (1.0 - F.cosine_similarity(a, b, dim=1, eps=1e-8))
            ).sum()
            direction_den = direction_den + weight.sum()
            magnitude_losses.append(
                F.smooth_l1_loss(
                    torch.log(mag_a + self.physical_eta),
                    torch.log(mag_b + self.physical_eta),
                )
            )
        direction = direction_num / direction_den.clamp_min(1e-8)
        magnitude = torch.stack(magnitude_losses).mean()
        return direction + self.physical_l1_magnitude_beta * magnitude, direction, magnitude

    def _physical_mask(self, image):
        payload = self.physical_masker({"image": image.clone()})
        return payload["image"], payload["mask"]

    def _physical_pair_step(self, batch, batch_idx, stage):
        source, clean_target = batch["image"], batch["label"]
        if self.physical_pair_mode == "identity":
            spacing = (1.0, 1.0, 1.0)
        elif stage == "val":
            spacing = self.physical_spacings[batch_idx % len(self.physical_spacings)]
        else:
            choice = int(torch.randint(len(self.physical_spacings), (), device=source.device))
            spacing = self.physical_spacings[choice]
        source_two = self._physical_regrid(source, spacing)
        target_two = self._physical_regrid(clean_target, spacing)
        masked_one, mask_one = self._physical_mask(source)
        masked_two, mask_two = self._physical_mask(source_two)
        pred_one, _enc_one, _semantic_one, dense_one = self._forward_ssl(masked_one)
        pred_two, _enc_two, _semantic_two, dense_two = self._forward_ssl(masked_two)
        rec_loss = 0.5 * (
            self._rec_loss(pred_one, clean_target, mask_one)
            + self._rec_loss(pred_two, target_two, mask_two)
        )
        feature_one = dense_one[self.physical_stage]
        feature_two = dense_two[self.physical_stage]
        l0 = self._physical_l0(feature_one, feature_two)
        l1, l1_direction, l1_magnitude = self._physical_l1(
            feature_one, feature_two, spacing
        )
        loss = rec_loss + self.physical_l0_weight * l0 + self.physical_l1_weight * l1
        self.log_dict(
            {
                f"{stage}/loss/total": loss,
                f"{stage}/loss/reconstruction": rec_loss,
                f"{stage}/physical/l0": l0,
                f"{stage}/physical/l1": l1,
                f"{stage}/physical/l1_direction": l1_direction,
                f"{stage}/physical/l1_magnitude": l1_magnitude,
                f"{stage}/physical/max_spacing_mm": float(max(spacing)),
            },
            sync_dist=True,
            batch_size=source.shape[0],
            prog_bar=False,
        )
        return loss

    def _forward_ssl(self, x):
        if hasattr(self.model, "forward_with_multiscale_features"):
            pred, semantic_features, encoder_features = self.model.forward_with_multiscale_features(x)
            return pred, encoder_features, semantic_features, encoder_features

        dense_features = self._encode_ssl_features(x)
        if hasattr(self.model, "decoder") and dense_features is not None:
            pred = self.model.decoder(dense_features)
            encoder_features = dense_features[-1] if isinstance(dense_features, (list, tuple)) else dense_features
            semantic_features = self._pool_features(dense_features)
            return pred, encoder_features, semantic_features, dense_features

        pred, encoder_features = self.model.forward_with_features(x)
        semantic_features = self._pool_features(encoder_features)
        return pred, encoder_features, semantic_features, encoder_features

    def _encode_ssl_features(self, x):
        if hasattr(self.model, "encoder"):
            return self.model.encoder(x)
        if hasattr(self.model, "forward_with_features"):
            _, features = self.model.forward_with_features(x)
            return features
        return None

    def _pool_features(self, features):
        if isinstance(features, (list, tuple)):
            return torch.cat([self._pool_feature_tensor(feature) for feature in features], dim=1)
        return self._pool_feature_tensor(features)

    @staticmethod
    def _pool_feature_tensor(feature):
        if feature.ndim <= 2:
            return feature
        spatial_dims = tuple(range(2, feature.ndim))
        return torch.cat([feature.mean(dim=spatial_dims), feature.amax(dim=spatial_dims)], dim=1)

    @staticmethod
    def _infer_semantic_feature_dim(model):
        model = getattr(model, "_orig_mod", model)
        if hasattr(model, "semantic_feature_dim"):
            return int(model.semantic_feature_dim)
        encoder = getattr(model, "encoder", None)
        if encoder is not None and hasattr(encoder, "filters"):
            filters = int(encoder.filters)
            return 2 * sum(filters * (2**stage) for stage in range(5))
        stage_channels = SelfSupervisedModule._infer_encoder_stage_channels(encoder)
        if stage_channels:
            return 2 * sum(stage_channels)
        return None

    @staticmethod
    def _infer_encoder_stage_channels(encoder):
        stages = getattr(encoder, "stages", None)
        if stages is None:
            return None
        channels = []
        for stage in stages:
            out_channels = None
            for module in stage.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    out_channels = module.out_channels
            if out_channels is not None:
                channels.append(int(out_channels))
        return channels or None

    def _needs_teacher_forward(self):
        return self.global_distill_weight > 0 or self.dense_consistency_weight > 0 or (self.global_weight > 0 and self.vicreg_sim_weight > 0)

    def _vicreg_weight(self):
        if self.global_weight <= 0:
            return 0.0
        if self.vicreg_warmup_steps <= 0:
            return self.global_weight
        progress = min(1.0, float(self.global_step + 1) / float(self.vicreg_warmup_steps))
        return self.global_weight * progress

    def _project_semantic(self, semantic_features):
        if semantic_features is None:
            return None, None
        z = semantic_features.float()
        if z.ndim > 2:
            z = self._pool_features(z)
        semantic_source_dim = z.shape[1]
        if self.semantic_projector is not None:
            z = self.semantic_projector(z)
        return F.layer_norm(z, (z.shape[1],)), semantic_source_dim

    def _global_loss(self, semantic_features, teacher_semantic_features=None):
        if self.global_weight <= 0 or semantic_features is None:
            return semantic_features.new_zeros(()), {}

        z, semantic_source_dim = self._project_semantic(semantic_features)
        if z is None:
            return semantic_features.new_zeros(()), {}
        if z.shape[0] < 2:
            zero = z.new_zeros(())
            return zero, {
                "vicreg_loss": zero,
                "vicreg_sim_loss": zero,
                "vicreg_var_loss": zero,
                "vicreg_cov_loss": zero,
                "vicreg_weight_effective": self._vicreg_weight(),
                "semantic_std": zero,
                "semantic_dim": float(z.shape[1]),
                "semantic_source_dim": float(semantic_source_dim),
                "global_weight": self.global_weight,
            }

        sim_loss = z.new_zeros(())
        if teacher_semantic_features is not None and self.vicreg_sim_weight > 0:
            z_teacher, _ = self._project_semantic(teacher_semantic_features)
            sim_loss = F.mse_loss(z, z_teacher.detach())

        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.vicreg_eps)
        var_loss = F.relu(1.0 - std).mean()

        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_centered.shape[0] - 1, 1)
        dim = cov.shape[0]
        off_diag = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        cov_loss = off_diag.pow(2).sum() / dim
        global_loss = self.vicreg_sim_weight * sim_loss + self.vicreg_var_weight * var_loss + self.vicreg_cov_weight * cov_loss

        return global_loss, {
            "vicreg_loss": global_loss.detach(),
            "vicreg_sim_loss": sim_loss.detach(),
            "vicreg_var_loss": var_loss.detach(),
            "vicreg_cov_loss": cov_loss.detach(),
            "vicreg_weight_effective": self._vicreg_weight(),
            "semantic_std": std.mean().detach(),
            "semantic_dim": float(z.shape[1]),
            "semantic_source_dim": float(semantic_source_dim),
            "global_weight": self.global_weight,
        }

    def _global_distill_loss(self, semantic_features, teacher_semantic_features):
        if self.global_distill_weight <= 0 or semantic_features is None or teacher_semantic_features is None:
            ref = semantic_features if semantic_features is not None else teacher_semantic_features
            return ref.new_zeros(()), {}
        z, _ = self._project_semantic(semantic_features)
        z_teacher, _ = self._project_semantic(teacher_semantic_features)
        z = F.normalize(z, dim=1)
        z_teacher = F.normalize(z_teacher.detach(), dim=1)
        loss = 1.0 - (z * z_teacher).sum(dim=1).mean()
        return loss, {"global_distill_loss": loss.detach(), "global_distill_weight": self.global_distill_weight}

    def _dense_consistency_loss(self, dense_features, teacher_dense_features):
        if self.dense_consistency_weight <= 0 or dense_features is None or teacher_dense_features is None:
            ref = dense_features if dense_features is not None else teacher_dense_features
            if isinstance(ref, (list, tuple)):
                ref = ref[-1]
            return ref.new_zeros(()), {}
        student = list(dense_features) if isinstance(dense_features, (list, tuple)) else [dense_features]
        teacher = list(teacher_dense_features) if isinstance(teacher_dense_features, (list, tuple)) else [teacher_dense_features]
        n = min(len(student), len(teacher))
        losses = []
        used = []
        for scale in self.dense_consistency_scales:
            idx = int(scale)
            if idx < 0:
                idx = n + idx
            if idx < 0 or idx >= n:
                continue
            s = F.normalize(student[idx].float(), dim=1)
            t = F.normalize(teacher[idx].float().detach(), dim=1)
            if s.shape != t.shape:
                continue
            losses.append(F.mse_loss(s, t))
            used.append(idx)
        if not losses:
            zero = student[-1].new_zeros(())
            return zero, {"dense_consistency_loss": zero, "dense_consistency_scales_used": 0.0}
        loss = torch.stack(losses).mean()
        return loss, {
            "dense_consistency_loss": loss.detach(),
            "dense_consistency_weight": self.dense_consistency_weight,
            "dense_consistency_scales_used": float(len(used)),
        }

    def _rec_loss(self, pred, y, mask=None):
        err = pred - y
        if self.reconstruction_loss == "l1":
            err = err.abs()
        elif self.reconstruction_loss == "smooth_l1":
            err = F.smooth_l1_loss(pred, y, reduction="none")
        else:
            err = err.pow(2)

        if mask is not None:
            selected = ~mask.bool()
            if selected.any():
                return err[selected].mean()
            return err.mean() * 0.0

        return err.mean()

    def on_after_backward(self):
        grad_clip_val = self.trainer.gradient_clip_val if hasattr(self.trainer, "gradient_clip_val") else None
        metrics_grouped = {
            "stability": stability_metrics.compute_on_backward(self.model, grad_clip_val),
            "performance": perf_metrics.compute_on_backward(self.trainer),
        }
        self.log_dict(
            self._format_metrics("train", metrics_grouped),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    def _format_metrics(self, stage, metric_groups):
        """
        Format metrics with hierarchical naming: stage/module/metric on wandb and stage_module/metric on MLflow.
        """
        #
        metric_separator = "_" if self.mlflow_logging else "/"  # mlflow only supports one / (sigh)
        metrics = {}
        for module_name, metric_dict in metric_groups.items():
            for key, value in metric_dict.items():
                metrics[f"{stage}{metric_separator}{module_name}/{key}"] = value
        return metrics
