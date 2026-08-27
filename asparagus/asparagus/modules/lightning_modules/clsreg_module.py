import logging
import os
import torch
import torch.nn as nn
import wandb
from abc import abstractmethod
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.paths.write import save_json
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchvision import transforms
from typing import Optional


class ClsRegBase(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        log_image_every_n_epochs: int = 50,
        test_output_path: str = None,
        load_decoder: bool = True,
        repeat_stem_weights: bool = True,
        freeze_backbone: bool = False,
        encoder_learning_rate: Optional[float] = None,
        decoder_learning_rate: Optional[float] = None,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            decoder_warmup_epochs=decoder_warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            load_decoder=load_decoder,
            repeat_stem_weights=repeat_stem_weights,
            freeze_backbone=freeze_backbone,
            encoder_learning_rate=encoder_learning_rate,
            decoder_learning_rate=decoder_learning_rate,
        )
        self.loss = None
        self.task_type = None
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.test_output_path = test_output_path
        self.ignore_index_in_metrics = -1
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_test_metrics()

    @abstractmethod
    def configure_test_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def configure_metrics(self, prefix: str):
        raise NotImplementedError

    def _prepare_loss_inputs(self, pred, target):
        return pred, target

    def _prepare_metric_inputs(self, pred, target):
        return pred, target

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]

        pred = self.model(x)
        loss_pred, loss_target = self._prepare_loss_inputs(pred, y)
        loss = self.loss(loss_pred, loss_target)

        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.trainer.datamodule.batch_size
        )

        metric_pred, metric_target = self._prepare_metric_inputs(pred, y)
        self.train_metrics.update(metric_pred, metric_target)

        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": metric_target.detach().cpu().to(torch.float32).numpy(),
                    "output": metric_pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="train",
                task_type=self.task_type,
            )

        return loss

    def on_train_epoch_end(self):
        metrics_results = self.train_metrics.compute()
        formatted_metrics = format_multilabel_metrics(metrics_results, ignore_index=self.ignore_index_in_metrics)
        self.log_dict(formatted_metrics, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]

        pred = self.model(x)
        loss_pred, loss_target = self._prepare_loss_inputs(pred, y)
        loss = self.loss(loss_pred, loss_target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.trainer.datamodule.batch_size)

        metric_pred, metric_target = self._prepare_metric_inputs(pred, y)
        self.val_metrics.update(metric_pred, metric_target)

        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": metric_target.detach().cpu().to(torch.float32).numpy(),
                    "output": metric_pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="val",
                task_type=self.task_type,
            )

    def on_validation_epoch_end(self):
        metrics_results = self.val_metrics.compute()
        formatted_metrics = format_multilabel_metrics(metrics_results, ignore_index=self.ignore_index_in_metrics)
        self.log_dict(formatted_metrics, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.results = {}
        self.predictions = []
        self.labels = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        outputs = self.model.forward(x)
        return outputs

    def on_test_epoch_end(self):
        avg_results = {}
        avg_results = self.test_metrics(
            torch.tensor(self.predictions, device=self.predictions[0].device),
            torch.tensor(self.labels, device=self.predictions[0].device),
        )
        avg_results = {key: value.cpu().numpy().tolist() for key, value in avg_results.items()}
        self.results["metrics"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)
        logging.info(f"Aggregated test results for {len(self.results)} files: {avg_results}")


class ClassificationModule(ClsRegBase):
    def __init__(self, label_smoothing: float = 0.0, loss_weight: list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor(loss_weight) if loss_weight else None, label_smoothing=label_smoothing
        )
        self.task_type = "classification"

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "Precision": MulticlassPrecision(num_classes=self.num_classes, average=None),
                "Recall": MulticlassRecall(num_classes=self.num_classes, average=None),
            }
        )

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/acc": MulticlassAccuracy(num_classes=self.num_classes, average=None),
                f"{prefix}/auroc": MulticlassAUROC(num_classes=self.num_classes, average=None),
            },
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch["CLSREG_label"] = batch["CLSREG_label"].squeeze().long()
        return batch

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = outputs.argmax(1).long()
        label = batch["CLSREG_label"]
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
        }
        self.predictions.append(prediction)
        self.labels.append(label)


class _ScaledRegressionLoss(torch.nn.Module):
    def __init__(self, loss_name: str, target_scale: float = 1.0):
        super().__init__()
        self.target_scale = float(target_scale)
        if loss_name == "mse":
            self.base_loss = torch.nn.MSELoss()
        elif loss_name == "mae":
            self.base_loss = torch.nn.L1Loss()
        elif loss_name == "smoothl1":
            self.base_loss = torch.nn.SmoothL1Loss(beta=float(os.environ.get("REGRESSION_SMOOTHL1_BETA", "0.05")))
        else:
            raise ValueError(f"Unknown REGRESSION_LOSS={loss_name!r}; expected mse, mae, or smoothl1")

    def forward(self, pred, target):
        if self.target_scale != 1.0:
            pred = pred / self.target_scale
            target = target / self.target_scale
        return self.base_loss(pred, target)


class RegressionModule(ClsRegBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_name = os.environ.get("REGRESSION_LOSS", "mse").lower()
        target_scale = float(os.environ.get("REGRESSION_TARGET_SCALE", "1.0"))
        self.target_normalize = os.environ.get("REGRESSION_TARGET_NORMALIZE", "false").lower() in {"1", "true", "yes"}
        self.target_mean = float(os.environ.get("REGRESSION_TARGET_MEAN", "0.0"))
        self.target_std = float(os.environ.get("REGRESSION_TARGET_STD", "1.0"))
        if self.target_normalize and self.target_std <= 0:
            raise ValueError("REGRESSION_TARGET_STD must be positive when REGRESSION_TARGET_NORMALIZE=true")
        self.loss = _ScaledRegressionLoss(loss_name=loss_name, target_scale=1.0 if self.target_normalize else target_scale)
        print(
            f"Regression loss: {loss_name}, target_scale={target_scale}, "
            f"target_normalize={self.target_normalize}, target_mean={self.target_mean}, target_std={self.target_std}",
            flush=True,
        )
        self.task_type = "regression"

    def _normalize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def _prediction_to_age(self, pred):
        if not self.target_normalize:
            return pred
        return pred * self.target_std + self.target_mean

    def _prepare_loss_inputs(self, pred, target):
        if not self.target_normalize:
            return pred, target
        return pred, self._normalize_target(target)

    def _prepare_metric_inputs(self, pred, target):
        return self._prediction_to_age(pred), target

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/MSE": MeanSquaredError(num_outputs=self.num_classes),
            },
        )

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.num_classes),
                "MAE": MeanAbsoluteError(num_outputs=self.num_classes),
            },
        )

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = self._prediction_to_age(outputs).squeeze().float()
        label = batch["CLSREG_label"].squeeze().float()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
        }
        self.predictions.append(prediction)
        self.labels.append(label)
