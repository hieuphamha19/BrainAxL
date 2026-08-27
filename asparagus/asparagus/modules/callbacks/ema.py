"""Opt-in exponential moving-average checkpoint callback."""
from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


class ExponentialMovingAverageCheckpoint(Callback):
    def __init__(self, decay: float = 0.999, start_epoch: int = 10, filename: str = "ema.ckpt"):
        super().__init__()
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.filename = str(filename)
        self.shadow: dict[str, torch.Tensor] = {}
        self.num_updates = 0
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA_DECAY must be in (0, 1)")
        if self.start_epoch < 0:
            raise ValueError("EMA_START_EPOCH must be non-negative")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.current_epoch < self.start_epoch:
            return
        parameters = dict(pl_module.named_parameters())
        if not self.shadow:
            self.shadow = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
                if parameter.dtype.is_floating_point
            }
            self.num_updates = 1
            return
        one_minus_decay = 1.0 - self.decay
        for name, average in self.shadow.items():
            average.mul_(self.decay).add_(parameters[name].detach(), alpha=one_minus_decay)
        self.num_updates += 1

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module) -> None:
        if not self.shadow or not trainer.is_global_zero:
            return
        parameters = dict(pl_module.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        for name, average in self.shadow.items():
            parameters[name].copy_(average)
        checkpoint_callbacks = getattr(trainer, "checkpoint_callbacks", ())
        if not checkpoint_callbacks:
            raise RuntimeError("EMA checkpoint requires at least one ModelCheckpoint callback")
        directory = Path(checkpoint_callbacks[0].dirpath)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / self.filename
        trainer.save_checkpoint(str(destination), weights_only=False)
        for name, value in backup.items():
            parameters[name].copy_(value)
        print(
            f"[ema] saved {destination} decay={self.decay} "
            f"start_epoch={self.start_epoch} updates={self.num_updates}",
            flush=True,
        )
