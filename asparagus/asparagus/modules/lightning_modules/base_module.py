import copy
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from asparagus.functional.lr_scheduling import (
    cosine_decay_schedule,
    sawtooth_warmup_cosine_decay_schedule,
    separate_encoder_decoder_weights,
    simple_warmup_cosine_decay_schedule,
)
from asparagus.functional.pos_embed import resize_pos_embed_3d
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from torch.optim import SGD, AdamW
from torchvision import transforms
from typing import Optional


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        warmup_epochs: int = None,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        load_decoder: bool = True,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        repeat_stem_weights: bool = True,
        freeze_backbone: bool = False,
        encoder_learning_rate: Optional[float] = None,
        decoder_learning_rate: Optional[float] = None,
        pretrained_target_size: Optional[tuple] = None,
        target_size: Optional[tuple] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.pretrained_target_size = pretrained_target_size
        self.target_size = target_size

        self.loss = None
        self.train_metrics = None
        self.val_metrics = None
        self.warmup_epochs = warmup_epochs
        self.decoder_warmup_epochs = decoder_warmup_epochs
        self.ignore_index_in_metrics = 0
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.repeat_stem_weights = repeat_stem_weights
        self.freeze_backbone = freeze_backbone
        self.encoder_learning_rate = encoder_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        assert 0 < cosine_period_ratio <= 1

        self.save_hyperparameters(ignore=["model", "weights", "train_transforms", "val_transforms", "test_transforms"])
        self.model = model

        if weights is not None:
            self.load_state_dict(weights, load_decoder=load_decoder, strict=False)

        if self.freeze_backbone:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
            elif hasattr(self.model, "encoder"):
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                self.model.encoder.eval()
            else:
                raise ValueError("freeze_backbone=True but model has no freeze_backbone() or encoder.")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print(f"Frozen backbone. Trainable params: {trainable_params}; frozen params: {frozen_params}")

        self.model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # Separate encoder and decoder parameters for warmup schedules or layer-wise learning rates.
        use_layerwise_lr = self.encoder_learning_rate is not None or self.decoder_learning_rate is not None
        if self.decoder_warmup_epochs > 0 or use_layerwise_lr:
            param_groups = separate_encoder_decoder_weights(self.named_parameters())
            param_groups[0]["lr"] = self.encoder_learning_rate or self.learning_rate
            param_groups[1]["lr"] = self.decoder_learning_rate or self.learning_rate
        else:
            param_groups = self.parameters()

        if self.optimizer == "SGD":
            optimizer = SGD(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            optimizer = AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if use_layerwise_lr:
            print(
                f"Using optimizer {optimizer.__class__.__name__} with encoder lr "
                f"{self.encoder_learning_rate or self.learning_rate} and decoder lr "
                f"{self.decoder_learning_rate or self.learning_rate}"
            )
        else:
            print(f"Using optimizer {optimizer.__class__.__name__} with learning rate {self.learning_rate}")

        # Calculate steps per epoch based on trainer configuration
        # if max_epochs is *not* set (i.e., set to -1), we are probably using max_steps
        # if max_epochs is set, we can calculate steps per epoch based on estimated_stepping_batches
        if self.trainer.max_epochs <= 0:
            optimizer_steps_per_epoch = self.trainer.limit_train_batches // self.trainer.accumulate_grad_batches
        else:
            optimizer_steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # Scheduler option 1: Three-phase schedule with separate decoder/joint warmup
        if self.decoder_warmup_epochs > 0:
            scheduler = sawtooth_warmup_cosine_decay_schedule(
                optimizer,
                self.decoder_warmup_epochs,
                self.warmup_epochs,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
            )
        # Scheduler option 2: Two-phase schedule with joint warmup
        elif self.warmup_epochs > 0:
            scheduler = simple_warmup_cosine_decay_schedule(
                optimizer,
                self.warmup_epochs,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
            )
        # Scheduler option 3: Just cosine annealing
        else:
            scheduler = cosine_decay_schedule(
                optimizer,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
            )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def load_state_dict(self, state_dict, load_decoder=True, *args, **kwargs):
        old_params = copy.deepcopy(self.state_dict())

        target_compiled = "_orig" in next(iter(old_params.keys()))
        source_compiled = "_orig" in next(iter(state_dict.keys()))

        print(f"Target compiled: {target_compiled}, source compiled: {source_compiled}")

        if not target_compiled and source_compiled:
            print("Source state_dict is compiled, but target model is not. Removing _orig suffix from state_dict keys.")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        semantic_projector_key_map = {
            "semantic_projector.0.weight": "model.decoder.projector.0.weight",
            "semantic_projector.0.bias": "model.decoder.projector.0.bias",
            "semantic_projector.1.weight": "model.decoder.projector.1.weight",
            "semantic_projector.1.bias": "model.decoder.projector.1.bias",
            "semantic_projector.4.weight": "model.decoder.projector.4.weight",
            "semantic_projector.4.bias": "model.decoder.projector.4.bias",
        }
        remapped_semantic_projector_keys = []
        remapped_semantic_projector_targets = set()
        for source_key, target_key in semantic_projector_key_map.items():
            if source_key in state_dict and target_key in old_params and state_dict[source_key].shape == old_params[target_key].shape:
                state_dict[target_key] = state_dict[source_key]
                remapped_semantic_projector_keys.append(f"{source_key}->{target_key}")
                remapped_semantic_projector_targets.add(target_key)
        if remapped_semantic_projector_keys:
            print(f"Remapped semantic projector keys: {remapped_semantic_projector_keys}")

        def adapt_input_channels(weight, target_channels):
            source_channels = weight.shape[1]
            if source_channels == target_channels:
                return weight
            if source_channels == 1:
                return weight.repeat(1, target_channels, *([1] * (weight.ndim - 2))) / target_channels
            if source_channels > target_channels:
                return weight.mean(dim=1, keepdim=True).repeat(
                    1, target_channels, *([1] * (weight.ndim - 2))
                )

            repeats = (target_channels + source_channels - 1) // source_channels
            repeated = weight.repeat(1, repeats, *([1] * (weight.ndim - 2)))
            return repeated[:, :target_channels] * (source_channels / target_channels)

        # Adapt stem weights when state_dict num_channels differs from new_state_dict num_channels.
        if hasattr(self.model, "stem_weight_name") and self.model.stem_weight_name is not None and self.repeat_stem_weights:
            prefix = "model._orig_mod." if "_orig_mod" in list(state_dict.keys())[0] else "model."
            stem_name = f"{prefix}{self.model.stem_weight_name}"
            if stem_name in state_dict and stem_name in old_params:
                pt_input_channels = state_dict[stem_name].shape[1]
                ft_input_channels = old_params[stem_name].shape[1]
                if (
                    state_dict[stem_name].ndim >= 3
                    and state_dict[stem_name].shape[0] == old_params[stem_name].shape[0]
                    and state_dict[stem_name].shape[2:] == old_params[stem_name].shape[2:]
                    and pt_input_channels != ft_input_channels
                ):
                    print(f"Adapting stem weights from {pt_input_channels} to {ft_input_channels} channels for {stem_name}.")
                    state_dict[stem_name] = adapt_input_channels(state_dict[stem_name], ft_input_channels)


        for key in list(state_dict.keys()):
            if key not in old_params:
                continue
            if not load_decoder and key.startswith("model.decoder"):
                continue
            if (
                state_dict[key].ndim >= 3
                and old_params[key].ndim == state_dict[key].ndim
                and state_dict[key].shape[0] == old_params[key].shape[0]
                and state_dict[key].shape[2:] == old_params[key].shape[2:]
                and state_dict[key].shape[1] != old_params[key].shape[1]
            ):
                print(
                    f"Adapting input-channel weights from {state_dict[key].shape[1]} "
                    f"to {old_params[key].shape[1]} channels for {key}."
                )
                state_dict[key] = adapt_input_channels(state_dict[key], old_params[key].shape[1])

        # Interpolate positional embeddings when spatial dimensions differ
        if self.pretrained_target_size is not None and self.target_size is not None:
            for key in list(state_dict.keys()):
                if key not in old_params or old_params[key].shape == state_dict[key].shape:
                    continue
                if key.endswith("pos_embed"):
                    num_prefix_tokens = getattr(self.model.eva, "num_prefix_tokens", 0)
                    patch_embed_size = tuple(self.model.encoder.proj.weight.shape[2:])
                    print(f"Interpolating {key}: {state_dict[key].shape} -> {old_params[key].shape}")
                    state_dict[key] = resize_pos_embed_3d(
                        state_dict[key],
                        old_params[key],
                        num_prefix_tokens=num_prefix_tokens,
                        pretrained_target_size=self.pretrained_target_size,
                        target_size=self.target_size,
                        patch_embed_size=patch_embed_size,
                    )

        # Filter out keys that are not in the old state dict or have different shapes
        def should_load_key(key, state_dict, old_params, load_decoder):
            # reject all decoder keys regardless of their shape
            if (
                not load_decoder
                and key.startswith("model.decoder")
                and key not in remapped_semantic_projector_targets
            ):
                return False
            # accept all keys that are in the old state dict and have the same shape
            return (key in old_params) and (old_params[key].shape == state_dict[key].shape)

        source_state_dict = state_dict
        rejected_keys_new = [key for key in source_state_dict if key not in old_params]
        rejected_keys_shape = [
            key for key in source_state_dict
            if key in old_params and old_params[key].shape != source_state_dict[key].shape
        ]
        rejected_keys_decoder = [
            key for key in source_state_dict
            if (
                not load_decoder
                and key.startswith("model.decoder")
                and key not in remapped_semantic_projector_targets
            )
        ]
        state_dict = {
            key: value for key, value in source_state_dict.items()
            if should_load_key(key, source_state_dict, old_params, load_decoder)
        }

        # Load the state dict
        kwargs["strict"] = False
        super().load_state_dict(state_dict, *args, **kwargs)

        new_params = self.state_dict()
        mismatched_after_load = [
            key for key, value in state_dict.items()
            if not torch.equal(new_params[key].detach().cpu(), value.detach().cpu())
        ]
        successful = len(state_dict) - len(mismatched_after_load)
        print(f"Successfully transferred weights for {successful}/{len(state_dict)} selected tensors")
        print(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post-load mismatches: {mismatched_after_load}."
        )
        if not load_decoder:
            print("Decoder weights were not loaded, as requested. If you want to load them, set `load_decoder=True`.")
            print(f"Rejected decoder keys: {rejected_keys_decoder}.")
        else:
            print("Warning! Also loaded the decoder. If you are finetuning, this might not be what you want.")

        assert successful > 0, "No weights were loaded. Check the state_dict and the model architecture."
        assert not mismatched_after_load, f"Checkpoint tensors failed post-load verification: {mismatched_after_load}"

    def _log_dict_of_images_to_wandb(self, imagedict: dict, log_key: str, task_type: str = ""):
        """
        Log a random image from the imagedict to wandb
        """
        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        image, output, target = get_logger_compatible_image_output_target(
            image=imagedict["input"][batch_idx],
            output=imagedict["output"][batch_idx],
            target=imagedict["target"][batch_idx],
            task_type=task_type,
        )
        for logger in self.trainer.loggers:
            if "WandbLogger" in logger.__class__.__name__:
                log_image_output_target_to_wandb(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )
            if "MLFlowLogger" in logger.__class__.__name__:
                log_image_output_target_to_mlflow(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        if (self.trainer.validating or self.trainer.sanity_checking) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        if (self.trainer.testing or self.trainer.predicting) and self.test_transforms is not None:
            batch = self.test_transforms(batch)
        return super().on_after_batch_transfer(batch, dataloader_idx)
