"""BrainAxL network definitions.

BrainAxL combines a five-stage 3D U-Net with bidirectional LSTMs that scan
features along each spatial axis at the two deepest encoder stages.  This file
contains the architecture used for both self-supervised pretraining and
downstream fine-tuning; experiment-specific variants live outside this module.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.blocks import (
    MultiLayerConvDropoutNormNonlin,
)
from gardening_tools.modules.networks.components.decoders import UNetDecoder


class AxialBiLSTM3D(nn.Module):
    """Apply independent bidirectional LSTMs along H, W, and D."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError(f"channels must be even, got {channels}")

        hidden_channels = channels // 2
        lstm_kwargs = {
            "input_size": channels,
            "hidden_size": hidden_channels,
            "batch_first": True,
            "bidirectional": True,
        }
        self.lstm_h = nn.LSTM(**lstm_kwargs)
        self.lstm_w = nn.LSTM(**lstm_kwargs)
        self.lstm_d = nn.LSTM(**lstm_kwargs)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _scan_axis(x: torch.Tensor, lstm: nn.LSTM, axis: int) -> torch.Tensor:
        """Move one spatial axis to sequence position and restore the layout."""
        if x.ndim != 5:
            raise ValueError(f"expected a 5D tensor [B,C,H,W,D], got {tuple(x.shape)}")

        # [B, C, H, W, D] -> [B, other_axis_1, other_axis_2, sequence, C]
        spatial_axis = axis + 2
        other_axes = [dim for dim in (2, 3, 4) if dim != spatial_axis]
        permutation = [0, *other_axes, spatial_axis, 1]
        sequence_view = x.permute(permutation).contiguous()
        leading_shape = sequence_view.shape[:-2]
        sequence_length, channels = sequence_view.shape[-2:]

        sequence = sequence_view.reshape(-1, sequence_length, channels)
        output, _ = lstm(sequence)
        output = output.reshape(*leading_shape, sequence_length, channels)

        # Invert the permutation used above.
        inverse = [permutation.index(dim) for dim in range(5)]
        return output.permute(inverse).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axial = self._scan_axis(x, self.lstm_h, axis=0)
        axial = axial + self._scan_axis(x, self.lstm_w, axis=1)
        axial = axial + self._scan_axis(x, self.lstm_d, axis=2)
        axial = axial / 3.0
        axial = self.norm(axial)
        return x + self.gate(axial) * axial


class BrainAxLEncoder(nn.Module):
    """Five-scale 3D convolutional encoder with deep axial BiLSTM blocks."""

    def __init__(
        self,
        input_channels: int,
        starting_filters: int = 40,
        xlstm_stages: Sequence[int] = (3, 4),
    ) -> None:
        super().__init__()
        if starting_filters <= 0 or starting_filters % 2:
            raise ValueError("starting_filters must be a positive even integer")

        self.filters = starting_filters
        self.stage_channels = tuple(starting_filters * (2**stage) for stage in range(5))
        selected_stages = frozenset(int(stage) for stage in xlstm_stages)
        invalid_stages = selected_stages.difference(range(5))
        if invalid_stages:
            raise ValueError(f"xlstm_stages must be in [0, 4], got {sorted(invalid_stages)}")

        block = MultiLayerConvDropoutNormNonlin.get_block_constructor(2)
        block_kwargs = {
            "conv_op": nn.Conv3d,
            "conv_kwargs": {"kernel_size": 3, "stride": 1, "bias": True},
            "norm_op": nn.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True, "momentum": 0.1},
            "dropout_op": nn.Dropout3d,
            "dropout_op_kwargs": {"p": 0.0, "inplace": True},
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"negative_slope": 1e-2, "inplace": True},
        }

        self.in_conv = block(
            input_channels=input_channels,
            output_channels=self.stage_channels[0],
            **block_kwargs,
        )
        self.pool1 = nn.MaxPool3d(2)
        self.encoder_conv1 = block(
            input_channels=self.stage_channels[0],
            output_channels=self.stage_channels[1],
            **block_kwargs,
        )
        self.pool2 = nn.MaxPool3d(2)
        self.encoder_conv2 = block(
            input_channels=self.stage_channels[1],
            output_channels=self.stage_channels[2],
            **block_kwargs,
        )
        self.pool3 = nn.MaxPool3d(2)
        self.encoder_conv3 = block(
            input_channels=self.stage_channels[2],
            output_channels=self.stage_channels[3],
            **block_kwargs,
        )
        self.pool4 = nn.MaxPool3d(2)
        self.encoder_conv4 = block(
            input_channels=self.stage_channels[3],
            output_channels=self.stage_channels[4],
            **block_kwargs,
        )

        for stage, channels in enumerate(self.stage_channels):
            module = AxialBiLSTM3D(channels) if stage in selected_stages else nn.Identity()
            # Keep the historical names so published checkpoints load directly.
            setattr(self, f"xlstm{stage}", module)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.xlstm0(self.in_conv(x))
        x1 = self.xlstm1(self.encoder_conv1(self.pool1(x0)))
        x2 = self.xlstm2(self.encoder_conv2(self.pool2(x1)))
        x3 = self.xlstm3(self.encoder_conv3(self.pool3(x2)))
        x4 = self.xlstm4(self.encoder_conv4(self.pool4(x3)))
        return [x0, x1, x2, x3, x4]


class BrainAxL(BaseNet):
    """BrainAxL encoder with a U-Net reconstruction/segmentation decoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        use_skip_connections: bool = True,
        deep_supervision: bool = False,
        xlstm_stages: Sequence[int] = (3, 4),
    ) -> None:
        super().__init__()
        if dimensions != "3D":
            raise ValueError("BrainAxL supports only 3D inputs")

        self.stem_weight_name = None
        self.num_classes = output_channels
        self.semantic_feature_dim = 2 * sum(starting_filters * (2**stage) for stage in range(5))
        self.encoder = BrainAxLEncoder(
            input_channels=input_channels,
            starting_filters=starting_filters,
            xlstm_stages=xlstm_stages,
        )
        self.decoder = UNetDecoder(
            output_channels=output_channels,
            starting_filters=starting_filters,
            conv_op=nn.Conv3d,
            dropout_op=nn.Dropout3d,
            norm_op=nn.InstanceNorm3d,
            upsample_op=nn.ConvTranspose3d,
            use_skip_connections=use_skip_connections,
            deep_supervision=deep_supervision,
        )

    @staticmethod
    def pool_multiscale(features: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feature in features:
            spatial_dims = tuple(range(2, feature.ndim))
            pooled.extend(
                [feature.mean(dim=spatial_dims), feature.amax(dim=spatial_dims)]
            )
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.decoder(features), features[-1]

    def forward_with_multiscale_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        features = self.encoder(x)
        return self.decoder(features), self.pool_multiscale(features), features


class SemanticHead(nn.Module):
    """Projection and prediction head for classification and regression."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        semantic_dim: int = 512,
        dropout_rate: float = 0.0,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.l2_normalize = l2_normalize
        self.projector = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(semantic_dim),
        )
        self.fc = nn.Linear(semantic_dim, output_channels)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(x)
        return F.normalize(embedding, dim=1) if self.l2_normalize else embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


class BrainAxLSemanticCLSREG(BaseNet):
    """BrainAxL encoder with a multiscale classification/regression head."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        xlstm_stages: Sequence[int] = (3, 4),
        semantic_dim: int = 512,
        dropout_op_kwargs: dict | None = None,
        late_fusion: bool = False,
        gated_late_fusion: bool = True,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        if dimensions != "3D":
            raise ValueError("BrainAxL supports only 3D inputs")

        dropout_op_kwargs = dropout_op_kwargs or {}
        self.stem_weight_name = None
        self.num_classes = output_channels
        self.late_fusion = late_fusion
        self.semantic_feature_dim = 2 * sum(starting_filters * (2**stage) for stage in range(5))
        self.encoder = BrainAxLEncoder(
            input_channels=1 if late_fusion else input_channels,
            starting_filters=starting_filters,
            xlstm_stages=xlstm_stages,
        )
        self.fusion_gate = (
            nn.Linear(self.semantic_feature_dim, 1)
            if late_fusion and gated_late_fusion
            else None
        )
        self.decoder = SemanticHead(
            input_channels=self.semantic_feature_dim,
            output_channels=output_channels,
            semantic_dim=semantic_dim,
            dropout_rate=float(dropout_op_kwargs.get("decoder_dropout_rate", 0.0)),
            l2_normalize=l2_normalize,
        )

    def _semantic_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.late_fusion:
            return BrainAxL.pool_multiscale(self.encoder(x))

        batch_size, modalities = x.shape[:2]
        features = self.encoder(x.reshape(batch_size * modalities, 1, *x.shape[2:]))
        pooled = BrainAxL.pool_multiscale(features).reshape(batch_size, modalities, -1)
        if self.fusion_gate is None:
            return pooled.mean(dim=1)
        weights = torch.softmax(self.fusion_gate(pooled).squeeze(-1), dim=1)
        return (pooled * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self._semantic_features(x))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_features = self._semantic_features(x)
        embedding = self.decoder.embed(semantic_features)
        return self.decoder.fc(embedding), embedding

    def freeze_backbone(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()


def brainaxl_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    use_skip_connections: bool = True,
    deep_supervision: bool = False,
    xlstm_stages: Sequence[int] = (3, 4),
) -> BrainAxL:
    """Build the BrainAxL-B pretraining or segmentation network."""
    return BrainAxL(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        deep_supervision=deep_supervision,
        xlstm_stages=xlstm_stages,
    )


def brainaxl_b_semantic_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    xlstm_stages: Sequence[int] = (3, 4),
    semantic_dim: int = 512,
    dropout_op_kwargs: dict | None = None,
    late_fusion: bool = False,
    gated_late_fusion: bool = True,
    l2_normalize: bool = True,
) -> BrainAxLSemanticCLSREG:
    """Build the BrainAxL-B classification/regression network."""
    return BrainAxLSemanticCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        xlstm_stages=xlstm_stages,
        semantic_dim=semantic_dim,
        dropout_op_kwargs=dropout_op_kwargs,
        late_fusion=late_fusion,
        gated_late_fusion=gated_late_fusion,
        l2_normalize=l2_normalize,
    )
