import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.blocks import MultiLayerConvDropoutNormNonlin
from gardening_tools.modules.networks.components.decoders import UNetDecoder
from gardening_tools.modules.networks.components.heads import ClsRegHead


class BidirectionalAxialLSTM3D(nn.Module):
    """Bidirectional LSTM block over each spatial axis for 3D feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = channels // 2
        if hidden_channels * 2 != channels:
            raise ValueError("BidirectionalAxialLSTM3D requires an even channel count.")

        self.lstm_h = nn.LSTM(channels, hidden_channels, batch_first=True, bidirectional=True)
        self.lstm_w = nn.LSTM(channels, hidden_channels, batch_first=True, bidirectional=True)
        self.lstm_d = nn.LSTM(channels, hidden_channels, batch_first=True, bidirectional=True)
        self.norm = nn.GroupNorm(1, channels)
        self.gate = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def _run_h(x: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        b, c, h, w, d = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * w * d, h, c)
        out, _ = lstm(seq)
        return out.reshape(b, w, d, h, c).permute(0, 4, 3, 1, 2).contiguous()

    @staticmethod
    def _run_w(x: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        b, c, h, w, d = x.shape
        seq = x.permute(0, 2, 4, 3, 1).reshape(b * h * d, w, c)
        out, _ = lstm(seq)
        return out.reshape(b, h, d, w, c).permute(0, 4, 1, 3, 2).contiguous()

    @staticmethod
    def _run_d(x: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        b, c, h, w, d = x.shape
        seq = x.permute(0, 2, 3, 4, 1).reshape(b * h * w, d, c)
        out, _ = lstm(seq)
        return out.reshape(b, h, w, d, c).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (self._run_h(x, self.lstm_h) + self._run_w(x, self.lstm_w) + self._run_d(x, self.lstm_d)) / 3.0
        y = self.norm(y)
        return x + self.gate(y) * y


class DolphinsXLSTMUNetEncoder(nn.Module):
    """3D U-Net encoder with bidirectional axial LSTM blocks at deep stages."""

    def __init__(
        self,
        input_channels: int,
        basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        conv_op=nn.Conv3d,
        conv_kwargs={"kernel_size": 3, "stride": 1, "bias": True},
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        pool_op=nn.MaxPool3d,
        starting_filters: int = 40,
        xlstm_stages=(3, 4),
    ) -> None:
        super().__init__()
        self.filters = starting_filters
        self.xlstm_stages = set(xlstm_stages)

        self.in_conv = basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.pool1 = pool_op(2)
        self.encoder_conv1 = basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.pool2 = pool_op(2)
        self.encoder_conv2 = basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.pool3 = pool_op(2)
        self.encoder_conv3 = basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.pool4 = pool_op(2)
        self.encoder_conv4 = basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.xlstm0 = BidirectionalAxialLSTM3D(self.filters) if 0 in self.xlstm_stages else nn.Identity()
        self.xlstm1 = BidirectionalAxialLSTM3D(self.filters * 2) if 1 in self.xlstm_stages else nn.Identity()
        self.xlstm2 = BidirectionalAxialLSTM3D(self.filters * 4) if 2 in self.xlstm_stages else nn.Identity()
        self.xlstm3 = BidirectionalAxialLSTM3D(self.filters * 8) if 3 in self.xlstm_stages else nn.Identity()
        self.xlstm4 = BidirectionalAxialLSTM3D(self.filters * 16) if 4 in self.xlstm_stages else nn.Identity()

    def forward(self, x):
        x0 = self.xlstm0(self.in_conv(x))

        x1 = self.pool1(x0)
        x1 = self.xlstm1(self.encoder_conv1(x1))

        x2 = self.pool2(x1)
        x2 = self.xlstm2(self.encoder_conv2(x2))

        x3 = self.pool3(x2)
        x3 = self.xlstm3(self.encoder_conv3(x3))

        x4 = self.pool4(x3)
        x4 = self.xlstm4(self.encoder_conv4(x4))

        return [x0, x1, x2, x3, x4]


class DolphinsXLSTMUNet(BaseNet):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        use_skip_connections: bool = True,
        deep_supervision: bool = False,
        xlstm_stages=(3, 4),
    ):
        super().__init__()
        self.stem_weight_name = None

        if dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            upsample_op = torch.nn.ConvTranspose3d
        else:
            logging.warning("DolphinsXLSTMUNet currently supports only 3D inputs.")
            raise ValueError("DolphinsXLSTMUNet currently supports only 3D inputs.")

        self.encoder = DolphinsXLSTMUNetEncoder(
            input_channels=input_channels,
            conv_op=conv_op,
            dropout_op=dropout_op,
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
            xlstm_stages=xlstm_stages,
        )
        self.decoder = UNetDecoder(
            output_channels=output_channels,
            starting_filters=starting_filters,
            conv_op=conv_op,
            dropout_op=dropout_op,
            norm_op=norm_op,
            upsample_op=upsample_op,
            use_skip_connections=use_skip_connections,
            deep_supervision=deep_supervision,
        )
        self.num_classes = output_channels

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)

    def forward_with_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, skips[-1]

    @staticmethod
    def _pool_skips(skips):
        pooled = []
        for skip in skips:
            spatial_dims = tuple(range(2, skip.ndim))
            pooled.append(skip.mean(dim=spatial_dims))
            pooled.append(skip.amax(dim=spatial_dims))
        return torch.cat(pooled, dim=1)


class DolphinsXLSTMUNetAnisotropic(DolphinsXLSTMUNet):
    """xLSTM U-Net that preserves shallow lesions along the final spatial axis.

    Task 2 lesions frequently occupy only a few through-plane slices.  The
    default architecture pools that axis four times (32 -> 2 for a 32-slice
    patch), so this variant keeps it intact in the two shallow encoder stages:
    32 -> 32 -> 32 -> 16 -> 8.  The decoder mirrors those strides exactly.

    Encoder weights, decoder convolutions, and the two deep upsamplers retain
    their pretrained shapes.  The two stride-(2, 2, 1) upsamplers intentionally
    have new shapes and are safely skipped by the existing non-strict loader.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        use_skip_connections: bool = True,
        deep_supervision: bool = False,
        xlstm_stages=(3, 4),
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            dimensions=dimensions,
            starting_filters=starting_filters,
            use_skip_connections=use_skip_connections,
            deep_supervision=deep_supervision,
            xlstm_stages=xlstm_stages,
        )
        if dimensions != "3D":
            raise ValueError("DolphinsXLSTMUNetAnisotropic supports only 3D inputs.")

        self.encoder.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.encoder.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        filters = starting_filters
        # Keep the deep pretrained transpose-convolutions shape-compatible.
        self.decoder.upsample1 = nn.ConvTranspose3d(
            filters * 16, filters * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder.upsample2 = nn.ConvTranspose3d(
            filters * 8, filters * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder.upsample3 = nn.ConvTranspose3d(
            filters * 4, filters * 2, kernel_size=(2, 2, 1), stride=(2, 2, 1)
        )
        self.decoder.upsample4 = nn.ConvTranspose3d(
            filters * 2, filters, kernel_size=(2, 2, 1), stride=(2, 2, 1)
        )

    def forward_with_multiscale_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, self._pool_skips(skips), skips[-1]


class DolphinsXLSTMUNetCLSREG(BaseNet):
    """Classification/regression head on top of the Dolphins xLSTM encoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        xlstm_stages=(3, 4),
        decoder: nn.Module = ClsRegHead,
        dropout_op_kwargs: dict = None,
        late_fusion: bool = False,
    ):
        super().__init__()
        self.stem_weight_name = None
        self.num_classes = output_channels
        self.late_fusion = late_fusion

        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        encoder_dropout_rate = dropout_op_kwargs.get("encoder_dropout_rate", 0.0)
        decoder_dropout_rate = dropout_op_kwargs.get("decoder_dropout_rate", 0.0)
        inplace = dropout_op_kwargs.get("inplace", True)

        if dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            clsreg_pool_op = nn.AdaptiveAvgPool3d
        else:
            logging.warning("DolphinsXLSTMUNetCLSREG currently supports only 3D inputs.")
            raise ValueError("DolphinsXLSTMUNetCLSREG currently supports only 3D inputs.")

        self.encoder = DolphinsXLSTMUNetEncoder(
            input_channels=1 if late_fusion else input_channels,
            conv_op=conv_op,
            dropout_op=dropout_op,
            dropout_op_kwargs={"p": encoder_dropout_rate, "inplace": inplace},
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
            xlstm_stages=xlstm_stages,
        )
        self.decoder = decoder(
            pool_op=clsreg_pool_op,
            input_channels=starting_filters * 16 * input_channels if late_fusion else starting_filters * 16,
            output_channels=output_channels,
            dropout_rate=decoder_dropout_rate,
        )

    def _encode(self, x):
        if not self.late_fusion:
            return self.encoder(x)

        b, n = x.shape[:2]
        skips = self.encoder(x.view(b * n, -1, *x.shape[2:]))
        return [s.view(b, n * s.shape[1], *s.shape[2:]) for s in skips]

    def forward(self, x):
        skips = self._encode(x)
        return self.decoder(skips)

    def forward_with_features(self, x):
        skips = self._encode(x)
        output = self.decoder(skips)
        return output, skips[-1]

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()


class SemanticVectorHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        semantic_dim: int = 768,
        dropout_rate: float = 0.0,
        l2_normalize: bool = True,
    ):
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

    def embed(self, x):
        z = self.projector(x)
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, x):
        return self.fc(self.embed(x))


class DolphinsXLSTMUNetSemanticCLSREG(BaseNet):
    """Multi-scale semantic vector head on top of the Dolphins xLSTM encoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        xlstm_stages=(3, 4),
        semantic_dim: int = 768,
        dropout_op_kwargs: dict = None,
        late_fusion: bool = False,
        gated_late_fusion: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.stem_weight_name = None
        self.num_classes = output_channels
        self.late_fusion = late_fusion
        self.gated_late_fusion = gated_late_fusion
        self.stage_channels = [starting_filters * (2 ** i) for i in range(5)]
        self.semantic_input_channels = 2 * sum(self.stage_channels)

        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        encoder_dropout_rate = dropout_op_kwargs.get("encoder_dropout_rate", 0.0)
        decoder_dropout_rate = dropout_op_kwargs.get("decoder_dropout_rate", 0.0)
        inplace = dropout_op_kwargs.get("inplace", True)

        if dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
        else:
            logging.warning("DolphinsXLSTMUNetSemanticCLSREG currently supports only 3D inputs.")
            raise ValueError("DolphinsXLSTMUNetSemanticCLSREG currently supports only 3D inputs.")

        self.encoder = DolphinsXLSTMUNetEncoder(
            input_channels=1 if late_fusion else input_channels,
            conv_op=conv_op,
            dropout_op=dropout_op,
            dropout_op_kwargs={"p": encoder_dropout_rate, "inplace": inplace},
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
            xlstm_stages=xlstm_stages,
        )
        self.fusion_gate = nn.Linear(self.semantic_input_channels, 1) if late_fusion and gated_late_fusion else None
        self.decoder = SemanticVectorHead(
            input_channels=self.semantic_input_channels,
            output_channels=output_channels,
            semantic_dim=semantic_dim,
            dropout_rate=decoder_dropout_rate,
            l2_normalize=l2_normalize,
        )

    @staticmethod
    def _pool_skips(skips):
        pooled = []
        for skip in skips:
            spatial_dims = tuple(range(2, skip.ndim))
            pooled.append(skip.mean(dim=spatial_dims))
            pooled.append(skip.amax(dim=spatial_dims))
        return torch.cat(pooled, dim=1)

    def _semantic_features(self, x):
        if not self.late_fusion:
            return self._pool_skips(self.encoder(x))

        b, n = x.shape[:2]
        skips = self.encoder(x.reshape(b * n, -1, *x.shape[2:]))
        features = self._pool_skips(skips).reshape(b, n, -1)
        if self.fusion_gate is None:
            return features.mean(dim=1)
        weights = torch.softmax(self.fusion_gate(features).squeeze(-1), dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x):
        return self.decoder(self._semantic_features(x))

    def forward_with_features(self, x):
        features = self._semantic_features(x)
        embedding = self.decoder.embed(features)
        output = self.decoder.fc(embedding)
        return output, embedding

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()


class DolphinsXLSTMUNetHiRes(DolphinsXLSTMUNet):
    """Adds capacity at the ONLY resolution where a 2-voxel target exists.

    Task 4's targets are 1.04 mm thick on a 0.4883 mm grid -- 2.13 voxels -- so
    they are sub-voxel from encoder stage 2 onward, and both xLSTM blocks (at
    1/8 and 1/16, i.e. 3.91 mm and 7.81 mm) see nothing of them
    (`RESCORE_VERDICT_2026-08-15.md` §16). Everything that can actually
    discriminate this structure lives in `in_conv` (2 convs), `decoder_conv4`
    (2 convs) and `out_conv` (1x1): roughly four convolutional layers at full
    resolution inside a 45 M-parameter network.

    This variant adds a residual refinement branch at 1/1 that sees the decoder's
    full-resolution features AND the raw image, which carries detail no
    downsampled feature map retains.

    ZERO-INITIALISED, DELIBERATELY. `refine_out` starts at exactly zero, so at
    epoch 0 this network is bit-identical to `dolphins_xlstm_unet_b` and the
    pretrained encoder/decoder load unchanged. The branch can only be switched on
    by training, which makes the change strictly non-destructive: it cannot be
    worse at initialisation, and any regression is attributable to optimisation
    rather than to a broken graph. Same trick, same reason, as the
    zero-initialised gated residual in `train_task2_bilateral_symmetry_c2f.py`.

    The decoder path is reimplemented here rather than called, because
    `UNetDecoder.forward` returns logits and discards `x8` -- the full-resolution
    feature map this branch needs. The ordering of the deep-supervision list is
    kept identical to the parent so the loss weighting still lines up.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        starting_filters: int = 40,
        use_skip_connections: bool = True,
        deep_supervision: bool = False,
        xlstm_stages=(3, 4),
        refine_channels: int = 32,
        refine_blocks: int = 2,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            dimensions=dimensions,
            starting_filters=starting_filters,
            use_skip_connections=use_skip_connections,
            deep_supervision=deep_supervision,
            xlstm_stages=xlstm_stages,
        )
        channels = starting_filters + input_channels
        layers = []
        for _ in range(refine_blocks):
            layers += [
                nn.Conv3d(channels, refine_channels, kernel_size=3, padding=1, bias=True),
                nn.InstanceNorm3d(refine_channels, eps=1e-5, affine=True),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            ]
            channels = refine_channels
        self.refine = nn.Sequential(*layers)
        self.refine_out = nn.Conv3d(refine_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.refine_out.weight)
        nn.init.zeros_(self.refine_out.bias)

    def forward(self, x):
        skips = self.encoder(x)
        dec = self.decoder

        if dec.use_skip_connections:
            x5 = dec.decoder_conv1(torch.cat([dec.upsample1(skips[4]), skips[3]], dim=1))
            x6 = dec.decoder_conv2(torch.cat([dec.upsample2(x5), skips[2]], dim=1))
            x7 = dec.decoder_conv3(torch.cat([dec.upsample3(x6), skips[1]], dim=1))
            x8 = dec.decoder_conv4(torch.cat([dec.upsample4(x7), skips[0]], dim=1))
        else:
            x5 = dec.decoder_conv1(dec.upsample1(skips[4]))
            x6 = dec.decoder_conv2(dec.upsample2(x5))
            x7 = dec.decoder_conv3(dec.upsample3(x6))
            x8 = dec.decoder_conv4(dec.upsample4(x7))

        logits = dec.out_conv(x8) + self.refine_out(self.refine(torch.cat([x8, x], dim=1)))

        if dec.deep_supervision and torch.is_grad_enabled():
            return [
                logits,
                dec.ds_out_conv3(x7),
                dec.ds_out_conv2(x6),
                dec.ds_out_conv1(x5),
                dec.ds_out_conv0(skips[4]),
            ]
        return logits


def dolphins_xlstm_unet_b_hires(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    use_skip_connections: bool = True,
    deep_supervision: bool = False,
    xlstm_stages=(3, 4),
):
    return DolphinsXLSTMUNetHiRes(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        deep_supervision=deep_supervision,
        xlstm_stages=xlstm_stages,
    )


def dolphins_xlstm_unet_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    use_skip_connections: bool = True,
    deep_supervision: bool = False,
    xlstm_stages=(3, 4),
):
    return DolphinsXLSTMUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        deep_supervision=deep_supervision,
        xlstm_stages=xlstm_stages,
    )


def dolphins_xlstm_unet_b_aniso(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    use_skip_connections: bool = True,
    deep_supervision: bool = False,
    xlstm_stages=(3, 4),
):
    return DolphinsXLSTMUNetAnisotropic(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        deep_supervision=deep_supervision,
        xlstm_stages=xlstm_stages,
    )

def dolphins_xlstm_unet_b_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    xlstm_stages=(3, 4),
    dropout_op_kwargs: dict = None,
    late_fusion: bool = False,
):
    return DolphinsXLSTMUNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        xlstm_stages=xlstm_stages,
        dropout_op_kwargs=dropout_op_kwargs,
        late_fusion=late_fusion,
    )

def dolphins_xlstm_unet_b_semantic_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    xlstm_stages=(3, 4),
    semantic_dim: int = 768,
    dropout_op_kwargs: dict = None,
    late_fusion: bool = False,
    gated_late_fusion: bool = True,
    l2_normalize: bool = True,
):
    return DolphinsXLSTMUNetSemanticCLSREG(
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
