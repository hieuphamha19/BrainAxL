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


class S512AxialUNetEncoder(nn.Module):
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


class S512AxialUNet(BaseNet):
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
            logging.warning("S512AxialUNet currently supports only 3D inputs.")
            raise ValueError("S512AxialUNet currently supports only 3D inputs.")

        self.encoder = S512AxialUNetEncoder(
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

    def forward_with_multiscale_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, self._pool_skips(skips), skips[-1]


class S512AxialUNetCLSREG(BaseNet):
    """Classification/regression head on top of the S512 axial recurrent encoder."""

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
            logging.warning("S512AxialUNetCLSREG currently supports only 3D inputs.")
            raise ValueError("S512AxialUNetCLSREG currently supports only 3D inputs.")

        self.encoder = S512AxialUNetEncoder(
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


class S512AxialUNetSemanticCLSREG(BaseNet):
    """Multi-scale semantic vector head on top of the S512 axial recurrent encoder."""

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
            logging.warning("S512AxialUNetSemanticCLSREG currently supports only 3D inputs.")
            raise ValueError("S512AxialUNetSemanticCLSREG currently supports only 3D inputs.")

        self.encoder = S512AxialUNetEncoder(
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


def create_s512_backbone(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    use_skip_connections: bool = True,
    deep_supervision: bool = False,
    xlstm_stages=(3, 4),
):
    return S512AxialUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        deep_supervision=deep_supervision,
        xlstm_stages=xlstm_stages,
    )

def create_s512_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    starting_filters: int = 40,
    xlstm_stages=(3, 4),
    dropout_op_kwargs: dict = None,
    late_fusion: bool = False,
):
    return S512AxialUNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=starting_filters,
        xlstm_stages=xlstm_stages,
        dropout_op_kwargs=dropout_op_kwargs,
        late_fusion=late_fusion,
    )

def create_s512_semantic_clsreg(
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
    return S512AxialUNetSemanticCLSREG(
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
