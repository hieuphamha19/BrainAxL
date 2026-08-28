import gc
import pytest
import torch
from asparagus.modules.networks.brainaxl import AxialBiLSTM3D, brainaxl_b
from asparagus.modules.networks.dolphins_xlstm_unet import (
    BidirectionalAxialLSTM3D,
    dolphins_xlstm_unet_b,
)


def test_axial_block_preserves_shape_and_gradient() -> None:
    block = AxialBiLSTM3D(channels=4)
    image = torch.randn(2, 4, 3, 4, 5, requires_grad=True)

    output = block(image)

    assert output.shape == image.shape
    output.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_axial_block_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError, match="even"):
        AxialBiLSTM3D(channels=3)


def test_axial_block_matches_checkpoint_era_implementation() -> None:
    torch.manual_seed(7)
    legacy = BidirectionalAxialLSTM3D(channels=4).eval()
    public = AxialBiLSTM3D(channels=4).eval()
    public.load_state_dict(legacy.state_dict(), strict=True)
    image = torch.randn(2, 4, 3, 4, 5)

    torch.testing.assert_close(public(image), legacy(image), rtol=0.0, atol=0.0)


def test_public_name_is_checkpoint_compatible() -> None:
    legacy = dolphins_xlstm_unet_b(starting_filters=2)
    legacy_shapes = {name: value.shape for name, value in legacy.state_dict().items()}
    del legacy
    gc.collect()

    public = brainaxl_b(starting_filters=2)
    public_shapes = {name: value.shape for name, value in public.state_dict().items()}

    assert public_shapes == legacy_shapes


def test_reported_parameter_counts() -> None:
    model = brainaxl_b()
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in model.decoder.parameters())

    assert encoder_parameters == 31_837_640
    assert decoder_parameters == 13_196_241
    assert model.semantic_feature_dim == 2_480
