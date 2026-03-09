"""Tests for MLX-specific configuration and conversion helpers."""

import torch

from rfdetr.mlx.config import (
    RFDETRLargeMLXConfig,
    RFDETRMediumMLXConfig,
    RFDETRSegLargeMLXConfig,
    RFDETRSegMediumMLXConfig,
)
from rfdetr.mlx.convert import _reshape_weight_for_mlx


class TestMLXConfig:
    """Validate static MLX model metadata."""

    def test_large_config_matches_checkpoint_layout(self) -> None:
        """Large detection uses 300 live queries and 300 stored query embeddings."""
        assert RFDETRLargeMLXConfig.num_queries == 300
        assert RFDETRLargeMLXConfig.num_query_embeddings == 300
        assert RFDETRLargeMLXConfig.group_detr == 13

    def test_medium_config_matches_checkpoint_layout(self) -> None:
        """Medium detection keeps the same query layout with a smaller resolution."""
        assert RFDETRMediumMLXConfig.resolution == 576
        assert RFDETRMediumMLXConfig.num_queries == 300
        assert RFDETRMediumMLXConfig.num_query_embeddings == 300

    def test_seg_large_config_distinguishes_live_and_stored_queries(self) -> None:
        """Segmentation uses 200 live queries but keeps 300 stored query embeddings."""
        assert RFDETRSegLargeMLXConfig.num_queries == 200
        assert RFDETRSegLargeMLXConfig.num_query_embeddings == 300
        assert RFDETRSegLargeMLXConfig.num_select == 200

    def test_seg_medium_config_distinguishes_live_and_stored_queries(self) -> None:
        """Medium segmentation matches the retained mask-query layout."""
        assert RFDETRSegMediumMLXConfig.resolution == 432
        assert RFDETRSegMediumMLXConfig.num_queries == 200
        assert RFDETRSegMediumMLXConfig.num_query_embeddings == 200


class TestConvertLayouts:
    """Validate PyTorch-to-MLX tensor layout conversion rules."""

    def test_conv2d_weights_convert_to_ohwi(self) -> None:
        """Standard 2D convolution weights transpose from OIHW to OHWI."""
        tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        converted = _reshape_weight_for_mlx("backbone.0.projector.stages.0.0.cv1.conv.weight", tensor)
        assert converted.shape == (2, 4, 5, 3)
        assert torch.equal(converted, tensor.permute(0, 2, 3, 1))

    def test_conv_transpose_weights_convert_to_ohwi(self) -> None:
        """Projector upsampling weights transpose from IOHW to OHWI."""
        tensor = torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5)
        converted = _reshape_weight_for_mlx("backbone.0.projector.stages_sampling.0.0.weight", tensor)
        assert converted.shape == (2, 4, 5, 3)
        assert torch.equal(converted, tensor.permute(1, 2, 3, 0))
