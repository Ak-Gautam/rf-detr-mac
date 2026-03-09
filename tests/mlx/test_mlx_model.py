# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Apple-Silicon smoke tests for the MLX RF-DETR model tree."""

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("MLX smoke tests run only on macOS.", allow_module_level=True)

pytest.importorskip("mlx.core")
tree_flatten = pytest.importorskip("mlx.utils").tree_flatten


class TestMLXModelTree:
    """Ensure the MLX parameter tree stays checkpoint-compatible."""

    def test_large_model_exposes_checkpoint_compatible_parameter_names(self) -> None:
        """Large detection exposes the expected flattened parameter names."""
        from rfdetr.mlx.config import RFDETRLargeMLXConfig
        from rfdetr.mlx.model import RFDETRForInference

        model = RFDETRForInference(RFDETRLargeMLXConfig)
        names = {name for name, _ in tree_flatten(model.parameters())}
        assert "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight" in names
        assert "transformer.decoder.layers.0.self_attn.in_proj_weight" in names
        assert "transformer.enc_out_bbox_embed.0.layers.2.bias" in names
        assert "class_embed.weight" in names

    def test_seg_large_model_exposes_segmentation_head_parameters(self) -> None:
        """Large segmentation exposes the expected mask-head parameter names."""
        from rfdetr.mlx.config import RFDETRSegLargeMLXConfig
        from rfdetr.mlx.model import RFDETRForInference

        model = RFDETRForInference(RFDETRSegLargeMLXConfig)
        names = {name for name, _ in tree_flatten(model.parameters())}
        assert "segmentation_head.blocks.0.dwconv.weight" in names
        assert "segmentation_head.query_features_block.layers.2.weight" in names
        assert "segmentation_head.query_features_proj.weight" in names
