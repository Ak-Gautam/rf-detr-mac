"""Inference-only MLX model configs for the Mac-only RF-DETR fork."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MLXModelConfig:
    """Static configuration required to run an MLX inference graph."""

    name: str
    num_classes: int
    resolution: int
    patch_size: int
    hidden_dim: int
    backbone_dim: int
    backbone_heads: int
    backbone_layers: int
    backbone_mlp_ratio: int
    dec_layers: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    num_windows: int
    num_queries: int
    num_query_embeddings: int
    num_select: int
    positional_encoding_size: int
    projector_scales: tuple[str, ...]
    out_feature_indexes: tuple[int, ...]
    group_detr: int = 13
    num_register_tokens: int = 0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    bbox_reparam: bool = True
    two_stage: bool = True
    lite_refpoint_refine: bool = True
    license: str = "Apache-2.0"
    checkpoint_name: str = ""


RFDETRLargeMLXConfig = MLXModelConfig(
    name="rfdetr-large-mlx",
    num_classes=90,
    resolution=704,
    patch_size=16,
    hidden_dim=256,
    backbone_dim=384,
    backbone_heads=6,
    backbone_layers=12,
    backbone_mlp_ratio=4,
    dec_layers=4,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_windows=2,
    num_queries=300,
    num_query_embeddings=300,
    num_select=300,
    positional_encoding_size=704 // 16,
    projector_scales=("P4",),
    out_feature_indexes=(3, 6, 9, 12),
    checkpoint_name="rf-detr-large-2026.pth",
)


RFDETRMediumMLXConfig = MLXModelConfig(
    name="rfdetr-medium-mlx",
    num_classes=90,
    resolution=576,
    patch_size=16,
    hidden_dim=256,
    backbone_dim=384,
    backbone_heads=6,
    backbone_layers=12,
    backbone_mlp_ratio=4,
    dec_layers=4,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_windows=2,
    num_queries=300,
    num_query_embeddings=300,
    num_select=300,
    positional_encoding_size=576 // 16,
    projector_scales=("P4",),
    out_feature_indexes=(3, 6, 9, 12),
    checkpoint_name="rf-detr-medium.pth",
)


RFDETRSegLargeMLXConfig = MLXModelConfig(
    name="rfdetr-seg-large-mlx",
    num_classes=90,
    resolution=504,
    patch_size=12,
    hidden_dim=256,
    backbone_dim=384,
    backbone_heads=6,
    backbone_layers=12,
    backbone_mlp_ratio=4,
    dec_layers=5,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_windows=2,
    num_queries=200,
    num_query_embeddings=300,
    num_select=200,
    positional_encoding_size=504 // 12,
    projector_scales=("P4",),
    out_feature_indexes=(3, 6, 9, 12),
    segmentation_head=True,
    checkpoint_name="rf-detr-seg-large.pt",
)


RFDETRSegMediumMLXConfig = MLXModelConfig(
    name="rfdetr-seg-medium-mlx",
    num_classes=90,
    resolution=432,
    patch_size=12,
    hidden_dim=256,
    backbone_dim=384,
    backbone_heads=6,
    backbone_layers=12,
    backbone_mlp_ratio=4,
    dec_layers=5,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_windows=2,
    num_queries=200,
    num_query_embeddings=200,
    num_select=200,
    positional_encoding_size=432 // 12,
    projector_scales=("P4",),
    out_feature_indexes=(3, 6, 9, 12),
    segmentation_head=True,
    checkpoint_name="rf-detr-seg-medium.pt",
)
