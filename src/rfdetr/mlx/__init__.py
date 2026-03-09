"""MLX inference port foundation for RF-DETR.

This package is intentionally inference-first for local Apple Silicon use.
"""

from rfdetr.mlx.config import (
    RFDETRLargeMLXConfig,
    RFDETRMediumMLXConfig,
    RFDETRSegLargeMLXConfig,
    RFDETRSegMediumMLXConfig,
)
from rfdetr.mlx.convert import convert_checkpoint, convert_pretrained_checkpoint
from rfdetr.mlx.model import (
    RFDETRMLX,
    RFDETRLargeMLX,
    RFDETRMediumMLX,
    RFDETRSegLargeMLX,
    RFDETRSegMediumMLX,
)

__all__ = [
    "RFDETRLargeMLX",
    "RFDETRLargeMLXConfig",
    "RFDETRMediumMLX",
    "RFDETRMediumMLXConfig",
    "RFDETRMLX",
    "RFDETRSegLargeMLX",
    "RFDETRSegLargeMLXConfig",
    "RFDETRSegMediumMLX",
    "RFDETRSegMediumMLXConfig",
    "convert_checkpoint",
    "convert_pretrained_checkpoint",
]
