# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MLX inference port foundation for RF-DETR.

This package is intentionally inference-first. The initial port target is local
Apple Silicon inference for ``RFDETRLarge`` and ``RFDETRSegLarge``.
"""

from rfdetr.mlx.config import RFDETRLargeMLXConfig, RFDETRSegLargeMLXConfig
from rfdetr.mlx.convert import convert_checkpoint, convert_pretrained_checkpoint
from rfdetr.mlx.model import RFDETRMLX, RFDETRLargeMLX, RFDETRSegLargeMLX

__all__ = [
    "RFDETRLargeMLX",
    "RFDETRLargeMLXConfig",
    "RFDETRMLX",
    "RFDETRSegLargeMLX",
    "RFDETRSegLargeMLXConfig",
    "convert_checkpoint",
    "convert_pretrained_checkpoint",
]
