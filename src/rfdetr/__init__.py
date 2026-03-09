import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetr.detr import (
    RFDETRLarge,
    RFDETRMedium,
    RFDETRSegLarge,
    RFDETRSegMedium,
)
from rfdetr.lit import RFDETRDataModule, RFDETRModule, build_trainer

__all__ = [
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRMediumMLX",
    "RFDETRLargeMLX",
    "RFDETRSegMediumMLX",
    "RFDETRSegLargeMLX",
    "RFDETRModule",
    "RFDETRDataModule",
    "build_trainer",
]


def __getattr__(name: str):
    """Resolve MLX exports lazily to avoid importing MLX unnecessarily."""
    if name in {"RFDETRMediumMLX", "RFDETRLargeMLX", "RFDETRSegMediumMLX", "RFDETRSegLargeMLX"}:
        from rfdetr import mlx as mlx_module

        value = getattr(mlx_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
