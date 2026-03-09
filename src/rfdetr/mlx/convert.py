# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint conversion utilities for the MLX inference port."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.mlx.config import (
    MLXModelConfig,
    RFDETRLargeMLXConfig,
    RFDETRMediumMLXConfig,
    RFDETRSegLargeMLXConfig,
    RFDETRSegMediumMLXConfig,
)


def _import_mlx() -> tuple[Any, Any]:
    """Import MLX lazily so non-MLX users can still import this module."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return mx, tree_flatten


def _reshape_weight_for_mlx(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch parameter layouts into MLX-friendly layouts.

    MLX ``Conv2d`` and ``ConvTranspose2d`` use ``OHWI`` weights and operate on
    ``NHWC`` tensors. Most other parameter layouts match PyTorch directly.
    """
    if tensor.ndim != 4:
        return tensor

    if "stages_sampling" in key:
        # PyTorch ConvTranspose2d: [in_channels, out_channels, kh, kw]
        # MLX ConvTranspose2d: [out_channels, kh, kw, in_channels]
        return tensor.permute(1, 2, 3, 0).contiguous()

    # PyTorch Conv2d: [out_channels, in_channels, kh, kw]
    # MLX Conv2d: [out_channels, kh, kw, in_channels]
    return tensor.permute(0, 2, 3, 1).contiguous()


def _to_mlx_weights(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, Any]]:
    """Convert a PyTorch state dict into MLX ``load_weights`` pairs."""
    mx, _ = _import_mlx()
    converted: list[tuple[str, Any]] = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = _reshape_weight_for_mlx(key, value.detach().cpu())
        converted.append((key, mx.array(tensor.numpy())))
    return converted


def convert_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert a PyTorch RF-DETR checkpoint into MLX safetensors.

    Args:
        checkpoint_path: Path to the original PyTorch checkpoint.
        output_path: Optional explicit output path. Defaults to replacing the
            source suffix with ``.mlx.safetensors``.
    """
    mx, _ = _import_mlx()

    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".mlx.safetensors")
    output_path = Path(output_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = dict(_to_mlx_weights(checkpoint["model"]))
    mx.save_safetensors(str(output_path), weights)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint_path),
                "format": "mlx-safetensors",
                "num_tensors": len(weights),
            },
            indent=2,
        )
    )
    return output_path


def convert_pretrained_checkpoint(
    config: MLXModelConfig,
    output_dir: str | os.PathLike[str] = ".",
) -> Path:
    """Download and convert one of the built-in Apache checkpoints."""
    checkpoint_name = config.checkpoint_name
    if not checkpoint_name:
        raise ValueError(f"Config {config.name} does not define checkpoint_name")
    download_pretrain_weights(checkpoint_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return convert_checkpoint(checkpoint_name, output_dir / f"{config.name}.safetensors")


def main() -> None:
    """CLI entry point for ad-hoc checkpoint conversion."""
    import argparse

    parser = argparse.ArgumentParser("Convert RF-DETR PyTorch checkpoints to MLX safetensors")
    parser.add_argument("checkpoint", nargs="?", help="Path to a PyTorch checkpoint to convert")
    parser.add_argument(
        "--model",
        choices=["medium", "large", "seg-medium", "seg-large"],
        help="Convert a built-in pretrained checkpoint",
    )
    parser.add_argument("--output", default=".", help="Output file or directory")
    args = parser.parse_args()

    if args.model:
        configs = {
            "medium": RFDETRMediumMLXConfig,
            "large": RFDETRLargeMLXConfig,
            "seg-medium": RFDETRSegMediumMLXConfig,
            "seg-large": RFDETRSegLargeMLXConfig,
        }
        config = configs[args.model]
        output = convert_pretrained_checkpoint(config, args.output)
    elif args.checkpoint:
        output = convert_checkpoint(args.checkpoint, args.output)
    else:
        raise SystemExit("Pass either a checkpoint path or --model")

    print(output)


if __name__ == "__main__":
    main()
