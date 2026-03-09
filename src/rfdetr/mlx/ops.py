"""Low-level MLX tensor ops used by the inference port."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def nchw_to_nhwc(x: mx.array) -> mx.array:
    """Convert a tensor from ``NCHW`` to ``NHWC`` layout."""
    return mx.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x: mx.array) -> mx.array:
    """Convert a tensor from ``NHWC`` to ``NCHW`` layout."""
    return mx.transpose(x, (0, 3, 1, 2))


def resize_nhwc(x: mx.array, target_size: tuple[int, int], mode: str = "linear") -> mx.array:
    """Resize an ``NHWC`` tensor to ``target_size`` using MLX upsampling."""
    height, width = x.shape[1], x.shape[2]
    target_h, target_w = target_size
    if (height, width) == (target_h, target_w):
        return x
    upsample = nn.Upsample(
        scale_factor=(target_h / height, target_w / width),
        mode=mode,
        align_corners=False,
    )
    return upsample(x)


def resize_nchw(x: mx.array, target_size: tuple[int, int], mode: str = "linear") -> mx.array:
    """Resize an ``NCHW`` tensor to ``target_size`` using MLX upsampling."""
    return nhwc_to_nchw(resize_nhwc(nchw_to_nhwc(x), target_size, mode=mode))


def _gather_spatial(flattened: mx.array, indices: mx.array) -> mx.array:
    """Gather ``flattened`` spatial rows at ``indices`` for each batch.

    Args:
        flattened: ``[B, HW, C]`` tensor.
        indices: ``[B, ...]`` integer indices into the ``HW`` dimension.
    """
    gathered = []
    for batch_idx in range(flattened.shape[0]):
        gathered.append(mx.take(flattened[batch_idx], indices[batch_idx], axis=0))
    return mx.stack(gathered, axis=0)


def bilinear_sample_nhwc(
    features: mx.array,
    sampling_locations: mx.array,
    *,
    padding_mode: str = "zeros",
) -> mx.array:
    """Sample ``NHWC`` features at normalized ``[0, 1]`` coordinates.

    This implements the subset of PyTorch ``grid_sample(..., align_corners=False)``
    needed by RF-DETR inference.

    Args:
        features: Tensor of shape ``[B, H, W, C]``.
        sampling_locations: Tensor of shape ``[B, ..., 2]`` with normalized x/y
            coordinates in ``[0, 1]`` order.
        padding_mode: Only ``"zeros"`` and ``"border"`` are supported.
    """
    batch, height, width, channels = features.shape
    coords_x = sampling_locations[..., 0] * width - 0.5
    coords_y = sampling_locations[..., 1] * height - 0.5

    x0 = mx.floor(coords_x).astype(mx.int32)
    y0 = mx.floor(coords_y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0_clip = mx.clip(x0, 0, width - 1)
    x1_clip = mx.clip(x1, 0, width - 1)
    y0_clip = mx.clip(y0, 0, height - 1)
    y1_clip = mx.clip(y1, 0, height - 1)

    flat = mx.reshape(features, (batch, height * width, channels))
    idx00 = y0_clip * width + x0_clip
    idx01 = y0_clip * width + x1_clip
    idx10 = y1_clip * width + x0_clip
    idx11 = y1_clip * width + x1_clip

    v00 = _gather_spatial(flat, idx00)
    v01 = _gather_spatial(flat, idx01)
    v10 = _gather_spatial(flat, idx10)
    v11 = _gather_spatial(flat, idx11)

    wx = coords_x - x0.astype(coords_x.dtype)
    wy = coords_y - y0.astype(coords_y.dtype)

    if padding_mode == "zeros":
        valid00 = ((x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)).astype(features.dtype)[..., None]
        valid01 = ((x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)).astype(features.dtype)[..., None]
        valid10 = ((x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)).astype(features.dtype)[..., None]
        valid11 = ((x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)).astype(features.dtype)[..., None]
        v00 = v00 * valid00
        v01 = v01 * valid01
        v10 = v10 * valid10
        v11 = v11 * valid11
    elif padding_mode != "border":
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")

    w00 = ((1.0 - wx) * (1.0 - wy))[..., None]
    w01 = (wx * (1.0 - wy))[..., None]
    w10 = ((1.0 - wx) * wy)[..., None]
    w11 = (wx * wy)[..., None]
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11


def ms_deform_attn_core(
    value: mx.array,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: mx.array,
    attention_weights: mx.array,
) -> mx.array:
    """Inference-only MLX port of RF-DETR's PyTorch deformable attention core.

    Args:
        value: Tensor shaped ``[B, n_heads, head_dim, N]``.
        value_spatial_shapes: Spatial shapes for each feature level.
        sampling_locations: Tensor shaped ``[B, Len_q, n_heads, L, P, 2]`` in
            normalized ``[0, 1]`` coordinates.
        attention_weights: Tensor shaped ``[B, Len_q, n_heads, L * P]``.
    """
    batch, n_heads, head_dim, _ = value.shape
    _, len_q, _, num_levels, num_points, _ = sampling_locations.shape
    level_sizes = [height * width for height, width in value_spatial_shapes]

    splits = []
    start = 0
    for size in level_sizes:
        splits.append(value[:, :, :, start : start + size])
        start += size

    sampled_values = []
    for level_idx, (height, width) in enumerate(value_spatial_shapes):
        level_value = mx.reshape(splits[level_idx], (batch, n_heads, head_dim, height, width))
        level_value = mx.transpose(level_value, (0, 1, 3, 4, 2))
        level_value = mx.reshape(level_value, (batch * n_heads, height, width, head_dim))

        level_grid = sampling_locations[:, :, :, level_idx]
        level_grid = mx.transpose(level_grid, (0, 2, 1, 3, 4))
        level_grid = mx.reshape(level_grid, (batch * n_heads, len_q, num_points, 2))

        sampled = bilinear_sample_nhwc(level_value, level_grid, padding_mode="zeros")
        sampled_values.append(sampled)

    stacked = mx.stack(sampled_values, axis=-2)
    stacked = mx.reshape(stacked, (batch * n_heads, len_q, num_levels * num_points, head_dim))
    weights = mx.transpose(attention_weights, (0, 2, 1, 3))
    weights = mx.reshape(weights, (batch * n_heads, len_q, num_levels * num_points, 1))
    output = mx.sum(stacked * weights, axis=2)
    output = mx.reshape(output, (batch, n_heads, len_q, head_dim))
    output = mx.transpose(output, (0, 2, 1, 3))
    return mx.reshape(output, (batch, len_q, n_heads * head_dim))


def gen_sineembed_for_position(pos_tensor: mx.array, dim: int = 128) -> mx.array:
    """MLX port of RF-DETR's reference point sine embedding."""
    scale = 2 * math.pi
    dim_t = mx.arange(dim, dtype=pos_tensor.dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = mx.reshape(mx.stack([mx.sin(pos_x[:, :, 0::2]), mx.cos(pos_x[:, :, 1::2])], axis=3), pos_x.shape[:2] + (-1,))
    pos_y = mx.reshape(mx.stack([mx.sin(pos_y[:, :, 0::2]), mx.cos(pos_y[:, :, 1::2])], axis=3), pos_y.shape[:2] + (-1,))

    if pos_tensor.shape[-1] == 2:
        return mx.concatenate([pos_y, pos_x], axis=2)

    if pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        h_embed = pos_tensor[:, :, 3] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_w = mx.reshape(mx.stack([mx.sin(pos_w[:, :, 0::2]), mx.cos(pos_w[:, :, 1::2])], axis=3), pos_w.shape[:2] + (-1,))
        pos_h = mx.reshape(mx.stack([mx.sin(pos_h[:, :, 0::2]), mx.cos(pos_h[:, :, 1::2])], axis=3), pos_h.shape[:2] + (-1,))
        return mx.concatenate([pos_y, pos_x, pos_w, pos_h], axis=2)

    raise ValueError(f"Unknown pos_tensor.shape[-1]: {pos_tensor.shape[-1]}")
