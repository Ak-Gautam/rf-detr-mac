# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MLX backbone and projector implementation for RF-DETR inference."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from rfdetr.mlx.config import MLXModelConfig
from rfdetr.mlx.layers import Conv2dNCHW, ConvTranspose2dNCHW, LayerNorm2d


def _interpolate_position_embeddings(
    position_embeddings: mx.array,
    *,
    patch_size: int,
    target_height: int,
    target_width: int,
) -> mx.array:
    """Resize pretrained patch position embeddings to the current patch grid."""
    num_positions = position_embeddings.shape[1] - 1
    height = target_height // patch_size
    width = target_width // patch_size
    class_pos_embed = position_embeddings[:, :1]
    patch_pos_embed = position_embeddings[:, 1:]
    if patch_pos_embed.shape[1] == height * width:
        return position_embeddings

    spatial_size = int(num_positions**0.5)
    patch_pos_embed = mx.reshape(patch_pos_embed, (1, spatial_size, spatial_size, patch_pos_embed.shape[-1]))
    patch_pos_embed = nn.Upsample(
        scale_factor=(height / spatial_size, width / spatial_size),
        mode="cubic",
        align_corners=False,
    )(patch_pos_embed)
    patch_pos_embed = mx.reshape(patch_pos_embed, (1, height * width, patch_pos_embed.shape[-1]))
    return mx.concatenate([class_pos_embed, patch_pos_embed], axis=1)


class Dinov2PatchEmbeddings(nn.Module):
    """Patch embedding stem with checkpoint-compatible parameter names."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the patch projection.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.projection = Conv2dNCHW(
            3,
            config.backbone_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Project pixels into patch tokens."""
        embeddings = self.projection(pixel_values)
        embeddings = mx.reshape(embeddings, (embeddings.shape[0], embeddings.shape[1], -1))
        return mx.transpose(embeddings, (0, 2, 1))


class WindowedDinov2Embeddings(nn.Module):
    """Construct DINOv2 tokens and apply window partitioning."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize learnable embedding parameters.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.cls_token = mx.zeros((1, 1, config.backbone_dim))
        self.mask_token = mx.zeros((1, config.backbone_dim))
        self.position_embeddings = mx.zeros((1, config.positional_encoding_size**2 + 1, config.backbone_dim))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.num_windows = config.num_windows
        self.num_register_tokens = config.num_register_tokens

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Build the token sequence consumed by the windowed ViT blocks."""
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.cls_token.shape[-1]))
        position_embeddings = _interpolate_position_embeddings(
            self.position_embeddings,
            patch_size=self.patch_size,
            target_height=height,
            target_width=width,
        )
        embeddings = mx.concatenate([cls_tokens, embeddings], axis=1) + position_embeddings

        if self.num_windows > 1:
            num_h_patches = height // self.patch_size
            num_w_patches = width // self.patch_size
            num_h_window = num_h_patches // self.num_windows
            num_w_window = num_w_patches // self.num_windows

            cls_with_pos = embeddings[:, :1]
            patch_tokens = embeddings[:, 1:]
            patch_tokens = mx.reshape(patch_tokens, (batch_size, num_h_patches, num_w_patches, -1))
            patch_tokens = mx.reshape(
                patch_tokens,
                (batch_size * self.num_windows, num_h_window, self.num_windows, num_w_window, patch_tokens.shape[-1]),
            )
            patch_tokens = mx.transpose(patch_tokens, (0, 2, 1, 3, 4))
            patch_tokens = mx.reshape(
                patch_tokens,
                (batch_size * self.num_windows**2, num_h_window * num_w_window, patch_tokens.shape[-1]),
            )
            cls_with_pos = mx.repeat(cls_with_pos, self.num_windows**2, axis=0)
            embeddings = mx.concatenate([cls_with_pos, patch_tokens], axis=1)

        return embeddings


class Dinov2SelfAttention(nn.Module):
    """DINOv2 self-attention with explicit query, key, and value projections."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        """Initialize the attention projections.

        Args:
            hidden_size: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply multi-head self-attention."""
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = mx.transpose(mx.reshape(query, (*query.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (*key.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (*value.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))

        scores = (query @ mx.transpose(key, (0, 1, 3, 2))) / (self.head_dim**0.5)
        weights = mx.softmax(scores, axis=-1)
        context = weights @ value
        context = mx.transpose(context, (0, 2, 1, 3))
        return mx.reshape(context, (*context.shape[:2], self.hidden_size))


class Dinov2SelfOutput(nn.Module):
    """Output projection wrapper used by the DINOv2 attention block."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize the output projection.

        Args:
            hidden_size: Embedding dimension.
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Project the attention output."""
        return self.dense(hidden_states)


class Dinov2Attention(nn.Module):
    """Composite DINOv2 attention block matching PyTorch parameter names."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        """Initialize the attention wrapper.

        Args:
            hidden_size: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.attention = Dinov2SelfAttention(hidden_size, num_heads)
        self.output = Dinov2SelfOutput(hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply self-attention followed by the output projection."""
        return self.output(self.attention(hidden_states))


class Dinov2MLP(nn.Module):
    """DINOv2 feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: int) -> None:
        """Initialize the MLP.

        Args:
            hidden_size: Embedding dimension.
            mlp_ratio: Hidden multiplier for the first projection.
        """
        super().__init__()
        mlp_hidden = hidden_size * mlp_ratio
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.act = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the MLP."""
        return self.fc2(self.act(self.fc1(hidden_states)))


class LayerScale(nn.Module):
    """Learned residual scaling used by DINOv2."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize the scale vector.

        Args:
            hidden_size: Embedding dimension.
        """
        super().__init__()
        self.lambda1 = mx.ones((hidden_size,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Scale the final dimension."""
        return hidden_states * self.lambda1


class WindowedDinov2Layer(nn.Module):
    """Single windowed DINOv2 encoder block."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the layer.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.num_windows = config.num_windows
        self.norm1 = nn.LayerNorm(config.backbone_dim, eps=1e-6)
        self.attention = Dinov2Attention(config.backbone_dim, config.backbone_heads)
        self.layer_scale1 = LayerScale(config.backbone_dim)
        self.norm2 = nn.LayerNorm(config.backbone_dim, eps=1e-6)
        self.mlp = Dinov2MLP(config.backbone_dim, config.backbone_mlp_ratio)
        self.layer_scale2 = LayerScale(config.backbone_dim)

    def __call__(self, hidden_states: mx.array, run_full_attention: bool) -> mx.array:
        """Run the block with either windowed or full attention."""
        shortcut = hidden_states
        if run_full_attention:
            batch_size, tokens, channels = hidden_states.shape
            hidden_states = mx.reshape(
                hidden_states,
                (batch_size // self.num_windows**2, self.num_windows**2 * tokens, channels),
            )

        attention_output = self.attention(self.norm1(hidden_states))
        if run_full_attention:
            batch_size, tokens, channels = hidden_states.shape
            attention_output = mx.reshape(
                attention_output,
                (batch_size * self.num_windows**2, tokens // self.num_windows**2, channels),
            )

        hidden_states = shortcut + self.layer_scale1(attention_output)
        hidden_states = hidden_states + self.layer_scale2(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class WindowedDinov2Encoder(nn.Module):
    """Inference-only stack of windowed DINOv2 blocks."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the encoder.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.layer = [WindowedDinov2Layer(config) for _ in range(config.backbone_layers)]
        window_block_indexes = set(range(config.out_feature_indexes[-1] + 1))
        window_block_indexes.difference_update(config.out_feature_indexes)
        self.window_block_indexes = window_block_indexes

    def __call__(self, hidden_states: mx.array) -> list[mx.array]:
        """Run the encoder and return hidden states for the requested stages."""
        all_hidden_states = [hidden_states]
        for index, layer_module in enumerate(self.layer):
            run_full_attention = index not in self.window_block_indexes
            hidden_states = layer_module(hidden_states, run_full_attention=run_full_attention)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class WindowedDinov2Backbone(nn.Module):
    """Windowed DINOv2 backbone that returns selected spatial feature maps."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the inference backbone.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.config = config
        self.embeddings = WindowedDinov2Embeddings(config)
        self.encoder = WindowedDinov2Encoder(config)
        self.layernorm = nn.LayerNorm(config.backbone_dim, eps=1e-6)

    def __call__(self, pixel_values: mx.array) -> list[mx.array]:
        """Return the feature maps used by the RF-DETR projector."""
        batch_size, _, height, width = pixel_values.shape
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        num_h_window = num_h_patches // self.config.num_windows
        num_w_window = num_w_patches // self.config.num_windows

        hidden_states = self.encoder(self.embeddings(pixel_values))
        feature_maps: list[mx.array] = []
        for stage_index in self.config.out_feature_indexes:
            hidden_state = self.layernorm(hidden_states[stage_index])
            hidden_state = hidden_state[:, 1:]
            if self.config.num_windows > 1:
                hidden_state = mx.reshape(
                    hidden_state,
                    (
                        (hidden_state.shape[0] // self.config.num_windows**2) * self.config.num_windows,
                        self.config.num_windows,
                        num_h_window,
                        num_w_window,
                        hidden_state.shape[-1],
                    ),
                )
                hidden_state = mx.transpose(hidden_state, (0, 2, 1, 3, 4))
                hidden_state = mx.reshape(
                    hidden_state,
                    (batch_size, num_h_patches, num_w_patches, self.config.backbone_dim),
                )
            else:
                hidden_state = mx.reshape(hidden_state, (batch_size, num_h_patches, num_w_patches, self.config.backbone_dim))

            feature_maps.append(mx.transpose(hidden_state, (0, 3, 1, 2)))
        return feature_maps


class DinoV2(nn.Module):
    """Thin wrapper around the windowed DINOv2 backbone."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the wrapper.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.encoder = WindowedDinov2Backbone(config)

    def __call__(self, x: mx.array) -> list[mx.array]:
        """Run the backbone."""
        return self.encoder(x)


class ConvX(nn.Module):
    """Projector convolution + layer norm + activation block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel: int | tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        act: str = "relu",
    ) -> None:
        """Initialize the block.

        Args:
            in_planes: Number of input channels.
            out_planes: Number of output channels.
            kernel: Kernel size.
            stride: Convolution stride.
            groups: Number of groups.
            act: Activation name.
        """
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.conv = Conv2dNCHW(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel[0] // 2, kernel[1] // 2),
            groups=groups,
            bias=False,
        )
        self.bn = LayerNorm2d(out_planes)
        self.act = {"relu": nn.ReLU(), "silu": nn.SiLU()}[act]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the block."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """CSP bottleneck used by the RF-DETR projector."""

    def __init__(self, c1: int, c2: int, *, shortcut: bool = True, e: float = 0.5) -> None:
        """Initialize the bottleneck.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            shortcut: Whether to add the residual input.
            e: Expansion ratio.
        """
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = ConvX(c1, hidden, kernel=3, stride=1, act="silu")
        self.cv2 = ConvX(hidden, c2, kernel=3, stride=1, act="silu")
        self.add = shortcut and c1 == c2

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the bottleneck."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP-style fusion block used in the projector."""

    def __init__(self, c1: int, c2: int, n: int) -> None:
        """Initialize the fusion block.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            n: Number of bottleneck blocks.
        """
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ConvX(c1, 2 * self.c, kernel=1, stride=1, act="silu")
        self.cv2 = ConvX((2 + n) * self.c, c2, kernel=1, stride=1, act="silu")
        self.m = [Bottleneck(self.c, self.c, shortcut=False, e=1.0) for _ in range(n)]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the fusion block."""
        split = mx.split(self.cv1(x), [self.c], axis=1)
        outputs = [split[0], split[1]]
        for block in self.m:
            outputs.append(block(outputs[-1]))
        return self.cv2(mx.concatenate(outputs, axis=1))


class MultiScaleProjector(nn.Module):
    """Project backbone features into the single-level RF-DETR feature map."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the projector.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        in_channels = [config.backbone_dim] * len(config.out_feature_indexes)
        scale_factors = {"P4": 1.0}
        self.stages_sampling = []
        self.stages = []
        for scale_name in config.projector_scales:
            stage_sampling = []
            scale = scale_factors[scale_name]
            for in_dim in in_channels:
                if scale == 1.0:
                    stage_sampling.append(nn.Identity())
                elif scale == 2.0:
                    stage_sampling.append(ConvTranspose2dNCHW(in_dim, in_dim // 2, kernel_size=2, stride=2))
                else:
                    raise NotImplementedError(f"Unsupported projector scale: {scale_name}")
            self.stages_sampling.append(stage_sampling)
            fused_in = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            self.stages.append([C2f(fused_in, config.hidden_dim, 3), LayerNorm2d(config.hidden_dim)])

    def __call__(self, features: list[mx.array]) -> list[mx.array]:
        """Fuse the selected backbone feature maps."""
        outputs: list[mx.array] = []
        for stage, samplers in zip(self.stages, self.stages_sampling):
            fused = [sampler(feature) for sampler, feature in zip(samplers, features)]
            stage_output = mx.concatenate(fused, axis=1)
            for module in stage:
                stage_output = module(stage_output)
            outputs.append(stage_output)
        return outputs


class Backbone(nn.Module):
    """Checkpoint-compatible MLX RF-DETR backbone wrapper."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the wrapper.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.encoder = DinoV2(config)
        self.projector = MultiScaleProjector(config)

    def __call__(self, pixel_values: mx.array) -> list[mx.array]:
        """Run the backbone and projector."""
        return self.projector(self.encoder(pixel_values))


@dataclass(frozen=True)
class FeatureMapSpec:
    """Simple feature-map description used by the positional encoding helper."""

    height: int
    width: int


def build_sine_position_encoding(spec: FeatureMapSpec, hidden_dim: int) -> mx.array:
    """Build the RF-DETR sine position encoding for a feature map.

    Args:
        spec: Spatial shape of the feature map.
        hidden_dim: Output channel count for the positional encoding.

    Returns:
        Positional encoding shaped ``[1, hidden_dim, H, W]``.
    """
    not_mask = mx.ones((1, spec.height, spec.width), dtype=mx.float32)
    y_embed = mx.cumsum(not_mask, axis=1)
    x_embed = mx.cumsum(not_mask, axis=2)
    eps = 1e-6
    scale = 2.0 * 3.141592653589793
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = hidden_dim // 2
    dim_t = mx.arange(num_pos_feats, dtype=mx.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = mx.reshape(mx.stack([mx.sin(pos_x[:, :, :, 0::2]), mx.cos(pos_x[:, :, :, 1::2])], axis=4), (1, spec.height, spec.width, -1))
    pos_y = mx.reshape(mx.stack([mx.sin(pos_y[:, :, :, 0::2]), mx.cos(pos_y[:, :, :, 1::2])], axis=4), (1, spec.height, spec.width, -1))
    return mx.transpose(mx.concatenate([pos_y, pos_x], axis=3), (0, 3, 1, 2))
