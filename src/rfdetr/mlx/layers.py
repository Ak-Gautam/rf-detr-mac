"""Shared MLX layers used by the inference-only RF-DETR port."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from rfdetr.mlx.ops import ms_deform_attn_core, nchw_to_nhwc, nhwc_to_nchw


def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """Apply a linear projection using PyTorch-compatible weight layout.

    Args:
        x: Input tensor whose final dimension is the input feature size.
        weight: Weight matrix shaped ``[out_features, in_features]``.
        bias: Optional bias vector shaped ``[out_features]``.

    Returns:
        The projected tensor.
    """
    output = x @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


class Embedding(nn.Module):
    """Minimal embedding layer whose weights match PyTorch checkpoint names."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """Initialize the embedding table.

        Args:
            num_embeddings: Number of rows in the embedding table.
            embedding_dim: Size of each embedding vector.
        """
        super().__init__()
        self.weight = mx.zeros((num_embeddings, embedding_dim))

    def __call__(self, indices: mx.array) -> mx.array:
        """Look up embeddings by integer index."""
        return mx.take(self.weight, indices, axis=0)


class Conv2dNCHW(nn.Module):
    """2D convolution that accepts and returns ``NCHW`` tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the convolution parameters.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            stride: Convolution stride.
            padding: Symmetric padding.
            dilation: Dilation factor.
            groups: Number of groups.
            bias: Whether to include a bias parameter.
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = mx.zeros((out_channels, kernel_size[0], kernel_size[1], in_channels // groups))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the convolution."""
        y = mx.conv2d(
            nchw_to_nhwc(x),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return nhwc_to_nchw(y)


class ConvTranspose2dNCHW(nn.Module):
    """Transposed 2D convolution that accepts and returns ``NCHW`` tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        """Initialize the transposed convolution parameters.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            stride: Convolution stride.
            padding: Symmetric padding.
            output_padding: Output padding.
            bias: Whether to include a bias parameter.
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = mx.zeros((out_channels, kernel_size[0], kernel_size[1], in_channels))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the transposed convolution."""
        y = mx.conv_transpose2d(
            nchw_to_nhwc(x),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if self.bias is not None:
            y = y + self.bias
        return nhwc_to_nchw(y)


class LayerNorm2d(nn.Module):
    """Channel-wise layer norm for ``NCHW`` tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize the layer norm.

        Args:
            num_channels: Number of channels in the input tensor.
            eps: Numerical stability epsilon.
        """
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Normalize the final channel dimension after converting to NHWC."""
        x_nhwc = nchw_to_nhwc(x)
        mean = mx.mean(x_nhwc, axis=-1, keepdims=True)
        var = mx.mean(mx.square(x_nhwc - mean), axis=-1, keepdims=True)
        y = (x_nhwc - mean) * mx.rsqrt(var + self.eps)
        y = y * self.weight + self.bias
        return nhwc_to_nchw(y)


class MLP(nn.Module):
    """Small feed-forward network used throughout RF-DETR."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        """Build the linear stack.

        Args:
            input_dim: Input feature size.
            hidden_dim: Hidden feature size.
            output_dim: Output feature size.
            num_layers: Number of linear layers.
        """
        super().__init__()
        hidden_sizes = [hidden_dim] * (num_layers - 1)
        dims = [input_dim] + hidden_sizes
        outputs = hidden_sizes + [output_dim]
        self.layers = [nn.Linear(in_features, out_features) for in_features, out_features in zip(dims, outputs)]

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP with ReLU on all but the last layer."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = nn.relu(x)
        return x


class LayerScale(nn.Module):
    """Learned channel-wise layer scaling used by DINOv2 blocks."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize the scaling parameter.

        Args:
            hidden_size: Size of the final feature dimension.
        """
        super().__init__()
        self.lambda1 = mx.ones((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        """Scale the final dimension elementwise."""
        return x * self.lambda1


class MultiheadSelfAttention(nn.Module):
    """PyTorch-compatible attention with fused ``in_proj`` weights."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize the attention module.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim))
        self.in_proj_bias = mx.zeros((3 * embed_dim,))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self,
        query_input: mx.array,
        key_input: mx.array | None = None,
        value_input: mx.array | None = None,
    ) -> mx.array:
        """Apply batch-first attention.

        Args:
            query_input: Query input shaped ``[B, N, C]``.
            key_input: Optional key input. Defaults to ``query_input``.
            value_input: Optional value input. Defaults to ``query_input``.
        """
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input

        q_weight, k_weight, v_weight = mx.split(self.in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = mx.split(self.in_proj_bias, 3, axis=0)
        query = linear(query_input, q_weight, q_bias)
        key = linear(key_input, k_weight, k_bias)
        value = linear(value_input, v_weight, v_bias)

        query = mx.transpose(mx.reshape(query, (*query.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (*key.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (*value.shape[:2], self.num_heads, self.head_dim)), (0, 2, 1, 3))

        scores = (query @ mx.transpose(key, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        weights = mx.softmax(scores, axis=-1)
        context = weights @ value
        context = mx.transpose(context, (0, 2, 1, 3))
        context = mx.reshape(context, (*context.shape[:2], self.embed_dim))
        return self.out_proj(context)


class MSDeformAttn(nn.Module):
    """Inference-only multi-scale deformable attention."""

    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int) -> None:
        """Initialize the deformable attention module.

        Args:
            d_model: Hidden dimension.
            n_levels: Number of feature levels.
            n_heads: Number of attention heads.
            n_points: Number of sampling points per head and level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        input_flatten: mx.array,
        input_spatial_shapes: list[tuple[int, int]],
    ) -> mx.array:
        """Apply deformable cross-attention.

        Args:
            query: Query tensor shaped ``[B, Nq, C]``.
            reference_points: Reference boxes shaped ``[B, Nq, L, 4]``.
            input_flatten: Flattened source features shaped ``[B, S, C]``.
            input_spatial_shapes: Spatial shape for each feature level.

        Returns:
            Updated query features shaped ``[B, Nq, C]``.
        """
        batch_size, len_q, _ = query.shape
        value = self.value_proj(input_flatten)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = mx.reshape(
            sampling_offsets,
            (batch_size, len_q, self.n_heads, self.n_levels, self.n_points, 2),
        )
        attention_weights = self.attention_weights(query)
        attention_weights = mx.reshape(
            attention_weights,
            (batch_size, len_q, self.n_heads, self.n_levels * self.n_points),
        )
        attention_weights = mx.softmax(attention_weights, axis=-1)

        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )

        value = mx.transpose(value, (0, 2, 1))
        value = mx.reshape(value, (batch_size, self.n_heads, self.head_dim, input_flatten.shape[1]))
        output = ms_deform_attn_core(value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
