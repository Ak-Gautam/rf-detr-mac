# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Inference-only MLX implementation of the Apache RF-DETR large models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
import supervision as sv
from PIL import Image

from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.mlx.backbone import Backbone, FeatureMapSpec, build_sine_position_encoding
from rfdetr.mlx.config import MLXModelConfig, RFDETRLargeMLXConfig, RFDETRSegLargeMLXConfig
from rfdetr.mlx.convert import convert_checkpoint, convert_pretrained_checkpoint
from rfdetr.mlx.layers import MLP, Conv2dNCHW, Embedding, MSDeformAttn, MultiheadSelfAttention
from rfdetr.mlx.ops import gen_sineembed_for_position, resize_nchw
from rfdetr.mlx.postprocess import postprocess


def _batch_gather(x: mx.array, indices: mx.array) -> mx.array:
    """Gather rows from a batch of tensors.

    Args:
        x: Tensor shaped ``[B, S, C]``.
        indices: Integer tensor shaped ``[B, K]``.

    Returns:
        Gathered tensor shaped ``[B, K, C]``.
    """
    return mx.stack([mx.take(x[batch_idx], indices[batch_idx], axis=0) for batch_idx in range(x.shape[0])], axis=0)


def _topk_with_indices(values: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Return top-k values and indices along axis 1."""
    indices = mx.argsort(-values, axis=1)[:, :k]
    topk_values = mx.take_along_axis(values, indices, axis=1)
    return topk_values, indices


def _prepare_image(image: Any, resolution: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Normalize one input image to the RF-DETR model resolution.

    Args:
        image: PIL image, numpy array, local path, URL, or CHW tensor-like object.
        resolution: Square inference resolution.

    Returns:
        A tuple of ``(chw_float32, original_hw)``.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True, timeout=30).raw)
        else:
            image = Image.open(image)

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image, received shape {array.shape}")
        pil_image = Image.fromarray(array.astype(np.uint8) if array.dtype != np.uint8 else array, mode="RGB")

    original_size = (pil_image.height, pil_image.width)
    resized = pil_image.resize((resolution, resolution), resample=Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))

    means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    stds = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (array - means) / stds, original_size


def _gen_encoder_output_proposals(
    memory: mx.array,
    spatial_shapes: list[tuple[int, int]],
    *,
    unsigmoid: bool,
) -> tuple[mx.array, mx.array]:
    """Generate encoder proposals for the RF-DETR two-stage decoder.

    Args:
        memory: Flattened source tensor shaped ``[B, S, C]``.
        spatial_shapes: Spatial shape for each feature level.
        unsigmoid: Whether to return logits instead of normalized boxes.

    Returns:
        Tuple of ``(output_memory, output_proposals)``.
    """
    batch_size = memory.shape[0]
    proposals = []
    for level, (height, width) in enumerate(spatial_shapes):
        grid_y, grid_x = mx.meshgrid(
            mx.arange(height, dtype=memory.dtype),
            mx.arange(width, dtype=memory.dtype),
            indexing="ij",
        )
        grid = mx.stack([grid_x, grid_y], axis=-1)
        scale = mx.array([width, height], dtype=memory.dtype)
        grid = (grid[None] + 0.5) / scale
        grid = mx.broadcast_to(grid, (batch_size, height, width, 2))
        wh = mx.ones_like(grid) * (0.05 * (2.0**level))
        proposals.append(mx.reshape(mx.concatenate([grid, wh], axis=-1), (batch_size, height * width, 4)))

    output_proposals = mx.concatenate(proposals, axis=1)
    valid = mx.all((output_proposals > 0.01) & (output_proposals < 0.99), axis=-1, keepdims=True)
    if unsigmoid:
        output_proposals = mx.log(output_proposals / (1.0 - output_proposals))
        output_proposals = mx.where(valid, output_proposals, mx.full_like(output_proposals, float("inf")))
    else:
        output_proposals = mx.where(valid, output_proposals, mx.zeros_like(output_proposals))
    output_memory = mx.where(valid, memory, mx.zeros_like(memory))
    return output_memory, output_proposals


class DepthwiseConvBlock(nn.Module):
    """Simplified ConvNeXt-style depthwise block for the segmentation head."""

    def __init__(self, dim: int) -> None:
        """Initialize the block.

        Args:
            dim: Channel dimension of the spatial feature map.
        """
        super().__init__()
        self.dwconv = Conv2dNCHW(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the block."""
        residual = x
        x = self.dwconv(x)
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.act(self.pwconv1(self.norm(x)))
        x = mx.transpose(x, (0, 3, 1, 2))
        return x + residual


class MLPBlock(nn.Module):
    """Query-side MLP block for the segmentation head."""

    def __init__(self, dim: int) -> None:
        """Initialize the block.

        Args:
            dim: Query embedding dimension.
        """
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the residual MLP."""
        residual = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        return x + residual


class SegmentationHead(nn.Module):
    """RF-DETR segmentation head for the large segmentation checkpoint."""

    def __init__(self, hidden_dim: int, num_blocks: int, downsample_ratio: int) -> None:
        """Initialize the segmentation head.

        Args:
            hidden_dim: Feature dimension of the decoder outputs.
            num_blocks: Number of depthwise blocks.
            downsample_ratio: Final mask downsample ratio relative to the image.
        """
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.blocks = [DepthwiseConvBlock(hidden_dim) for _ in range(num_blocks)]
        self.spatial_features_proj = Conv2dNCHW(hidden_dim, hidden_dim, kernel_size=1)
        self.query_features_block = MLPBlock(hidden_dim)
        self.query_features_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bias = mx.zeros((1,))

    def __call__(
        self,
        spatial_features: mx.array,
        query_features: list[mx.array],
        image_size: tuple[int, int],
        *,
        skip_blocks: bool = False,
    ) -> list[mx.array]:
        """Predict coarse masks from decoder or encoder query features.

        Args:
            spatial_features: Backbone feature map shaped ``[B, C, H, W]``.
            query_features: Query features per decoder layer.
            image_size: Original resized model input shape ``(H, W)``.
            skip_blocks: Whether to skip the spatial refinement stack.

        Returns:
            List of mask logits shaped ``[B, N, Hm, Wm]``.
        """
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = resize_nchw(spatial_features, target_size, mode="linear")
        outputs: list[mx.array] = []

        if skip_blocks:
            query_feature = self.query_features_proj(self.query_features_block(query_features[0]))
            projected = spatial_features
            spatial_flat = mx.reshape(projected, (projected.shape[0], projected.shape[1], -1))
            masks = query_feature @ spatial_flat
            outputs.append(mx.reshape(masks, (masks.shape[0], masks.shape[1], *target_size)) + self.bias)
            return outputs

        if len(query_features) == 1:
            for block in self.blocks:
                spatial_features = block(spatial_features)
            projected = self.spatial_features_proj(spatial_features)
            query_feature = self.query_features_proj(self.query_features_block(query_features[0]))
            spatial_flat = mx.reshape(projected, (projected.shape[0], projected.shape[1], -1))
            masks = query_feature @ spatial_flat
            outputs.append(mx.reshape(masks, (masks.shape[0], masks.shape[1], *target_size)) + self.bias)
            return outputs

        for block, query_feature in zip(self.blocks, query_features):
            spatial_features = block(spatial_features)
            projected = self.spatial_features_proj(spatial_features)
            query_feature = self.query_features_proj(self.query_features_block(query_feature))
            spatial_flat = mx.reshape(projected, (projected.shape[0], projected.shape[1], -1))
            masks = query_feature @ spatial_flat
            outputs.append(mx.reshape(masks, (masks.shape[0], masks.shape[1], *target_size)) + self.bias)
        return outputs


class TransformerDecoderLayer(nn.Module):
    """Single RF-DETR decoder layer."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the decoder layer.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.self_attn = MultiheadSelfAttention(config.hidden_dim, config.sa_nheads)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.cross_attn = MSDeformAttn(config.hidden_dim, len(config.projector_scales), config.ca_nheads, config.dec_n_points)
        self.linear1 = nn.Linear(config.hidden_dim, 2048)
        self.linear2 = nn.Linear(2048, config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.act = nn.ReLU()

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        pos: mx.array,
        query_pos: mx.array,
        reference_points: mx.array,
        spatial_shapes: list[tuple[int, int]],
    ) -> mx.array:
        """Apply one decoder layer.

        Args:
            tgt: Current query features shaped ``[B, N, C]``.
            memory: Flattened source features shaped ``[B, S, C]``.
            pos: Flattened positional encodings shaped ``[B, S, C]``.
            query_pos: Query positional encoding shaped ``[B, N, C]``.
            reference_points: Reference boxes shaped ``[B, N, L, 4]``.
            spatial_shapes: Spatial shape for each feature level.

        Returns:
            Updated query features shaped ``[B, N, C]``.
        """
        q = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, q, tgt))
        del pos
        tgt = self.norm2(tgt + self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes))
        tgt = self.norm3(tgt + self.linear2(self.act(self.linear1(tgt))))
        return tgt


class TransformerDecoder(nn.Module):
    """RF-DETR decoder used in the large inference checkpoints."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the decoder.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.layers = [TransformerDecoderLayer(config) for _ in range(config.dec_layers)]
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.ref_point_head = MLP(2 * config.hidden_dim, config.hidden_dim, config.hidden_dim, 2)

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        pos: mx.array,
        refpoints_unsigmoid: mx.array,
        spatial_shapes: list[tuple[int, int]],
    ) -> tuple[mx.array, mx.array]:
        """Run the inference-time decoder.

        Args:
            tgt: Initial query embeddings shaped ``[B, N, C]``.
            memory: Flattened source features shaped ``[B, S, C]``.
            pos: Flattened positional encodings shaped ``[B, S, C]``.
            refpoints_unsigmoid: Reference boxes shaped ``[B, N, 4]``.
            spatial_shapes: Spatial shape for each feature level.

        Returns:
            Tuple of ``(decoder_output, reference_boxes)``.
        """
        output = tgt
        obj_center = refpoints_unsigmoid
        query_sine_embed = gen_sineembed_for_position(obj_center, output.shape[-1] // 2)
        refpoints_input = obj_center[:, :, None, :]
        query_pos = self.ref_point_head(query_sine_embed)

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos, refpoints_input, spatial_shapes)

        return self.norm(output), refpoints_unsigmoid


class Transformer(nn.Module):
    """RF-DETR transformer wrapper with two-stage proposal selection."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the transformer.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.config = config
        self.decoder = TransformerDecoder(config)
        self.enc_output = [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.group_detr)]
        self.enc_output_norm = [nn.LayerNorm(config.hidden_dim) for _ in range(config.group_detr)]
        self.enc_out_bbox_embed = [MLP(config.hidden_dim, config.hidden_dim, 4, 3) for _ in range(config.group_detr)]
        self.enc_out_class_embed = [nn.Linear(config.hidden_dim, config.num_classes + 1) for _ in range(config.group_detr)]

    def __call__(
        self,
        srcs: list[mx.array],
        poss: list[mx.array],
        refpoint_embed: mx.array,
        query_feat: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Run the inference-time transformer stack.

        Args:
            srcs: Projected source feature maps shaped ``[B, C, H, W]``.
            poss: Positional encodings matched to ``srcs``.
            refpoint_embed: Learned reference embeddings shaped ``[N, 4]``.
            query_feat: Learned query embeddings shaped ``[N, C]``.

        Returns:
            Tuple of ``(decoder_hs, decoder_refs, encoder_memory, encoder_boxes)``.
        """
        src_flatten = []
        pos_flatten = []
        spatial_shapes: list[tuple[int, int]] = []
        for src, pos in zip(srcs, poss):
            _, channels, height, width = src.shape
            spatial_shapes.append((height, width))
            src_flatten.append(mx.transpose(mx.reshape(src, (src.shape[0], channels, height * width)), (0, 2, 1)))
            pos_flatten.append(mx.transpose(mx.reshape(pos, (pos.shape[0], channels, height * width)), (0, 2, 1)))

        memory = mx.concatenate(src_flatten, axis=1)
        pos = mx.concatenate(pos_flatten, axis=1)

        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory,
            spatial_shapes,
            unsigmoid=not self.config.bbox_reparam,
        )
        output_memory = self.enc_output_norm[0](self.enc_output[0](output_memory))
        enc_outputs_class = self.enc_out_class_embed[0](output_memory)
        enc_outputs_delta = self.enc_out_bbox_embed[0](output_memory)
        enc_outputs_coord = mx.concatenate(
            [
                enc_outputs_delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2],
                mx.exp(enc_outputs_delta[..., 2:]) * output_proposals[..., 2:],
            ],
            axis=-1,
        )

        proposal_scores = mx.max(enc_outputs_class, axis=-1)
        _, topk_indices = _topk_with_indices(proposal_scores, min(self.config.num_queries, proposal_scores.shape[1]))
        refpoint_embed_ts = _batch_gather(enc_outputs_coord, topk_indices)
        memory_ts = _batch_gather(output_memory, topk_indices)

        batch_size = memory.shape[0]
        tgt = mx.broadcast_to(query_feat[None, :, :], (batch_size, query_feat.shape[0], query_feat.shape[1]))
        refpoint_embed = mx.broadcast_to(refpoint_embed[None, :, :], (batch_size, refpoint_embed.shape[0], refpoint_embed.shape[1]))

        ts_len = refpoint_embed_ts.shape[1]
        refpoint_embed_ts_subset = mx.concatenate(
            [
                refpoint_embed[:, :ts_len, :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2],
                mx.exp(refpoint_embed[:, :ts_len, 2:]) * refpoint_embed_ts[..., 2:],
            ],
            axis=-1,
        )
        refpoint_embed = mx.concatenate([refpoint_embed_ts_subset, refpoint_embed[:, ts_len:, :]], axis=1)

        hs, references = self.decoder(tgt, memory, pos, refpoint_embed, spatial_shapes)
        return hs, references, memory_ts, refpoint_embed_ts


class RFDETRForInference(nn.Module):
    """Checkpoint-compatible MLX RF-DETR inference model."""

    def __init__(self, config: MLXModelConfig) -> None:
        """Initialize the model.

        Args:
            config: Static MLX model configuration.
        """
        super().__init__()
        self.config = config
        self.num_queries = config.num_queries
        self.backbone = [Backbone(config)]
        self.transformer = Transformer(config)
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        self.refpoint_embed = Embedding(config.num_query_embeddings * config.group_detr, 4)
        self.query_feat = Embedding(config.num_query_embeddings * config.group_detr, config.hidden_dim)
        self.segmentation_head = (
            SegmentationHead(config.hidden_dim, config.dec_layers, config.mask_downsample_ratio)
            if config.segmentation_head
            else None
        )

    def __call__(self, tensors: mx.array) -> dict[str, mx.array]:
        """Run the inference graph on a preprocessed batch."""
        srcs = self.backbone[0](tensors)
        poss = [
            mx.broadcast_to(
                build_sine_position_encoding(FeatureMapSpec(height=src.shape[2], width=src.shape[3]), self.config.hidden_dim),
                (src.shape[0], self.config.hidden_dim, src.shape[2], src.shape[3]),
            )
            for src in srcs
        ]
        refpoint_embed = self.refpoint_embed.weight[: self.num_queries]
        query_feat = self.query_feat.weight[: self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(srcs, poss, refpoint_embed, query_feat)

        outputs_coord_delta = self.bbox_embed(hs)
        outputs_coord = mx.concatenate(
            [
                outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2],
                mx.exp(outputs_coord_delta[..., 2:]) * ref_unsigmoid[..., 2:],
            ],
            axis=-1,
        )
        outputs_class = self.class_embed(hs)

        outputs: dict[str, mx.array] = {"pred_boxes": outputs_coord, "pred_logits": outputs_class}
        if self.segmentation_head is not None:
            outputs["pred_masks"] = self.segmentation_head(srcs[0], [hs], tensors.shape[-2:])[0]
        return outputs


class RFDETRMLX:
    """User-facing MLX inference wrapper for RF-DETR large checkpoints."""

    means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    stds = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, config: MLXModelConfig, weights_path: str | None = None) -> None:
        """Create an MLX RF-DETR model.

        Args:
            config: Static MLX model configuration.
            weights_path: Optional path to converted MLX safetensors.
        """
        self.config = config
        self.model = RFDETRForInference(config)
        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str | Path) -> None:
        """Load converted MLX safetensors.

        Args:
            weights_path: Path to a converted MLX checkpoint.
        """
        self.model.load_weights(str(weights_path), strict=True)

    @classmethod
    def from_pretrained(
        cls,
        config: MLXModelConfig,
        *,
        weights_path: str | Path | None = None,
        output_dir: str | Path = ".",
    ) -> "RFDETRMLX":
        """Create a model from a converted or original pretrained checkpoint.

        Args:
            config: Static MLX model configuration.
            weights_path: Optional path to either MLX safetensors or a PyTorch checkpoint.
            output_dir: Directory used when converting a built-in checkpoint.

        Returns:
            Loaded MLX model wrapper.
        """
        if weights_path is None:
            converted = convert_pretrained_checkpoint(config, output_dir)
        else:
            weights_path = Path(weights_path)
            if weights_path.suffix == ".safetensors":
                converted = weights_path
            else:
                converted = convert_checkpoint(weights_path, output_path=weights_path.with_suffix(".mlx.safetensors"))
        return cls(config, str(converted))

    def predict(self, images: Any, threshold: float = 0.5) -> sv.Detections | list[sv.Detections]:
        """Run detection or segmentation inference.

        Args:
            images: One image or a list of images.
            threshold: Confidence threshold applied after top-k selection.

        Returns:
            One ``supervision.Detections`` object or a list of them.
        """
        if not isinstance(images, list):
            images = [images]

        processed = []
        original_sizes = []
        for image in images:
            tensor, original_size = _prepare_image(image, self.config.resolution)
            processed.append(tensor)
            original_sizes.append(original_size)

        batch = mx.array(np.stack(processed, axis=0))
        outputs = self.model(batch)
        target_sizes = mx.array(np.asarray(original_sizes, dtype=np.int32))
        results = postprocess(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            target_sizes,
            num_select=self.config.num_select,
            pred_masks=outputs.get("pred_masks"),
        )

        detections = []
        for result in results:
            scores = np.asarray(result["scores"])
            labels = np.asarray(result["labels"])
            boxes = np.asarray(result["boxes"])
            keep = scores > threshold
            if "masks" in result:
                masks = np.asarray(result["masks"])[:, 0]
                detections.append(
                    sv.Detections(
                        xyxy=boxes[keep].astype(np.float32),
                        confidence=scores[keep].astype(np.float32),
                        class_id=labels[keep].astype(np.int32),
                        mask=masks[keep].astype(bool),
                    )
                )
            else:
                detections.append(
                    sv.Detections(
                        xyxy=boxes[keep].astype(np.float32),
                        confidence=scores[keep].astype(np.float32),
                        class_id=labels[keep].astype(np.int32),
                    )
                )
        return detections[0] if len(detections) == 1 else detections


class RFDETRLargeMLX(RFDETRMLX):
    """MLX wrapper for the Apache RF-DETR large detection checkpoint."""

    def __init__(self, weights_path: str | None = None) -> None:
        """Create a large detection model.

        Args:
            weights_path: Optional converted checkpoint path.
        """
        super().__init__(RFDETRLargeMLXConfig, weights_path=weights_path)


class RFDETRSegLargeMLX(RFDETRMLX):
    """MLX wrapper for the Apache RF-DETR large segmentation checkpoint."""

    def __init__(self, weights_path: str | None = None) -> None:
        """Create a large segmentation model.

        Args:
            weights_path: Optional converted checkpoint path.
        """
        super().__init__(RFDETRSegLargeMLXConfig, weights_path=weights_path)


def download_and_prepare_builtin_weights(config: MLXModelConfig, output_dir: str | Path = ".") -> Path:
    """Download and convert one of the built-in Apache checkpoints.

    Args:
        config: Static MLX model configuration.
        output_dir: Conversion output directory.

    Returns:
        Path to the converted MLX safetensors file.
    """
    download_pretrain_weights(config.checkpoint_name)
    return convert_pretrained_checkpoint(config, output_dir=output_dir)
