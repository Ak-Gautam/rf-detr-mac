"""Inference post-processing shared by the MLX detection and segmentation ports."""

from __future__ import annotations

import mlx.core as mx

from rfdetr.mlx.ops import resize_nchw


def _topk_with_indices(values: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Return top-k values and indices along axis 1."""
    indices = mx.argsort(-values, axis=1)[:, :k]
    topk_values = mx.take_along_axis(values, indices, axis=1)
    return topk_values, indices


def box_cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    """Convert center-based boxes to ``xyxy`` boxes."""
    cx, cy, w, h = mx.split(boxes, 4, axis=-1)
    half_w = w / 2.0
    half_h = h / 2.0
    return mx.concatenate([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=-1)


def postprocess(
    pred_logits: mx.array,
    pred_boxes: mx.array,
    target_sizes: mx.array,
    *,
    num_select: int,
    pred_masks: mx.array | None = None,
) -> list[dict[str, mx.array]]:
    """Convert raw MLX model outputs into per-image detection dictionaries."""
    prob = mx.sigmoid(pred_logits)
    batch, _, num_classes = prob.shape
    topk_values, topk_indices = _topk_with_indices(mx.reshape(prob, (batch, -1)), num_select)

    scores = topk_values
    topk_boxes = topk_indices // num_classes
    labels = topk_indices % num_classes

    boxes = box_cxcywh_to_xyxy(pred_boxes)
    gathered_boxes = []
    gathered_masks = []
    for batch_idx in range(batch):
        gathered_boxes.append(mx.take(boxes[batch_idx], topk_boxes[batch_idx], axis=0))
        if pred_masks is not None:
            gathered_masks.append(mx.take(pred_masks[batch_idx], topk_boxes[batch_idx], axis=0))
    boxes = mx.stack(gathered_boxes, axis=0)

    img_h = target_sizes[:, 0]
    img_w = target_sizes[:, 1]
    scale = mx.stack([img_w, img_h, img_w, img_h], axis=1)
    boxes = boxes * scale[:, None, :]

    results: list[dict[str, mx.array]] = []
    for batch_idx in range(batch):
        result = {
            "scores": scores[batch_idx],
            "labels": labels[batch_idx],
            "boxes": boxes[batch_idx],
        }
        if pred_masks is not None:
            target_size = tuple(int(v) for v in target_sizes[batch_idx].tolist())
            resized_masks = resize_nchw(gathered_masks[batch_idx][:, None, :, :], target_size, mode="linear")
            result["masks"] = resized_masks > 0.0
        results.append(result)
    return results
