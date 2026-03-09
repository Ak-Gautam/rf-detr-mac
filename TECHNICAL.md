# RF-DETR Technical Notes

## What RF-DETR Does

RF-DETR is a transformer-based model family for object detection and instance segmentation.

At a high level:

1. An input image is resized to a fixed square resolution.
2. A DINOv2-based vision transformer backbone extracts dense visual features.
3. A lightweight feature projector exposes the feature level used by the decoder.
4. A two-stage transformer decoder proposes object queries and refines them into boxes and class logits.
5. Segmentation variants add a mask head that combines decoder query features with spatial features to predict per-instance masks.

## Retained Architecture In This Fork

This fork keeps only the Apache-2.0 Medium and Large checkpoints:

- `RFDETRMedium`
- `RFDETRLarge`
- `RFDETRSegMedium`
- `RFDETRSegLarge`

The MLX port mirrors the retained inference graph only. It does not reimplement the training stack.

## Detection Path

Detection inference in this fork follows the same broad structure as upstream:

1. Normalize the RGB image with ImageNet-style mean and std.
2. Run the DINOv2 windowed-small backbone.
3. Flatten projected features and positional encodings.
4. Generate encoder proposals.
5. Select top proposals for decoder queries.
6. Decode into class logits and normalized boxes.
7. Postprocess to image-space `xyxy` boxes and confidence scores.

## Segmentation Path

Segmentation models reuse the same detection backbone and decoder, then add:

1. Spatial feature refinement over the projected backbone feature map.
2. Query-side feature projection for each kept instance query.
3. Query-by-spatial multiplication to produce coarse mask logits.
4. Resize and threshold during postprocessing.

## MLX Port

The MLX code in this fork is original to this fork's Apple Silicon effort.

It includes:

- checkpoint conversion from PyTorch tensors to MLX-friendly layouts
- MLX layers for the retained inference graph
- MLX wrappers for Medium/Large detection and segmentation
- MLX postprocessing compatible with `supervision.Detections`

The MLX port is intended for local inference on M-series Macs.

## PyTorch/MPS vs MLX

PyTorch/MPS:

- keeps more upstream behavior intact
- remains the training path
- is easier to compare directly against upstream checkpoints

MLX:

- is the preferred local inference path in this fork for Apple Silicon
- avoids carrying CUDA-specific optimizations and packaging
- is implemented only for the retained Medium/Large Apache checkpoints

## What Was Removed

To keep the fork focused and legally simple, this repository removes:

- non-Apache upstream extensions and platform models
- smaller and larger variant families outside Medium/Large
- ONNX/TensorRT export
- Roboflow deployment helpers
- CUDA-specific benchmarking and CI plumbing

## Provenance

This fork is based on [roboflow/rf-detr](https://github.com/roboflow/rf-detr).

Not all code here is original to this fork:

- the RF-DETR model family and upstream PyTorch implementation come from Roboflow and contributors
- the Apple Silicon narrowing and MLX inference port are the main original contributions in this fork
