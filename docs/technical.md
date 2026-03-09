# Technical Overview

RF-DETR is a transformer-based detection and segmentation architecture.

Inference in this fork works like this:

1. Resize and normalize the image.
2. Run the DINOv2-based backbone.
3. Project the retained feature level.
4. Build encoder proposals.
5. Use transformer decoder queries to refine boxes and class scores.
6. For segmentation variants, combine query features with spatial features to produce masks.

This fork keeps only:

- `RFDETRMedium`
- `RFDETRLarge`
- `RFDETRSegMedium`
- `RFDETRSegLarge`
- their MLX inference wrappers

The MLX path is inference-only. Training remains in the PyTorch path.
