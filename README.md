# RF-DETR Mac Fork

This repository is an Apple Silicon-focused non-official fork of [RF-DETR by Roboflow](https://github.com/roboflow/rf-detr).

The original RF-DETR architecture, pretrained checkpoints, and most of the PyTorch implementation come from the Roboflow team and upstream contributors under the Apache License 2.0. The original work in this fork is the cleanup for M-series Macs and the MLX inference port for local execution on Apple Silicon.

This repository intentionally keeps only Apache-2.0 components. Because of that, this fork as distributed here is Apache-2.0.

Fork maintainer:

- GitHub: [Ak-Gautam](https://github.com/Ak-Gautam)
- Repository: [Ak-Gautam/rf-detr-mac](https://github.com/Ak-Gautam/rf-detr-mac)
- X: [@Gautam_A_k](https://x.com/Gautam_A_k)

## Scope

Retained models:

- `RFDETRMedium`
- `RFDETRLarge`
- `RFDETRSegMedium`
- `RFDETRSegLarge`
- `RFDETRMediumMLX`
- `RFDETRLargeMLX`
- `RFDETRSegMediumMLX`
- `RFDETRSegLargeMLX`

Removed from this fork:

- plus/platform models
- nano/small/xlarge/2xlarge variants
- ONNX and TensorRT export
- Roboflow deployment helpers
- CUDA-specific workflows and packaging

## Provenance

- Upstream repository: [roboflow/rf-detr](https://github.com/roboflow/rf-detr)
- Upstream license: Apache License 2.0
- This fork preserves upstream copyright and license notices for upstream-derived files
- The MLX port and Apple Silicon-specific cleanup in this fork are additional work on top of the upstream Apache-2.0 codebase

## Local Setup

This repo is meant to run locally on an M-series Mac with `uv`.

```bash
uv sync --all-groups --extra mlx
```

If you only want runtime dependencies:

```bash
uv sync --extra mlx
```

## Convert Checkpoints to MLX

Built-in conversion commands:

```bash
uv run --no-sync python -m rfdetr.mlx.convert --model medium --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model large --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model seg-medium --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model seg-large --output mlx_out
```

This writes:

- `mlx_out/rfdetr-medium-mlx.safetensors`
- `mlx_out/rfdetr-large-mlx.safetensors`
- `mlx_out/rfdetr-seg-medium-mlx.safetensors`
- `mlx_out/rfdetr-seg-large-mlx.safetensors`

## Run Locally

PyTorch/MPS detection:

```python
from PIL import Image
import numpy as np
from rfdetr import RFDETRMedium

image = np.array(Image.open("input.jpg").convert("RGB"))
model = RFDETRMedium(device="mps")
detections = model.predict(image, threshold=0.2)
```

MLX segmentation:

```python
from PIL import Image
import numpy as np
from rfdetr.mlx import RFDETRSegMediumMLX

image = np.array(Image.open("input.jpg").convert("RGB"))
model = RFDETRSegMediumMLX(weights_path="mlx_out/rfdetr-seg-medium-mlx.safetensors")
detections = model.predict(image, threshold=0.2)
```

The MLX wrappers are inference-only. Training remains in the PyTorch path.

## Export OmnimatteZero Masks

Use the built-in Omnimatte exporter to write `video.mp4` and `object_mask.mp4`
in the folder layout expected by [OmnimatteZeroEfficient](https://github.com/Ak-Gautam/OmnimatteZeroEfficient).

```bash
uv run --no-sync python -m rfdetr.omnimatte /path/to/input.mp4 \
  --omnimatte-root ../OmnimatteZeroEfficient/example_videos \
  --clip-name my_clip \
  --model seg-medium \
  --class-name person
```

This writes:

- `../OmnimatteZeroEfficient/example_videos/my_clip/video.mp4`
- `../OmnimatteZeroEfficient/example_videos/my_clip/object_mask.mp4`

The default output root is the sibling `OmnimatteZeroEfficient/example_videos`
directory if it exists next to this repo. Use `--output-dir` to override it with
an explicit clip directory. The first version is intentionally narrow: it exports
the object mask only, so Omnimatte can generate `total_mask.mp4` with its own
self-attention stage afterward.

## Model Summary

| Model | Task | Backend |
| :-- | :-- | :-- |
| `RFDETRMedium` | detection | PyTorch / MPS |
| `RFDETRLarge` | detection | PyTorch / MPS |
| `RFDETRSegMedium` | segmentation | PyTorch / MPS |
| `RFDETRSegLarge` | segmentation | PyTorch / MPS |
| `RFDETRMediumMLX` | detection | MLX |
| `RFDETRLargeMLX` | detection | MLX |
| `RFDETRSegMediumMLX` | segmentation | MLX |
| `RFDETRSegLargeMLX` | segmentation | MLX |

## Documentation

- [Technical Notes](TECHNICAL.md)
- [Minimal docs site entry](docs/index.md)

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Visit our [documentation website](https://rfdetr.roboflow.com) to learn more about how to use RF-DETR.

## License

Licensing is split by component:

- The open-source `rfdetr` package and Apache-designated model weights are licensed under Apache License 2.0. See [`LICENSE`](LICENSE).
- This fork ships only Apache 2.0 components and checkpoints.

## Acknowledgements

Our work is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159). Thanks to their authors for their excellent work!

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@misc{rf-detr,
    title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
    author={Isaac Robinson and Peter Robicheaux and Matvei Popov and Deva Ramanan and Neehar Peri},
    year={2025},
    eprint={2511.09554},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2511.09554},
}
```
