# Local Run Guide

## Setup

```bash
uv sync --all-groups --extra mlx
```

## Convert Built-in Checkpoints

```bash
uv run --no-sync python -m rfdetr.mlx.convert --model medium --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model large --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model seg-medium --output mlx_out
uv run --no-sync python -m rfdetr.mlx.convert --model seg-large --output mlx_out
```

## Run MLX Detection

```python
from PIL import Image
import numpy as np
from rfdetr.mlx import RFDETRMediumMLX

image = np.array(Image.open("input.jpg").convert("RGB"))
model = RFDETRMediumMLX(weights_path="mlx_out/rfdetr-medium-mlx.safetensors")
detections = model.predict(image, threshold=0.2)
```

## Run MLX Segmentation

```python
from PIL import Image
import numpy as np
from rfdetr.mlx import RFDETRSegMediumMLX

image = np.array(Image.open("input.jpg").convert("RGB"))
model = RFDETRSegMediumMLX(weights_path="mlx_out/rfdetr-seg-medium-mlx.safetensors")
detections = model.predict(image, threshold=0.2)
```

## Run PyTorch/MPS Detection

```python
from PIL import Image
import numpy as np
from rfdetr import RFDETRMedium

image = np.array(Image.open("input.jpg").convert("RGB"))
model = RFDETRMedium(device="mps")
detections = model.predict(image, threshold=0.2)
```
