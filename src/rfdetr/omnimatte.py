# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Export RF-DETR segmentation masks in OmnimatteZero-ready layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from rfdetr import RFDETRSegLarge, RFDETRSegMedium
from rfdetr.util.coco_classes import COCO_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OMNIMATTE_ROOT = REPO_ROOT.parent / "OmnimatteZeroEfficient" / "example_videos"
DEFAULT_MLX_OUTPUT_DIR = REPO_ROOT / "mlx_out"


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def resolve_output_dir(
    *,
    input_path: str | Path,
    output_dir: str | Path | None,
    omnimatte_root: str | Path | None,
    clip_name: str | None,
) -> Path:
    """Resolve the Omnimatte clip directory for exported assets."""
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    root = Path(omnimatte_root).expanduser() if omnimatte_root is not None else DEFAULT_OMNIMATTE_ROOT
    resolved_clip_name = clip_name or Path(input_path).stem
    return root.resolve() / resolved_clip_name


def get_class_name_map(model: Any) -> dict[int, str]:
    """Return a best-effort class-id to class-name mapping for filtering."""
    class_names = getattr(model, "class_names", None)
    if isinstance(class_names, dict) and class_names:
        return {int(class_id): str(name) for class_id, name in class_names.items()}
    return dict(COCO_CLASSES)


def select_mask(
    detections: sv.Detections,
    *,
    class_id: int | None,
    class_name: str | None,
    class_name_map: dict[int, str],
    strategy: str,
) -> np.ndarray | None:
    """Select one mask from an RF-DETR segmentation result."""
    masks = getattr(detections, "mask", None)
    if masks is None:
        raise ValueError("Detections do not include masks. Use an RF-DETR segmentation model.")

    class_ids = detections.class_id
    confidences = detections.confidence
    boxes = detections.xyxy
    requested_name = _normalize_label(class_name) if class_name is not None else None

    candidates: list[tuple[float, int]] = []
    for index, mask in enumerate(masks):
        candidate_class_id = None if class_ids is None else int(class_ids[index])
        if class_id is not None and candidate_class_id != class_id:
            continue
        if requested_name is not None:
            candidate_name = class_name_map.get(candidate_class_id)
            if candidate_name is None or _normalize_label(candidate_name) != requested_name:
                continue

        if strategy == "largest":
            priority = float(np.count_nonzero(mask))
            tie_breaker = float(confidences[index]) if confidences is not None else 0.0
        elif strategy == "score":
            priority = float(confidences[index]) if confidences is not None else 0.0
            if boxes is not None:
                x1, y1, x2, y2 = boxes[index]
                tie_breaker = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            else:
                tie_breaker = float(np.count_nonzero(mask))
        else:
            raise ValueError(f"Unsupported selection strategy: {strategy}")

        candidates.append((priority * 1_000_000.0 + tie_breaker, index))

    if not candidates:
        return None

    _, selected_index = max(candidates)
    return np.asarray(masks[selected_index], dtype=bool)


def build_mask_frame(mask: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a boolean mask into Omnimatte's white-on-black RGB frame format."""
    height, width = frame_shape[:2]
    if mask.shape != (height, width):
        raise ValueError(f"Mask shape {mask.shape} does not match frame size {(height, width)}")

    binary = np.where(mask, 255, 0).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def create_model(model_name: str, *, weights_path: str | None, device: str) -> Any:
    """Instantiate one of the retained segmentation models."""
    if model_name == "seg-medium":
        kwargs = {"device": device}
        if weights_path is not None:
            kwargs["pretrain_weights"] = weights_path
        return RFDETRSegMedium(**kwargs)

    if model_name == "seg-large":
        kwargs = {"device": device}
        if weights_path is not None:
            kwargs["pretrain_weights"] = weights_path
        return RFDETRSegLarge(**kwargs)

    if model_name == "seg-medium-mlx":
        from rfdetr.mlx import RFDETRSegMediumMLX, RFDETRSegMediumMLXConfig

        return RFDETRSegMediumMLX.from_pretrained(
            RFDETRSegMediumMLXConfig,
            weights_path=weights_path,
            output_dir=DEFAULT_MLX_OUTPUT_DIR,
        )

    if model_name == "seg-large-mlx":
        from rfdetr.mlx import RFDETRSegLargeMLX, RFDETRSegLargeMLXConfig

        return RFDETRSegLargeMLX.from_pretrained(
            RFDETRSegLargeMLXConfig,
            weights_path=weights_path,
            output_dir=DEFAULT_MLX_OUTPUT_DIR,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def export_omnimatte_assets(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model: Any,
    threshold: float,
    class_id: int | None,
    class_name: str | None,
    strategy: str,
) -> tuple[Path, Path]:
    """Run segmentation on a video and export Omnimatte-compatible assets."""
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    clip_dir = Path(output_dir).expanduser().resolve()
    clip_dir.mkdir(parents=True, exist_ok=True)

    target_video_path = clip_dir / "video.mp4"
    target_mask_path = clip_dir / "object_mask.mp4"
    if source_path != target_video_path:
        shutil.copy2(source_path, target_video_path)

    video_info = sv.VideoInfo.from_video_path(str(source_path))
    class_name_map = get_class_name_map(model)
    selected_frames = 0

    with sv.VideoSink(str(target_mask_path), video_info=video_info) as sink:
        for frame_bgr in sv.get_video_frames_generator(str(source_path)):
            frame_rgb = frame_bgr[:, :, ::-1]
            detections = model.predict(frame_rgb, threshold=threshold)
            mask = select_mask(
                detections,
                class_id=class_id,
                class_name=class_name,
                class_name_map=class_name_map,
                strategy=strategy,
            )
            if mask is None:
                mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
            else:
                selected_frames += 1

            sink.write_frame(build_mask_frame(mask, frame_bgr.shape))

    if selected_frames == 0:
        raise RuntimeError(
            "No matching detections were found across the video. "
            "Try lowering --threshold or adjusting --class-id / --class-name."
        )

    return target_video_path, target_mask_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Omnimatte export."""
    parser = argparse.ArgumentParser(
        description="Export RF-DETR segmentation masks in OmnimatteZero-ready layout."
    )
    parser.add_argument("input", help="Path to the source video.")
    parser.add_argument(
        "--output-dir",
        help="Explicit Omnimatte clip directory to write. Defaults to the sibling OmnimatteZeroEfficient repo.",
    )
    parser.add_argument(
        "--omnimatte-root",
        help="Root directory containing Omnimatte clip folders, usually OmnimatteZeroEfficient/example_videos.",
    )
    parser.add_argument("--clip-name", help="Clip folder name under --omnimatte-root. Defaults to the input stem.")
    parser.add_argument(
        "--model",
        choices=["seg-medium", "seg-large", "seg-medium-mlx", "seg-large-mlx"],
        default="seg-medium",
        help="Segmentation model to run.",
    )
    parser.add_argument("--weights", help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Device for PyTorch segmentation models. Ignored for MLX models.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum confidence score.")
    parser.add_argument("--class-id", type=int, help="Optional class-id filter.")
    parser.add_argument("--class-name", help="Optional class-name filter, e.g. 'person'.")
    parser.add_argument(
        "--strategy",
        choices=("largest", "score"),
        default="largest",
        help="How to pick one instance when multiple masks match the filters.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    output_dir = resolve_output_dir(
        input_path=args.input,
        output_dir=args.output_dir,
        omnimatte_root=args.omnimatte_root,
        clip_name=args.clip_name,
    )
    model = create_model(args.model, weights_path=args.weights, device=args.device)
    video_path, mask_path = export_omnimatte_assets(
        input_path=args.input,
        output_dir=output_dir,
        model=model,
        threshold=args.threshold,
        class_id=args.class_id,
        class_name=args.class_name,
        strategy=args.strategy,
    )
    print(f"Wrote Omnimatte assets to {output_dir}")
    print(f"Video: {video_path}")
    print(f"Object mask: {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
