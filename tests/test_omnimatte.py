# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from rfdetr.omnimatte import build_mask_frame, export_omnimatte_assets, resolve_output_dir, select_mask


def test_resolve_output_dir_uses_omnimatte_root_and_clip_name(tmp_path: Path) -> None:
    resolved = resolve_output_dir(
        input_path=tmp_path / "demo.mp4",
        output_dir=None,
        omnimatte_root=tmp_path / "example_videos",
        clip_name="custom-clip",
    )

    assert resolved == (tmp_path / "example_videos" / "custom-clip").resolve()


def test_select_mask_filters_by_class_name_and_uses_largest_strategy() -> None:
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 3, 3], [0, 0, 2, 2]], dtype=np.float32),
        confidence=np.asarray([0.4, 0.9], dtype=np.float32),
        class_id=np.asarray([1, 1], dtype=np.int32),
        mask=np.asarray(
            [
                [[True, True, False], [True, True, False], [False, False, False]],
                [[True, False, False], [False, False, False], [False, False, False]],
            ],
            dtype=bool,
        ),
    )

    selected = select_mask(
        detections,
        class_id=None,
        class_name="person",
        class_name_map={1: "person"},
        strategy="largest",
    )

    assert selected is not None
    assert int(np.count_nonzero(selected)) == 4


def test_build_mask_frame_returns_three_channel_uint8() -> None:
    frame = build_mask_frame(
        np.asarray([[True, False], [False, True]], dtype=bool),
        frame_shape=(2, 2, 3),
    )

    assert frame.dtype == np.uint8
    assert frame.shape == (2, 2, 3)
    assert frame[0, 0].tolist() == [255, 255, 255]
    assert frame[0, 1].tolist() == [0, 0, 0]


def test_export_omnimatte_assets_copies_video_and_writes_mask_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"video-bytes")
    output_dir = tmp_path / "example_videos" / "clip"
    written_frames: list[np.ndarray] = []

    class _FakeModel:
        class_names = {1: "person"}

        def __init__(self) -> None:
            self._calls = 0

        def predict(self, frame_rgb: np.ndarray, threshold: float = 0.5) -> sv.Detections:
            del frame_rgb, threshold
            self._calls += 1
            if self._calls == 1:
                return sv.Detections(
                    xyxy=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
                    confidence=np.asarray([0.9], dtype=np.float32),
                    class_id=np.asarray([1], dtype=np.int32),
                    mask=np.asarray([[[True, False], [False, False]]], dtype=bool),
                )
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                mask=np.empty((0, 2, 2), dtype=bool),
            )

    class _FakeVideoSink:
        def __init__(self, target_path: str, video_info: sv.VideoInfo, codec: str = "mp4v") -> None:
            del video_info, codec
            self.target_path = Path(target_path)

        def __enter__(self) -> "_FakeVideoSink":
            return self

        def write_frame(self, frame: np.ndarray) -> None:
            written_frames.append(frame.copy())

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
            del exc_type, exc_value, traceback
            self.target_path.write_bytes(b"mask-video")
            return False

    monkeypatch.setattr("rfdetr.omnimatte.sv.VideoInfo.from_video_path", lambda _path: sv.VideoInfo(2, 2, 24, 2))
    monkeypatch.setattr(
        "rfdetr.omnimatte.sv.get_video_frames_generator",
        lambda _path: iter(
            [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2, 3), dtype=np.uint8) * 127,
            ]
        ),
    )
    monkeypatch.setattr("rfdetr.omnimatte.sv.VideoSink", _FakeVideoSink)

    video_path, mask_path = export_omnimatte_assets(
        input_path=source_video,
        output_dir=output_dir,
        model=_FakeModel(),
        threshold=0.35,
        class_id=None,
        class_name="person",
        strategy="largest",
    )

    assert video_path == output_dir / "video.mp4"
    assert mask_path == output_dir / "object_mask.mp4"
    assert video_path.read_bytes() == b"video-bytes"
    assert mask_path.read_bytes() == b"mask-video"
    assert len(written_frames) == 2
    assert written_frames[0][0, 0].tolist() == [255, 255, 255]
    assert np.count_nonzero(written_frames[1]) == 0
