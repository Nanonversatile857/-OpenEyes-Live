"""Frame Sampler Engine — Stage 1 of the video pipeline.

Source: docs/VIDEO_PIPELINE.md — "Stage 1: Frame Sampler Engine" section;
        docs/ENGINE_SPEC.md — "Video Engine Specifications / 1. FrameSamplerEngine".

Reduces the input frame rate to a manageable level while preserving
temporal information. Pure NumPy implementation — no model download needed.

Deviation note: the doc's dynamic-mode formula
``adaptive = max(min_interval, max_interval - motion_score)`` samples *less*
when motion is high, which contradicts its stated intent ("Sample more when
motion is high"). This implementation uses
``adaptive = max_interval - motion * (max_interval - min_interval)``
so that higher motion yields a shorter sampling interval.
"""

import time
from typing import Any, Dict, Optional, Union

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError

FrameInput = Union[np.ndarray, Dict[str, Any]]


class FrameSamplerEngine(BaseEngine):
    """Samples frames from a video stream.

    Input:  np.ndarray (HWC, BGR) or Dict {"frame": np.ndarray, "timestamp": float}
    Output: EngineResult whose ``data`` is the sampled frame, or None (skip).

    Config:
        mode: "uniform" | "dynamic" | "keyframe_only" (default: "uniform")
        target_fps: int (default: 10) — uniform mode
        source_fps: int (default: 30) — assumed input fps for uniform mode
        min_interval: float (default: 0.5) — dynamic mode
        max_interval: float (default: 5.0) — dynamic mode
        keyframe_threshold: float (default: 0.15) — keyframe_only mode
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "mode": "uniform",
        "target_fps": 10,
        "source_fps": 30,
        "min_interval": 0.5,
        "max_interval": 5.0,
        "keyframe_threshold": 0.15,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self.frame_count: int = 0
        self.last_sample_time: float = 0.0
        self._last_gray: Optional[np.ndarray] = None

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="sampler",
            version="0.1.0",
            description="Samples frames from a video stream (uniform / dynamic / keyframe_only).",
            author="OpenEyes-Live",
            input_type="video_frame",
            output_type="video_frame",
            input_schema={"type": "np.ndarray (H, W, 3) BGR, or dict{frame, timestamp}"},
            output_schema={"type": "np.ndarray or None (skipped)"},
            size_mb=5,
            memory_mb=10,
            tags=["video", "sampling"],
        )

    def load(self) -> None:
        """Initialize sampler state. Idempotent."""
        self._loaded = True

    def process(self, input_data: FrameInput) -> EngineResult:
        """Process a single frame and decide whether to emit it.

        Returns:
            EngineResult with ``data`` = the frame (sampled) or None (skipped),
            and ``metadata["sampled"]`` indicating the decision.
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")

        frame, timestamp = self._unpack(input_data)
        self.frame_count += 1

        mode = str(self.config["mode"])
        if mode == "uniform":
            sampled = self._uniform_sample()
        elif mode == "dynamic":
            sampled = self._dynamic_sample(frame, timestamp)
        elif mode == "keyframe_only":
            sampled = self._keyframe_sample(frame)
        else:
            raise EngineProcessError(f"unknown sampler mode: {mode!r}")

        return EngineResult(
            data=frame if sampled else None,
            metadata={
                "engine": "sampler",
                "mode": mode,
                "frame_count": self.frame_count,
                "sampled": sampled,
                "timestamp": timestamp,
            },
        )

    def unload(self) -> None:
        """Reset sampler state. Idempotent."""
        self.frame_count = 0
        self.last_sample_time = 0.0
        self._last_gray = None
        self._loaded = False

    # === Sampling strategies ===

    def _uniform_sample(self) -> bool:
        """Emit every Nth frame, N = source_fps / target_fps."""
        interval = max(1, round(int(self.config["source_fps"]) / int(self.config["target_fps"])))
        return self.frame_count % interval == 0

    def _dynamic_sample(self, frame: np.ndarray, timestamp: float) -> bool:
        """Emit more often when motion is high, less when static."""
        motion = self._compute_motion(frame)
        min_iv = float(self.config["min_interval"])
        max_iv = float(self.config["max_interval"])
        adaptive = max(min_iv, max_iv - motion * (max_iv - min_iv))
        if timestamp - self.last_sample_time >= adaptive:
            self.last_sample_time = timestamp
            return True
        return False

    def _keyframe_sample(self, frame: np.ndarray) -> bool:
        """Emit only on scene changes (first frame always emitted)."""
        if self._last_gray is None:
            self._compute_motion(frame)  # initializes _last_gray
            return True
        motion = self._compute_motion(frame)
        return motion >= float(self.config["keyframe_threshold"])

    # === Helpers ===

    @staticmethod
    def _unpack(input_data: FrameInput) -> tuple[np.ndarray, float]:
        if isinstance(input_data, dict):
            frame = input_data.get("frame")
            timestamp = float(input_data.get("timestamp", time.time()))
        else:
            frame, timestamp = input_data, time.time()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise EngineProcessError("sampler expects a frame np.ndarray (H, W, C)")
        return frame, timestamp

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Downscale + grayscale via NumPy (keeps the engine dependency-free)."""
        small = frame[::8, ::8]
        return small.astype(np.float32).mean(axis=2)

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Motion score in [0, 1] from mean absolute frame difference."""
        gray = self._to_gray(frame)
        if self._last_gray is None or self._last_gray.shape != gray.shape:
            self._last_gray = gray
            return 0.0
        motion = float(np.abs(gray - self._last_gray).mean() / 255.0)
        self._last_gray = gray
        return motion
