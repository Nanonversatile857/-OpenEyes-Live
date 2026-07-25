"""Visual Encoder Engine — Stage 3 of the video pipeline.

Source: docs/VIDEO_PIPELINE.md — "Stage 3: Visual Encoder Engine" section;
        docs/ENGINE_SPEC.md — "Video Engine Specifications / 3. VisualEncoderEngine".

v0.1.0 scope: MOCK implementation. ``process()`` returns random vectors to
validate the interface; a real VideoMamba/CLIP model is plugged in later.
"""

import time
from typing import Any, Dict, List

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError


class _MockEncoderModel:
    """Stand-in for a real visual encoder (VideoMamba-T / CLIP).

    Returns a deterministic-shape random embedding per frame.
    """

    def __init__(self, model_name: str, embedding_dim: int, seed: int = 42) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Mock-encode a single frame into an embedding vector."""
        return self._rng.standard_normal(self.embedding_dim).astype(np.float32)


class VisualEncoderEngine(BaseEngine):
    """Encodes key frames into visual embeddings.

    Input:  List[np.ndarray] (key frames, RGB, HWC)
    Output: EngineResult with visual embeddings, shape (T, embedding_dim)

    Config:
        model_name: "clip" | "videomamba_t" | "videomamba_m" (default: "videomamba_t")
        embedding_dim: int (default: 512)
        use_fp16: bool (default: True)
        temporal_aggregation: "mean" | "max" | "attention" (default: "attention")
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_name": "videomamba_t",
        "embedding_dim": 512,
        "use_fp16": True,
        "temporal_aggregation": "attention",
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._model: _MockEncoderModel | None = None

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="encoder",
            version="0.1.0",
            description="Encodes frames into visual embeddings (mock in v0.1.0).",
            author="OpenEyes-Live",
            input_type="frames_rgb",
            output_type="visual_embeddings",
            input_schema={"type": "array", "items": "np.ndarray (H, W, 3) RGB"},
            output_schema={"type": "np.ndarray", "shape": "(T, embedding_dim)"},
            size_mb=self.size_mb,
            memory_mb=self.memory_usage_mb,
            tags=["video", "encoding", "vision"],
        )

    @property
    def size_mb(self) -> int:
        sizes = {"clip": 200, "videomamba_t": 150, "videomamba_m": 280}
        return sizes.get(str(self.config["model_name"]), 200)

    @property
    def memory_usage_mb(self) -> int:
        return self.size_mb + 50  # Model + batch memory

    def load(self) -> None:
        """Load the visual encoder model (mock). Idempotent."""
        if self._loaded:
            return
        try:
            self._model = _MockEncoderModel(
                model_name=str(self.config["model_name"]),
                embedding_dim=int(self.config["embedding_dim"]),
            )
        except Exception as exc:
            raise EngineLoadError(f"encoder load failed: {exc}") from exc
        self._loaded = True

    def process(self, input_data: List[np.ndarray]) -> EngineResult:
        """Encode frames into visual embeddings.

        Args:
            input_data: List of key frames (RGB, HWC).

        Returns:
            EngineResult whose ``data`` is an np.ndarray of shape
            (T, embedding_dim) where T = len(input_data).
        """
        if not self._loaded or self._model is None:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, list) or len(input_data) == 0:
            raise EngineProcessError("encoder expects a non-empty list of frames")

        start = time.perf_counter()
        embeddings = [self._model.encode(frame) for frame in input_data]
        data = np.stack(embeddings)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=data,
            metadata={
                "engine": "encoder",
                "model_name": self.config["model_name"],
                "num_frames": len(input_data),
                "embedding_dim": data.shape[1],
                "mock": True,
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free model resources. Idempotent."""
        self._model = None
        self._loaded = False
