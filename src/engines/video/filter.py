"""Frame Filter Engine — Stage 2 of the video pipeline.

Source: docs/VIDEO_PIPELINE.md — "Stage 2: Frame Filter Engine" section;
        docs/ENGINE_SPEC.md — "Video Engine Specifications / 2. FrameFilterEngine".

Selects the most informative frames for encoding. The doc specifies a ~15MB
attention scoring model (Mobile-VideoGPT style); this v0.1.x implementation
uses a dependency-free NumPy heuristic scorer (sharpness + colorfulness) as
a stand-in until the real model ships. The selection methods
(attention / diversity / hybrid) follow the documented semantics.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError


class FrameFilterEngine(BaseEngine):
    """Selects key frames using attention-style scoring.

    Input:  List[np.ndarray] (sampled frames)
    Output: EngineResult whose ``data`` is the list of selected key frames.

    Config:
        top_k: int (default: 8) — number of frames to select
        score_threshold: float (default: 0.3) — minimum score (attention/hybrid)
        method: "attention" | "diversity" | "hybrid" (default: "attention")
        buffer_size: int (default: 30) — max frames accepted per process call
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "top_k": 8,
        "score_threshold": 0.3,
        "method": "attention",
        "buffer_size": 30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="filter",
            version="0.1.0",
            description="Selects key frames via heuristic attention scoring (mock scorer).",
            author="OpenEyes-Live",
            input_type="video_frames",
            output_type="video_frames",
            input_schema={"type": "array", "items": "np.ndarray (H, W, 3)"},
            output_schema={"type": "array", "items": "np.ndarray (selected, <= top_k)"},
            size_mb=15,
            memory_mb=50,
            tags=["video", "filtering", "attention"],
        )

    def load(self) -> None:
        """Load the scoring model (mock scorer — nothing to load). Idempotent."""
        self._loaded = True

    def process(self, input_data: List[np.ndarray]) -> EngineResult:
        """Score a batch of frames and select the key frames.

        Args:
            input_data: List of sampled frames.

        Returns:
            EngineResult whose ``data`` is the selected frames (time-ordered),
            with per-frame scores in ``metadata["scores"]``.
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, list) or len(input_data) == 0:
            raise EngineProcessError("filter expects a non-empty list of frames")

        frames = input_data[: int(self.config["buffer_size"])]
        top_k = int(self.config["top_k"])

        start = time.perf_counter()
        self._frames = frames  # needed by diversity selection
        scores = self._compute_scores(frames)

        if len(frames) <= top_k:
            indices = list(range(len(frames)))
        else:
            indices = self._select_top_k(scores)
        latency_ms = (time.perf_counter() - start) * 1000.0

        selected = [frames[i] for i in indices]
        return EngineResult(
            data=selected,
            metadata={
                "engine": "filter",
                "method": self.config["method"],
                "input_frames": len(frames),
                "selected_indices": indices,
                "scores": scores.tolist(),
                "scorer": "heuristic_mock",
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free resources. Idempotent."""
        self._loaded = False

    # === Scoring (mock scorer, NumPy only) ===

    def _compute_scores(self, frames: List[np.ndarray]) -> np.ndarray:
        """Score each frame by information content.

        Heuristic stand-in for the documented attention model:
        60% sharpness (gradient energy) + 40% colorfulness (channel std),
        normalized to [0, 1] across the batch.
        """
        raw = np.array([self._score_frame(f) for f in frames], dtype=np.float64)
        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-12:
            return np.full(len(frames), 0.5, dtype=np.float64)
        return (raw - lo) / (hi - lo)

    @staticmethod
    def _score_frame(frame: np.ndarray) -> float:
        small = frame[::4, ::4].astype(np.float32)
        gray = small.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        sharpness = float(gx + gy)
        colorfulness = float(small.std(axis=2).mean())
        return 0.6 * sharpness + 0.4 * colorfulness

    # === Selection strategies ===

    def _select_top_k(self, scores: np.ndarray) -> List[int]:
        method = str(self.config["method"])
        if method == "attention":
            return self._attention_selection(scores)
        if method == "diversity":
            return self._diversity_selection(scores, list(range(len(scores))))
        if method == "hybrid":
            return self._hybrid_selection(scores)
        raise EngineProcessError(f"unknown filter method: {method!r}")

    def _attention_selection(self, scores: np.ndarray) -> List[int]:
        """Top-k frames above the score threshold, by score."""
        top_k = int(self.config["top_k"])
        threshold = float(self.config["score_threshold"])
        order = np.argsort(scores)[::-1]
        chosen = [i for i in order if scores[i] >= threshold][:top_k]
        if not chosen:  # guarantee at least one frame
            chosen = [int(order[0])]
        return sorted(int(i) for i in chosen)

    def _diversity_selection(self, scores: np.ndarray, candidates: List[int]) -> List[int]:
        """Greedy max-min-distance selection seeded by the best score."""
        top_k = int(self.config["top_k"])
        if len(candidates) <= top_k:
            return sorted(int(i) for i in candidates)

        features = {i: self._feature(self._frames[i]) for i in candidates}
        seed = max(candidates, key=lambda i: scores[i])
        selected = [seed]
        remaining = [i for i in candidates if i != seed]

        while remaining and len(selected) < top_k:
            nxt = max(
                remaining,
                key=lambda i: min(
                    float(np.abs(features[i] - features[j]).mean()) for j in selected
                ),
            )
            selected.append(nxt)
            remaining.remove(nxt)
        return sorted(int(i) for i in selected)

    def _hybrid_selection(self, scores: np.ndarray) -> List[int]:
        """Attention shortlist (2 * top_k, thresholded) then diversity pick."""
        top_k = int(self.config["top_k"])
        threshold = float(self.config["score_threshold"])
        order = [int(i) for i in np.argsort(scores)[::-1]]
        shortlist = [i for i in order if scores[i] >= threshold][: 2 * top_k]
        if len(shortlist) < top_k:
            shortlist = order[: min(2 * top_k, len(order))]
        return self._diversity_selection(scores, shortlist)

    # === Helpers ===

    # Kept on the instance so _diversity_selection can re-derive features
    # for the candidate pool without re-passing frames around.
    _frames: List[np.ndarray]

    @staticmethod
    def _feature(frame: np.ndarray) -> np.ndarray:
        """Compact frame signature for diversity distance (16x16 gray)."""
        h, w = frame.shape[:2]
        ys = np.linspace(0, h - 1, 16).astype(int)
        xs = np.linspace(0, w - 1, 16).astype(int)
        small = frame[np.ix_(ys, xs)].astype(np.float32).mean(axis=2)
        return small.flatten() / 255.0

