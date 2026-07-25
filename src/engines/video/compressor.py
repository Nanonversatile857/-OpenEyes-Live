"""Token Compressor Engine — Stage 4 of the video pipeline.

Source: docs/VIDEO_PIPELINE.md — "Stage 4: Token Compressor Engine" section;
        docs/ENGINE_SPEC.md — "Video Engine Specifications / 4. TokenCompressorEngine".

Reduces the number of visual tokens fed to the LLM, speeding up inference.
The doc specifies a learnable projector; this v0.1.x implementation uses
deterministic NumPy stand-ins (chunk averaging for "projection", L2-norm
importance for "selection") until the real model ships.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError


class TokenCompressorEngine(BaseEngine):
    """Compresses visual tokens for LLM input.

    Input:  np.ndarray (visual tokens, shape (T, D))
    Output: EngineResult whose ``data`` has shape (target_tokens, D);
            passes through unchanged when T <= target_tokens.

    Config:
        target_tokens: int (default: 64)
        method: "projection" | "selection" | "hybrid" (default: "hybrid")
        preserve_temporal: bool (default: True) — keep time order in output
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "target_tokens": 64,
        "method": "hybrid",
        "preserve_temporal": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="compressor",
            version="0.1.0",
            description="Compresses visual tokens for LLM input (mock projector).",
            author="OpenEyes-Live",
            input_type="visual_tokens",
            output_type="visual_tokens_compressed",
            input_schema={"type": "np.ndarray", "shape": "(T, D)"},
            output_schema={"type": "np.ndarray", "shape": "(target_tokens, D)"},
            size_mb=20,
            memory_mb=30,
            tags=["video", "compression", "efficiency"],
        )

    def load(self) -> None:
        """Load the token projector (mock — nothing to load). Idempotent."""
        self._loaded = True

    def process(self, input_data: np.ndarray) -> EngineResult:
        """Compress visual tokens.

        Args:
            input_data: Visual tokens of shape (T, D).

        Returns:
            EngineResult whose ``data`` has shape (min(T, target_tokens), D).
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, np.ndarray) or input_data.ndim != 2:
            raise EngineProcessError("compressor expects tokens np.ndarray of shape (T, D)")

        tokens = input_data
        target = int(self.config["target_tokens"])

        start = time.perf_counter()
        if len(tokens) <= target:
            compressed, method_used = tokens, "passthrough"
        else:
            method = str(self.config["method"])
            if method == "projection":
                compressed = self._project(tokens, target)
            elif method == "selection":
                indices = self._select(tokens, target)
                compressed = tokens[indices]
            elif method == "hybrid":
                indices = self._select(tokens, min(2 * target, len(tokens)))
                compressed = self._project(tokens[indices], target)
            else:
                raise EngineProcessError(f"unknown compressor method: {method!r}")
            method_used = method
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=compressed.astype(np.float32, copy=False),
            metadata={
                "engine": "compressor",
                "method": method_used,
                "input_tokens": int(len(tokens)),
                "output_tokens": int(len(compressed)),
                "reduction": f"{1 - len(compressed) / len(tokens):.0%}",
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free projector resources. Idempotent."""
        self._loaded = False

    # === Compression strategies (mock) ===

    def _compute_importance(self, tokens: np.ndarray) -> np.ndarray:
        """Per-token importance via L2 norm (stand-in for learned scoring)."""
        return np.linalg.norm(tokens, axis=1)

    def _select(self, tokens: np.ndarray, k: int) -> List[int]:
        """Indices of the k most important tokens (time-ordered if configured)."""
        importance = self._compute_importance(tokens)
        indices = [int(i) for i in np.argsort(importance)[::-1][:k]]
        if bool(self.config["preserve_temporal"]):
            indices.sort()
        return indices

    def _project(self, tokens: np.ndarray, target: int) -> np.ndarray:
        """Chunk-average projection to ``target`` tokens (mock projector).

        A real implementation would use a learnable projection matrix;
        contiguous chunk averaging preserves temporal structure and is
        deterministic for testing.
        """
        boundaries = np.linspace(0, len(tokens), target + 1).astype(int)
        out = []
        for i in range(target):
            lo, hi = boundaries[i], max(boundaries[i + 1], boundaries[i] + 1)
            out.append(tokens[lo:hi].mean(axis=0))
        return np.stack(out)
