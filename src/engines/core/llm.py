"""Language Engine — multi-modal language reasoning.

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 1. LanguageEngine";
        docs/API_REFERENCE.md — "Core Engines / LanguageEngine".

v0.1.0 scope: MOCK implementation. ``process()`` returns a fixed text
description to validate the interface; a real GGUF LLM (via
llama-cpp-python) is plugged in later.
"""

import time
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError

MOCK_RESPONSE = (
    "[mock] I see a scene with several visual features. "
    "A real language model will describe it in v0.2.0."
)


class LanguageEngine(BaseEngine):
    """Multi-modal language reasoning engine.

    Input:  Dict with ``visual_tokens`` (np.ndarray), optional ``audio_text``,
            ``speaker_id`` and ``prompt``.
    Output: EngineResult whose ``data`` is a natural-language string.

    Config:
        model_path: str (default: "./models/llm/qwen2.5-2b-q4.gguf")
        context_length: int (default: 4096)
        temperature: float (default: 0.7)
        max_tokens: int (default: 256)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/llm/qwen2.5-2b-q4.gguf",
        "context_length": 4096,
        "temperature": 0.7,
        "max_tokens": 256,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._model: Any = None  # llama_cpp.Llama instance in a future release

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="llm",
            version="0.1.0",
            description="Multi-modal language reasoning (mock in v0.1.0).",
            author="OpenEyes-Live",
            input_type="multi_modal_features",
            output_type="text",
            input_schema={
                "type": "object",
                "properties": {
                    "visual_tokens": "np.ndarray (T, D)",
                    "audio_text": "str, optional",
                    "speaker_id": "str, optional",
                    "prompt": "str, optional",
                },
                "required": ["visual_tokens"],
            },
            output_schema={"type": "string"},
            size_mb=400,
            memory_mb=500,
            tags=["core", "reasoning", "llm"],
        )

    def load(self) -> None:
        """Load the language model (mock). Idempotent."""
        if self._loaded:
            return
        # Mock: a real implementation would do
        #   from llama_cpp import Llama
        #   self._model = Llama(model_path=self.config["model_path"], ...)
        try:
            self._model = object()  # placeholder for the real model handle
        except Exception as exc:
            raise EngineLoadError(f"llm load failed: {exc}") from exc
        self._loaded = True

    def process(self, input_data: Dict[str, Any]) -> EngineResult:
        """Generate a natural-language description from multi-modal features.

        Args:
            input_data: Dict containing at least ``visual_tokens``
                (np.ndarray of shape (T, D)).

        Returns:
            EngineResult whose ``data`` is the response text.
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, dict) or "visual_tokens" not in input_data:
            raise EngineProcessError("llm expects a dict with a 'visual_tokens' key")

        visual_tokens = input_data["visual_tokens"]
        if not isinstance(visual_tokens, np.ndarray) or visual_tokens.ndim != 2:
            raise EngineProcessError("visual_tokens must be an np.ndarray of shape (T, D)")

        start = time.perf_counter()

        # Mock: summarize token statistics into a fixed response.
        num_tokens, dim = visual_tokens.shape
        prompt = input_data.get("prompt", "Describe what you see.")
        text = (
            f"{MOCK_RESPONSE} "
            f"(received {num_tokens} visual tokens of dim {dim}; prompt: '{prompt}')"
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=text,
            metadata={
                "engine": "llm",
                "num_visual_tokens": int(num_tokens),
                "has_audio_text": bool(input_data.get("audio_text")),
                "speaker_id": input_data.get("speaker_id"),
                "mock": True,
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free model resources. Idempotent."""
        self._model = None
        self._loaded = False
