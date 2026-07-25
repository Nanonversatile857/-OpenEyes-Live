"""Speaker Engine — speaker recognition (real ERes2Net embeddings).

Source: docs/AUDIO_PIPELINE.md — "Stage 3: Speaker Engine";
        docs/ENGINE_SPEC.md — "Audio Engine Specifications".

REAL implementation: 3D-Speaker ERes2Net base (ONNX, ~38MB, zh-cn, 16kHz)
via sherpa-onnx's SpeakerEmbeddingExtractor, fully offline. The engine
extracts 512-d speaker embeddings and provides enroll/identify on top of
sherpa-onnx's SpeakerEmbeddingManager (cosine-similarity search).

Model: models/speaker/3dspeaker_eres2net_base_16k.onnx
  (https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)

Measured on desktop CPU: ~164ms per 5.6s utterance (~34x real-time);
same-speaker score 0.87 vs noise score 0.10 at the default 0.5 threshold.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError


class SpeakerEngine(BaseEngine):
    """Speaker recognition (real ERes2Net speaker embeddings).

    Input:  Dict with ``audio`` — np.ndarray of 16kHz mono PCM
            (float32 in [-1, 1] or int16).
    Output: EngineResult whose ``data`` is the 512-d embedding (np.ndarray,
            float32). Use :meth:`enroll` / :meth:`identify` for recognition.

    Config:
        model_path: str (default: models/speaker/3dspeaker_eres2net_base_16k.onnx)
        sample_rate: int (default: 16000)
        num_threads: int (default: 2)
        threshold: float (default: 0.5) — cosine-similarity accept threshold
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/speaker/3dspeaker_eres2net_base_16k.onnx",
        "sample_rate": 16000,
        "num_threads": 2,
        "threshold": 0.5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._extractor: Any = None  # sherpa_onnx.SpeakerEmbeddingExtractor
        self._manager: Any = None  # sherpa_onnx.SpeakerEmbeddingManager

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="speaker",
            version="0.3.0",
            description="Speaker recognition (3D-Speaker ERes2Net base, offline).",
            author="OpenEyes-Live",
            input_type="audio_pcm",
            output_type="speaker_embedding",
            input_schema={
                "type": "object",
                "properties": {
                    "audio": "np.ndarray 16kHz mono (float32 [-1,1] or int16)",
                },
                "required": ["audio"],
            },
            output_schema={"type": "np.ndarray", "shape": [512]},
            size_mb=38,
            memory_mb=100,
            tags=["audio", "speaker", "embedding", "recognition"],
        )

    def load(self) -> None:
        """Load the embedding model (38MB). Idempotent."""
        if self._loaded:
            return
        model_path = Path(str(self.config["model_path"]))
        if not model_path.exists():
            raise EngineLoadError(
                f"speaker model not found: {model_path} — "
                f"run `openeyes install speaker`"
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineLoadError(
                "sherpa-onnx is required for the speaker engine; "
                "install it with `pip install sherpa-onnx`"
            ) from exc
        try:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path),
                num_threads=int(self.config["num_threads"]),
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
        except EngineLoadError:
            raise
        except Exception as exc:
            raise EngineLoadError(f"failed to load speaker model: {exc}") from exc
        self._loaded = True

    def process(self, input_data: Dict[str, Any]) -> EngineResult:
        """Extract a 512-d speaker embedding from an utterance.

        Args:
            input_data: Dict containing ``audio`` (16kHz mono PCM,
                float32 in [-1, 1] or int16).

        Returns:
            EngineResult whose ``data`` is the embedding (np.ndarray float32).
        """
        if not self._loaded or self._extractor is None:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, dict) or "audio" not in input_data:
            raise EngineProcessError("speaker expects a dict with an 'audio' key")

        audio = self._normalize_audio(input_data["audio"])
        start = time.perf_counter()
        embedding = self._embed(audio)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=embedding,
            metadata={
                "engine": "speaker",
                "embedding_dim": int(embedding.shape[0]),
                "audio_seconds": round(len(audio) / int(self.config["sample_rate"]), 2),
            },
            latency_ms=latency_ms,
        )

    def enroll(self, name: str, audio: np.ndarray) -> np.ndarray:
        """Enroll a speaker under ``name``. Returns the embedding."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not name:
            raise EngineProcessError("speaker name must be non-empty")
        embedding = self._embed(self._normalize_audio(audio))
        self.add_embedding(name, embedding)
        return embedding

    def add_embedding(self, name: str, embedding: Any) -> None:
        """Register a precomputed embedding (e.g. loaded from a speaker DB)."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not name:
            raise EngineProcessError("speaker name must be non-empty")
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.shape[0] != self._extractor.dim:
            raise EngineProcessError(
                f"embedding dim mismatch: got {vec.shape[0]}, "
                f"expected {self._extractor.dim}"
            )
        ok = self._manager.add(name, vec.tolist())
        if not ok:
            raise EngineProcessError(f"failed to enroll speaker '{name}'")

    def identify(self, audio: np.ndarray) -> Tuple[str, float]:
        """Identify the closest enrolled speaker.

        Returns:
            (name, score) — name is "" when no enrolled speaker clears the
            configured threshold. score is the cosine similarity with the
            named speaker (or with the best-matching one when rejected).
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if self._manager.num_speakers == 0:
            raise EngineProcessError("no enrolled speakers")
        embedding = self._embed(self._normalize_audio(audio))
        name = self._manager.search(embedding.tolist(),
                                    threshold=float(self.config["threshold"]))
        if name:
            return name, float(self._manager.score(name, embedding.tolist()))
        # Rejected — report the best raw score for observability.
        best = max(
            (float(self._manager.score(n, embedding.tolist()))
             for n in self._manager.all_speakers),
            default=0.0,
        )
        return "", best

    def remove(self, name: str) -> bool:
        """Remove an enrolled speaker. Returns True when it existed."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        return bool(self._manager.remove(name))

    @property
    def speakers(self) -> List[str]:
        """Names of all enrolled speakers."""
        if not self._loaded or self._manager is None:
            return []
        return list(self._manager.all_speakers)

    def unload(self) -> None:
        """Free the model. Idempotent."""
        self._manager = None
        self._extractor = None
        self._loaded = False

    # === Internals ===

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(int(self.config["sample_rate"]), audio)
        stream.input_finished()
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)

    @staticmethod
    def _normalize_audio(audio: Any) -> np.ndarray:
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or len(audio) == 0:
            raise EngineProcessError(
                "audio must be a non-empty 1-D np.ndarray (16kHz mono)"
            )
        if audio.dtype == np.int16:
            return (audio.astype(np.float32) / 32768.0)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio
