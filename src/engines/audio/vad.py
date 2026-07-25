"""VAD Engine — Stage 1 of the audio pipeline (Voice Activity Detection).

Source: docs/AUDIO_PIPELINE.md — "Stage 1: VAD Engine" section;
        docs/ENGINE_SPEC.md — "Audio Engine Specifications / 1. VADEngine".

Detects when someone is speaking and drops silence, preventing unnecessary
ASR processing. The doc specifies Silero VAD (ONNX, ~2MB); this v0.2.0
implementation uses an RMS-energy-based mock scorer (pure NumPy, fully
offline) until the real model ships. The state machine follows the
documented interface exactly.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError

# Chunks of trailing silence before a speech segment is considered ended
# (doc: ~90 ms of silence at 30 ms frames).
SILENCE_CHUNKS_TO_END = 3


class VADEngine(BaseEngine):
    """Voice Activity Detection.

    Input:  np.ndarray (PCM audio, float32 in [-1, 1] or int16)
    Output: EngineResult whose ``data`` is the audio chunk when voice is
            detected, or None when silence. ``metadata["speech_prob"]``
            carries the mock probability; ``metadata["segment_ended"]``
            marks the end of a speech segment.

    Config:
        threshold: float (0.0-1.0, default: 0.5)
        sample_rate: int (default: 16000)
        frame_duration_ms: int (default: 30)
        min_speech_duration_ms: int (default: 250)
        max_speech_duration_s: int (default: 30)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "threshold": 0.5,
        "sample_rate": 16000,
        "frame_duration_ms": 30,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._speech_buffer: List[np.ndarray] = []
        self._is_speaking: bool = False
        self._silence_count: int = 0
        self._speech_samples: int = 0

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="vad",
            version="0.1.0",
            description="Voice Activity Detection (RMS-energy mock scorer).",
            author="OpenEyes-Live",
            input_type="audio_pcm",
            output_type="audio_pcm_voice",
            input_schema={"type": "np.ndarray", "dtype": "float32 | int16"},
            output_schema={"type": "np.ndarray or None (silence)"},
            size_mb=2,
            memory_mb=10,
            tags=["audio", "vad", "detection"],
        )

    def load(self) -> None:
        """Load the VAD model (mock scorer — nothing to load). Idempotent."""
        self._loaded = True

    def process(self, input_data: np.ndarray) -> EngineResult:
        """Process one audio chunk and detect voice activity.

        Follows the documented state machine: speech chunks are passed
        through and buffered; trailing silence ends the segment after
        ~90 ms; segments longer than ``max_speech_duration_s`` are cut.
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, np.ndarray) or input_data.ndim != 1:
            raise EngineProcessError("vad expects a 1-D PCM audio np.ndarray")

        start = time.perf_counter()

        # Convert to float32 if needed (doc interface).
        chunk = input_data
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0

        speech_prob = self._mock_vad_score(chunk)
        segment_ended = False
        passed: Optional[np.ndarray] = None

        if speech_prob > float(self.config["threshold"]):
            # Voice detected.
            self._is_speaking = True
            self._silence_count = 0
            self._speech_buffer.append(chunk)
            self._speech_samples += len(chunk)
            passed = chunk

            max_samples = int(
                float(self.config["max_speech_duration_s"])
                * int(self.config["sample_rate"])
            )
            if self._speech_samples >= max_samples:
                segment_ended = True
                self._reset_segment()
        else:
            # Silence.
            if self._is_speaking:
                self._silence_count += 1
                self._speech_buffer.append(chunk)  # keep for context
                if self._silence_count > SILENCE_CHUNKS_TO_END:
                    segment_ended = True
                    self._reset_segment()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return EngineResult(
            data=passed,
            metadata={
                "engine": "vad",
                "speech_prob": round(speech_prob, 4),
                "is_speaking": self._is_speaking,
                "segment_ended": segment_ended,
                "scorer": "rms_energy_mock",
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free resources and reset the state machine. Idempotent."""
        self._reset_segment()
        self._loaded = False

    # === Mock scorer ===

    @staticmethod
    def _mock_vad_score(chunk: np.ndarray) -> float:
        """Speech probability proxy from RMS energy.

        Calibration: typical speech RMS ≈ 0.02–0.3 maps to ~0.4–1.0;
        background noise RMS < 0.005 maps below the default 0.5 threshold.
        """
        if len(chunk) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        return min(1.0, rms * 20.0)

    def _reset_segment(self) -> None:
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_count = 0
        self._speech_samples = 0

    @property
    def buffered_speech(self) -> Optional[np.ndarray]:
        """Concatenated audio of the current/last speech segment, if any."""
        if not self._speech_buffer:
            return None
        return np.concatenate(self._speech_buffer)
