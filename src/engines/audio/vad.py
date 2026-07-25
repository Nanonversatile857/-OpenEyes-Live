"""VAD Engine — Stage 1 of the audio pipeline (Voice Activity Detection).

Source: docs/AUDIO_PIPELINE.md — "Stage 1: VAD Engine" section;
        docs/ENGINE_SPEC.md — "Audio Engine Specifications / 1. VADEngine".

REAL implementation: Silero VAD (ONNX, ~2MB) running on onnxruntime,
fully offline. The documented state machine (speech passthrough,
trailing-silence segment end, max-duration cut) is preserved, so
downstream stages see no interface change.

Model: models/vad/silero_vad.onnx
  (https://huggingface.co/onnx-community/silero-vad)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError

# Chunks of trailing silence before a speech segment is considered ended
# (doc: ~90 ms of silence at 30 ms frames).
SILENCE_CHUNKS_TO_END = 3

# Silero VAD v5 expects exactly 512 samples (32 ms) per window at 16 kHz.
SILERO_WINDOW_16K = 512


class VADEngine(BaseEngine):
    """Voice Activity Detection (real Silero VAD on onnxruntime).

    Input:  np.ndarray (PCM audio, float32 in [-1, 1] or int16)
    Output: EngineResult whose ``data`` is the audio chunk when voice is
            detected, or None when silence. ``metadata["speech_prob"]``
            carries the model probability; ``metadata["segment_ended"]``
            marks the end of a speech segment.

    Config:
        model_path: str (default: "./models/vad/silero_vad.onnx")
        threshold: float (0.0-1.0, default: 0.5)
        sample_rate: int (default: 16000)
        frame_duration_ms: int (default: 30)
        min_speech_duration_ms: int (default: 250)
        max_speech_duration_s: int (default: 30)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/vad/silero_vad.onnx",
        "threshold": 0.5,
        "sample_rate": 16000,
        "frame_duration_ms": 30,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._session: Any = None  # onnxruntime.InferenceSession
        self._state: Optional[np.ndarray] = None
        self._sr: Optional[np.ndarray] = None
        self._window_buffer = np.zeros(0, dtype=np.float32)
        self._speech_buffer: List[np.ndarray] = []
        self._is_speaking: bool = False
        self._silence_count: int = 0
        self._speech_samples: int = 0

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="vad",
            version="0.1.0",
            description="Voice Activity Detection (Silero VAD, ONNX, offline).",
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
        """Load the Silero VAD ONNX model. Idempotent."""
        if self._loaded:
            return
        model_path = Path(str(self.config["model_path"]))
        if not model_path.exists():
            raise EngineLoadError(
                f"Silero VAD model not found: {model_path} — "
                f"run `openeyes install vad` or see models/README"
            )
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except ImportError as exc:
            raise EngineLoadError(
                "onnxruntime is required for the VAD engine; "
                "install it with `pip install onnxruntime`"
            ) from exc
        except Exception as exc:
            raise EngineLoadError(f"failed to load Silero VAD: {exc}") from exc

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(int(self.config["sample_rate"]), dtype=np.int64)
        self._loaded = True

    def process(self, input_data: np.ndarray) -> EngineResult:
        """Process one audio chunk and detect voice activity.

        Speech chunks pass through and are buffered; trailing silence ends
        the segment after ~90 ms; segments longer than
        ``max_speech_duration_s`` are cut.
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, np.ndarray) or input_data.ndim != 1:
            raise EngineProcessError("vad expects a 1-D PCM audio np.ndarray")

        start = time.perf_counter()

        chunk = input_data
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0
        chunk = chunk.astype(np.float32, copy=False)

        speech_prob = self._infer_prob(chunk)
        segment_ended = False
        passed: Optional[np.ndarray] = None

        if speech_prob > float(self.config["threshold"]):
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
                "model": "silero_vad",
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free the ONNX session and reset the state machine. Idempotent."""
        self._session = None
        self._state = None
        self._sr = None
        self._window_buffer = np.zeros(0, dtype=np.float32)
        self._reset_segment()
        self._loaded = False

    # === Real Silero inference ===

    def _infer_prob(self, chunk: np.ndarray) -> float:
        """Max speech probability across all complete 512-sample windows.

        Partial trailing windows stay buffered for the next chunk, so
        arbitrary chunk sizes are handled. Returns 0.0 when no complete
        window is available yet.
        """
        self._window_buffer = np.concatenate([self._window_buffer, chunk])
        n_windows = len(self._window_buffer) // SILERO_WINDOW_16K
        if n_windows == 0:
            return 0.0

        usable = n_windows * SILERO_WINDOW_16K
        samples, self._window_buffer = (
            self._window_buffer[:usable],
            self._window_buffer[usable:],
        )

        assert self._session is not None and self._state is not None
        best = 0.0
        for i in range(n_windows):
            window = samples[i * SILERO_WINDOW_16K:(i + 1) * SILERO_WINDOW_16K]
            out, self._state = self._session.run(
                ["output", "stateN"],
                {
                    "input": window[None, :],
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            best = max(best, float(out[0][0]))
        return best

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
