"""ASR Engine — Stage 2 of the audio pipeline (Speech-to-Text).

Source: docs/AUDIO_PIPELINE.md — "Stage 2: ASR Engine" section;
        docs/ENGINE_SPEC.md — "Audio Engine Specifications / 2. ASREngine";
        docs/API_REFERENCE.md — "Audio Engines / ASREngine".

REAL implementation: SenseVoice (int8 ONNX, ~234MB) via sherpa-onnx,
fully offline — models load from local paths, zero network calls at
runtime. Punctuation is produced natively by SenseVoice.

Model: models/asr/sense-voice/model.int8.onnx + tokens.txt
  (https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError


class ASREngine(BaseEngine):
    """Speech-to-Text with punctuation (real SenseVoice via sherpa-onnx).

    Input:  np.ndarray (voice PCM, float32, 16 kHz mono)
    Output: EngineResult whose ``data`` is the transcribed text (str).

    Config:
        model_path: str (default: "./models/asr/sense-voice/model.int8.onnx")
        tokens_path: str (default: "./models/asr/sense-voice/tokens.txt")
        model_name: "sense_voice" | "parakeet" | "whisper_tiny"
            (default: "sense_voice"; only sense_voice is implemented)
        language: "zh" | "en" | "ja" | "ko" | "yue" | "auto" (default: "auto")
        punctuation_enabled: bool (default: True) — SenseVoice punctuates natively
        use_itn: bool (default: True) — inverse text normalization
        num_threads: int (default: 4)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/asr/sense-voice/model.int8.onnx",
        "tokens_path": "./models/asr/sense-voice/tokens.txt",
        "model_name": "sense_voice",
        "language": "auto",
        "punctuation_enabled": True,
        "use_itn": True,
        "num_threads": 4,
    }

    # Rough per-model sizes (docs/AUDIO_PIPELINE.md), MB.
    _SIZES = {"sense_voice": 229, "parakeet": 150, "whisper_tiny": 80}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._recognizer: Any = None  # sherpa_onnx.OfflineRecognizer

    @property
    def metadata(self) -> EngineMetadata:
        size = self._SIZES.get(str(self.config["model_name"]), 229) + 5
        return EngineMetadata(
            name="asr",
            version="0.1.0",
            description="Speech-to-Text with punctuation (SenseVoice, sherpa-onnx, offline).",
            author="OpenEyes-Live",
            input_type="audio_pcm_voice",
            output_type="text",
            input_schema={"type": "np.ndarray", "dtype": "float32", "sample_rate": 16000},
            output_schema={"type": "string"},
            size_mb=size,
            memory_mb=size + 30,
            tags=["audio", "asr", "speech-to-text"],
        )

    def load(self) -> None:
        """Load the ASR model. Idempotent."""
        if self._loaded:
            return
        if str(self.config["model_name"]) != "sense_voice":
            raise EngineLoadError(
                f"model '{self.config['model_name']}' not implemented; "
                f"v0.2.0 ships 'sense_voice' only"
            )
        model_path = Path(str(self.config["model_path"]))
        tokens_path = Path(str(self.config["tokens_path"]))
        for p in (model_path, tokens_path):
            if not p.exists():
                raise EngineLoadError(
                    f"ASR model file not found: {p} — "
                    f"run `openeyes install asr` or see models/README"
                )
        try:
            import sherpa_onnx

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=int(self.config["num_threads"]),
                use_itn=bool(self.config["use_itn"]),
            )
        except ImportError as exc:
            raise EngineLoadError(
                "sherpa-onnx is required for the ASR engine; "
                "install it with `pip install sherpa-onnx`"
            ) from exc
        except Exception as exc:
            raise EngineLoadError(f"failed to load SenseVoice: {exc}") from exc
        self._loaded = True

    def process(self, input_data: np.ndarray) -> EngineResult:
        """Transcribe voice audio to punctuated text.

        Args:
            input_data: Voice PCM (float32, 16 kHz mono). int16 input is
                accepted and converted automatically.
        """
        if not self._loaded or self._recognizer is None:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, np.ndarray) or input_data.ndim != 1:
            raise EngineProcessError("asr expects a 1-D PCM audio np.ndarray")
        if len(input_data) == 0:
            raise EngineProcessError("asr received empty audio")

        audio = input_data
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = audio.astype(np.float32, copy=False)

        start = time.perf_counter()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(int(self.config.get("sample_rate", 16000)), audio)
        self._recognizer.decode_stream(stream)
        text = str(stream.result.text).strip()
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=text,
            metadata={
                "engine": "asr",
                "model": "sense_voice",
                "language": self.config["language"],
                "duration_s": round(len(audio) / 16000.0, 2),
                "rtf": round(latency_ms / max(len(audio) / 16.0, 1.0), 3),
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free the recognizer. Idempotent."""
        self._recognizer = None
        self._loaded = False
