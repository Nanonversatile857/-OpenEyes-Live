# 🎤 Audio Pipeline Specification

> Detailed technical specification of OpenEyes-Live's modular audio understanding pipeline.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [Pipeline Overview](#-pipeline-overview)
2. [Stage 1: VAD Engine](#-stage-1-vad-engine)
3. [Stage 2: ASR Engine](#-stage-2-asr-engine)
4. [Stage 3: Speaker Recognition Engine](#-stage-3-speaker-recognition-engine)
5. [Audio Pipeline Integration](#-audio-pipeline-integration)
6. [Configuration & Performance](#-configuration--performance)
7. [Extending the Pipeline](#-extending-the-pipeline)

---

## 🎤 Pipeline Overview

The audio pipeline transforms a continuous audio stream into transcribed text with speaker identification and punctuation recovery.

**Core Design Principle:** All processing runs **fully offline** — zero network calls, zero cloud dependencies. Speech data never leaves the device.

```mermaid
flowchart LR
    subgraph AudioPipeline["🎤 Audio Pipeline"]
        Raw["Audio Stream<br>(48kHz)"] -->|VAD| V["VAD引擎<br>(Voice Activity)"]
        V -->|Voice| ASR["ASR引擎<br>(Speech-to-Text)"]
        ASR -->|Text| Punc["标点恢复<br>(Punctuation)"]
        Punc -->|Text| Speaker["声纹引擎<br>(Speaker ID)"]
    end

    V -->|Silence| Drop["(Discard)"]
    Punc --> LLM["🧠 Language Reasoning Engine"]
    Speaker --> LLM
Stage	Engine	Model	Size	Input → Output	Description
1	VAD引擎	Silero VAD	~2MB	Audio → Voice segments	Detects voice activity, drops silence
2	ASR引擎	SenseVoice / Parakeet	~229MB	Voice → Text	Transcribes speech to text
2b	标点恢复	CT-Transformer	~5MB	Raw text → Punctuated text	Adds punctuation, formatting
3	声纹引擎	ERes2Net (3D-Speaker)	~38MB	Voice → Speaker ID	Identifies who is speaking (v0.3.0 已实现)
🎯 Stage 1: VAD Engine (Voice Activity Detection)
File: src/engines/audio/vad.py

Purpose: Detect when someone is speaking. Prevents unnecessary ASR processing and reduces power consumption by 90%+ in silence periods.

Interface
python
from src.core.base_engine import BaseEngine
from typing import Optional
import numpy as np

class VADEngine(BaseEngine):
    def __init__(self, config: VADConfig):
        """
        Initialize VAD engine.

        Args:
            config: VAD configuration
                - threshold: float (0.0-1.0, default: 0.5)
                - sample_rate: int (default: 16000)
                - frame_duration_ms: int (default: 30)
                - min_speech_duration_ms: int (default: 250)
                - max_speech_duration_s: int (default: 30)
        """
        self.config = config
        self._model = None
        self._loaded = False
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_count = 0

    def load(self) -> None:
        """Load Silero VAD model."""
        self._model = self._load_silero_vad()
        self._loaded = True

    def process(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk and detect voice activity.

        Args:
            audio_chunk: Raw PCM audio (int16 or float32)

        Returns:
            Audio chunk if voice detected, None if silence
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        # Run VAD
        speech_prob = self._model(audio_chunk, self.config.sample_rate)

        if speech_prob > self.config.threshold:
            # Voice detected
            self._is_speaking = True
            self._silence_count = 0
            self._speech_buffer.append(audio_chunk)
            return audio_chunk
        else:
            # Silence
            if self._is_speaking:
                self._silence_count += 1
                # Keep buffer for context
                self._speech_buffer.append(audio_chunk)
                # If enough silence, consider speech segment ended
                if self._silence_count > 3:  # ~90ms of silence
                    self._is_speaking = False
                    self._speech_buffer = []
            return None

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        self._speech_buffer = []
        self._is_speaking = False

    @property
    def size_mb(self) -> int:
        return 2

    @property
    def memory_usage_mb(self) -> int:
        return 10  # Model + buffer

    @property
    def input_type(self) -> str:
        return "audio_pcm"

    @property
    def output_type(self) -> str:
        return "audio_pcm_voice"

    def _load_silero_vad(self):
        # Silero VAD - ~2MB, runs on CPU
        # Implementation uses ONNX Runtime
        pass
Configuration Options
Parameter	Type	Default	Description
threshold	float	0.5	Speech probability threshold (0.3-0.7)
sample_rate	int	16000	Target sample rate
frame_duration_ms	int	30	Frame size in ms
min_speech_duration_ms	int	250	Minimum speech segment length
max_speech_duration_s	int	30	Maximum speech segment length
VAD Model Options
Model	Size	Accuracy	Platform	Description
Silero VAD	2MB	⭐⭐⭐⭐	All	Most popular, well-optimized
WebRTC VAD	0.1MB	⭐⭐⭐	All	Very lightweight, lower accuracy
Pyannote VAD	50MB	⭐⭐⭐⭐⭐	Linux/macOS	Higher accuracy, larger size
🎯 Stage 2: ASR Engine (Speech-to-Text + Punctuation)
File: src/engines/audio/asr.py

Purpose: Convert speech to text with proper punctuation and formatting. All processing is fully offline — no ModelScope or cloud dependencies.

Interface
python
from src.core.base_engine import BaseEngine
from typing import Optional, Tuple
import numpy as np

class ASREngine(BaseEngine):
    def __init__(self, config: ASRConfig):
        """
        Initialize ASR engine.

        Args:
            config: ASR configuration
                - model_name: str ("sense_voice" | "parakeet" | "whisper_tiny")
                - language: str ("zh" | "en" | "ja" | "ko" | "yue" | "auto")
                - use_itn: bool (inverse text normalization)
                - num_threads: int
                - punctuation_enabled: bool
        """
        self.config = config
        self._model = None
        self._punctuation_model = None
        self._loaded = False

    def load(self) -> None:
        """Load ASR model and punctuation model."""
        # Load main ASR model (ONNX Runtime based)
        self._model = self._load_asr_model(self.config.model_name)

        # Load punctuation recovery model if enabled
        if self.config.punctuation_enabled:
            self._punctuation_model = self._load_punctuation_model()

        self._loaded = True

    def process(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text with punctuation.

        Args:
            audio: Voice audio (float32, 16kHz)

        Returns:
            Transcribed text with punctuation
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # Run ASR
        raw_text = self._transcribe(audio)

        # Apply punctuation recovery
        if self.config.punctuation_enabled and self._punctuation_model:
            text = self._punctuation_model.restore(raw_text)
        else:
            text = raw_text

        return text

    def unload(self) -> None:
        self._model = None
        self._punctuation_model = None
        self._loaded = False

    @property
    def size_mb(self) -> int:
        sizes = {
            "sense_voice": 229,   # int8 quantized
            "parakeet": 150,
            "whisper_tiny": 80
        }
        return sizes.get(self.config.model_name, 229) + 5  # + punctuation model

    @property
    def memory_usage_mb(self) -> int:
        return self.size_mb + 30  # Model + buffer

    @property
    def input_type(self) -> str:
        return "audio_pcm_voice"

    @property
    def output_type(self) -> str:
        return "text"

    def _load_asr_model(self, model_name: str):
        """Load ASR model from local path."""
        # Uses sherpa-onnx for offline inference[citation:2]
        # Model loaded from local file system, zero network calls
        pass

    def _load_punctuation_model(self):
        """Load punctuation recovery model."""
        # CT-Transformer based, ~5MB[citation:5]
        pass
Available Models
Model	Params	Size (Q4)	Speed (SD 660)	Languages	CER
SenseVoice int8	—	229MB	60-90ms latency	zh/en/ja/ko/yue	⭐⭐⭐⭐
Parakeet TDT	0.6B	150MB	25x real-time	en	⭐⭐⭐⭐
Whisper Tiny	39M	80MB	50x real-time	20+ languages	⭐⭐⭐
Key Difference from FunASR/SenseVoice: The sherpa-onnx-based implementation does not require ModelScope SDK or any network access. Models are loaded directly from local file paths. This is critical for privacy-sensitive and offline deployments.

Punctuation Recovery
Purpose: Add punctuation to raw ASR output (e.g., "hello my name is john" → "Hello, my name is John.")

Available models:

Model	Size	Accuracy	Source
CT-Transformer	~5MB	⭐⭐⭐⭐	FunASR
sherpa-onnx Punctuation	~5MB	⭐⭐⭐⭐	sherpa-onnx
python
# Punctuation recovery example
class PunctuationRecoverer:
    def restore(self, text: str) -> str:
        # Input:  "hello my name is john"
        # Output: "Hello, my name is John."
        pass
🎯 Stage 3: Speaker Recognition Engine
File: src/engines/audio/speaker.py

> **v0.3.0 status: IMPLEMENTED (real model).** Uses 3D-Speaker ERes2Net
> base (ONNX, ~38MB, zh-cn, 16kHz) via sherpa-onnx — 512-d embeddings,
> ~34x real-time on desktop CPU. The interface below is the original
> design sketch; see `src/engines/audio/speaker.py` for the shipped API
> (`process` → embedding, plus `enroll` / `identify` / `remove`).

Purpose: Identify who is speaking based on voice characteristics (voiceprint / 声纹识别).

Interface
python
from src.core.base_engine import BaseEngine
from typing import Optional, List, Dict
import numpy as np

class SpeakerEngine(BaseEngine):
    def __init__(self, config: SpeakerConfig):
        """
        Initialize speaker recognition engine.

        Args:
            config: Speaker configuration
                - model_path: str (path to embedding model)
                - db_path: str (path to speaker database)
                - threshold: float (similarity threshold, default: 0.5)
                - embedding_dim: int (default: 192)
        """
        self.config = config
        self._embedder = None
        self._db = SpeakerDatabase(config.db_path)
        self._loaded = False

    def load(self) -> None:
        """Load speaker embedding model."""
        self._embedder = self._load_embedder(self.config.model_path)
        self._loaded = True

    def process(self, audio: np.ndarray) -> Optional[str]:
        """
        Identify speaker from audio.

        Args:
            audio: Voice audio

        Returns:
            Speaker name or None if unknown
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        embedding = self._embedder.extract(audio)

        # Find closest match in database
        match = self._db.find_closest(embedding, self.config.threshold)
        return match.name if match else None

    def enroll(self, name: str, audio: np.ndarray) -> None:
        """
        Enroll a new speaker.

        Args:
            name: Speaker identifier
            audio: Voice audio sample
        """
        embedding = self._embedder.extract(audio)
        self._db.add(name, embedding)

    def verify(self, audio: np.ndarray, expected_name: str) -> bool:
        """
        Verify if audio matches expected speaker.

        Args:
            audio: Voice audio
            expected_name: Expected speaker name

        Returns:
            True if verified
        """
        embedding = self._embedder.extract(audio)
        return self._db.verify(expected_name, embedding, self.config.threshold)

    def unload(self) -> None:
        self._embedder = None
        self._loaded = False
        self._db.save()

    @property
    def size_mb(self) -> int:
        return 30

    @property
    def memory_usage_mb(self) -> int:
        return 50  # Model + database

    @property
    def input_type(self) -> str:
        return "audio_pcm_voice"

    @property
    def output_type(self) -> str:
        return "speaker_id"
Speaker Embedding Models
Model	Architecture	Size	Embedding Dim	Accuracy
3D-Speaker ERES2Net	ECAPA-TDNN	~30MB	192	⭐⭐⭐⭐⭐
WeSpeaker	ResNet-based	~20MB	512	⭐⭐⭐⭐
ECAPA-TDNN is the current mainstream architecture for speaker verification in 2024-2026, still used in judicial identification and forensic applications.

Speaker Database
python
import sqlite3
import numpy as np

class SpeakerDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                embedding BLOB,
                created_at TIMESTAMP
            )
        """)

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding, created_at) VALUES (?, ?, ?)",
            (name, embedding.tobytes(), datetime.now())
        )
        self.conn.commit()

    def find_closest(self, embedding: np.ndarray, threshold: float) -> Optional[Match]:
        # Compute cosine similarity with all registered speakers
        # Return closest match if similarity > threshold
        pass

    def verify(self, name: str, embedding: np.ndarray, threshold: float) -> bool:
        # Compute cosine similarity with specific speaker
        # Return True if similarity > threshold
        pass
🔗 Audio Pipeline Integration
The complete audio pipeline ties all stages together:

python
from src.engines.audio.vad import VADEngine
from src.engines.audio.asr import ASREngine
from src.engines.audio.speaker import SpeakerEngine

class AudioPipeline:
    def __init__(self, config: AudioConfig):
        self.vad = VADEngine(config.vad)
        self.asr = ASREngine(config.asr)
        self.speaker = SpeakerEngine(config.speaker)
        self.buffer = []
        self._loaded = False

    def load(self) -> None:
        self.vad.load()
        self.asr.load()
        self.speaker.load()
        self._loaded = True

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[AudioResult]:
        """
        Process a single audio chunk through the full pipeline.

        Returns:
            AudioResult with transcribed text and speaker ID, or None if silence
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded")

        # Stage 1: VAD
        voice_audio = self.vad.process(audio_chunk)
        if voice_audio is None:
            return None

        # Accumulate voice audio
        self.buffer.append(voice_audio)

        # Stage 2: ASR (when speech segment ends)
        if not self.vad._is_speaking and len(self.buffer) > 0:
            full_audio = np.concatenate(self.buffer)
            text = self.asr.process(full_audio)

            # Stage 3: Speaker Recognition
            speaker_id = self.speaker.process(full_audio)

            self.buffer = []

            return AudioResult(
                text=text,
                speaker_id=speaker_id,
                duration_sec=len(full_audio) / 16000
            )

        return None

    def unload(self) -> None:
        self.vad.unload()
        self.asr.unload()
        self.speaker.unload()
        self.buffer = []
        self._loaded = False

    @property
    def total_size_mb(self) -> int:
        return self.vad.size_mb + self.asr.size_mb + self.speaker.size_mb

@dataclass
class AudioResult:
    text: str
    speaker_id: Optional[str]
    duration_sec: float
⚙️ Configuration
Complete pipeline configuration:

python
# config/audio.yaml

audio_pipeline:
  vad:
    threshold: 0.5
    sample_rate: 16000
    frame_duration_ms: 30
    min_speech_duration_ms: 250
    max_speech_duration_s: 30

  asr:
    model_name: "sense_voice"      # sense_voice | parakeet | whisper_tiny
    language: "zh"                 # zh | en | ja | ko | yue | auto
    use_itn: True                  # Inverse Text Normalization
    num_threads: 4
    punctuation_enabled: True

  speaker:
    model_path: "./models/3dspeaker_eres2net.onnx"
    db_path: "./speaker_db.sqlite"
    threshold: 0.5
📊 Performance Benchmarks
End-to-End Pipeline Performance (SD 660)
Stage	Latency	Memory	Notes
VAD	< 1ms	2MB	Per chunk
ASR (SenseVoice int8)	60-90ms	229MB	Per 5-7s audio clip
Punctuation	5ms	5MB	Per text segment
Speaker Recognition	10ms	30MB	Per audio clip
Total Pipeline Size
Configuration	Total Size	RAM Usage	Device Fit
VAD + ASR (SenseVoice) + Punctuation	~236MB	~300MB	All devices
VAD + ASR + Punctuation + Speaker	~266MB	~350MB	3GB+ devices
VAD + ASR (Whisper Tiny)	~82MB	~120MB	Very low-end devices
Comparison: sherpa-onnx vs FunASR/SenseVoice
Dimension	sherpa-onnx	FunASR/SenseVoice
Network Dependency	None — local file path only	ModelScope SDK checks
Offline Support	✅ Full	⚠️ Requires startup check
Model Loading	Direct ONNX	PyTorch + ModelScope
Cross-Platform	✅ 12 languages, ARM/RISC-V	Primarily Linux
🔌 Extending the Pipeline
Adding a New ASR Model
python
# src/engines/audio/asr/my_asr.py

from src.engines.audio.asr import ASREngine

class MyCustomASR(ASREngine):
    def _load_asr_model(self, model_name: str):
        # Load your custom model using sherpa-onnx
        # Models must be in ONNX format
        return sherpa_onnx.OfflineRecognizer(...)

    def _transcribe(self, audio: np.ndarray) -> str:
        return self.model.recognize(audio)
Adding Custom Speaker Model
python
# src/engines/audio/speaker/my_speaker.py

from src.engines.audio.speaker import SpeakerEngine

class MyCustomSpeaker(SpeakerEngine):
    def _load_embedder(self, model_path: str):
        # Load your custom embedding model
        return CustomEmbedder(model_path)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        return self.embedder.extract(audio)
📁 Related Documentation
Document	Description
ARCHITECTURE.md	Overall architecture
VIDEO_PIPELINE.md	Video pipeline specification
ENGINE_SPEC.md	Engine interface specification
PERFORMANCE_BENCHMARK.md	Complete benchmark data