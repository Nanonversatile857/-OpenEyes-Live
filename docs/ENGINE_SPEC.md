# 📐 Engine Specification

> Complete interface specification for OpenEyes-Live engines.  
> **All engines must implement this specification to be pluggable.**

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [Core Philosophy](#-core-philosophy)
2. [BaseEngine Interface](#-baseengine-interface)
3. [Engine Registration](#-engine-registration)
4. [Engine Lifecycle](#-engine-lifecycle)
5. [Video Engine Specifications](#-video-engine-specifications)
6. [Audio Engine Specifications](#-audio-engine-specifications)
7. [Core Engine Specifications](#-core-engine-specifications)
8. [Engine Registry Format](#-engine-registry-format)
9. [Error Handling](#-error-handling)
10. [Testing Requirements](#-testing-requirements)

---

## 🧭 Core Philosophy

Every engine in OpenEyes-Live follows the **"One Engine, One Job"** principle:

| Principle | Description |
| :--- | :--- |
| **Single Responsibility** | Each engine does exactly one thing and does it well |
| **Pluggable** | Any engine can be swapped in/out without affecting others |
| **Composable** | Engines can be combined in arbitrary pipelines |
| **Discoverable** | Engines self-describe their capabilities and requirements |
| **Testable** | Each engine has standalone unit tests |

---

## 🔧 BaseEngine Interface

**File:** `src/core/base_engine.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import numpy as np

@dataclass
class EngineMetadata:
    """Self-describing engine metadata."""
    name: str                          # Unique engine name
    version: str                       # Semver version
    description: str                   # Human-readable description
    author: str                        # Author/organization
    input_type: str                    # MIME type or type name
    output_type: str                   # MIME type or type name
    input_schema: Dict[str, Any]       # JSON schema for inputs
    output_schema: Dict[str, Any]      # JSON schema for outputs
    size_mb: int                       # Disk size in MB
    memory_mb: int                     # RAM usage in MB
    tags: List[str]                    # Tags for discovery

@dataclass
class EngineResult:
    """Standardized engine output."""
    data: Any                          # Main output data
    metadata: Dict[str, Any]           # Additional metadata
    confidence: Optional[float] = None # Confidence score (0-1)
    latency_ms: Optional[float] = None # Processing latency

class BaseEngine(ABC):
    """
    Abstract base class for all OpenEyes-Live engines.
    All engines must inherit from this class.
    """

    # === Required Properties ===

    @property
    @abstractmethod
    def metadata(self) -> EngineMetadata:
        """Self-describing engine metadata."""
        pass

    # === Required Lifecycle Methods ===

    @abstractmethod
    def load(self) -> None:
        """
        Load engine into memory.
        Called once before any process() calls.
        Should be idempotent.
        """
        pass

    @abstractmethod
    def process(self, input_data: Any) -> EngineResult:
        """
        Process input and return result.
        Must be thread-safe.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload engine from memory.
        Free all resources.
        Should be idempotent.
        """
        pass

    # === Optional Methods ===

    def validate(self, input_data: Any) -> bool:
        """
        Validate input before processing.
        Default: always True.
        """
        return True

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update engine configuration at runtime.
        Default: no-op.
        """
        pass

    def health_check(self) -> bool:
        """
        Check if engine is healthy.
        Default: returns loaded state.
        """
        return self.is_loaded()

    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        # Engine must track this state internally
        pass

    def reset(self) -> None:
        """
        Reset engine to initial state.
        Default: unload and reload.
        """
        self.unload()
        self.load()
📋 Engine Registration
All engines must register themselves in src/engines/__init__.py:

python
# src/engines/__init__.py

from src.core.base_engine import BaseEngine
from src.core.engine_registry import EngineRegistry

# Video engines
from .video.sampler import FrameSamplerEngine
from .video.filter import FrameFilterEngine
from .video.encoder import VisualEncoderEngine
from .video.compressor import TokenCompressorEngine

# Audio engines
from .audio.vad import VADEngine
from .audio.asr import ASREngine
from .audio.speaker import SpeakerEngine

# Core engines
from .core.llm import LanguageEngine
from .core.memory import MemoryEngine
from .core.mcp_gateway import MCPGateway

# Register all engines
ENGINE_REGISTRY = EngineRegistry()

ENGINE_REGISTRY.register("sampler", FrameSamplerEngine)
ENGINE_REGISTRY.register("filter", FrameFilterEngine)
ENGINE_REGISTRY.register("encoder", VisualEncoderEngine)
ENGINE_REGISTRY.register("compressor", TokenCompressorEngine)
ENGINE_REGISTRY.register("vad", VADEngine)
ENGINE_REGISTRY.register("asr", ASREngine)
ENGINE_REGISTRY.register("speaker", SpeakerEngine)
ENGINE_REGISTRY.register("llm", LanguageEngine)
ENGINE_REGISTRY.register("memory", MemoryEngine)
ENGINE_REGISTRY.register("mcp", MCPGateway)

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "ENGINE_REGISTRY",
    # ... all engine classes
]
🔄 Engine Lifecycle

stateDiagram-v2
    [*] --> REGISTERED: Registry registration
    REGISTERED --> DOWNLOADED: engine_manager.download()
    DOWNLOADED --> LOADED: engine_manager.load()
    LOADED --> PROCESSING: process() called
    PROCESSING --> LOADED: process() returns
    LOADED --> UNLOADED: engine_manager.unload()
    UNLOADED --> LOADED: engine_manager.load()
    UNLOADED --> DELETED: engine_manager.uninstall()
    DELETED --> [*]



Lifecycle Code Example
python
from src.core.engine_manager import EngineManager

manager = EngineManager()

# 1. Download (first time only)
if not manager.is_installed("encoder"):
    manager.download("encoder")  # Downloads ~200MB model

# 2. Load
engine = manager.load("encoder")

# 3. Process
result = engine.process(frames)

# 4. Unload (when done)
manager.unload("encoder")
🎬 Video Engine Specifications
1. FrameSamplerEngine
File: src/engines/video/sampler.py

python
class FrameSamplerEngine(BaseEngine):
    """
    Samples frames from video stream.

    Input:  np.ndarray (HWC, BGR) or list of frames
    Output: EngineResult with filtered frames

    Config:
        mode: "uniform" | "dynamic" | "keyframe_only"
        target_fps: int (default: 10)
        min_interval: float (default: 0.5)
        max_interval: float (default: 5.0)

    Metadata:
        name: "sampler"
        version: "0.1.0"
        input_type: "video_frame"
        output_type: "video_frame"
        size_mb: 5
        memory_mb: 10
        tags: ["video", "sampling"]
    """
2. FrameFilterEngine
File: src/engines/video/filter.py

python
class FrameFilterEngine(BaseEngine):
    """
    Selects key frames using attention scoring.

    Input:  List[np.ndarray] (sampled frames)
    Output: EngineResult with selected key frames

    Config:
        top_k: int (default: 8)
        method: "attention" | "diversity" | "hybrid"
        score_threshold: float (default: 0.3)
        buffer_size: int (default: 30)

    Metadata:
        name: "filter"
        version: "0.1.0"
        input_type: "video_frames"
        output_type: "video_frames"
        size_mb: 15
        memory_mb: 50
        tags: ["video", "filtering", "attention"]
    """
3. VisualEncoderEngine
File: src/engines/video/encoder.py

python
class VisualEncoderEngine(BaseEngine):
    """
    Encodes frames into visual embeddings.

    Input:  List[np.ndarray] (key frames, RGB)
    Output: EngineResult with visual embeddings

    Config:
        model_name: "clip" | "videomamba_t" | "videomamba_m"
        embedding_dim: int (default: 512)
        use_fp16: bool (default: True)

    Metadata:
        name: "encoder"
        version: "0.1.0"
        input_type: "frames_rgb"
        output_type: "visual_embeddings"
        size_mb: 200
        memory_mb: 250
        tags: ["video", "encoding", "vision"]
    """
4. TokenCompressorEngine
File: src/engines/video/compressor.py

python
class TokenCompressorEngine(BaseEngine):
    """
    Compresses visual tokens for LLM input.

    Input:  np.ndarray (visual tokens, T, D)
    Output: EngineResult with compressed tokens

    Config:
        target_tokens: int (default: 64)
        method: "projection" | "selection" | "hybrid"

    Metadata:
        name: "compressor"
        version: "0.1.0"
        input_type: "visual_tokens"
        output_type: "visual_tokens_compressed"
        size_mb: 20
        memory_mb: 30
        tags: ["video", "compression", "efficiency"]
    """
🎤 Audio Engine Specifications
1. VADEngine
File: src/engines/audio/vad.py

python
class VADEngine(BaseEngine):
    """
    Voice Activity Detection.

    Input:  np.ndarray (PCM audio, float32 or int16)
    Output: EngineResult with voice audio or None

    Config:
        threshold: float (default: 0.5)
        sample_rate: int (default: 16000)
        min_speech_duration_ms: int (default: 250)
        max_speech_duration_s: int (default: 30)

    Metadata:
        name: "vad"
        version: "0.1.0"
        input_type: "audio_pcm"
        output_type: "audio_pcm_voice"
        size_mb: 2
        memory_mb: 10
        tags: ["audio", "vad", "detection"]
    """
2. ASREngine
File: src/engines/audio/asr.py

python
class ASREngine(BaseEngine):
    """
    Speech-to-Text with punctuation recovery.
    Fully offline — no network calls.

    Input:  np.ndarray (voice PCM, 16kHz)
    Output: EngineResult with transcribed text

    Config:
        model_name: "sense_voice" | "parakeet" | "whisper_tiny"
        language: "zh" | "en" | "ja" | "ko" | "yue" | "auto"
        punctuation_enabled: bool (default: True)
        num_threads: int (default: 4)

    Metadata:
        name: "asr"
        version: "0.1.0"
        input_type: "audio_pcm_voice"
        output_type: "text"
        size_mb: 234
        memory_mb: 260
        tags: ["audio", "asr", "speech-to-text"]
    """
3. SpeakerEngine
File: src/engines/audio/speaker.py

python
class SpeakerEngine(BaseEngine):
    """
    Speaker Recognition / Verification.

    Input:  np.ndarray (voice PCM)
    Output: EngineResult with speaker_id or verification result

    Config:
        model_path: str
        db_path: str
        threshold: float (default: 0.5)

    Methods:
        enroll(name: str, audio: np.ndarray) -> None
        verify(audio: np.ndarray, expected_name: str) -> bool

    Metadata:
        name: "speaker"
        version: "0.1.0"
        input_type: "audio_pcm_voice"
        output_type: "speaker_id"
        size_mb: 30
        memory_mb: 50
        tags: ["audio", "speaker", "identification"]
    """
🧠 Core Engine Specifications
1. LanguageEngine
File: src/engines/core/llm.py

python
class LanguageEngine(BaseEngine):
    """
    Multi-modal language reasoning.

    Input:  Dict with visual_tokens, audio_text, speaker_id
    Output: EngineResult with natural language response

    Config:
        model_path: str
        context_length: int (default: 4096)
        temperature: float (default: 0.7)
        max_tokens: int (default: 256)

    Metadata:
        name: "llm"
        version: "0.1.0"
        input_type: "multi_modal_features"
        output_type: "text"
        size_mb: 400
        memory_mb: 500
        tags: ["core", "reasoning", "llm"]
    """
2. MemoryEngine
File: src/engines/core/memory.py

python
class MemoryEngine(BaseEngine):
    """
    Vector memory with timeline support.

    Input:  Dict with query or observation
    Output: EngineResult with memory matches

    Methods:
        store(observation: str, timestamp: str, metadata: dict) -> None
        query(query: str, limit: int = 5) -> List[MemoryResult]
        get_timeline(start: str, end: str) -> List[MemoryResult]

    Metadata:
        name: "memory"
        version: "0.1.0"
        input_type: "memory_operation"
        output_type: "memory_result"
        size_mb: 50
        memory_mb: 100
        tags: ["core", "memory", "retrieval"]
    """
3. MCPGateway
File: src/engines/core/mcp_gateway.py

python
class MCPGateway(BaseEngine):
    """
    MCP protocol gateway.

    Input:  JSON-RPC request
    Output: EngineResult with JSON-RPC response

    Methods:
        register_tool(name: str, handler: Callable) -> None
        list_tools() -> List[ToolSpec]

    Metadata:
        name: "mcp"
        version: "0.1.0"
        input_type: "json_rpc"
        output_type: "json_rpc"
        size_mb: 10
        memory_mb: 20
        tags: ["core", "mcp", "protocol"]
    """
📦 Engine Registry Format
File: src/core/registry.yaml

yaml
# Engine Registry
# All engines and their model download manifests (schema v2, v0.3.0).
# Model engines declare `hf_repo` + a `files` list; each file records its
# remote path in the HF repo, the local path under models/<engine>/, and
# the expected byte size (integrity check). `source: internal` engines
# are pure code and need no download.

registry:
  version: "2.0"
  # Download sources, tried in order unless --mirror pins one.
  sources:
    huggingface: "https://huggingface.co"
    hf-mirror: "https://hf-mirror.com"
  default_source: "huggingface"
  fallback_source: "hf-mirror"

engines:
  # Video Pipeline
  sampler:
    name: "sampler"
    version: "0.1.0"
    size_mb: 5
    source: "internal"

  encoder:
    name: "encoder"
    version: "0.2.0"
    size_mb: 89
    source: "huggingface"
    hf_repo: "Xenova/clip-vit-base-patch32"
    files:
      - remote: "onnx/vision_model_quantized.onnx"
        local: "clip-vit-b32/vision_model_quantized.onnx"
        size: 89117001
      - remote: "preprocessor_config.json"
        local: "clip-vit-b32/preprocessor_config.json"
        size: 520

  # Audio Pipeline
  vad:
    name: "vad"
    version: "0.2.0"
    size_mb: 2
    source: "huggingface"
    hf_repo: "onnx-community/silero-vad"
    files:
      - remote: "onnx/model.onnx"
        local: "silero_vad.onnx"
        size: 2243022

  # ... asr (SenseVoice int8, 2 files), llm (Phi-3.5-vision int4, 11
  # files) follow the same pattern — see src/core/registry.yaml.

  speaker:
    name: "speaker"
    version: "0.3.0"
    size_mb: 38
    source: "github"
    files:
      # GitHub release asset — absolute URL (upstream tag spelled
      # "speaker-recongition-models", sic); mirrors don't apply.
      - url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        local: "3dspeaker_eres2net_base_16k.onnx"
        size: 39593761
❌ Error Handling
All engines must use standardized error codes:

python
# src/core/errors.py

class EngineError(Exception):
    """Base engine exception."""

class EngineLoadError(EngineError):
    """Raised when engine fails to load."""
    pass

class EngineProcessError(EngineError):
    """Raised when engine fails to process input."""
    pass

class EngineUnloadError(EngineError):
    """Raised when engine fails to unload."""
    pass

class EngineInvalidInputError(EngineError):
    """Raised when input validation fails."""
    pass

class EngineOutOfMemoryError(EngineError):
    """Raised when engine runs out of memory."""
    pass

class EngineNotFoundError(EngineError):
    """Raised when engine not found in registry."""
    pass

class EngineDownloadError(EngineError):
    """Raised when engine download fails."""
    pass
🧪 Testing Requirements
Each engine must have:

1. Unit Tests
python
# tests/test_engine_<name>.py

import pytest
from src.engines import <EngineName>

class Test<EngineName>:
    def test_load(self):
        engine = <EngineName>(config)
        engine.load()
        assert engine.is_loaded()

    def test_process(self):
        engine = <EngineName>(config)
        engine.load()
        result = engine.process(test_input)
        assert result is not None
        # Check output format

    def test_unload(self):
        engine = <EngineName>(config)
        engine.load()
        engine.unload()
        assert not engine.is_loaded()
2. Integration Tests
python
# tests/test_pipeline_integration.py

def test_video_pipeline_integration():
    """Test full video pipeline."""
    sampler = FrameSamplerEngine(sampler_config)
    filter = FrameFilterEngine(filter_config)
    encoder = VisualEncoderEngine(encoder_config)

    sampler.load()
    filter.load()
    encoder.load()

    # Process through pipeline
    sampled = sampler.process(test_frames)
    filtered = filter.process(sampled.data)
    encoded = encoder.process(filtered.data)

    # Verify output
    assert encoded.data.shape == (8, 512)
3. Performance Tests
python
# tests/test_engine_performance.py

def test_engine_latency():
    """Engine must meet latency requirements."""
    engine = <EngineName>(config)
    engine.load()

    latencies = []
    for _ in range(100):
        start = time.time()
        engine.process(test_input)
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < MAX_ALLOWED_LATENCY
✅ Engine Development Checklist
When creating a new engine:

□ Inherits from BaseEngine
□ Implements all abstract methods
□ Provides complete EngineMetadata
□ Follows the directory structure: src/engines/<category>/
□ Registered in src/engines/__init__.py
□ Added to registry.yaml
□ Includes unit tests (>80% coverage)
□ Includes docstrings for all public methods
□ Handles errors with standardized error codes
□ Meets size and memory requirements
□ Documented in this specification
📁 Related Documentation
Document	Description
ARCHITECTURE.md	Overall architecture
VIDEO_PIPELINE.md	Video pipeline specification
AUDIO_PIPELINE.md	Audio pipeline specification
API_REFERENCE.md	Complete Python API reference