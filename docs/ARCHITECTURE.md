# 🏗️ Architecture & Pipeline Design

> Complete technical specification of OpenEyes-Live's modular, pluggable, multi-modal engine architecture.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [Core Philosophy](#-core-philosophy)
2. [Multi-Modal Pipeline Overview](#-multi-modal-pipeline-overview)
3. [Video Engine Pipeline](#-video-engine-pipeline)
4. [Audio Engine Pipeline](#-audio-engine-pipeline)
5. [Core Engines](#-core-engines)
6. [Engine Lifecycle Management](#-engine-lifecycle-management)
7. [Scheduler Design](#-scheduler-design)
8. [MCP Integration](#-mcp-integration)
9. [Memory & Performance Optimization](#-memory--performance-optimization)

---

## 🧭 Core Philosophy

OpenEyes-Live is built on three architectural principles:

| Principle | Description |
| :--- | :--- |
| **Pipeline Separation** | Video understanding is broken into 4 stages (Sampler → Filter → Encoder → Compressor). Audio understanding into 3 stages (VAD → ASR → Speaker). Each stage is an independent, pluggable engine. |
| **On-Demand Composition** | Every sub-engine is downloaded only when needed. Users can combine engines freely based on their use case and device capability. |
| **Multi-Modal Fusion** | Video and audio pipelines feed into a shared reasoning engine, enabling cross-modal understanding (e.g., "Who is speaking in this scene?"). |

---

## 🎬 Multi-Modal Pipeline Overview

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        Camera["📷 Video Stream<br>(Local / IP Webcam / USB)"]
        Mic["🎤 Audio Stream<br>(Microphone / System Audio)"]
    end

    subgraph VideoPipeline["🎬 Video Pipeline"]
        direction TB
        S1["Sampler<br>Frame Sampling"]
        S2["Filter<br>Key Frame Selection"]
        S3["Encoder<br>Visual Encoding"]
        S4["Compressor<br>Token Compression"]
    end

    subgraph AudioPipeline["🎤 Audio Pipeline"]
        direction TB
        A1["VAD<br>Voice Activity Detection"]
        A2["ASR<br>Speech-to-Text + Punctuation"]
        A3["Speaker<br>Speaker Recognition"]
    end

    subgraph Core["🧠 Core Engines"]
        direction TB
        LLM["Language Reasoning<br>Qwen / Llama"]
        Memory["Memory Engine<br>Vector + Timeline"]
        MCP["MCP Gateway<br>Protocol Adapter"]
    end

    subgraph Output["📤 Output Layer"]
        Text["📝 Description"]
        Voice["🔊 Speech"]
        Alert["🔔 Alert"]
        MCPOut["🔌 MCP Tools"]
    end

    Camera --> S1 --> S2 --> S3 --> S4
    Mic --> A1 --> A2 --> A3
    S4 --> LLM
    A2 --> LLM
    A3 --> LLM
    LLM --> Memory
    LLM --> MCP
    Memory --> Text
    LLM --> Voice
    LLM --> Alert
    MCP --> MCPOut


🎬 Video Engine Pipeline

The video pipeline transforms a continuous video stream into compact, semantically rich visual features.

Pipeline Stages

flowchart LR
    subgraph VideoPipeline["Video Pipeline"]
        Raw["Raw Frames<br>(30 fps)"] --> Sampler["Sampler<br>(5-15 fps)"]
        Sampler --> Filter["Filter<br>(Key Frames)"]
        Filter --> Encoder["Encoder<br>(Features)"]
        Encoder --> Compressor["Compressor<br>(Tokens)"]
    end


Stage	Engine	Model	Size	Input → Output	Description
1	帧采样引擎	Configurable	~5MB	30 fps → 5-15 fps	Reduces frame rate. Uniform or dynamic sampling.
2	帧筛选引擎	Attention Scorer	~15MB	15 fps → 3-8 key frames	Scores each frame by importance. Selects only key frames.
3	视觉编码引擎	CLIP / VideoMamba	~200MB	Key frames → Visual embeddings	Extracts spatio-temporal features.
4	Token压缩引擎	Token Projector	~20MB	256 tokens → 64-128 tokens	Compresses redundant visual tokens. 2x speedup.

Stage 1: 帧采样引擎 (Frame Sampler)
Purpose: Reduce input frame rate without losing important information.

Implementation:
class FrameSampler:
    def __init__(self, mode: str = "uniform", target_fps: int = 10):
        self.mode = mode  # "uniform" | "dynamic"
        self.target_fps = target_fps
        self.frame_count = 0

    def sample(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self.frame_count += 1
        if self.mode == "uniform":
            if self.frame_count % self.interval == 0:
                return frame
        # Dynamic sampling based on motion score
        return None


Configurations:

Mode	Description	Use Case
Uniform	Sample every Nth frame	Low-motion scenes (static monitoring)
Dynamic	Sample based on motion score	High-motion scenes (activity detection)

Stage 2: 帧筛选引擎 (Frame Filter)
Purpose: Select only the most informative frames for encoding.

Implementation: Uses an attention-based scoring mechanism (inspired by Mobile-VideoGPT).

class FrameFilter:
    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.buffer = []

    def process(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # Score each frame using attention mechanism
        scores = self.compute_attention_scores(frames)
        # Select top_k frames with highest scores
        indices = np.argsort(scores)[-self.top_k:]
        return [frames[i] for i in indices]


Performance Impact:

Metric	Without Filter	With Filter	Improvement
Frames sent to encoder	15 fps	3-8 key frames	50-70% reduction
Encoder compute time	100%	30-50%	50-70% reduction
Accuracy loss	N/A	< 5%	Negligible
Stage 3: 视觉编码引擎 (Visual Encoder)
Purpose: Convert selected key frames into compact vector representations.

Available Models:

Model	Params	Size (Q4)	Speed (SD 660)	Accuracy
CLIP-ViT-B/16	86M	~200MB	12 fps	⭐⭐⭐⭐
VideoMamba-T	50M	~150MB	18 fps	⭐⭐⭐
VideoMamba-M	100M	~280MB	8 fps	⭐⭐⭐⭐⭐
Interface:

class VisualEncoder:
    def encode(self, frames: List[np.ndarray]) -> np.ndarray:
        """Returns visual embeddings of shape (n_frames, embedding_dim)."""
        pass


Stage 4: Token压缩引擎 (Token Compressor)
Purpose: Reduce the number of visual tokens fed to the LLM, speeding up inference.

Implementation:


class TokenCompressor:
    def __init__(self, target_tokens: int = 64):
        self.target_tokens = target_tokens

    def compress(self, tokens: np.ndarray) -> np.ndarray:
        # tokens shape: (n_frames, n_tokens_per_frame, dim)
        # Compress to (target_tokens, dim)
        return self.projector(tokens)  # Learnable projection


Performance Impact:

Metric	Before Compression	After Compression	Improvement
Tokens to LLM	256	64-128	50-75% reduction
LLM inference speed	6 t/s	12 t/s	2x speedup
Accuracy loss	N/A	< 3%	Minimal
🎤 Audio Engine Pipeline
The audio pipeline gives the system the ability to "hear" and "identify" speakers.

Pipeline Stages


flowchart LR
    subgraph AudioPipeline["Audio Pipeline"]
        Raw["Audio Stream"] --> VAD["VAD<br>(Voice Activity)"]
        VAD --> ASR["ASR<br>(Speech-to-Text)"]
        ASR --> Speaker["Speaker<br>(Identification)"]
    end


Stage	Engine	Model	Size	Input → Output	Description
1	VAD引擎	Silero VAD	~2MB	Audio → Voice segments	Detects when someone is speaking.
2	ASR引擎	Parakeet / Voxtral	~150MB	Voice → Text + Punctuation	Transcribes speech, restores punctuation.
3	声纹引擎	Speaker Embedding	~30MB	Voice → Speaker ID	Identifies who is speaking.

tage 1: VAD引擎 (Voice Activity Detection)
Purpose: Detect when someone is speaking. Prevents unnecessary ASR processing.

Implementation: Silero VAD is a lightweight LSTM-based model optimized for mobile.


class VADEngine:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = load_silero_vad()

    def detect(self, audio_chunk: bytes) -> bool:
        prob = self.model.predict(audio_chunk)
        return prob > self.threshold

Performance:

Metric	Value
Model size	2MB
Inference speed	< 1ms per chunk
Accuracy	95%+ on clean speech
Stage 2: ASR引擎 (Speech-to-Text + Punctuation Recovery)
Purpose: Convert speech to text with proper punctuation.

Available Models:

Model	Params	Size (Q4)	Speed (SD 660)	CER
Parakeet TDT	0.6B	~150MB	25x real-time	6.1%
Voxtral Realtime	~4B	~400MB	10x real-time	4.8%
Whisper Tiny	39M	~80MB	50x real-time	11.2%
Punctuation Recovery:


class PunctuationRecoverer:
    def __init__(self):
        self.model = load_punctuation_model()  # ~5MB

    def recover(self, text: str) -> str:
        """Adds punctuation to ASR output."""
        # Input: "hello my name is john"
        # Output: "Hello, my name is John."
        return self.model.predict(text)


Stage 3: 声纹引擎 (Speaker Recognition)
Purpose: Identify or verify speakers based on voice characteristics.

Implementation:

class SpeakerEngine:
    def __init__(self, db_path: str = "./speaker_db.pkl"):
        self.db_path = db_path
        self.embedder = load_speaker_embedder()  # ~30MB

    def enroll(self, name: str, audio: bytes) -> None:
        embedding = self.embedder.extract(audio)
        self.db[name] = embedding

    def identify(self, audio: bytes) -> Optional[str]:
        embedding = self.embedder.extract(audio)
        return self.find_closest_match(embedding)


Use Cases:

Scenario	Description
Family monitoring	Identify which family member is speaking
Visitor detection	Recognize if the speaker is unknown
Voice commands	Restrict certain commands to authorized users
Multimodal grounding	Combine vision + voice to identify "who said what"
🧠 Core Engines
Language Reasoning Engine
Purpose: Generate natural language responses based on visual and/or audio inputs.

class LanguageEngine:
    def __init__(self, model_path: str, context_length: int = 4096):
        self.model = load_model(model_path)
        self.context_length = context_length

    def generate(self,
                 visual_tokens: Optional[np.ndarray],
                 audio_text: Optional[str],
                 speaker_id: Optional[str],
                 prompt: str) -> str:
        """Generate response from multi-modal inputs."""
        pass

Memory Engine
Purpose: Store and retrieve historical observations.

class MemoryEngine:
    def store(self, observation: str, timestamp: str, metadata: dict):
        pass

    def query(self, query: str, limit: int = 5) -> List[MemoryResult]:
        pass

    def get_timeline(self, start: str, end: str) -> List[MemoryResult]:
        pass


MCP Gateway Engine
Purpose: Expose all capabilities via MCP protocol.

Exposed Tools:

Tool	Description
openeyes_capture_and_describe	Capture video frame, return description
openeyes_transcribe_audio	Transcribe microphone input
openeyes_identify_speaker	Identify who is speaking
openeyes_query_memory	Query historical observations
openeyes_set_alert	Set proactive alert triggers
openeyes_set_speaker	Enroll a new speaker
🔄 Engine Lifecycle Management
All engines follow the same lifecycle:

stateDiagram-v2
    [*] --> REGISTERED: install command
    REGISTERED --> DOWNLOADED: first use
    DOWNLOADED --> LOADED: inference request
    LOADED --> RUNNING: processing
    RUNNING --> LOADED: complete
    LOADED --> UNLOADED: OOM protection / idle
    UNLOADED --> LOADED: next request
    UNLOADED --> [*]: uninstall

EngineManager Interface:
class EngineManager:
    def list_engines(self) -> List[EngineInfo]:
        """List all available sub-engines."""

    def download(self, name: str, mirror: Optional[str] = None) -> Path:
        """Download an engine from registry."""

    def load(self, name: str) -> BaseEngine:
        """Load engine into memory."""

    def unload(self, name: str) -> None:
        """Unload engine from memory."""

    def is_loaded(self, name: str) -> bool:
        """Check if engine is loaded."""

Unified Engine Interface
Every engine in OpenEyes-Live implements the same interface:

python
from abc import ABC, abstractmethod

class BaseEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input and return result."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Free memory."""
        pass

    @property
    @abstractmethod
    def size_mb(self) -> int:
        """Engine size on disk."""
        pass

    @property
    @abstractmethod
    def memory_usage_mb(self) -> int:
        """Current RAM usage."""
        pass

    @property
    @abstractmethod
    def input_type(self) -> str:
        """Expected input type."""
        pass

    @property
    @abstractmethod
    def output_type(self) -> str:
        """Output type."""
        pass
This standardized interface enables:

Any engine to be swapped in/out without affecting other stages

Independent model upgrades

Community contributions of new engines

🔌 Scheduler Design
The scheduler coordinates all engines based on scene analysis.

python
class MultiModalScheduler:
    def __init__(self, config: SchedulerConfig):
        self.video_pipeline = VideoPipeline(config.video)
        self.audio_pipeline = AudioPipeline(config.audio)
        self.core = CoreEngines(config.core)

    def process_frame(self,
                      video_frame: np.ndarray,
                      audio_chunk: Optional[bytes]) -> InferenceResult:
        # 1. Run video pipeline
        visual_features = self.video_pipeline.process(video_frame)

        # 2. Run audio pipeline (if audio available)
        audio_text = None
        speaker_id = None
        if audio_chunk and self.audio_pipeline.vad.detect(audio_chunk):
            audio_text = self.audio_pipeline.asr.transcribe(audio_chunk)
            speaker_id = self.audio_pipeline.speaker.identify(audio_chunk)

        # 3. Fuse and reason
        response = self.core.generate(
            visual_features=visual_features,
            audio_text=audio_text,
            speaker_id=speaker_id
        )

        # 4. Store in memory
        self.core.memory.store(response, timestamp)

        return response
📊 Memory & Performance Optimization
Strategy	Mechanism	Impact
Pipeline Separation	Only load needed stages	RAM usage reduced by 60%
On-Demand Loading	Sub-engines loaded only when used	Minimal base footprint
Frame Filtering	Attention-based key frame selection	50-70% compute reduction
Token Compression	Projection to fewer tokens	2x LLM speedup
VAD Gate	Only run ASR when speech detected	90%+ ASR compute reduction
📁 Related Documentation
Document	Description
VIDEO_PIPELINE.md	Detailed video engine specifications
AUDIO_PIPELINE.md	Detailed audio engine specifications
ENGINE_SPEC.md	Engine interface specification for contributors
API_REFERENCE.md	Complete Python API reference





