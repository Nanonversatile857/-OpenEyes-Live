# 🎬 Video Pipeline Specification

> Detailed technical specification of OpenEyes-Live's modular video understanding pipeline.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [Pipeline Overview](#-pipeline-overview)
2. [Stage 1: Frame Sampler Engine](#-stage-1-frame-sampler-engine)
3. [Stage 2: Frame Filter Engine](#-stage-2-frame-filter-engine)
4. [Stage 3: Visual Encoder Engine](#-stage-3-visual-encoder-engine)
5. [Stage 4: Token Compressor Engine](#-stage-4-token-compressor-engine)
6. [Pipeline Configuration](#-pipeline-configuration)
7. [Performance Benchmarks](#-performance-benchmarks)
8. [Extending the Pipeline](#-extending-the-pipeline)

---

## 🎬 Pipeline Overview

The video pipeline transforms a continuous video stream into compact, semantically rich visual features that can be consumed by the Language Reasoning Engine.

```mermaid
flowchart LR
    subgraph VideoPipeline["🎬 Video Pipeline"]
        Raw["Raw Frames<br>(30 fps)"] -->|Sampler| S["帧采样引擎<br>(5-15 fps)"]
        S -->|Filter| F["帧筛选引擎<br>(Key Frames)"]
        F -->|Encoder| E["视觉编码引擎<br>(Features)"]
        E -->|Compressor| C["Token压缩引擎<br>(Tokens)"]
    end

    C --> LLM["🧠 Language Reasoning Engine"]
Stage	Engine	File	Size	Input → Output	Reduction
1	帧采样引擎	sampler.py	~5MB	30 fps → 5-15 fps	50-83%
2	帧筛选引擎	filter.py	~15MB	15 fps → 3-8 key frames	50-70%
3	视觉编码引擎	encoder.py	~200MB	Key frames → Visual embeddings	N/A
4	Token压缩引擎	compressor.py	~20MB	256 tokens → 64-128 tokens	50-75%
Total			~240MB		~90%+ reduction
🔧 Stage 1: Frame Sampler Engine
File: src/engines/video/sampler.py

Purpose: Reduce input frame rate to a manageable level while preserving temporal information.

Interface
python
from src.core.base_engine import BaseEngine
from typing import Optional, List
import numpy as np

class FrameSamplerEngine(BaseEngine):
    def __init__(self, config: SamplerConfig):
        """
        Initialize frame sampler.

        Args:
            config: Sampler configuration
                - mode: "uniform" | "dynamic" | "keyframe_only"
                - target_fps: int (default: 10)
                - min_interval: float (seconds between samples)
                - max_interval: float (max seconds between samples)
        """
        self.config = config
        self.frame_count = 0
        self.last_sample_time = 0
        self._loaded = False

    def load(self) -> None:
        """Initialize sampler state."""
        self._loaded = True

    def process(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Args:
            frame: Raw frame (HWC, BGR)
            timestamp: Frame timestamp in seconds

        Returns:
            Sampled frame or None (skip)
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        self.frame_count += 1

        if self.config.mode == "uniform":
            return self._uniform_sample(frame)

        elif self.config.mode == "dynamic":
            return self._dynamic_sample(frame, timestamp)

        elif self.config.mode == "keyframe_only":
            return self._keyframe_sample(frame)

        return frame

    def unload(self) -> None:
        """Reset sampler state."""
        self.frame_count = 0
        self.last_sample_time = 0
        self._loaded = False

    def _uniform_sample(self, frame: np.ndarray) -> Optional[np.ndarray]:
        interval = int(30 / self.config.target_fps)
        if self.frame_count % interval == 0:
            return frame
        return None

    def _dynamic_sample(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        # Sample more when motion is high, less when static
        motion_score = self._compute_motion(frame)
        adaptive_interval = max(
            self.config.min_interval,
            self.config.max_interval - motion_score
        )
        if timestamp - self.last_sample_time >= adaptive_interval:
            self.last_sample_time = timestamp
            return frame
        return None

    @property
    def size_mb(self) -> int:
        return 5

    @property
    def memory_usage_mb(self) -> int:
        return 10  # Buffer for frame storage

    @property
    def input_type(self) -> str:
        return "video_frame"

    @property
    def output_type(self) -> str:
        return "video_frame"

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Compute motion score from frame difference."""
        # Implementation uses frame differencing
        pass
Configuration Options
Parameter	Type	Default	Description
mode	str	"uniform"	Sampling mode: uniform/dynamic/keyframe_only
target_fps	int	10	Target frames per second
min_interval	float	0.5	Minimum seconds between samples
max_interval	float	5.0	Maximum seconds between samples
Sampler Modes
Mode	Description	Use Case
Uniform	Sample every Nth frame	Low-motion scenes, predictable frame rate
Dynamic	Sample based on motion score	High-motion scenes, varying activity
Keyframe Only	Sample only scene changes	Scene detection, summarization
🎯 Stage 2: Frame Filter Engine
File: src/engines/video/filter.py

Purpose: Select only the most informative frames for encoding. This is the most critical stage for performance optimization.

Interface
python
from src.core.base_engine import BaseEngine
from typing import List
import numpy as np

class FrameFilterEngine(BaseEngine):
    def __init__(self, config: FilterConfig):
        """
        Initialize frame filter.

        Args:
            config: Filter configuration
                - top_k: int (number of frames to select)
                - score_threshold: float (minimum score)
                - method: "attention" | "diversity" | "hybrid"
                - buffer_size: int (frames to keep in buffer)
        """
        self.config = config
        self.buffer = []
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load attention scoring model."""
        self._model = self._load_attention_model()
        self._loaded = True

    def process(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of frames and select key frames.

        Args:
            frames: List of sampled frames

        Returns:
            List of selected key frames (top_k)
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # Compute scores for each frame
        scores = self._compute_scores(frames)

        # Select top_k frames
        indices = self._select_top_k(scores)

        return [frames[i] for i in indices]

    def unload(self) -> None:
        """Free memory."""
        self._model = None
        self._loaded = False
        self.buffer = []

    def _compute_scores(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute attention scores for each frame."""
        # Mobile-VideoGPT style attention scoring
        # Returns score per frame
        return self._model.score(frames)

    def _select_top_k(self, scores: np.ndarray) -> List[int]:
        """Select indices of top-k frames."""
        if self.config.method == "attention":
            indices = np.argsort(scores)[-self.config.top_k:]
        elif self.config.method == "diversity":
            # Maximize diversity among selected frames
            indices = self._diversity_selection(scores)
        else:
            # Hybrid: attention + diversity
            indices = self._hybrid_selection(scores)
        return sorted(indices)

    @property
    def size_mb(self) -> int:
        return 15

    @property
    def memory_usage_mb(self) -> int:
        return 50  # Model + buffer

    @property
    def input_type(self) -> str:
        return "video_frames"

    @property
    def output_type(self) -> str:
        return "video_frames"
Attention Scoring Mechanism
The frame filter uses a lightweight attention-based scorer (inspired by Mobile-VideoGPT):

python
class AttentionScorer:
    """
    Lightweight attention-based frame scoring.
    ~15MB model size.
    """

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def score(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Score each frame by information content.

        Algorithm:
        1. Encode each frame with lightweight CNN
        2. Compute attention weights across time
        3. Scores indicate how representative each frame is
        """
        # Encode frames
        features = [self._encode(f) for f in frames]

        # Compute attention (self-attention across time)
        features = np.stack(features)  # (T, D)
        scores = self._attention(features)  # (T,)

        return scores
Configuration Options
Parameter	Type	Default	Description
top_k	int	8	Number of frames to select
score_threshold	float	0.3	Minimum score threshold
method	str	"attention"	Selection method
buffer_size	int	30	Frames to buffer before selection
🖼️ Stage 3: Visual Encoder Engine
File: src/engines/video/encoder.py

Purpose: Convert selected key frames into compact vector representations.

Interface
python
from src.core.base_engine import BaseEngine
from typing import List
import numpy as np

class VisualEncoderEngine(BaseEngine):
    def __init__(self, config: EncoderConfig):
        """
        Initialize visual encoder.

        Args:
            config: Encoder configuration
                - model_name: str ("clip" | "videomamba_t" | "videomamba_m")
                - embedding_dim: int (default: 512)
                - use_fp16: bool (default: True)
                - temporal_aggregation: str ("mean" | "max" | "attention")
        """
        self.config = config
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load visual encoder model."""
        self._model = self._load_encoder(self.config.model_name)
        self._loaded = True

    def process(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Encode frames into visual embeddings.

        Args:
            frames: List of key frames (RGB, HWC)

        Returns:
            Visual embeddings (T, embedding_dim) where T = len(frames)
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        embeddings = []
        for frame in frames:
            emb = self._model.encode(frame)
            embeddings.append(emb)

        return np.stack(embeddings)

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    @property
    def size_mb(self) -> int:
        sizes = {
            "clip": 200,
            "videomamba_t": 150,
            "videomamba_m": 280
        }
        return sizes.get(self.config.model_name, 200)

    @property
    def memory_usage_mb(self) -> int:
        return self.size_mb + 50  # Model + batch memory

    @property
    def input_type(self) -> str:
        return "frames_rgb"

    @property
    def output_type(self) -> str:
        return "visual_embeddings"
Available Models
Model	Params	Size (Q4)	Embedding Dim	Speed (SD 660)	Accuracy
CLIP-ViT-B/16	86M	~200MB	512	12 fps	⭐⭐⭐⭐
VideoMamba-T	50M	~150MB	512	18 fps	⭐⭐⭐
VideoMamba-M	100M	~280MB	768	8 fps	⭐⭐⭐⭐⭐
MobileNet-V3	5M	~20MB	256	30 fps	⭐⭐
Temporal Aggregation
When multiple frames are encoded, they can be aggregated into a single representation:

Method	Description	Use Case
Mean	Average across frames	Stable scenes
Max	Max pooling across frames	Highlight important features
Attention	Weighted average	All scenes
💾 Stage 4: Token Compressor Engine
File: src/engines/video/compressor.py

Purpose: Reduce the number of visual tokens fed to the LLM, speeding up inference.

Interface
python
from src.core.base_engine import BaseEngine
import numpy as np

class TokenCompressorEngine(BaseEngine):
    def __init__(self, config: CompressorConfig):
        """
        Initialize token compressor.

        Args:
            config: Compressor configuration
                - target_tokens: int (default: 64)
                - method: str ("projection" | "selection" | "hybrid")
                - preserve_temporal: bool (default: True)
        """
        self.config = config
        self._projector = None
        self._loaded = False

    def load(self) -> None:
        """Load token projector."""
        self._projector = self._load_projector()
        self._loaded = True

    def process(self, tokens: np.ndarray) -> np.ndarray:
        """
        Compress visual tokens.

        Args:
            tokens: Visual tokens (T, D) where T is number of tokens

        Returns:
            Compressed tokens (target_tokens, D)
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        if len(tokens) <= self.config.target_tokens:
            return tokens

        if self.config.method == "projection":
            # Learnable projection
            return self._projector(tokens)

        elif self.config.method == "selection":
            # Select most important tokens
            importance = self._compute_importance(tokens)
            indices = np.argsort(importance)[-self.config.target_tokens:]
            return tokens[indices]

        else:  # hybrid
            # Selection + projection
            indices = self._select(tokens)
            tokens = tokens[indices]
            return self._projector(tokens)

    def unload(self) -> None:
        self._projector = None
        self._loaded = False

    @property
    def size_mb(self) -> int:
        return 20

    @property
    def memory_usage_mb(self) -> int:
        return 30  # Projector model + buffer

    @property
    def input_type(self) -> str:
        return "visual_tokens"

    @property
    def output_type(self) -> str:
        return "visual_tokens_compressed"
Token Compression Impact
Token Count	LLM Speed (t/s)	Memory (MB)	Accuracy Loss
256 (original)	6	400	0%
128 (2x compression)	9	320	< 2%
64 (4x compression)	12	280	< 5%
32 (8x compression)	15	240	~10%
⚙️ Pipeline Configuration
Complete pipeline configuration:

python
# config/pipeline.yaml

video_pipeline:
  sampler:
    mode: "dynamic"           # uniform | dynamic | keyframe_only
    target_fps: 10
    min_interval: 0.5
    max_interval: 5.0

  filter:
    top_k: 8
    score_threshold: 0.3
    method: "attention"       # attention | diversity | hybrid
    buffer_size: 30

  encoder:
    model_name: "videomamba_t"  # clip | videomamba_t | videomamba_m
    embedding_dim: 512
    use_fp16: true
    temporal_aggregation: "attention"  # mean | max | attention

  compressor:
    target_tokens: 64
    method: "hybrid"          # projection | selection | hybrid
    preserve_temporal: true
📊 Performance Benchmarks
End-to-End Pipeline Performance
Stage	SD 660 (3GB)	SD 845 (6GB)	SD 888 (8GB)	A11 (2GB)
Sampler	< 1ms	< 1ms	< 1ms	< 1ms
Filter	15ms	8ms	5ms	20ms
Encoder	55ms	35ms	20ms	70ms
Compressor	10ms	6ms	4ms	15ms
Total	81ms	50ms	30ms	106ms
Memory Usage by Configuration
Configuration	Total Size	RAM Usage	Device Fit
Sampler + Encoder + Compressor	~240MB	~350MB	All devices
Sampler + Filter + Encoder + Compressor	~255MB	~400MB	All devices
Sampler + Filter + Encoder (no Compressor)	~235MB	~450MB	All devices
Sampler + Encoder (no Filter, no Compressor)	~220MB	~350MB	All devices
🔌 Extending the Pipeline
Adding a New Encoder Model
python
# src/engines/video/encoders/my_encoder.py

from src.engines.video.encoder import VisualEncoderEngine

class MyCustomEncoder(VisualEncoderEngine):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "my_custom_model"

    def _load_encoder(self, model_name: str):
        # Load your custom model
        return MyModel()

    def encode(self, frame: np.ndarray) -> np.ndarray:
        # Custom encoding logic
        return self.model.encode(frame)
Adding a New Filter Method
python
# src/engines/video/filters/my_filter.py

from src.engines.video.filter import FrameFilterEngine

class MyCustomFilter(FrameFilterEngine):
    def _compute_scores(self, frames: List[np.ndarray]) -> np.ndarray:
        # Your custom scoring logic
        return self._my_scoring(frames)

    def _select_top_k(self, scores: np.ndarray) -> List[int]:
        # Your custom selection logic
        return self._my_selection(scores)
Register Your Engine
python
# src/engines/__init__.py

from .video.encoder import VisualEncoderEngine
from .video.encoders.my_encoder import MyCustomEncoder

__all__ = [
    "VisualEncoderEngine",
    "MyCustomEncoder",
    # ...
]
📁 Related Documentation
Document	Description
ARCHITECTURE.md	Overall architecture and multi-modal design
AUDIO_PIPELINE.md	Audio pipeline specification
ENGINE_SPEC.md	Engine interface specification
PERFORMANCE_BENCHMARK.md	Complete benchmark data