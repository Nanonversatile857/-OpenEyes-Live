# 🔧 API Reference

> Complete Python API documentation for OpenEyes-Live developers.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [EngineManager](#enginemanager)
2. [Pipeline](#pipeline)
3. [Video Engines](#video-engines)
4. [Audio Engines](#audio-engines)
5. [Core Engines](#core-engines)
6. [CLI Reference](#cli-reference)

---

## 🧩 EngineManager

Manages engine lifecycle (download/load/unload).

```python
from src.core.engine_manager import EngineManager

manager = EngineManager(cache_dir="./models")

# List available engines
manager.list_engines()
# ['sampler', 'filter', 'encoder', 'compressor', 'vad', 'asr', 'speaker', 'llm', 'memory', 'mcp']

# Download an engine
manager.download("encoder", mirror="modelscope")

# Load an engine
engine = manager.load("encoder")

# Check status
manager.is_loaded("encoder")  # True

# Unload
manager.unload("encoder")
🎬 Video Engines
FrameSamplerEngine
python
from src.engines.video import FrameSamplerEngine

engine = FrameSamplerEngine({
    "mode": "uniform",     # uniform | dynamic | keyframe_only
    "target_fps": 10,
})

engine.load()
result = engine.process(frame)  # np.ndarray or None
engine.unload()
FrameFilterEngine
python
from src.engines.video import FrameFilterEngine

engine = FrameFilterEngine({
    "top_k": 8,
    "method": "attention",  # attention | diversity | hybrid
})

engine.load()
result = engine.process(frames)  # List[np.ndarray]
engine.unload()
VisualEncoderEngine
python
from src.engines.video import VisualEncoderEngine

engine = VisualEncoderEngine({
    "model_name": "videomamba_t",  # clip | videomamba_t | videomamba_m
    "embedding_dim": 512,
})

engine.load()
result = engine.process(frames)  # np.ndarray (T, 512)
engine.unload()
TokenCompressorEngine
python
from src.engines.video import TokenCompressorEngine

engine = TokenCompressorEngine({
    "target_tokens": 64,
    "method": "hybrid",  # projection | selection | hybrid
})

engine.load()
result = engine.process(tokens)  # np.ndarray (64, D)
engine.unload()
🎤 Audio Engines
VADEngine
python
from src.engines.audio import VADEngine

engine = VADEngine({
    "threshold": 0.5,
    "sample_rate": 16000,
})

engine.load()
result = engine.process(audio_chunk)  # np.ndarray or None
engine.unload()
ASREngine
python
from src.engines.audio import ASREngine

engine = ASREngine({
    "model_name": "sense_voice",  # sense_voice | parakeet | whisper_tiny
    "language": "zh",
    "punctuation_enabled": True,
})

engine.load()
result = engine.process(audio)  # str with punctuation
engine.unload()
SpeakerEngine
python
from src.engines.audio import SpeakerEngine

engine = SpeakerEngine({
    "threshold": 0.5,
    "db_path": "./speakers.db",
})

engine.load()

# Enroll a speaker
engine.enroll("Alice", audio_sample)

# Identify speaker
result = engine.process(audio)  # "Alice" or None

# Verify speaker
engine.verify(audio, "Alice")  # True/False

engine.unload()
🧠 Core Engines
LanguageEngine
python
from src.engines.core import LanguageEngine

engine = LanguageEngine({
    "model_path": "./qwen2.5-2b-q4.gguf",
    "context_length": 4096,
})

engine.load()

result = engine.process({
    "visual_tokens": visual_tokens,
    "audio_text": "Hello, I'm Alice.",
    "speaker_id": "Alice",
    "prompt": "Describe what you see."
})

print(result.data)
# "I see a person sitting on a sofa..."
engine.unload()
MemoryEngine
python
from src.engines.core import MemoryEngine

engine = MemoryEngine({
    "vector_dim": 384,
    "db_path": "./memory.db",
})

engine.load()

# Store observation
engine.store("Cat is sleeping", "2026-07-25T14:30:00Z", {"location": "sofa"})

# Query memory
results = engine.query("cat")
for r in results.data:
    print(r.description, r.timestamp)

engine.unload()
MCPGateway
python
from src.engines.core import MCPGateway

gateway = MCPGateway({"port": 3000})
gateway.load()

# Register custom tool
gateway.register_tool("custom_tool", handler_function)

# Stop server
gateway.unload()
🖥️ CLI Reference
Command	Description
openeyes list	List all available engines
openeyes install <engine>	Download an engine
openeyes watch	Start visual understanding
openeyes watch --engines <engines>	Use specific engines
openeyes watch --audio	Enable audio pipeline
openeyes watch --mode proactive	Proactive alerts
openeyes mcp	Start MCP server
openeyes --version	Show version