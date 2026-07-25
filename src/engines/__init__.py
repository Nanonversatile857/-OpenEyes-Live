"""OpenEyes-Live engines package.

Source: docs/ENGINE_SPEC.md — "Engine Registration" section.

Implemented: the full video pipeline (``sampler``, ``filter``, ``encoder``,
``compressor``), the core engines (``llm``, ``memory``, ``mcp``) and the
audio engines ``vad`` (real Silero VAD) and ``asr`` (real SenseVoice).
``speaker`` is planned.
"""

from .audio.asr import ASREngine
from .audio.vad import VADEngine
from .core.llm import LanguageEngine
from .core.mcp_gateway import MCPGateway
from .core.memory import MemoryEngine
from .video.compressor import TokenCompressorEngine
from .video.encoder import VisualEncoderEngine
from .video.filter import FrameFilterEngine
from .video.sampler import FrameSamplerEngine

__all__ = [
    "FrameSamplerEngine",
    "FrameFilterEngine",
    "VisualEncoderEngine",
    "TokenCompressorEngine",
    "LanguageEngine",
    "MemoryEngine",
    "MCPGateway",
    "VADEngine",
    "ASREngine",
]
