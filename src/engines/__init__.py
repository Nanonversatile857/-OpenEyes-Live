"""OpenEyes-Live engines package.

Source: docs/ENGINE_SPEC.md — "Engine Registration" section.

v0.1.x implements the full video pipeline (``sampler``, ``filter``,
``encoder``, ``compressor``) plus the core ``llm``, ``memory`` and ``mcp``
engines; audio engines are reserved for v0.2.0.
"""

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
]
