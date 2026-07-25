"""OpenEyes-Live engines package.

Source: docs/ENGINE_SPEC.md — "Engine Registration" section.

v0.1.0 implements only the video ``encoder`` and core ``llm`` engines;
audio engines are reserved for v0.2.0.
"""

from .core.llm import LanguageEngine
from .video.encoder import VisualEncoderEngine

__all__ = [
    "VisualEncoderEngine",
    "LanguageEngine",
]
