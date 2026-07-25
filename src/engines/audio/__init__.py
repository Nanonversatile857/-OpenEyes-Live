"""Audio engines (docs/AUDIO_PIPELINE.md).

v0.2.0 adds VADEngine; ASR and Speaker engines are planned next.
"""

from .vad import VADEngine

__all__ = ["VADEngine"]
