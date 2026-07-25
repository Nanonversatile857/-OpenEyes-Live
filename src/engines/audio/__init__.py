"""Audio engines (docs/AUDIO_PIPELINE.md).

v0.2.0: VADEngine (real Silero VAD) and ASREngine (real SenseVoice);
the Speaker engine is planned next.
"""

from .asr import ASREngine
from .vad import VADEngine

__all__ = ["VADEngine", "ASREngine"]
