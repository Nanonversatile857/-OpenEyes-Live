"""Unit tests for VADEngine.

Source: docs/AUDIO_PIPELINE.md — "Stage 1: VAD Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.audio.vad import SILENCE_CHUNKS_TO_END, VADEngine

SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE * 0.03)  # 30 ms frames, per doc default


def _speech_chunk(amplitude: float = 0.3) -> np.ndarray:
    """Loud sine wave — RMS well above the VAD threshold."""
    t = np.arange(CHUNK) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence_chunk() -> np.ndarray:
    """Near-silent background — RMS far below the threshold."""
    return np.full(CHUNK, 0.001, dtype=np.float32)


class TestVADEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = VADEngine()
        self.engine.load()

    def tearDown(self) -> None:
        self.engine.unload()

    def test_metadata(self) -> None:
        meta = self.engine.metadata
        self.assertEqual(meta.name, "vad")
        self.assertEqual(meta.input_type, "audio_pcm")
        self.assertEqual(meta.output_type, "audio_pcm_voice")
        self.assertIn("detection", meta.tags)

    def test_speech_passes_through(self) -> None:
        result = self.engine.process(_speech_chunk())
        self.assertIsNotNone(result.data)
        self.assertGreater(result.metadata["speech_prob"], 0.5)
        self.assertTrue(result.metadata["is_speaking"])

    def test_silence_returns_none(self) -> None:
        result = self.engine.process(_silence_chunk())
        self.assertIsNone(result.data)
        self.assertLess(result.metadata["speech_prob"], 0.5)

    def test_segment_ends_after_trailing_silence(self) -> None:
        self.engine.process(_speech_chunk())
        ended_at = None
        for i in range(SILENCE_CHUNKS_TO_END + 2):
            res = self.engine.process(_silence_chunk())
            if res.metadata["segment_ended"]:
                ended_at = i
                break
        self.assertEqual(ended_at, SILENCE_CHUNKS_TO_END)
        self.assertFalse(self.engine.is_loaded() is False)
        self.assertFalse(self.engine._is_speaking)

    def test_int16_input_accepted(self) -> None:
        chunk = (_speech_chunk() * 32768).astype(np.int16)
        result = self.engine.process(chunk)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.data.dtype, np.float32)

    def test_max_speech_duration_cuts_segment(self) -> None:
        engine = VADEngine({"max_speech_duration_s": 0.06})  # = 2 chunks
        engine.load()
        engine.process(_speech_chunk())
        res = engine.process(_speech_chunk())
        self.assertTrue(res.metadata["segment_ended"])
        engine.unload()

    def test_buffered_speech_accumulates(self) -> None:
        for _ in range(3):
            self.engine.process(_speech_chunk())
        buffered = self.engine.buffered_speech
        self.assertIsNotNone(buffered)
        self.assertEqual(len(buffered), 3 * CHUNK)

    def test_process_requires_load(self) -> None:
        engine = VADEngine()
        with self.assertRaises(EngineProcessError):
            engine.process(_speech_chunk())

    def test_process_rejects_bad_shape(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process(np.zeros((CHUNK, 2), dtype=np.float32))

    def test_unload_resets_state(self) -> None:
        self.engine.process(_speech_chunk())
        self.engine.unload()
        self.assertFalse(self.engine.is_loaded())
        self.assertIsNone(self.engine.buffered_speech)
        self.assertFalse(self.engine._is_speaking)


class TestAudioPipelineStage1(unittest.TestCase):
    """VAD feeding downstream stages (docs/AUDIO_PIPELINE.md integration)."""

    def test_only_speech_reaches_asr_input(self) -> None:
        vad = VADEngine()
        vad.load()
        voice_chunks = []
        stream = ([_silence_chunk()] * 5 + [_speech_chunk()] * 10
                  + [_silence_chunk()] * 5)
        for chunk in stream:
            res = vad.process(chunk)
            if res.data is not None:
                voice_chunks.append(res.data)
        self.assertEqual(len(voice_chunks), 10)  # silence fully dropped
        vad.unload()


if __name__ == "__main__":
    unittest.main()
