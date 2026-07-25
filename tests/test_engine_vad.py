"""Unit tests for VADEngine (real Silero VAD).

Source: docs/AUDIO_PIPELINE.md — "Stage 1: VAD Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".

Model-dependent tests are skipped when the ONNX model is not present
(e.g. CI, where models/ is gitignored).
"""

import unittest
import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from src.core.errors import EngineLoadError, EngineProcessError
from src.engines.audio.vad import SILENCE_CHUNKS_TO_END, VADEngine

MODEL_PATH = Path("./models/vad/silero_vad.onnx")
TEST_WAV = Path("./models/asr/sense-voice/test_zh.wav")
HAS_MODEL = (
    MODEL_PATH.exists()
    and TEST_WAV.exists()
    and find_spec("onnxruntime") is not None
)

SAMPLE_RATE = 16000
CHUNK = 480  # 30 ms @ 16 kHz


def _speech_chunks() -> list:
    """Real speech (Chinese test wav) split into 30 ms chunks."""
    with wave.open(str(TEST_WAV), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK, CHUNK)]


def _silence() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.float32)


class TestVADEngineInterface(unittest.TestCase):
    """Interface tests — no model required."""

    def test_metadata(self) -> None:
        engine = VADEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "vad")
        self.assertEqual(meta.input_type, "audio_pcm")
        self.assertEqual(meta.output_type, "audio_pcm_voice")
        self.assertIn("detection", meta.tags)

    def test_load_missing_model_raises(self) -> None:
        engine = VADEngine({"model_path": "./models/vad/does_not_exist.onnx"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_process_requires_load(self) -> None:
        engine = VADEngine()
        with self.assertRaises(EngineProcessError):
            engine.process(_silence())

    def test_unload_idempotent(self) -> None:
        engine = VADEngine()
        engine.unload()  # never loaded — must not raise


@unittest.skipUnless(HAS_MODEL, "Silero VAD model / test wav not downloaded")
class TestVADEngineReal(unittest.TestCase):
    """Real-model tests — require models/vad/silero_vad.onnx."""

    def setUp(self) -> None:
        self.engine = VADEngine()
        self.engine.load()

    def tearDown(self) -> None:
        self.engine.unload()

    def test_real_speech_detected(self) -> None:
        probs = [self.engine.process(c).metadata["speech_prob"]
                 for c in _speech_chunks()[:40]]
        self.assertGreater(max(probs), 0.5)

    def test_silence_dropped(self) -> None:
        for _ in range(10):
            result = self.engine.process(_silence())
            self.assertIsNone(result.data)
            self.assertLess(result.metadata["speech_prob"], 0.5)

    def test_segment_ends_after_trailing_silence(self) -> None:
        for c in _speech_chunks()[:30]:
            self.engine.process(c)
        ended_at = None
        for i in range(SILENCE_CHUNKS_TO_END + 2):
            res = self.engine.process(_silence())
            if res.metadata["segment_ended"]:
                ended_at = i
                break
        self.assertEqual(ended_at, SILENCE_CHUNKS_TO_END)
        self.assertFalse(self.engine._is_speaking)

    def test_int16_input_accepted(self) -> None:
        chunk = (_silence() * 32767).astype(np.int16)
        result = self.engine.process(chunk)
        self.assertIsNone(result.data)  # silence, but no crash on dtype

    def test_max_speech_duration_cuts_segment(self) -> None:
        engine = VADEngine({"max_speech_duration_s": 0.5})
        engine.load()
        ended = False
        for c in _speech_chunks():  # full wav: ~3.8s of speech, leading silence
            if engine.process(c).metadata["segment_ended"]:
                ended = True
                break
        self.assertTrue(ended)
        engine.unload()

    def test_process_rejects_bad_shape(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process(np.zeros((CHUNK, 2), dtype=np.float32))

    def test_unload_resets_state(self) -> None:
        self.engine.process(_speech_chunks()[0])
        self.engine.unload()
        self.assertFalse(self.engine.is_loaded())
        self.assertIsNone(self.engine.buffered_speech)

    def test_only_speech_reaches_downstream(self) -> None:
        voice = 0
        for c in [_silence()] * 10 + _speech_chunks()[:30]:
            if self.engine.process(c).data is not None:
                voice += 1
        self.assertGreater(voice, 0)
        self.assertLessEqual(voice, 30)


if __name__ == "__main__":
    unittest.main()
