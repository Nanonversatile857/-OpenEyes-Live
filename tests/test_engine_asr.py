"""Unit tests for ASREngine (real SenseVoice via sherpa-onnx).

Source: docs/AUDIO_PIPELINE.md — "Stage 2: ASR Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".

Model-dependent tests are skipped when the model is not present
(e.g. CI, where models/ is gitignored).
"""

import unittest
import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from src.core.errors import EngineLoadError, EngineProcessError
from src.engines.audio.asr import ASREngine

MODEL_PATH = Path("./models/asr/sense-voice/model.int8.onnx")
TOKENS_PATH = Path("./models/asr/sense-voice/tokens.txt")
TEST_WAV = Path("./models/asr/sense-voice/test_zh.wav")
HAS_MODEL = (
    MODEL_PATH.exists()
    and TOKENS_PATH.exists()
    and TEST_WAV.exists()
    and find_spec("sherpa_onnx") is not None
)


def _load_test_audio() -> np.ndarray:
    with wave.open(str(TEST_WAV), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class TestASREngineInterface(unittest.TestCase):
    """Interface tests — no model required."""

    def test_metadata(self) -> None:
        engine = ASREngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "asr")
        self.assertEqual(meta.input_type, "audio_pcm_voice")
        self.assertEqual(meta.output_type, "text")
        self.assertIn("speech-to-text", meta.tags)

    def test_load_missing_model_raises(self) -> None:
        engine = ASREngine({"model_path": "./nope.onnx", "tokens_path": "./nope.txt"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_unsupported_model_name(self) -> None:
        engine = ASREngine({"model_name": "parakeet"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_process_requires_load(self) -> None:
        engine = ASREngine()
        with self.assertRaises(EngineProcessError):
            engine.process(np.zeros(16000, dtype=np.float32))

    def test_unload_idempotent(self) -> None:
        engine = ASREngine()
        engine.unload()


@unittest.skipUnless(HAS_MODEL, "SenseVoice model / test wav not downloaded")
class TestASREngineReal(unittest.TestCase):
    """Real-model tests — require models/asr/sense-voice/."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ASREngine()
        cls.engine.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()

    def test_transcribes_chinese_speech(self) -> None:
        result = self.engine.process(_load_test_audio())
        self.assertIn("开放时间", result.data)
        self.assertIn("下午5点", result.data)
        self.assertGreater(result.metadata["duration_s"], 4.0)

    def test_faster_than_realtime(self) -> None:
        result = self.engine.process(_load_test_audio())
        # SenseVoice int8 should do well under 1.0 RTF on a desktop CPU.
        self.assertLess(result.metadata["rtf"], 1.0)

    def test_int16_input_accepted(self) -> None:
        audio = (_load_test_audio() * 32767).astype(np.int16)
        result = self.engine.process(audio)
        self.assertIn("开放时间", result.data)

    def test_empty_audio_rejected(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process(np.zeros(0, dtype=np.float32))

    def test_bad_shape_rejected(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process(np.zeros((16000, 2), dtype=np.float32))

    def test_silence_yields_empty_or_short_text(self) -> None:
        result = self.engine.process(np.zeros(16000, dtype=np.float32))
        self.assertLessEqual(len(result.data), 4)


if __name__ == "__main__":
    unittest.main()
