"""Unit tests for SpeakerEngine (real ERes2Net speaker embeddings).

Source: docs/AUDIO_PIPELINE.md — "Stage 3: Speaker Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".

Model-dependent tests are skipped when the ONNX model or the sherpa-onnx
runtime is not present (e.g. CI, where models/ is gitignored).
"""

import unittest
import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from src.core.errors import EngineLoadError, EngineProcessError
from src.engines.audio.speaker import SpeakerEngine

MODEL_PATH = Path("./models/speaker/3dspeaker_eres2net_base_16k.onnx")
TEST_WAV = Path("./models/asr/sense-voice/test_zh.wav")
HAS_MODEL = (
    MODEL_PATH.exists()
    and TEST_WAV.exists()
    and find_spec("sherpa_onnx") is not None
)

SAMPLE_RATE = 16000


def _speech(seconds: float = 0.0) -> np.ndarray:
    """Real speech (Chinese test wav), optionally truncated to ``seconds``."""
    with wave.open(str(TEST_WAV), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if seconds > 0:
        audio = audio[: int(SAMPLE_RATE * seconds)]
    return audio


def _noise(seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32)
            * 0.01)


class TestSpeakerEngineInterface(unittest.TestCase):
    """Interface tests — no model required."""

    def test_metadata(self) -> None:
        meta = SpeakerEngine().metadata
        self.assertEqual(meta.name, "speaker")
        self.assertEqual(meta.version, "0.3.0")
        self.assertIn("speaker", meta.tags)

    def test_load_missing_model_raises(self) -> None:
        engine = SpeakerEngine({"model_path": "./models/speaker/nope.onnx"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_process_requires_load(self) -> None:
        engine = SpeakerEngine()
        with self.assertRaises(EngineProcessError):
            engine.process({"audio": np.zeros(SAMPLE_RATE, dtype=np.float32)})

    def test_identify_requires_load(self) -> None:
        engine = SpeakerEngine()
        with self.assertRaises(EngineProcessError):
            engine.identify(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def test_unload_idempotent(self) -> None:
        SpeakerEngine().unload()  # never loaded — must not raise

    def test_speakers_empty_when_not_loaded(self) -> None:
        self.assertEqual(SpeakerEngine().speakers, [])


@unittest.skipUnless(HAS_MODEL, "speaker model / test wav not downloaded")
class TestSpeakerEngineReal(unittest.TestCase):
    """Real model tests — require models/speaker + sherpa-onnx."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SpeakerEngine()
        cls.engine.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()

    def setUp(self) -> None:
        for name in list(self.engine.speakers):
            self.engine.remove(name)

    def test_embedding_dim_and_dtype(self) -> None:
        result = self.engine.process({"audio": _speech()})
        emb = result.data
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertGreater(float(np.linalg.norm(emb)), 0.0)

    def test_int16_input_accepted(self) -> None:
        audio_int16 = (_speech() * 32768.0).astype(np.int16)
        result = self.engine.process({"audio": audio_int16})
        self.assertEqual(result.data.shape, (512,))

    def test_enroll_and_identify_same_speaker(self) -> None:
        self.engine.enroll("zhu", _speech())
        name, score = self.engine.identify(_speech(2.0))
        self.assertEqual(name, "zhu")
        self.assertGreater(score, 0.5)

    def test_identify_rejects_noise(self) -> None:
        self.engine.enroll("zhu", _speech())
        name, score = self.engine.identify(_noise())
        self.assertEqual(name, "")
        self.assertLess(score, 0.5)

    def test_two_speakers_best_match_wins(self) -> None:
        self.engine.enroll("speech", _speech())
        self.engine.enroll("noise", _noise())
        name, _ = self.engine.identify(_speech(2.0))
        self.assertEqual(name, "speech")

    def test_identify_without_enrollment_raises(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.identify(_speech(1.0))

    def test_enroll_empty_name_raises(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.enroll("", _speech(1.0))

    def test_remove_speaker(self) -> None:
        self.engine.enroll("temp", _speech())
        self.assertIn("temp", self.engine.speakers)
        self.assertTrue(self.engine.remove("temp"))
        self.assertNotIn("temp", self.engine.speakers)
        self.assertFalse(self.engine.remove("temp"))

    def test_process_rejects_bad_input(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process({"audio": np.zeros((10, 2), dtype=np.float32)})
        with self.assertRaises(EngineProcessError):
            self.engine.process({"audio": np.array([], dtype=np.float32)})
        with self.assertRaises(EngineProcessError):
            self.engine.process({"samples": _speech(1.0)})


if __name__ == "__main__":
    unittest.main()
