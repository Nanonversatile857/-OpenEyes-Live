"""Unit tests for LanguageEngine (real Phi-3.5-vision VLM).

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 1. LanguageEngine"
        and "Testing Requirements"; docs/API_REFERENCE.md — LanguageEngine.

Model-dependent tests are skipped when the model is not present
(e.g. CI, where models/ is gitignored) — the real test loads a 3.2GB
model and generates text on CPU, so it is slow by nature.
"""

import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from src.core.errors import EngineLoadError, EngineProcessError
from src.engines.core.llm import LanguageEngine

MODEL_DIR = Path("./models/llm/phi-3.5-vision-int4")
HAS_MODEL = (
    (MODEL_DIR / "genai_config.json").exists()
    and find_spec("onnxruntime_genai") is not None
)


def _frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (240, 320, 3), dtype=np.uint8)


class TestLanguageEngineInterface(unittest.TestCase):
    """Interface tests — no model required."""

    def test_metadata(self) -> None:
        engine = LanguageEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "llm")
        self.assertEqual(meta.input_type, "multi_modal_features")
        self.assertEqual(meta.output_type, "text")
        self.assertIn("vlm", meta.tags)

    def test_load_missing_model_raises(self) -> None:
        engine = LanguageEngine({"model_path": "./models/llm/nope"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_process_requires_load(self) -> None:
        engine = LanguageEngine()
        with self.assertRaises(EngineProcessError):
            engine.process({"frames": [_frame()]})

    def test_unload_idempotent(self) -> None:
        engine = LanguageEngine()
        engine.unload()

    def test_downscale_keeps_small_frames(self) -> None:
        frame = _frame()  # 240x320 — already under 336
        out = LanguageEngine._downscale(frame, 336, _FakeCv2)
        self.assertIs(out, frame)

    def test_downscale_disabled_with_zero(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = LanguageEngine._downscale(frame, 0, _FakeCv2)
        self.assertIs(out, frame)

    def test_downscale_preserves_aspect(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = LanguageEngine._downscale(frame, 336, _FakeCv2)
        self.assertEqual(_FakeCv2.last_size, (336, 252))  # (w, h), 4:3 kept


class _FakeCv2:
    """Stand-in for the cv2 module inside _downscale (no real resize)."""

    INTER_AREA = 3
    last_size = None

    @staticmethod
    def resize(frame, size, interpolation=0):  # noqa: ARG004
        _FakeCv2.last_size = size
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


@unittest.skipUnless(HAS_MODEL, "Phi-3.5-vision model not downloaded")
class TestLanguageEngineReal(unittest.TestCase):
    """Real-model tests — slow (CPU inference), local only.

    Only one generation test is kept: each VLM call costs ~1 minute on CPU.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = LanguageEngine({"max_tokens": 16})
        cls.engine.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()

    def test_describes_frame(self) -> None:
        result = self.engine.process(
            {"frames": [_frame(1)], "prompt": "Describe this image."}
        )
        self.assertIsInstance(result.data, str)
        self.assertTrue(result.data.strip())
        self.assertGreater(result.metadata["generated_tokens"], 0)

    def test_visual_tokens_only_raises(self) -> None:
        tokens = np.zeros((8, 512), dtype=np.float32)
        with self.assertRaises(EngineProcessError):
            self.engine.process({"visual_tokens": tokens})

    def test_empty_frames_raises(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process({"frames": []})


if __name__ == "__main__":
    unittest.main()
