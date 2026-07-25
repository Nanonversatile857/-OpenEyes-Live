"""Unit tests for VisualEncoderEngine (real CLIP ViT-B/32, quantized ONNX).

Source: docs/VIDEO_PIPELINE.md — "Stage 3: Visual Encoder Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".

Model-dependent tests are skipped when the ONNX model is not present
(e.g. CI, where models/ is gitignored).
"""

import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from src.core.errors import EngineLoadError, EngineProcessError
from src.engines.video.encoder import VisualEncoderEngine

MODEL_PATH = Path("./models/encoder/clip-vit-b32/vision_model_quantized.onnx")
HAS_MODEL = MODEL_PATH.exists() and find_spec("onnxruntime") is not None


def _frame(seed: int, size=(64, 64, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size, dtype=np.uint8)


class TestVisualEncoderInterface(unittest.TestCase):
    """Interface tests — no model required."""

    def test_metadata(self) -> None:
        engine = VisualEncoderEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "encoder")
        self.assertEqual(meta.version, "0.2.0")
        self.assertEqual(meta.input_type, "frames_rgb")
        self.assertEqual(meta.output_type, "visual_embeddings")
        self.assertIn("video", meta.tags)

    def test_load_missing_model_raises(self) -> None:
        engine = VisualEncoderEngine({"model_path": "./models/encoder/nope.onnx"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_unsupported_model_name(self) -> None:
        engine = VisualEncoderEngine({"model_name": "videomamba_t"})
        with self.assertRaises(EngineLoadError):
            engine.load()

    def test_process_requires_load(self) -> None:
        engine = VisualEncoderEngine()
        with self.assertRaises(EngineProcessError):
            engine.process([_frame(0)])

    def test_unload_idempotent(self) -> None:
        engine = VisualEncoderEngine()
        engine.unload()  # never loaded — must not raise


@unittest.skipUnless(HAS_MODEL, "CLIP vision model not downloaded")
class TestVisualEncoderReal(unittest.TestCase):
    """Real-model tests — require models/encoder/clip-vit-b32/."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = VisualEncoderEngine()
        cls.engine.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()

    def test_output_shape_and_dtype(self) -> None:
        result = self.engine.process([_frame(1) for _ in range(4)])
        self.assertEqual(result.data.shape, (4, 512))
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(result.metadata["num_frames"], 4)
        self.assertIsNotNone(result.latency_ms)

    def test_embeddings_are_l2_normalized(self) -> None:
        result = self.engine.process([_frame(1), _frame(2)])
        norms = np.linalg.norm(result.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_deterministic(self) -> None:
        frame = _frame(42)
        r1 = self.engine.process([frame]).data
        r2 = self.engine.process([frame]).data
        np.testing.assert_array_equal(r1, r2)

    def test_identical_frame_similarity_is_one(self) -> None:
        frame = _frame(7)
        self.assertAlmostEqual(self.engine.similarity(frame, frame), 1.0, places=4)

    def test_similar_frames_score_higher_than_random(self) -> None:
        base = _frame(10, size=(128, 128, 3))
        slightly_changed = np.clip(base.astype(int) + 3, 0, 255).astype(np.uint8)
        unrelated = _frame(99, size=(128, 128, 3))
        sim_close = self.engine.similarity(base, slightly_changed)
        sim_far = self.engine.similarity(base, unrelated)
        self.assertGreater(sim_close, sim_far)

    def test_process_rejects_empty(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process([])

    def test_process_rejects_bad_frame_shape(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process([np.zeros(512, dtype=np.float32)])


if __name__ == "__main__":
    unittest.main()
