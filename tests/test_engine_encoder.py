"""Unit tests for VisualEncoderEngine (mock).

Source: docs/ENGINE_SPEC.md — "Testing Requirements" section.
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.video.encoder import VisualEncoderEngine


def _fake_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestVisualEncoderEngine(unittest.TestCase):
    def test_metadata(self) -> None:
        engine = VisualEncoderEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "encoder")
        self.assertEqual(meta.version, "0.1.0")
        self.assertEqual(meta.input_type, "frames_rgb")
        self.assertEqual(meta.output_type, "visual_embeddings")
        self.assertIn("video", meta.tags)

    def test_load(self) -> None:
        engine = VisualEncoderEngine({"embedding_dim": 512})
        engine.load()
        self.assertTrue(engine.is_loaded())

    def test_load_idempotent(self) -> None:
        engine = VisualEncoderEngine()
        engine.load()
        engine.load()
        self.assertTrue(engine.is_loaded())

    def test_process_shape(self) -> None:
        engine = VisualEncoderEngine({"embedding_dim": 512})
        engine.load()
        frames = [_fake_frame() for _ in range(4)]
        result = engine.process(frames)
        self.assertIsInstance(result.data, np.ndarray)
        self.assertEqual(result.data.shape, (4, 512))
        self.assertEqual(result.metadata["num_frames"], 4)
        self.assertIsNotNone(result.latency_ms)

    def test_process_requires_load(self) -> None:
        engine = VisualEncoderEngine()
        with self.assertRaises(EngineProcessError):
            engine.process([_fake_frame()])

    def test_process_rejects_empty(self) -> None:
        engine = VisualEncoderEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process([])

    def test_unload(self) -> None:
        engine = VisualEncoderEngine()
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded())


if __name__ == "__main__":
    unittest.main()
