"""Unit tests for FrameFilterEngine.

Source: docs/VIDEO_PIPELINE.md — "Stage 2: Frame Filter Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.video.filter import FrameFilterEngine


def _busy_frame(seed: int) -> np.ndarray:
    """High-information frame (random noise -> high sharpness/colorfulness)."""
    return np.random.default_rng(seed).integers(
        0, 255, (64, 64, 3), dtype=np.uint8
    )


def _flat_frame(value: int = 128) -> np.ndarray:
    """Low-information frame (uniform color)."""
    return np.full((64, 64, 3), value, dtype=np.uint8)


class TestFrameFilterEngine(unittest.TestCase):
    def test_metadata(self) -> None:
        engine = FrameFilterEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "filter")
        self.assertEqual(meta.input_type, "video_frames")
        self.assertEqual(meta.output_type, "video_frames")
        self.assertIn("attention", meta.tags)

    def test_attention_prefers_busy_frames(self) -> None:
        engine = FrameFilterEngine({"top_k": 2, "method": "attention",
                                    "score_threshold": 0.0})
        engine.load()
        frames = [_flat_frame(), _busy_frame(1), _flat_frame(), _busy_frame(2)]
        result = engine.process(frames)
        self.assertEqual(len(result.data), 2)
        # Selected frames must be the two noisy (high-information) ones.
        self.assertEqual(result.metadata["selected_indices"], [1, 3])

    def test_attention_threshold_keeps_at_least_one(self) -> None:
        engine = FrameFilterEngine({"top_k": 4, "method": "attention",
                                    "score_threshold": 1.1})
        engine.load()
        result = engine.process([_flat_frame(), _flat_frame(200)])
        self.assertGreaterEqual(len(result.data), 1)

    def test_diversity_selection(self) -> None:
        engine = FrameFilterEngine({"top_k": 2, "method": "diversity"})
        engine.load()
        frames = [_busy_frame(s) for s in range(6)]
        result = engine.process(frames)
        self.assertEqual(len(result.data), 2)
        indices = result.metadata["selected_indices"]
        self.assertEqual(indices, sorted(indices))  # time-ordered output

    def test_hybrid_selection(self) -> None:
        engine = FrameFilterEngine({"top_k": 2, "method": "hybrid",
                                    "score_threshold": 0.0})
        engine.load()
        frames = [_busy_frame(s) for s in range(8)]
        result = engine.process(frames)
        self.assertEqual(len(result.data), 2)

    def test_passthrough_when_fewer_than_top_k(self) -> None:
        engine = FrameFilterEngine({"top_k": 8})
        engine.load()
        frames = [_busy_frame(1), _busy_frame(2)]
        result = engine.process(frames)
        self.assertEqual(len(result.data), 2)

    def test_buffer_size_caps_input(self) -> None:
        engine = FrameFilterEngine({"top_k": 8, "buffer_size": 3})
        engine.load()
        frames = [_busy_frame(s) for s in range(10)]
        result = engine.process(frames)
        self.assertEqual(result.metadata["input_frames"], 3)

    def test_process_requires_load(self) -> None:
        engine = FrameFilterEngine()
        with self.assertRaises(EngineProcessError):
            engine.process([_flat_frame()])

    def test_process_rejects_empty(self) -> None:
        engine = FrameFilterEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process([])

    def test_unknown_method(self) -> None:
        engine = FrameFilterEngine({"method": "bogus", "top_k": 1})
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process([_busy_frame(1), _busy_frame(2)])

    def test_unload(self) -> None:
        engine = FrameFilterEngine()
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded())


class TestVideoPipelineIntegration(unittest.TestCase):
    """sampler -> filter -> encoder chain (docs/ENGINE_SPEC.md integration tests)."""

    def test_chain(self) -> None:
        from src.engines.video.encoder import VisualEncoderEngine
        from src.engines.video.sampler import FrameSamplerEngine

        sampler = FrameSamplerEngine({"mode": "uniform", "target_fps": 30,
                                      "source_fps": 30})
        filt = FrameFilterEngine({"top_k": 2, "score_threshold": 0.0})
        encoder = VisualEncoderEngine({"embedding_dim": 512})
        for e in (sampler, filt, encoder):
            e.load()

        sampled = []
        for t in range(6):
            res = sampler.process({"frame": _busy_frame(t), "timestamp": t / 30.0})
            if res.data is not None:
                sampled.append(res.data)
        self.assertEqual(len(sampled), 6)  # 30/30 -> every frame

        selected = filt.process(sampled).data
        self.assertEqual(len(selected), 2)

        encoded = encoder.process(selected)
        self.assertEqual(encoded.data.shape, (2, 512))


if __name__ == "__main__":
    unittest.main()
