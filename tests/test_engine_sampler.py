"""Unit tests for FrameSamplerEngine.

Source: docs/VIDEO_PIPELINE.md — "Stage 1: Frame Sampler Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.video.sampler import FrameSamplerEngine


def _frame(value: int = 0) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


class TestFrameSamplerEngine(unittest.TestCase):
    def test_metadata(self) -> None:
        engine = FrameSamplerEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "sampler")
        self.assertEqual(meta.input_type, "video_frame")
        self.assertEqual(meta.output_type, "video_frame")
        self.assertIn("sampling", meta.tags)

    def test_uniform_sampling_interval(self) -> None:
        # source 30 fps / target 10 fps -> every 3rd frame.
        engine = FrameSamplerEngine({"mode": "uniform", "target_fps": 10, "source_fps": 30})
        engine.load()
        decisions = [
            engine.process({"frame": _frame(), "timestamp": t / 30.0}).data is not None
            for t in range(9)
        ]
        self.assertEqual(decisions, [False, False, True, False, False, True,
                                     False, False, True])

    def test_uniform_accepts_raw_ndarray(self) -> None:
        engine = FrameSamplerEngine({"mode": "uniform", "target_fps": 30, "source_fps": 30})
        engine.load()
        result = engine.process(_frame(10))
        self.assertIsNotNone(result.data)
        self.assertTrue(result.metadata["sampled"])

    def test_dynamic_samples_first_frame(self) -> None:
        engine = FrameSamplerEngine({"mode": "dynamic"})
        engine.load()
        # First frame: last_sample_time = 0 -> sampled.
        result = engine.process({"frame": _frame(), "timestamp": 100.0})
        self.assertTrue(result.metadata["sampled"])

    def test_dynamic_suppresses_rapid_frames(self) -> None:
        engine = FrameSamplerEngine({"mode": "dynamic", "min_interval": 0.5,
                                     "max_interval": 5.0})
        engine.load()
        # Identical frames -> motion 0 -> interval stays max (5s).
        engine.process({"frame": _frame(), "timestamp": 100.0})
        r2 = engine.process({"frame": _frame(), "timestamp": 101.0})
        self.assertIsNone(r2.data)

    def test_keyframe_detects_scene_change(self) -> None:
        engine = FrameSamplerEngine({"mode": "keyframe_only", "keyframe_threshold": 0.15})
        engine.load()
        r1 = engine.process({"frame": _frame(0), "timestamp": 0.0})
        self.assertTrue(r1.metadata["sampled"])  # first frame always emitted
        r2 = engine.process({"frame": _frame(0), "timestamp": 1.0})
        self.assertIsNone(r2.data)  # static scene skipped
        r3 = engine.process({"frame": _frame(255), "timestamp": 2.0})
        self.assertIsNotNone(r3.data)  # scene change emitted

    def test_process_requires_load(self) -> None:
        engine = FrameSamplerEngine()
        with self.assertRaises(EngineProcessError):
            engine.process(_frame())

    def test_process_rejects_bad_input(self) -> None:
        engine = FrameSamplerEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process({"frame": "not a frame"})

    def test_unknown_mode(self) -> None:
        engine = FrameSamplerEngine({"mode": "bogus"})
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process(_frame())

    def test_unload_resets_state(self) -> None:
        engine = FrameSamplerEngine()
        engine.load()
        engine.process(_frame())
        engine.unload()
        self.assertFalse(engine.is_loaded())
        self.assertEqual(engine.frame_count, 0)


if __name__ == "__main__":
    unittest.main()
