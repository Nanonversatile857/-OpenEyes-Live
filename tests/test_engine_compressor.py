"""Unit tests for TokenCompressorEngine.

Source: docs/VIDEO_PIPELINE.md — "Stage 4: Token Compressor Engine";
        docs/ENGINE_SPEC.md — "Testing Requirements".
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.video.compressor import TokenCompressorEngine


def _tokens(t: int = 128, d: int = 512, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, d)).astype(np.float32)


class TestTokenCompressorEngine(unittest.TestCase):
    def test_metadata(self) -> None:
        engine = TokenCompressorEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "compressor")
        self.assertEqual(meta.input_type, "visual_tokens")
        self.assertEqual(meta.output_type, "visual_tokens_compressed")
        self.assertIn("compression", meta.tags)

    def test_passthrough_below_target(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 64})
        engine.load()
        tokens = _tokens(t=32)
        result = engine.process(tokens)
        self.assertEqual(result.data.shape, (32, 512))
        self.assertEqual(result.metadata["method"], "passthrough")
        np.testing.assert_array_equal(result.data, tokens)

    def test_selection_output_shape(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 32, "method": "selection"})
        engine.load()
        result = engine.process(_tokens(t=128))
        self.assertEqual(result.data.shape, (32, 512))
        self.assertEqual(result.metadata["output_tokens"], 32)

    def test_selection_preserves_temporal_order(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 16, "method": "selection",
                                        "preserve_temporal": True})
        engine.load()
        tokens = _tokens(t=64)
        result = engine.process(tokens)
        # Every output row must be an exact row of the input (no averaging).
        for row in result.data:
            self.assertTrue(any(np.array_equal(row, tokens[i]) for i in range(64)))

    def test_projection_output_shape(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 64, "method": "projection"})
        engine.load()
        result = engine.process(_tokens(t=256))
        self.assertEqual(result.data.shape, (64, 512))

    def test_projection_is_deterministic(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 32, "method": "projection"})
        engine.load()
        tokens = _tokens(t=128)
        r1 = engine.process(tokens).data
        r2 = engine.process(tokens).data
        np.testing.assert_array_equal(r1, r2)

    def test_hybrid_output_shape(self) -> None:
        engine = TokenCompressorEngine({"target_tokens": 32, "method": "hybrid"})
        engine.load()
        result = engine.process(_tokens(t=256))
        self.assertEqual(result.data.shape, (32, 512))

    def test_process_requires_load(self) -> None:
        engine = TokenCompressorEngine()
        with self.assertRaises(EngineProcessError):
            engine.process(_tokens())

    def test_process_requires_2d(self) -> None:
        engine = TokenCompressorEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process(np.zeros(512, dtype=np.float32))

    def test_unknown_method(self) -> None:
        engine = TokenCompressorEngine({"method": "bogus", "target_tokens": 8})
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process(_tokens(t=64))

    def test_unload(self) -> None:
        engine = TokenCompressorEngine()
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded())


class TestFullVideoPipeline(unittest.TestCase):
    """sampler -> filter -> encoder -> compressor -> llm end-to-end."""

    def test_full_chain(self) -> None:
        from src.engines.core.llm import LanguageEngine
        from src.engines.video.encoder import VisualEncoderEngine
        from src.engines.video.filter import FrameFilterEngine
        from src.engines.video.sampler import FrameSamplerEngine

        sampler = FrameSamplerEngine({"mode": "uniform", "target_fps": 15,
                                      "source_fps": 30})
        filt = FrameFilterEngine({"top_k": 4, "score_threshold": 0.0})
        encoder = VisualEncoderEngine({"embedding_dim": 512})
        compressor = TokenCompressorEngine({"target_tokens": 2, "method": "hybrid"})
        llm = LanguageEngine()
        for e in (sampler, filt, encoder, compressor, llm):
            e.load()

        frames = []
        rng = np.random.default_rng(0)
        for t in range(16):
            frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            res = sampler.process({"frame": frame, "timestamp": t / 30.0})
            if res.data is not None:
                frames.append(res.data)
        self.assertEqual(len(frames), 8)  # every 2nd frame

        selected = filt.process(frames).data
        self.assertEqual(len(selected), 4)

        tokens = encoder.process(selected).data
        self.assertEqual(tokens.shape, (4, 512))

        compressed = compressor.process(tokens).data
        self.assertEqual(compressed.shape, (2, 512))

        answer = llm.process({"visual_tokens": compressed})
        self.assertIsInstance(answer.data, str)
        self.assertEqual(answer.metadata["num_visual_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
