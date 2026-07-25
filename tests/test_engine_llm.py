"""Unit tests for LanguageEngine (mock).

Source: docs/ENGINE_SPEC.md — "Testing Requirements" section.
"""

import unittest

import numpy as np

from src.core.errors import EngineProcessError
from src.engines.core.llm import LanguageEngine


class TestLanguageEngine(unittest.TestCase):
    def test_metadata(self) -> None:
        engine = LanguageEngine()
        meta = engine.metadata
        self.assertEqual(meta.name, "llm")
        self.assertEqual(meta.input_type, "multi_modal_features")
        self.assertEqual(meta.output_type, "text")
        self.assertIn("llm", meta.tags)

    def test_load(self) -> None:
        engine = LanguageEngine()
        engine.load()
        self.assertTrue(engine.is_loaded())

    def test_process_returns_text(self) -> None:
        engine = LanguageEngine()
        engine.load()
        tokens = np.zeros((8, 512), dtype=np.float32)
        result = engine.process({"visual_tokens": tokens, "prompt": "What do you see?"})
        self.assertIsInstance(result.data, str)
        self.assertTrue(result.data)
        self.assertEqual(result.metadata["num_visual_tokens"], 8)
        self.assertIsNotNone(result.latency_ms)

    def test_process_requires_load(self) -> None:
        engine = LanguageEngine()
        with self.assertRaises(EngineProcessError):
            engine.process({"visual_tokens": np.zeros((2, 512), dtype=np.float32)})

    def test_process_requires_visual_tokens(self) -> None:
        engine = LanguageEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process({"prompt": "hello"})

    def test_process_requires_2d_tokens(self) -> None:
        engine = LanguageEngine()
        engine.load()
        with self.assertRaises(EngineProcessError):
            engine.process({"visual_tokens": np.zeros(512, dtype=np.float32)})

    def test_unload(self) -> None:
        engine = LanguageEngine()
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded())


if __name__ == "__main__":
    unittest.main()
