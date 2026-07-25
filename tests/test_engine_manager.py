"""Unit tests for EngineManager lifecycle.

Source: docs/ENGINE_SPEC.md — "Engine Lifecycle" section.
"""

import tempfile
import unittest
from pathlib import Path

from src.core.engine_manager import EngineManager
from src.core.errors import EngineNotFoundError


class TestEngineManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manager = EngineManager(cache_dir=self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_engines(self) -> None:
        engines = self.manager.list_engines()
        self.assertEqual(
            engines,
            ["asr", "compressor", "encoder", "filter", "llm",
             "mcp", "memory", "sampler", "speaker", "vad"],
        )

    def test_engine_info(self) -> None:
        info = self.manager.engine_info("encoder")
        self.assertEqual(info["version"], "0.1.0")
        self.assertEqual(info["size_mb"], 200)

    def test_engine_info_unknown(self) -> None:
        with self.assertRaises(EngineNotFoundError):
            self.manager.engine_info("does_not_exist")

    def test_download_mock_creates_dir(self) -> None:
        path = self.manager.download("encoder", mirror="modelscope")
        self.assertTrue(Path(path).is_dir())
        self.assertTrue((Path(path) / ".downloaded").exists())
        self.assertTrue(self.manager.is_installed("encoder"))

    def test_download_unknown_raises(self) -> None:
        with self.assertRaises(EngineNotFoundError):
            self.manager.download("does_not_exist")

    def test_load_and_unload(self) -> None:
        engine = self.manager.load("sampler")
        self.assertTrue(engine.is_loaded())
        self.assertTrue(self.manager.is_loaded("sampler"))
        self.manager.unload("sampler")
        self.assertFalse(self.manager.is_loaded("sampler"))

    def test_load_idempotent(self) -> None:
        first = self.manager.load("memory")
        second = self.manager.load("memory")
        self.assertIs(first, second)

    def test_load_unimplemented_engine(self) -> None:
        # Registered in registry.yaml but not implemented yet.
        with self.assertRaises(EngineNotFoundError):
            self.manager.load("speaker")

    def test_unload_idempotent(self) -> None:
        self.manager.unload("encoder")  # never loaded — must not raise


if __name__ == "__main__":
    unittest.main()
