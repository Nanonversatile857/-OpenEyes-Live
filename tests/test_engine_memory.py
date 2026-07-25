"""Unit tests for MemoryEngine.

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 2. MemoryEngine"
        and "Testing Requirements"; docs/API_REFERENCE.md — MemoryEngine usage.
"""

import tempfile
import unittest
from pathlib import Path

from src.core.errors import EngineProcessError
from src.engines.core.memory import MemoryEngine, MemoryResult


class TestMemoryEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "memory.json")
        self.engine = MemoryEngine({"db_path": self.db})
        self.engine.load()

    def tearDown(self) -> None:
        self.engine.unload()
        self._tmp.cleanup()

    def test_metadata(self) -> None:
        meta = self.engine.metadata
        self.assertEqual(meta.name, "memory")
        self.assertEqual(meta.input_type, "memory_operation")
        self.assertEqual(meta.output_type, "memory_result")
        self.assertIn("retrieval", meta.tags)

    def test_store_and_query(self) -> None:
        self.engine.store("Cat is sleeping on the sofa",
                          "2026-07-25T14:30:00", {"location": "sofa"})
        self.engine.store("A car drives past the window",
                          "2026-07-25T14:31:00")
        results = self.engine.query("cat").data
        self.assertTrue(results)
        top = results[0]
        self.assertIsInstance(top, MemoryResult)
        self.assertIn("Cat", top.description)
        self.assertEqual(top.metadata["location"], "sofa")
        self.assertIsNotNone(top.score)
        # The cat record must outrank the unrelated one.
        self.assertGreater(results[0].score, results[-1].score)

    def test_query_limit(self) -> None:
        for i in range(10):
            self.engine.store(f"observation number {i}", f"2026-07-25T14:{30+i}:00")
        results = self.engine.query("observation", limit=3).data
        self.assertEqual(len(results), 3)

    def test_query_empty_db(self) -> None:
        results = self.engine.query("anything").data
        self.assertEqual(results, [])

    def test_get_timeline(self) -> None:
        self.engine.store("morning", "2026-07-25T08:00:00")
        self.engine.store("noon", "2026-07-25T12:00:00")
        self.engine.store("night", "2026-07-25T22:00:00")
        hits = self.engine.get_timeline("2026-07-25T09:00:00", "2026-07-25T23:00:00")
        self.assertEqual([h.description for h in hits], ["noon", "night"])

    def test_process_store_dispatch(self) -> None:
        result = self.engine.process({"observation": "hello world"})
        self.assertEqual(result.data, "stored")
        self.assertEqual(result.metadata["total_records"], 1)

    def test_process_query_dispatch(self) -> None:
        self.engine.store("a dog barks loudly", "2026-07-25T10:00:00")
        result = self.engine.process({"query": "dog", "limit": 1})
        self.assertEqual(len(result.data), 1)

    def test_process_rejects_bad_op(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.process({"nonsense": True})

    def test_store_requires_load(self) -> None:
        engine = MemoryEngine({"db_path": ""})
        with self.assertRaises(EngineProcessError):
            engine.store("x")

    def test_store_rejects_empty_observation(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.engine.store("   ")

    def test_persistence_across_reload(self) -> None:
        self.engine.store("persistent memory", "2026-07-25T15:00:00")
        self.engine.unload()

        fresh = MemoryEngine({"db_path": self.db})
        fresh.load()
        results = fresh.query("persistent").data
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].description, "persistent memory")
        fresh.unload()

    def test_unload_idempotent(self) -> None:
        self.engine.unload()
        self.engine.unload()  # must not raise
        self.assertFalse(self.engine.is_loaded())


if __name__ == "__main__":
    unittest.main()
