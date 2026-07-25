"""Unit tests for EngineManager lifecycle.

Source: docs/ENGINE_SPEC.md — "Engine Lifecycle" section.

Download tests mock ``urllib.request.urlopen`` — no real network access.
"""

import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src.core.engine_manager import EngineManager
from src.core.errors import EngineDownloadError, EngineNotFoundError

#: Declared size of models/vad/silero_vad.onnx in registry.yaml.
VAD_MODEL_SIZE = 2243022


class _FakeResponse:
    """Minimal stand-in for an http.client.HTTPResponse."""

    def __init__(self, data: bytes, status: int = 200) -> None:
        self._buf = io.BytesIO(data)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self._buf.close()


def _urlopen_returning(data: bytes, status: int = 200):
    """Build a urlopen replacement that serves ``data`` (any Range honored)."""

    def fake_urlopen(request, timeout=0):  # noqa: ARG001
        range_header = request.headers.get("Range")
        if range_header and status == 206:
            start = int(range_header.replace("bytes=", "").split("-")[0])
            return _FakeResponse(data[start:], status=206)
        return _FakeResponse(data, status=200)

    return fake_urlopen


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
        self.assertEqual(info["version"], "0.2.0")
        self.assertEqual(info["size_mb"], 89)

    def test_engine_info_unknown(self) -> None:
        with self.assertRaises(EngineNotFoundError):
            self.manager.engine_info("does_not_exist")

    def test_download_internal_engine_needs_no_network(self) -> None:
        # sampler is `source: internal` — download() must not touch urlopen.
        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen"
        ) as mocked:
            path = self.manager.download("sampler")
            mocked.assert_not_called()
        self.assertTrue(Path(path).is_dir())
        self.assertTrue((Path(path) / ".downloaded").exists())
        self.assertTrue(self.manager.is_installed("sampler"))

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

    def test_all_registry_engines_have_implementations(self) -> None:
        # Every registry entry must resolve to an engine class (speaker
        # landed in v0.3.0; there are no planned-only entries left).
        for name in self.manager.list_engines():
            cls = self.manager._resolve_engine_class(name)
            self.assertTrue(callable(cls), name)

    def test_unload_idempotent(self) -> None:
        self.manager.unload("encoder")  # never loaded — must not raise


class TestEngineDownload(unittest.TestCase):
    """Real download logic against a mocked network layer."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manager = EngineManager(cache_dir=self._tmp.name)
        self.model_data = b"\x00" * VAD_MODEL_SIZE

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _download_vad(self, status: int = 200, **kwargs) -> Path:
        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=_urlopen_returning(self.model_data, status=status),
        ):
            return self.manager.download("vad", **kwargs)

    def test_download_fetches_and_verifies(self) -> None:
        path = self._download_vad()
        dest = Path(path) / "silero_vad.onnx"
        self.assertTrue(dest.exists())
        self.assertEqual(dest.stat().st_size, VAD_MODEL_SIZE)
        self.assertFalse(dest.with_name(dest.name + ".part").exists())
        self.assertTrue(self.manager.is_installed("vad"))
        marker = (Path(path) / ".downloaded").read_text(encoding="utf-8")
        self.assertIn("real=true", marker)

    def test_download_skips_complete_files(self) -> None:
        self._download_vad()
        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen"
        ) as mocked:
            self.manager.download("vad")
            mocked.assert_not_called()

    def test_download_resumes_partial_file(self) -> None:
        # Simulate an interrupted download: half the file is already there.
        engine_dir = Path(self._tmp.name) / "vad"
        engine_dir.mkdir(parents=True)
        part = engine_dir / "silero_vad.onnx.part"
        part.write_bytes(self.model_data[:1000])

        captured = []

        def spy_urlopen(request, timeout=0):  # noqa: ARG001
            captured.append(request.headers.get("Range"))
            start = int(request.headers["Range"].replace("bytes=", "").split("-")[0])
            return _FakeResponse(self.model_data[start:], status=206)

        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=spy_urlopen,
        ):
            self.manager.download("vad")

        self.assertEqual(captured, ["bytes=1000-"])
        dest = engine_dir / "silero_vad.onnx"
        self.assertEqual(dest.stat().st_size, VAD_MODEL_SIZE)

    def test_download_falls_back_to_mirror(self) -> None:
        calls = []

        def flaky_urlopen(request, timeout=0):  # noqa: ARG001
            calls.append(request.full_url)
            if "huggingface.co" in request.full_url:
                raise urllib.error.URLError("connection reset")
            return _FakeResponse(self.model_data)

        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=flaky_urlopen,
        ):
            path = self.manager.download("vad")

        self.assertEqual(len(calls), 2)
        self.assertIn("hf-mirror.com", calls[1])
        marker = (Path(path) / ".downloaded").read_text(encoding="utf-8")
        self.assertIn("source=hf-mirror", marker)

    def test_download_size_mismatch_raises(self) -> None:
        short = self.model_data[:-10]
        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=_urlopen_returning(short),
        ):
            with self.assertRaises(EngineDownloadError):
                self.manager.download("vad", mirror="hf-mirror")
        # Partial file is kept for the next resume attempt.
        part = Path(self._tmp.name) / "vad" / "silero_vad.onnx.part"
        self.assertTrue(part.exists())
        self.assertFalse(self.manager.is_installed("vad"))

    def test_download_all_sources_fail(self) -> None:
        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=urllib.error.URLError("no route"),
        ):
            with self.assertRaises(EngineDownloadError):
                self.manager.download("vad")

    def test_download_unknown_mirror_raises(self) -> None:
        with self.assertRaises(EngineDownloadError):
            self.manager.download("vad", mirror="modelscope")

    def test_download_direct_url_ignores_mirrors(self) -> None:
        # speaker uses an absolute GitHub release URL — mirror sources only
        # apply to hf_repo files, so urlopen must be called exactly once
        # with the registry URL even when huggingface is "down".
        engine_dir_url = (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/"
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        )
        data = b"\x00" * 39593761
        calls = []

        def fake_urlopen(request, timeout=0):  # noqa: ARG001
            calls.append(request.full_url)
            return _FakeResponse(data)

        with mock.patch(
            "src.core.engine_manager.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            path = self.manager.download("speaker", mirror="hf-mirror")

        self.assertEqual(calls, [engine_dir_url])
        self.assertTrue(self.manager.is_installed("speaker"))
        dest = Path(path) / "3dspeaker_eres2net_base_16k.onnx"
        self.assertEqual(dest.stat().st_size, 39593761)

    def test_is_installed_false_without_marker(self) -> None:
        self.assertFalse(self.manager.is_installed("vad"))

    def test_is_installed_true_for_manually_fetched_model(self) -> None:
        # No marker, but the model file is complete (e.g. downloaded by hand).
        engine_dir = Path(self._tmp.name) / "vad"
        engine_dir.mkdir(parents=True)
        (engine_dir / "silero_vad.onnx").write_bytes(self.model_data)
        self.assertTrue(self.manager.is_installed("vad"))

    def test_is_installed_detects_truncated_file(self) -> None:
        self._download_vad()
        dest = Path(self._tmp.name) / "vad" / "silero_vad.onnx"
        dest.write_bytes(b"\x00" * 10)  # corrupt it
        self.assertFalse(self.manager.is_installed("vad"))


if __name__ == "__main__":
    unittest.main()
