"""Unit tests for the CLI entry point.

Source: docs/API_REFERENCE.md — "CLI Reference" section.
"""

import io
import unittest
from contextlib import redirect_stdout

from src.cli.main import __version__, main
from src.runtime.camera import Camera


class TestCli(unittest.TestCase):
    def test_list_outputs_engines(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["list"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        for name in ("encoder", "llm", "asr", "sampler"):
            self.assertIn(name, out)
        self.assertIn(__version__, out)

    def test_install_internal_engine(self) -> None:
        # Internal engines need no model files — no network access involved.
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["install", "sampler"])
        self.assertEqual(rc, 0)
        self.assertIn("sampler", buf.getvalue())

    def test_install_unknown_engine(self) -> None:
        with redirect_stdout(io.StringIO()):
            rc = main(["install", "nope"])
        self.assertEqual(rc, 1)

    def test_listen_help_mentions_speaker(self) -> None:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main(["listen", "--help"])
        except SystemExit:
            pass
        out = buf.getvalue()
        self.assertIn("--speaker", out)
        self.assertIn("--enroll", out)


class _StubSpeakerEngine:
    """Records add_embedding calls without a real model."""

    def __init__(self) -> None:
        self.added = []

    def add_embedding(self, name, vec) -> None:
        self.added.append((name, list(vec)))


class TestSpeakerDatabase(unittest.TestCase):
    """Speaker DB persistence helpers — no model required."""

    def test_save_load_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        from src.cli.main import _load_speaker_db, _save_speaker_db

        db = {"张三": [[0.1] * 512, [0.2] * 512], "lisa": [[0.3] * 512]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "enrolled.json"
            _save_speaker_db(db, path)
            engine = _StubSpeakerEngine()
            loaded = _load_speaker_db(engine, path)
        self.assertEqual(loaded, db)
        # 2 embeddings for 张三 + 1 for lisa
        self.assertEqual(len(engine.added), 3)
        self.assertEqual(engine.added[0][0], "张三")

    def test_load_missing_file_returns_empty(self) -> None:
        from pathlib import Path

        from src.cli.main import _load_speaker_db

        engine = _StubSpeakerEngine()
        self.assertEqual(_load_speaker_db(engine, Path("./nope.json")), {})
        self.assertEqual(engine.added, [])

    def test_read_wav_16k_rejects_wrong_rate(self) -> None:
        import tempfile
        import wave
        from pathlib import Path

        import numpy as np

        from src.cli.main import _read_wav_16k
        from src.core.errors import EngineError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)  # wrong rate
                w.writeframes(np.zeros(800, dtype=np.int16).tobytes())
            with self.assertRaises(EngineError):
                _read_wav_16k(str(path))


class TestCameraSourceNormalization(unittest.TestCase):
    def test_local_aliases(self) -> None:
        self.assertEqual(Camera("camera").source, 0)
        self.assertEqual(Camera(0).source, 0)
        self.assertFalse(Camera(0).is_network_source)

    def test_ip_webcam_url(self) -> None:
        cam = Camera("http://192.168.1.100:8080/video")
        self.assertEqual(cam.source, "http://192.168.1.100:8080/video")
        self.assertTrue(cam.is_network_source)

    def test_rtsp_url(self) -> None:
        cam = Camera("rtsp://phone.local:8554")
        self.assertTrue(cam.is_network_source)


if __name__ == "__main__":
    unittest.main()
