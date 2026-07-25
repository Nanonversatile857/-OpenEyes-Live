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

    def test_install_mock(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["install", "encoder"])
        self.assertEqual(rc, 0)
        self.assertIn("encoder", buf.getvalue())

    def test_install_unknown_engine(self) -> None:
        with redirect_stdout(io.StringIO()):
            rc = main(["install", "nope"])
        self.assertEqual(rc, 1)


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
