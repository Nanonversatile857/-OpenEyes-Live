"""Camera access: local webcam and IP Webcam (HTTP/RTSP) sources.

Source: docs/QUICKSTART_MOBILE.md — "Option A: IP Camera Mode" and
        "Option B: Standalone Android Mode" sections.

OpenCV (``cv2``) is imported lazily so that the rest of the package works
on machines where opencv-python is not installed yet.
"""

from typing import Any, Optional, Tuple, Union

Source = Union[str, int]


class Camera:
    """Unified camera handle for local webcams and IP camera streams.

    Args:
        source: ``0`` (or ``"camera"``) for the local webcam, or a URL such as
            ``http://192.168.1.100:8080/video`` (IP Webcam MJPEG) or
            ``rtsp://phone.local:8554`` (RTSP stream).
        width: Optional capture width in pixels.
        height: Optional capture height in pixels.

    Example:
        >>> cam = Camera("http://192.168.1.100:8080/video")
        >>> cam.start()
        >>> ok, frame = cam.read()
        >>> cam.stop()
    """

    def __init__(
        self,
        source: Source = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.source = self._normalize_source(source)
        self.width = width
        self.height = height
        self._cap: Any = None  # cv2.VideoCapture, created in start()

    @staticmethod
    def _normalize_source(source: Source) -> Source:
        """Map friendly aliases to OpenCV-compatible sources."""
        if isinstance(source, str):
            s = source.strip()
            if s.lower() in ("camera", "webcam", "local"):
                return 0
            if s.isdigit():
                return int(s)
            return s  # http(s):// or rtsp:// URL
        return source

    @property
    def is_network_source(self) -> bool:
        """True when the source is an IP Webcam / RTSP URL."""
        return isinstance(self.source, str)

    def start(self) -> None:
        """Open the capture device. Idempotent."""
        if self._cap is not None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for camera access; "
                "install it with `pip install opencv-python`"
            ) from exc

        self._cap = cv2.VideoCapture(self.source)
        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Cannot open camera source: {self.source!r}")

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Read one frame.

        Returns:
            ``(True, frame)`` on success — frame is a BGR np.ndarray (HWC);
            ``(False, None)`` when the stream ended or a frame was dropped.
        """
        if self._cap is None:
            raise RuntimeError("Camera not started; call start() first")
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, frame

    def stop(self) -> None:
        """Release the capture device. Idempotent."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
