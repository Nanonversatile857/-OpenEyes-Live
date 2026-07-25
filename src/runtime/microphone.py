"""Microphone capture runtime (real audio input).

Source: docs/AUDIO_PIPELINE.md — audio stream input;
        docs/QUICKSTART_MOBILE.md — Option C `openeyes listen --source microphone`.

Uses sounddevice (PortAudio) with a callback queue so the processing loop
can pull fixed-size PCM chunks. sounddevice is imported lazily.
"""

import queue
from typing import Any, Optional

import numpy as np


class Microphone:
    """Capture PCM chunks from the default microphone.

    Args:
        sample_rate: Capture rate in Hz (default: 16000, what the audio
            pipeline expects).
        chunk_ms: Chunk size in milliseconds (default: 30, per the VAD spec).
        device: Optional sounddevice device index/name (default: system default).

    Example:
        >>> mic = Microphone()
        >>> mic.start()
        >>> chunk = mic.read()      # float32 PCM, 480 samples @ 16 kHz
        >>> mic.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        device: Optional[Any] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream: Any = None

    def start(self) -> None:
        """Open the input stream. Idempotent."""
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone capture; "
                "install it with `pip install sounddevice`"
            ) from exc

        def _callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                pass  # overflow/underflow notices — chunks are best-effort
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # consumer is behind — drop rather than block audio

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_samples,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=_callback,
        )
        self._stream.start()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read one PCM chunk (float32, chunk_ms long).

        Returns None on timeout (e.g. no audio arriving).
        """
        if self._stream is None:
            raise RuntimeError("Microphone not started; call start() first")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Close the input stream. Idempotent."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
