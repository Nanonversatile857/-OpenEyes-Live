"""Engine lifecycle management (download / load / unload).

Source: docs/ENGINE_SPEC.md — "Engine Lifecycle" section;
        docs/API_REFERENCE.md — "EngineManager" section.

v0.3.0: ``download()`` fetches real model files from Hugging Face (with
hf-mirror.com fallback for networks where huggingface.co is unreachable),
supports resuming interrupted downloads, and verifies each file against
the byte size declared in ``registry.yaml``.
"""

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import yaml

from src.core.base_engine import BaseEngine
from src.core.errors import (
    EngineDownloadError,
    EngineLoadError,
    EngineNotFoundError,
)

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"

_CHUNK_SIZE = 1 << 20  # 1 MiB
_CONNECT_TIMEOUT = 15  # seconds
_PROGRESS_INTERVAL = 0.5  # seconds between progress callbacks

#: Signature: (engine_name, file_local_path, downloaded_bytes, total_bytes)
ProgressCallback = Callable[[str, str, int, int], None]


class EngineManager:
    """Manages engine lifecycle: download, load, unload.

    Args:
        cache_dir: Directory where engine files are cached (default ``./models``).
    """

    def __init__(self, cache_dir: str = "./models") -> None:
        self.cache_dir = Path(cache_dir)
        self._loaded_engines: Dict[str, BaseEngine] = {}
        self._registry: Dict[str, Any] = self._load_registry()

    # === Registry ===

    def _load_registry(self) -> Dict[str, Any]:
        """Read the engine registry YAML file."""
        if not _REGISTRY_PATH.exists():
            raise EngineNotFoundError(f"Registry file not found: {_REGISTRY_PATH}")
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def list_engines(self) -> List[str]:
        """List all engine names available in the registry."""
        return sorted(self._registry.get("engines", {}).keys())

    def engine_info(self, name: str) -> Dict[str, Any]:
        """Return registry metadata for a single engine."""
        engines = self._registry.get("engines", {})
        if name not in engines:
            raise EngineNotFoundError(f"Engine '{name}' not found in registry")
        return engines[name]

    # === Download ===

    def download(
        self,
        name: str,
        mirror: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download an engine's model files.

        Internal (pure-code) engines need no files: this only creates the
        cache directory and marker. Model engines fetch every file declared
        in the registry, resuming partial downloads and verifying byte
        sizes. When ``mirror`` is not given, the default source is tried
        first and the fallback source (hf-mirror) is used if it fails.

        Args:
            name: Engine name from the registry.
            mirror: Optional source name pinning a single download source
                (e.g. ``hf-mirror``); see ``registry.sources``.
            progress: Optional callback
                ``(engine, file, downloaded_bytes, total_bytes)``.

        Returns:
            Path to the engine's cache directory.

        Raises:
            EngineNotFoundError: Unknown engine name.
            EngineDownloadError: A file could not be fetched or verified.
        """
        info = self.engine_info(name)  # raises EngineNotFoundError if unknown

        engine_dir = self.cache_dir / name
        try:
            engine_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EngineDownloadError(
                f"Failed to create cache dir for '{name}': {exc}"
            ) from exc

        if info.get("source") == "planned":
            raise EngineDownloadError(
                f"Engine '{name}' is planned but has no published model yet"
            )

        files: List[Dict[str, Any]] = info.get("files") or []
        if not files:
            # Internal engine — pure code, nothing to fetch.
            self._write_marker(engine_dir, info, mirror=mirror, real=True)
            return engine_dir

        repo = info.get("hf_repo")
        if not repo:
            raise EngineDownloadError(
                f"Engine '{name}' declares files but no hf_repo in registry"
            )

        used_source: Optional[str] = None
        for entry in files:
            dest = engine_dir / entry["local"]
            expected = int(entry.get("size") or 0)
            if expected and dest.exists() and dest.stat().st_size == expected:
                continue  # already complete
            dest.parent.mkdir(parents=True, exist_ok=True)
            used_source = self._fetch_file(
                name, repo, entry, dest, mirror=mirror, progress=progress
            )

        self._write_marker(
            engine_dir, info, mirror=used_source or mirror, real=True
        )
        return engine_dir

    def _source_bases(self, mirror: Optional[str]) -> List[tuple]:
        """Ordered (name, base_url) download sources to try."""
        reg = self._registry.get("registry", {})
        sources: Dict[str, str] = reg.get("sources") or {}
        if mirror:
            if mirror not in sources:
                raise EngineDownloadError(
                    f"Unknown mirror '{mirror}'; available: "
                    f"{', '.join(sorted(sources))}"
                )
            return [(mirror, sources[mirror])]
        ordered = [
            (reg.get("default_source"), None),
            (reg.get("fallback_source"), None),
        ]
        return [
            (name, sources[name])
            for name, _ in ordered
            if name and name in sources
        ]

    def _fetch_file(
        self,
        engine: str,
        repo: str,
        entry: Dict[str, Any],
        dest: Path,
        mirror: Optional[str],
        progress: Optional[ProgressCallback],
    ) -> str:
        """Fetch one file, trying each source in turn. Returns source used."""
        last_error: Optional[Exception] = None
        for source_name, base in self._source_bases(mirror):
            url = f"{base}/{repo}/resolve/main/{entry['remote']}"
            try:
                self._stream_to_disk(
                    engine, url, dest,
                    expected_size=int(entry.get("size") or 0),
                    progress=progress,
                )
                return source_name
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                continue  # try the next source
        raise EngineDownloadError(
            f"Failed to download '{entry['remote']}' for engine "
            f"'{engine}': {last_error}"
        ) from last_error

    @staticmethod
    def _stream_to_disk(
        engine: str,
        url: str,
        dest: Path,
        expected_size: int,
        progress: Optional[ProgressCallback],
    ) -> None:
        """Stream one URL to ``dest`` with resume support and size check.

        Interrupted downloads leave a ``<name>.part`` file; the next call
        resumes it with an HTTP Range request.
        """
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if expected_size and offset > expected_size:
            part.unlink()  # stale/garbage partial file
            offset = 0

        headers = {"User-Agent": "openeyes-live"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=_CONNECT_TIMEOUT) as resp:
            # A server ignoring Range returns 200 — restart from scratch.
            if offset and resp.status == 200:
                offset = 0
            mode = "ab" if offset else "wb"
            downloaded = offset
            last_report = 0.0
            with open(part, mode) as fh:
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if progress and now - last_report >= _PROGRESS_INTERVAL:
                        last_report = now
                        progress(engine, dest.name, downloaded, expected_size)

        if expected_size and downloaded != expected_size:
            raise EngineDownloadError(
                f"Size mismatch for {dest.name}: got {downloaded} bytes, "
                f"expected {expected_size} (partial file kept at {part})"
            )
        part.replace(dest)
        if progress:
            progress(engine, dest.name, downloaded, expected_size)

    def _write_marker(
        self,
        engine_dir: Path,
        info: Dict[str, Any],
        mirror: Optional[str],
        real: bool,
    ) -> None:
        marker = engine_dir / ".downloaded"
        marker.write_text(
            f"name={info['name']}\nversion={info['version']}\n"
            f"source={mirror or 'internal'}\nreal={str(real).lower()}\n",
            encoding="utf-8",
        )

    def is_installed(self, name: str) -> bool:
        """Check whether an engine's files are fully downloaded.

        Model engines are installed when every declared file exists with
        the expected size — regardless of the marker, so manually fetched
        models count too. Internal engines only need the marker.
        """
        try:
            info = self.engine_info(name)
        except EngineNotFoundError:
            return False
        engine_dir = self.cache_dir / name
        files = info.get("files") or []
        if not files:
            return (engine_dir / ".downloaded").exists()
        for entry in files:
            dest = engine_dir / entry["local"]
            expected = int(entry.get("size") or 0)
            if not dest.exists():
                return False
            if expected and dest.stat().st_size != expected:
                return False
        return True

    # === Load / Unload ===

    def _resolve_engine_class(self, name: str) -> Type[BaseEngine]:
        """Map a registry name to an engine class.

        v0.2.x implements the full video pipeline (``sampler``, ``filter``,
        ``encoder``, ``compressor``), the audio pipeline front-end
        (``vad``, ``asr``) plus ``llm``, ``memory`` and ``mcp``; other
        registry entries raise EngineNotFoundError (planned for v0.3.0+).
        """
        if name == "sampler":
            from src.engines.video.sampler import FrameSamplerEngine

            return FrameSamplerEngine
        if name == "filter":
            from src.engines.video.filter import FrameFilterEngine

            return FrameFilterEngine
        if name == "encoder":
            from src.engines.video.encoder import VisualEncoderEngine

            return VisualEncoderEngine
        if name == "compressor":
            from src.engines.video.compressor import TokenCompressorEngine

            return TokenCompressorEngine
        if name == "llm":
            from src.engines.core.llm import LanguageEngine

            return LanguageEngine
        if name == "memory":
            from src.engines.core.memory import MemoryEngine

            return MemoryEngine
        if name == "mcp":
            from src.engines.core.mcp_gateway import MCPGateway

            return MCPGateway
        if name == "vad":
            from src.engines.audio.vad import VADEngine

            return VADEngine
        if name == "asr":
            from src.engines.audio.asr import ASREngine

            return ASREngine
        if name in self._registry.get("engines", {}):
            raise EngineNotFoundError(
                f"Engine '{name}' is registered but not implemented in v0.2.x"
            )
        raise EngineNotFoundError(f"Engine '{name}' not found in registry")

    def load(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseEngine:
        """Instantiate and load an engine.

        Returns the already-loaded instance if called repeatedly (idempotent).
        """
        if name in self._loaded_engines:
            return self._loaded_engines[name]

        engine_cls = self._resolve_engine_class(name)
        engine = engine_cls(config or {})
        try:
            engine.load()
        except Exception as exc:
            raise EngineLoadError(f"Failed to load engine '{name}': {exc}") from exc

        self._loaded_engines[name] = engine
        return engine

    def unload(self, name: str) -> None:
        """Unload an engine and free its resources (idempotent)."""
        engine = self._loaded_engines.pop(name, None)
        if engine is not None:
            engine.unload()

    def is_loaded(self, name: str) -> bool:
        """Check whether an engine is currently loaded."""
        engine = self._loaded_engines.get(name)
        return engine is not None and engine.is_loaded()
