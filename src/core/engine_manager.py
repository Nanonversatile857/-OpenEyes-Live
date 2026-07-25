"""Engine lifecycle management (download / load / unload).

Source: docs/ENGINE_SPEC.md — "Engine Lifecycle" section;
        docs/API_REFERENCE.md — "EngineManager" section.

v0.1.0 scope: ``download()`` is a mock implementation — it does not fetch
any model files over the network, it only creates the engine cache
directory. Real downloading arrives in a later release.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from src.core.base_engine import BaseEngine
from src.core.errors import (
    EngineDownloadError,
    EngineLoadError,
    EngineNotFoundError,
)

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


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

    # === Download (mock in v0.1.0) ===

    def download(self, name: str, mirror: Optional[str] = None) -> Path:
        """Download an engine's model files.

        v0.1.0: mock implementation — creates the cache directory only,
        no network access.

        Args:
            name: Engine name from the registry.
            mirror: Optional mirror name (e.g. ``modelscope``). Recorded but
                not used by the mock implementation.

        Returns:
            Path to the engine's cache directory.
        """
        info = self.engine_info(name)  # raises EngineNotFoundError if unknown

        engine_dir = self.cache_dir / name
        try:
            engine_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EngineDownloadError(f"Failed to create cache dir for '{name}': {exc}") from exc

        # Mock: a real implementation would fetch `info["file"]` from
        # `info["source"]` (or the mirror) and verify `info["checksum"]`.
        marker = engine_dir / ".downloaded"
        marker.write_text(
            f"name={info['name']}\nversion={info['version']}\n"
            f"file={info['file']}\nmirror={mirror or 'primary'}\nmock=true\n",
            encoding="utf-8",
        )
        return engine_dir

    def is_installed(self, name: str) -> bool:
        """Check whether an engine has been downloaded (cache dir exists)."""
        return (self.cache_dir / name / ".downloaded").exists()

    # === Load / Unload ===

    def _resolve_engine_class(self, name: str) -> Type[BaseEngine]:
        """Map a registry name to an engine class.

        v0.1.x implements ``sampler``, ``filter``, ``encoder`` and ``llm``;
        other registry entries raise EngineNotFoundError (planned for v0.2.0+).
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
        if name == "llm":
            from src.engines.core.llm import LanguageEngine

            return LanguageEngine
        if name in self._registry.get("engines", {}):
            raise EngineNotFoundError(
                f"Engine '{name}' is registered but not implemented in v0.1.x"
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
