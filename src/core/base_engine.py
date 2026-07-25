"""Base engine interface for OpenEyes-Live.

Source: docs/ENGINE_SPEC.md — "BaseEngine Interface" section.
All engines must inherit from BaseEngine and implement this specification
to be pluggable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineMetadata:
    """Self-describing engine metadata (ENGINE_SPEC.md)."""

    name: str  # Unique engine name
    version: str  # Semver version
    description: str  # Human-readable description
    author: str  # Author/organization
    input_type: str  # MIME type or type name
    output_type: str  # MIME type or type name
    input_schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema for inputs
    output_schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema for outputs
    size_mb: int = 0  # Disk size in MB
    memory_mb: int = 0  # RAM usage in MB
    tags: List[str] = field(default_factory=list)  # Tags for discovery


@dataclass
class EngineResult:
    """Standardized engine output (ENGINE_SPEC.md)."""

    data: Any  # Main output data
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    confidence: Optional[float] = None  # Confidence score (0-1)
    latency_ms: Optional[float] = None  # Processing latency


class BaseEngine(ABC):
    """Abstract base class for all OpenEyes-Live engines.

    All engines must inherit from this class and implement the abstract
    lifecycle methods. Subclasses should track their loaded state via
    ``self._loaded``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._loaded: bool = False

    # === Required Properties ===

    @property
    @abstractmethod
    def metadata(self) -> EngineMetadata:
        """Self-describing engine metadata."""

    # === Required Lifecycle Methods ===

    @abstractmethod
    def load(self) -> None:
        """Load engine into memory.

        Called once before any process() calls. Should be idempotent.
        """

    @abstractmethod
    def process(self, input_data: Any) -> EngineResult:
        """Process input and return result. Must be thread-safe."""

    @abstractmethod
    def unload(self) -> None:
        """Unload engine from memory. Free all resources. Should be idempotent."""

    # === Optional Methods ===

    def validate(self, input_data: Any) -> bool:
        """Validate input before processing. Default: always True."""
        return True

    def configure(self, config: Dict[str, Any]) -> None:
        """Update engine configuration at runtime."""
        self.config.update(config)

    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self._loaded

    def health_check(self) -> bool:
        """Check if engine is healthy. Default: returns loaded state."""
        return self.is_loaded()

    def reset(self) -> None:
        """Reset engine to initial state. Default: unload and reload."""
        self.unload()
        self.load()
