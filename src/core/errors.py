"""Standardized engine error codes.

Source: docs/ENGINE_SPEC.md — "Error Handling" section.
"""


class EngineError(Exception):
    """Base engine exception."""


class EngineLoadError(EngineError):
    """Raised when engine fails to load."""


class EngineProcessError(EngineError):
    """Raised when engine fails to process input."""


class EngineUnloadError(EngineError):
    """Raised when engine fails to unload."""


class EngineInvalidInputError(EngineError):
    """Raised when input validation fails."""


class EngineOutOfMemoryError(EngineError):
    """Raised when engine runs out of memory."""


class EngineNotFoundError(EngineError):
    """Raised when engine not found in registry."""


class EngineDownloadError(EngineError):
    """Raised when engine download fails."""
