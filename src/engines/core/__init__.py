"""Core engines (docs/ENGINE_SPEC.md — Core Engine Specifications)."""

from .llm import LanguageEngine
from .mcp_gateway import MCPGateway
from .memory import MemoryEngine

__all__ = ["LanguageEngine", "MemoryEngine", "MCPGateway"]
