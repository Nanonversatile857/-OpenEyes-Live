"""MCP Gateway — JSON-RPC 2.0 tool gateway.

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 3. MCPGateway";
        docs/API_REFERENCE.md — "Core Engines / MCPGateway".

v0.1.x scope: full JSON-RPC 2.0 request handling in-process (``tools/list``
and ``tools/call``), plus a stdio server loop used by ``openeyes mcp``.
A networked (HTTP/WebSocket) transport is planned for v0.2.0.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TextIO

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError

# JSON-RPC 2.0 error codes.
METHOD_NOT_FOUND = -32601
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ToolHandler = Callable[[Dict[str, Any]], Any]


@dataclass
class ToolSpec:
    """Self-describing tool specification."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


class MCPGateway(BaseEngine):
    """MCP protocol gateway.

    Input (process): JSON-RPC 2.0 request dict, e.g.
        ``{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "ping", "arguments": {}}}``
    Output: EngineResult whose ``data`` is the JSON-RPC 2.0 response dict.

    Config:
        port: int (default: 3000) — reserved for the future HTTP transport.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {"port": 3000}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._tools: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, ToolHandler] = {}

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="mcp",
            version="0.1.0",
            description="MCP protocol gateway (JSON-RPC 2.0, stdio transport).",
            author="OpenEyes-Live",
            input_type="json_rpc",
            output_type="json_rpc",
            input_schema={"type": "object", "description": "JSON-RPC 2.0 request"},
            output_schema={"type": "object", "description": "JSON-RPC 2.0 response"},
            size_mb=10,
            memory_mb=20,
            tags=["core", "mcp", "protocol"],
        )

    # === Lifecycle ===

    def load(self) -> None:
        """Start the gateway and register built-in tools. Idempotent."""
        if self._loaded:
            return
        self.register_tool("ping", self._tool_ping,
                           description="Health check — returns 'pong'.")
        self.register_tool("server_info", self._tool_server_info,
                           description="Gateway name, version and tool count.")
        self._loaded = True

    def unload(self) -> None:
        """Stop the gateway and drop all tools. Idempotent."""
        self._tools.clear()
        self._handlers.clear()
        self._loaded = False

    # === Public API (ENGINE_SPEC.md) ===

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool handler. Callable before or after load()."""
        if not callable(handler):
            raise EngineProcessError(f"handler for tool '{name}' is not callable")
        self._tools[name] = ToolSpec(
            name=name, description=description, input_schema=input_schema or {}
        )
        self._handlers[name] = handler

    def list_tools(self) -> List[ToolSpec]:
        """List all registered tools."""
        return [self._tools[name] for name in sorted(self._tools)]

    # === JSON-RPC handling ===

    def process(self, input_data: Dict[str, Any]) -> EngineResult:
        """Handle one JSON-RPC 2.0 request and return the response."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")

        start = time.perf_counter()
        response = self._handle(input_data)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return EngineResult(
            data=response,
            metadata={"engine": "mcp", "method": (input_data or {}).get("method")
                      if isinstance(input_data, dict) else None},
            latency_ms=latency_ms,
        )

    def _handle(self, request: Any) -> Dict[str, Any]:
        if not isinstance(request, dict) or "method" not in request:
            return self._error(None, INVALID_REQUEST, "Invalid JSON-RPC request")

        req_id = request.get("id")
        method = request["method"]
        params = request.get("params") or {}

        if method == "tools/list":
            result = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.input_schema}
                for t in self.list_tools()
            ]
            return self._ok(req_id, result)

        if method == "tools/call":
            name = params.get("name")
            if name not in self._handlers:
                return self._error(req_id, METHOD_NOT_FOUND, f"Unknown tool: {name!r}")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(req_id, INVALID_PARAMS,
                                   "'arguments' must be an object")
            try:
                result = self._handlers[name](arguments)
            except EngineProcessError as exc:
                return self._error(req_id, INVALID_PARAMS, str(exc))
            except Exception as exc:  # noqa: BLE001 — tool isolation
                return self._error(req_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            return self._ok(req_id, result)

        return self._error(req_id, METHOD_NOT_FOUND, f"Unknown method: {method!r}")

    @staticmethod
    def _ok(req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": code, "message": message}}

    # === Built-in tools ===

    @staticmethod
    def _tool_ping(_args: Dict[str, Any]) -> str:
        return "pong"

    def _tool_server_info(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": "openeyes-mcp-gateway",
            "version": self.metadata.version,
            "transport": "stdio",
            "tools": len(self._tools),
        }

    # === stdio server (used by `openeyes mcp`) ===

    def serve_stdio(
        self,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
    ) -> None:
        """Serve JSON-RPC requests line-by-line over stdio until EOF."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        inp = stdin or sys.stdin
        out = stdout or sys.stdout
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                response = self._error(None, INVALID_REQUEST, "Parse error")
            else:
                response = self.process(request).data
            out.write(json.dumps(response, ensure_ascii=False) + "\n")
            out.flush()
