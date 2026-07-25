"""Unit tests for MCPGateway.

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 3. MCPGateway"
        and "Testing Requirements"; docs/API_REFERENCE.md — MCPGateway usage.
"""

import io
import json
import unittest

from src.core.errors import EngineProcessError
from src.engines.core.mcp_gateway import MCPGateway


def _rpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    req = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        req["params"] = params
    return req


class TestMCPGateway(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = MCPGateway({"port": 3000})
        self.gateway.load()

    def tearDown(self) -> None:
        self.gateway.unload()

    def test_metadata(self) -> None:
        meta = self.gateway.metadata
        self.assertEqual(meta.name, "mcp")
        self.assertEqual(meta.input_type, "json_rpc")
        self.assertEqual(meta.output_type, "json_rpc")
        self.assertIn("protocol", meta.tags)

    def test_builtin_tools_registered(self) -> None:
        names = [t.name for t in self.gateway.list_tools()]
        self.assertIn("ping", names)
        self.assertIn("server_info", names)

    def test_register_and_list_tool(self) -> None:
        self.gateway.register_tool("echo", lambda a: a.get("text", ""),
                                   description="Echo back text.")
        specs = {t.name: t for t in self.gateway.list_tools()}
        self.assertIn("echo", specs)
        self.assertEqual(specs["echo"].description, "Echo back text.")

    def test_tools_list_rpc(self) -> None:
        resp = self.gateway.process(_rpc("tools/list")).data
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 1)
        names = [t["name"] for t in resp["result"]]
        self.assertIn("ping", names)

    def test_tools_call_ping(self) -> None:
        resp = self.gateway.process(
            _rpc("tools/call", {"name": "ping", "arguments": {}})).data
        self.assertEqual(resp["result"], "pong")

    def test_tools_call_custom_handler(self) -> None:
        self.gateway.register_tool("add", lambda a: a["x"] + a["y"])
        resp = self.gateway.process(
            _rpc("tools/call", {"name": "add", "arguments": {"x": 2, "y": 3}})).data
        self.assertEqual(resp["result"], 5)

    def test_unknown_method(self) -> None:
        resp = self.gateway.process(_rpc("bogus/method")).data
        self.assertEqual(resp["error"]["code"], -32601)

    def test_unknown_tool(self) -> None:
        resp = self.gateway.process(
            _rpc("tools/call", {"name": "nope", "arguments": {}})).data
        self.assertEqual(resp["error"]["code"], -32601)
        self.assertIn("nope", resp["error"]["message"])

    def test_invalid_request(self) -> None:
        resp = self.gateway.process({"jsonrpc": "2.0", "id": 9}).data
        self.assertEqual(resp["error"]["code"], -32600)

    def test_invalid_arguments_type(self) -> None:
        resp = self.gateway.process(
            _rpc("tools/call", {"name": "ping", "arguments": [1, 2]})).data
        self.assertEqual(resp["error"]["code"], -32602)

    def test_handler_exception_isolated(self) -> None:
        def boom(_args: dict) -> None:
            raise RuntimeError("kaboom")

        self.gateway.register_tool("boom", boom)
        resp = self.gateway.process(
            _rpc("tools/call", {"name": "boom", "arguments": {}})).data
        self.assertEqual(resp["error"]["code"], -32603)
        self.assertIn("kaboom", resp["error"]["message"])
        # Gateway stays usable after a tool crash.
        self.assertTrue(self.gateway.health_check())

    def test_process_requires_load(self) -> None:
        gateway = MCPGateway()
        with self.assertRaises(EngineProcessError):
            gateway.process(_rpc("tools/list"))

    def test_register_rejects_non_callable(self) -> None:
        with self.assertRaises(EngineProcessError):
            self.gateway.register_tool("bad", "not callable")  # type: ignore[arg-type]

    def test_serve_stdio(self) -> None:
        stdin = io.StringIO(
            json.dumps(_rpc("tools/call", {"name": "ping", "arguments": {}})) + "\n"
            + "not json\n"
        )
        stdout = io.StringIO()
        self.gateway.serve_stdio(stdin=stdin, stdout=stdout)
        lines = stdout.getvalue().strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["result"], "pong")
        self.assertEqual(json.loads(lines[1])["error"]["code"], -32600)

    def test_unload_clears_tools(self) -> None:
        self.gateway.unload()
        self.assertEqual(self.gateway.list_tools(), [])


if __name__ == "__main__":
    unittest.main()
