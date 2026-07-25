"""Memory Engine — vector memory with timeline support.

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 2. MemoryEngine";
        docs/API_REFERENCE.md — "Core Engines / MemoryEngine".

v0.1.x scope: the doc specifies Sentence-BERT + FAISS; this implementation
uses a deterministic hash-based bag-of-words embedding (NumPy) and in-memory
cosine-similarity search, with optional JSON persistence. The interface
matches the spec so the real vector backend can drop in later.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineProcessError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class MemoryResult:
    """A single memory record returned by query / get_timeline."""

    description: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # cosine similarity, query() only


class MemoryEngine(BaseEngine):
    """Vector memory with timeline support.

    Input (process): Dict with either
        - ``observation`` (+ optional ``timestamp``, ``metadata``) → store, or
        - ``query`` (+ optional ``limit``) → search.
    Output: EngineResult with stored record or list[MemoryResult].

    Config:
        vector_dim: int (default: 384)
        db_path: str (default: "./models/memory/memory.json") — set to ""
            to disable persistence.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "vector_dim": 384,
        "db_path": "./models/memory/memory.json",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._records: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, vector_dim)

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="memory",
            version="0.1.0",
            description="Vector memory with timeline support (mock hash embeddings).",
            author="OpenEyes-Live",
            input_type="memory_operation",
            output_type="memory_result",
            input_schema={
                "type": "object",
                "properties": {
                    "observation": "str (store)",
                    "timestamp": "ISO-8601 str, optional",
                    "metadata": "dict, optional",
                    "query": "str (search)",
                    "limit": "int, optional",
                },
            },
            output_schema={"type": "array", "items": "MemoryResult"},
            size_mb=50,
            memory_mb=100,
            tags=["core", "memory", "retrieval"],
        )

    # === Lifecycle ===

    def load(self) -> None:
        """Load persisted records (if any) into memory. Idempotent."""
        if self._loaded:
            return
        db_path = str(self.config["db_path"])
        if db_path and Path(db_path).exists():
            try:
                data = json.loads(Path(db_path).read_text(encoding="utf-8"))
                self._records = data.get("records", [])
            except (json.JSONDecodeError, OSError) as exc:
                raise EngineProcessError(f"corrupt memory db {db_path}: {exc}") from exc
            self._rebuild_index()
        self._loaded = True

    def unload(self) -> None:
        """Persist records and free the in-memory index. Idempotent."""
        if self._loaded:
            self._persist()
        self._records = []
        self._embeddings = None
        self._loaded = False

    # === Public API (ENGINE_SPEC.md) ===

    def store(
        self,
        observation: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an observation with its embedding."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(observation, str) or not observation.strip():
            raise EngineProcessError("observation must be a non-empty string")

        record = {
            "description": observation,
            "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadata": metadata or {},
        }
        self._records.append(record)
        self._rebuild_index()
        self._persist()

    def query(self, query: str, limit: int = 5) -> EngineResult:
        """Search memories by cosine similarity against the query text."""
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(query, str) or not query.strip():
            raise EngineProcessError("query must be a non-empty string")

        results: List[MemoryResult] = []
        if self._records and self._embeddings is not None:
            q = self._embed(query)
            sims = self._embeddings @ q  # embeddings are L2-normalized
            order = np.argsort(sims)[::-1][:limit]
            results = [
                MemoryResult(
                    description=self._records[i]["description"],
                    timestamp=self._records[i]["timestamp"],
                    metadata=self._records[i]["metadata"],
                    score=float(sims[i]),
                )
                for i in order
            ]
        return EngineResult(
            data=results,
            metadata={"engine": "memory", "op": "query", "query": query,
                      "total_records": len(self._records)},
        )

    def get_timeline(self, start: str, end: str) -> List[MemoryResult]:
        """Return memories whose timestamp falls within [start, end].

        Timestamps are compared as ISO-8601 strings (lexicographic order).
        """
        if not self._loaded:
            raise EngineProcessError("Engine not loaded")
        return [
            MemoryResult(
                description=r["description"],
                timestamp=r["timestamp"],
                metadata=r["metadata"],
            )
            for r in self._records
            if start <= r["timestamp"] <= end
        ]

    def process(self, input_data: Dict[str, Any]) -> EngineResult:
        """Dispatch a memory operation (store or query)."""
        if not isinstance(input_data, dict):
            raise EngineProcessError("memory expects a dict operation")
        if "observation" in input_data:
            self.store(
                input_data["observation"],
                timestamp=input_data.get("timestamp"),
                metadata=input_data.get("metadata"),
            )
            return EngineResult(
                data="stored",
                metadata={"engine": "memory", "op": "store",
                          "total_records": len(self._records)},
            )
        if "query" in input_data:
            return self.query(input_data["query"], limit=int(input_data.get("limit", 5)))
        raise EngineProcessError("memory op requires 'observation' or 'query' key")

    # === Mock vector backend ===

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based bag-of-words embedding (mock encoder).

        Each token seeds its own RNG, so identical tokens always contribute
        the identical sub-vector — shared vocabulary yields high cosine
        similarity without any model download.
        """
        dim = int(self.config["vector_dim"])
        vec = np.zeros(dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            seed = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            vec += np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _rebuild_index(self) -> None:
        if self._records:
            self._embeddings = np.stack(
                [self._embed(r["description"]) for r in self._records]
            )
        else:
            self._embeddings = None

    def _persist(self) -> None:
        db_path = str(self.config["db_path"])
        if not db_path:
            return
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"records": self._records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
