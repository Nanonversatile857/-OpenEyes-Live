# Changelog

All notable changes to OpenEyes-Live will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- VADEngine — audio pipeline Stage 1 (RMS-energy mock scorer, documented
  state machine: trailing-silence segment end + max-duration cut)

---

## [0.1.0] - 2026-07-25

### Added
- 🎉 Initial release — pluggable engine architecture with the full video pipeline skeleton
- Initial project scaffolding
- BaseEngine interface, error codes and engine registry (`registry.yaml`)
- Engine lifecycle management (download/load/unload), mock download
- FrameSamplerEngine — Stage 1 (uniform / dynamic / keyframe_only, NumPy motion scoring)
- FrameFilterEngine — Stage 2 (attention / diversity / hybrid selection, heuristic mock scorer)
- VisualEncoderEngine — Stage 3 (mock embeddings, interface validated)
- TokenCompressorEngine — Stage 4 (projection / selection / hybrid, mock projector)
- LanguageEngine (mock responses, interface validated)
- MemoryEngine — vector memory with timeline (mock hash embeddings, JSON persistence)
- MCPGateway — JSON-RPC 2.0 tool gateway (stdio transport; `openeyes mcp` with
  ping / server_info / query_memory / capture_frame tools)
- Camera runtime (local webcam + IP Webcam / RTSP)
- CLI interface (`openeyes` command): `list`, `install`, `watch`, `mcp`, `--version`
- GitHub Actions CI (ubuntu/windows × Python 3.10–3.12, 90 unit tests + CLI smoke)
- 90 unit tests + real-camera smoke test of the full video pipeline

### Documentation
- README.md (EN + ZH-CN)
- PROJECT_DOC.md
- ARCHITECTURE.md
- PERFORMANCE_BENCHMARK.md
- QUICKSTART_MOBILE.md
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- API_REFERENCE.md
- CHANGELOG.md

### Known Issues
- All engines are mock implementations (random embeddings / fixed text) — real
  model weights arrive in v0.2.0
- Engine download is a mock: it creates the cache directory without network fetch
- Audio pipeline (vad / asr / speaker) not implemented yet
- MCP gateway serves stdio only; HTTP/WebSocket transport planned for v0.2.0

---

## [0.2.0] - Planned (2026 Q4)

### Planned Features
- Real model weights for encoder (VideoMamba-T / CLIP) and llm (Qwen2.5-2B GGUF)
- Base Vision Engine (SmolVLM2-256M), Scene Understanding Engine (SmolVLM2-500M)
- Audio pipeline engines (vad / asr / speaker)
- Memory Engine (Sentence-BERT + FAISS) and MCP Gateway Engine
- Scheduler with motion detection trigger, proactive alert system
- Termux (Android) support and native app wrappers
- ModelScope mirror downloads, auto-downgrade for low-end devices, OOM protection
- Additional model quantization formats (Q4_0_4_4)
- Community-contributed engines

---

## [1.0.0] - Planned (2027 Q2)

### Planned Features
- Stable release
- Full MCP support with all tools
- Cross-platform support (Windows, macOS, Linux, Android, iOS)
- Production-ready proactive alerts
- Comprehensive test suite
- Performance benchmarks for all supported devices

---

## Legend

| Symbol | Meaning |
| :--- | :--- |
| `Added` | New features |
| `Changed` | Changes to existing functionality |
| `Deprecated` | Soon-to-be removed features |
| `Removed` | Removed features |
| `Fixed` | Bug fixes |
| `Security` | Security fixes |
| `Documentation` | Documentation updates |

---

> **Document Version:** v0.1.0
> **Last Updated:** 2026-07-25
> **Compatible with:** OpenEyes-Live v0.1.x