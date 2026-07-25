# Changelog

All notable changes to OpenEyes-Live will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `openeyes listen --speaker` — real-time speaker identification on every
  transcribed segment ("张三 (0.98) 说：……", unknown speakers labelled
  未知说话人 with their best score)
- Speaker enrollment: `--enroll NAME:WAV` (16kHz mono wav file) and
  `--enroll-mic NAME` (speak one sentence live); the enrolled database
  persists to models/speaker/enrolled.json and auto-loads on later runs
- `SpeakerEngine.add_embedding()` for loading precomputed embeddings
- SpeakerEngine — audio pipeline Stage 3, REAL implementation (3D-Speaker
  ERes2Net base, ONNX, ~38MB, zh-cn, 16kHz, via sherpa-onnx): 512-d
  speaker embeddings plus `enroll` / `identify` / `remove` with a
  cosine-similarity threshold. Measured ~34x real-time on desktop CPU;
  same-speaker score 0.87 vs noise 0.10 at the default 0.5 threshold
- Download support for absolute file URLs (GitHub release assets) in
  `registry.yaml` — used by the speaker model (upstream tag
  "speaker-recongition-models", sic); hf-mirror fallback still applies
  to `hf_repo` files only
- LanguageEngine: `max_image_side` config (default 336) downscales frames
  to a single 336px VLM tile before inference, cutting prefill cost on
  capable CPUs at zero quality cost for scene description

### Changed
- All ten registry engines now have real implementations — no
  planned-only entries remain

---

## [0.3.0] - 2026-07-25

### Added
- `openeyes install` now performs REAL model downloads: per-file manifests
  in `registry.yaml` (schema v2: `hf_repo` + `files` with expected byte
  sizes), HTTP Range resume for interrupted downloads, byte-size
  verification, automatic huggingface → hf-mirror fallback, `--mirror`
  to pin a source, and a live progress display
- Download tests with a mocked network layer (resume, fallback, size
  mismatch, unknown mirror, planned engine)

### Changed
- `EngineManager.is_installed()` verifies actual model files (path + size)
  instead of only the marker file, so manually fetched models count too
- Registry/engine versions of the real engines (vad, asr, encoder, llm)
  bumped to 0.2.0 to match their model manifests
- Docs: ENGINE_SPEC registry format updated to schema v2; README and
  QUICKSTART_MOBILE now reference `--mirror=hf-mirror` (was modelscope)

---

## [0.2.0] - 2026-07-25

### Added
- ASREngine — audio pipeline Stage 2, REAL implementation (SenseVoice int8
  via sherpa-onnx, ~234MB, fully offline, punctuation + ITN built in;
  verified 20x real-time on desktop CPU)
- Microphone capture runtime (`src/runtime/microphone.py`, sounddevice/PortAudio)
- `openeyes listen` — real-time microphone speech transcription (VAD + ASR)

### Changed
- LanguageEngine — REPLACED mock with real Phi-3.5-vision-instruct VLM
  (int4 ONNX, 3.2GB, onnxruntime-genai, offline): takes camera frames
  directly and generates real scene descriptions. The visual_tokens input
  path now raises a clear error (no trained CLIP->LLM projector exists).
  watch passes one key frame per batch to keep CPU latency manageable
  (~0.4 t/s on desktop CPU — a smaller VLM is planned)
- VisualEncoderEngine — video pipeline Stage 3, REPLACED mock with real
  CLIP ViT-B/32 vision encoder (quantized ONNX, 89MB, onnxruntime, offline);
  L2-normalized 512-d embeddings, ~50ms/frame on desktop CPU. watch pipeline
  now converts camera BGR frames to RGB per the engine spec
- VADEngine — audio pipeline Stage 1, REPLACED mock with real Silero VAD
  (ONNX, 2MB, onnxruntime, fully offline); documented state machine unchanged
- requirements.txt / setup.py: added onnxruntime, sherpa-onnx, sounddevice
- Model-dependent tests skip automatically when models are not downloaded
  (CI stays green without the ~325MB of model weights); 118 unit tests total

### Fixed
- CI: model-dependent tests are now gated on BOTH model files and runtime
  dependencies (onnxruntime / sherpa_onnx / onnxruntime_genai), and the
  sampler->filter->encoder integration test skips without the CLIP model —
  the model-less CI matrix is green again

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

> **Document Version:** v0.3.0
> **Last Updated:** 2026-07-25
> **Compatible with:** OpenEyes-Live v0.3.x