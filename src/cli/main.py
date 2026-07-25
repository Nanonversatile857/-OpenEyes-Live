"""OpenEyes-Live command-line interface.

Source: docs/API_REFERENCE.md — "CLI Reference" section;
        docs/QUICKSTART_MOBILE.md — usage examples.

Commands:
    openeyes list                                  List available engines
    openeyes install <engine> [--mirror <name>]    Download an engine's model files
    openeyes watch --source <src> --engines <e+e>  Start visual understanding
    openeyes listen [--speaker] [--enroll N:WAV]   Transcribe mic speech (+ speaker ID)
    openeyes --version                             Show version
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.engine_manager import EngineManager
from src.core.errors import EngineError
from src.runtime.camera import Camera

__version__ = "0.3.0"

# Engines usable in the `watch` pipeline in v0.1.x.
IMPLEMENTED_ENGINES = {"sampler", "filter", "encoder", "compressor", "llm", "memory"}

# Engines implemented but launched via their own command or pipeline stage
# (not wired into the watch pipeline).
STANDALONE_ENGINES = {"mcp", "vad", "asr", "speaker"}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openeyes",
        description="OpenEyes-Live — on-device visual understanding.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all available engines")

    p_install = sub.add_parser("install", help="Download an engine's model files")
    p_install.add_argument("engine", help="Engine name (see `openeyes list`)")
    p_install.add_argument(
        "--mirror",
        default=None,
        help="Pin a download source, e.g. 'hf-mirror' "
             "(default: huggingface with automatic hf-mirror fallback)",
    )

    p_watch = sub.add_parser("watch", help="Start visual understanding")
    p_watch.add_argument(
        "--source",
        default="camera",
        help="'camera' for local webcam, or an IP Webcam / RTSP URL",
    )
    p_watch.add_argument(
        "--engines",
        default="encoder+llm",
        help="Engines to use, joined by '+' (default: encoder+llm)",
    )
    p_watch.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between descriptions (default: 2.0)",
    )
    p_watch.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N processed frames (default: 0 = run until Ctrl+C)",
    )

    p_mcp = sub.add_parser("mcp", help="Start the MCP gateway (stdio JSON-RPC)")
    p_mcp.add_argument(
        "--source",
        default="camera",
        help="Camera source used by the 'capture_frame' tool",
    )
    p_mcp.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Reserved for the future HTTP transport (default: 3000)",
    )

    p_listen = sub.add_parser("listen", help="Transcribe microphone speech (VAD + ASR)")
    p_listen.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Stop after N transcribed segments (default: 0 = run until Ctrl+C)",
    )
    p_listen.add_argument(
        "--speaker",
        action="store_true",
        help="Identify who is speaking (loads the speaker engine and the "
             "enrolled speaker database at models/speaker/enrolled.json)",
    )
    p_listen.add_argument(
        "--enroll",
        action="append",
        default=[],
        metavar="NAME:WAV",
        help="Enroll a speaker from a 16kHz mono wav file; repeatable. "
             "Implies --speaker.",
    )
    p_listen.add_argument(
        "--enroll-mic",
        action="append",
        default=[],
        metavar="NAME",
        help="Enroll a speaker by speaking one sentence into the "
             "microphone; repeatable. Implies --speaker.",
    )
    return parser


def cmd_list(manager: EngineManager) -> int:
    """List available engines and their status."""
    print(f"OpenEyes-Live v{__version__} — engine registry\n")
    print(f"{'NAME':<12} {'VERSION':<8} {'SIZE':>7}  STATUS")
    print("-" * 46)
    for name in manager.list_engines():
        info = manager.engine_info(name)
        if manager.is_loaded(name):
            status = "loaded"
        elif manager.is_installed(name):
            status = "installed"
        elif name in IMPLEMENTED_ENGINES or name in STANDALONE_ENGINES:
            status = "available"
        else:
            status = "planned (v0.3.0+)"
        print(f"{name:<12} {info['version']:<8} {info['size_mb']:>5}MB  {status}")
    return 0


def cmd_install(manager: EngineManager, engine: str, mirror: Optional[str]) -> int:
    """Download an engine's model files with a progress display."""
    info = manager.engine_info(engine)
    needs_files = bool(info.get("files"))

    def report(eng: str, fname: str, done: int, total: int) -> None:
        if total:
            pct = done / total * 100
            line = (f"  {fname}: {done / 1e6:.1f}/{total / 1e6:.1f} MB "
                    f"({pct:5.1f}%)")
        else:
            line = f"  {fname}: {done / 1e6:.1f} MB"
        print("\r" + line.ljust(72), end="", flush=True)
        if total and done >= total:
            print()  # newline when the file finishes

    if needs_files:
        print(f"Downloading '{engine}' v{info['version']} "
              f"(~{info['size_mb']}MB) ...")
    path = manager.download(engine, mirror=mirror, progress=report)
    if needs_files:
        print(f"[ok] '{engine}' installed at {path}")
    else:
        print(f"[ok] '{engine}' v{info['version']} is a built-in engine — "
              f"no model files needed, registered at {path}")
    return 0


def cmd_watch(
    manager: EngineManager,
    source: str,
    engines: str,
    interval: float,
    max_frames: int,
) -> int:
    """Run the watch loop: camera -> sampler -> filter -> encoder -> compressor -> llm.

    Frames are accumulated into a batch (size = filter top_k, or 1 when no
    filter stage is used). Each full batch flows through the remaining stages
    and produces one printed description.
    """
    requested: List[str] = [e.strip() for e in engines.split("+") if e.strip()]
    unsupported = [e for e in requested if e not in IMPLEMENTED_ENGINES]
    if unsupported:
        print(
            f"error: engines not implemented in v0.1.x: {', '.join(unsupported)}",
            file=sys.stderr,
        )
        return 2

    # Load requested engines.
    engine_objs = {name: manager.load(name) for name in requested}
    print(f"engines loaded: {', '.join(requested)}")
    print(f"source: {source}  (Ctrl+C to stop)")

    batch_size = (
        int(engine_objs["filter"].config["top_k"]) if "filter" in engine_objs else 1
    )
    batch: List[np.ndarray] = []
    stats = {"read": 0, "sampled": 0, "described": 0}

    def run_batch(frames: List[np.ndarray]) -> None:
        """Push one batch through filter -> encoder -> compressor -> llm."""
        if "filter" in engine_objs and len(frames) > 1:
            frames = engine_objs["filter"].process(frames).data
        # Camera frames are BGR; encoder and llm specs expect RGB.
        rgb_frames = [f[:, :, ::-1] for f in frames]

        visual_tokens = None
        if "encoder" in engine_objs:
            visual_tokens = engine_objs["encoder"].process(rgb_frames).data
        if "compressor" in engine_objs and visual_tokens is not None:
            visual_tokens = engine_objs["compressor"].process(visual_tokens).data

        if "llm" in engine_objs:
            # The VLM sees key frames directly (no trained token projector).
            # One frame keeps latency sane on CPU (~0.4 t/s vision+text).
            result = engine_objs["llm"].process({"frames": rgb_frames[:1]})
            print(f"[{time.strftime('%H:%M:%S')}] {result.data}")
            # Store the observation in long-term memory, if enabled.
            if "memory" in engine_objs:
                engine_objs["memory"].store(
                    result.data,
                    metadata={"source": source, "num_frames": len(frames)},
                )
        elif visual_tokens is not None:
            print(
                f"[{time.strftime('%H:%M:%S')}] encoded {len(frames)} frame(s) -> "
                f"{visual_tokens.shape}"
            )
        stats["described"] += len(frames)

    try:
        with Camera(source) as cam:
            while True:
                ok, frame = cam.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                stats["read"] += 1

                # Stage 1: sampling (skipped frames are dropped here).
                if "sampler" in engine_objs:
                    frame = engine_objs["sampler"].process(
                        {"frame": frame, "timestamp": time.time()}
                    ).data
                    if frame is None:
                        continue
                stats["sampled"] += 1

                batch.append(frame)
                if len(batch) >= batch_size:
                    run_batch(batch)
                    batch = []

                if max_frames and stats["described"] >= max_frames:
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        if batch:  # flush any leftover frames
            run_batch(batch)
        for name in requested:
            manager.unload(name)

    print(
        f"read {stats['read']} frame(s), sampled {stats['sampled']}, "
        f"described {stats['described']}."
    )
    return 0


def cmd_mcp(manager: EngineManager, source: str, port: int) -> int:
    """Start the MCP gateway as a stdio JSON-RPC server.

    Built-in tools: ping, server_info. Additionally wires two OpenEyes tools:
    ``query_memory`` (search the MemoryEngine) and ``capture_frame`` (grab one
    frame from the camera source and report its shape).
    """
    gateway = manager.load("mcp", {"port": port})
    memory = manager.load("memory")

    def query_memory(args: dict) -> list:
        result = memory.query(str(args.get("query", "")),
                              limit=int(args.get("limit", 5)))
        return [
            {"description": r.description, "timestamp": r.timestamp,
             "metadata": r.metadata, "score": r.score}
            for r in result.data
        ]

    def capture_frame(_args: dict) -> dict:
        with Camera(source) as cam:
            ok, frame = cam.read()
        if not ok:
            raise EngineError(f"no frame from source {source!r}")
        return {"source": source, "shape": list(frame.shape),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    gateway.register_tool(
        "query_memory", query_memory,
        description="Search long-term visual memory.",
        input_schema={"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["query"]},
    )
    gateway.register_tool(
        "capture_frame", capture_frame,
        description="Capture one frame from the camera and report its shape.",
        input_schema={"type": "object", "properties": {}},
    )

    tools = ", ".join(t.name for t in gateway.list_tools())
    print(f"MCP gateway ready (stdio). tools: {tools}", file=sys.stderr)
    print("waiting for JSON-RPC requests on stdin (one per line, Ctrl+C to stop)",
          file=sys.stderr)
    try:
        gateway.serve_stdio()
    except KeyboardInterrupt:
        print("\nstopped by user.", file=sys.stderr)
    finally:
        manager.unload("mcp")
        manager.unload("memory")
    return 0


SPEAKER_DB_PATH = Path("./models/speaker/enrolled.json")
UNKNOWN_SPEAKER = "未知说话人"


def _load_speaker_db(engine: Any, path: Path) -> Dict[str, List[List[float]]]:
    """Load enrolled embeddings from JSON into the speaker engine."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    db: Dict[str, List[List[float]]] = {}
    for name, embeddings in raw.items():
        vecs = [list(map(float, v)) for v in embeddings]
        for vec in vecs:
            engine.add_embedding(name, vec)
        db[name] = vecs
    return db


def _save_speaker_db(db: Dict[str, List[List[float]]], path: Path) -> None:
    """Persist enrolled embeddings (name -> list of 512-d vectors)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(db, ensure_ascii=False), encoding="utf-8"
    )


def _read_wav_16k(wav_path: str) -> np.ndarray:
    """Read a 16kHz mono PCM wav into float32 samples."""
    import wave

    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise EngineError(
                f"{wav_path}: expected a 16kHz mono wav, got "
                f"{w.getframerate()}Hz x {w.getnchannels()}ch"
            )
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if len(pcm) == 0:
        raise EngineError(f"{wav_path}: empty wav file")
    return pcm.astype(np.float32) / 32768.0


def cmd_listen(
    manager: EngineManager,
    max_segments: int,
    speaker: bool = False,
    enroll: Optional[List[str]] = None,
    enroll_mic: Optional[List[str]] = None,
) -> int:
    """Listen to the microphone: VAD segments speech, ASR transcribes it.

    With ``--speaker`` (implied by --enroll/--enroll-mic), each segment is
    also matched against the enrolled speaker database and printed as
    "张三说：……" (unknown speakers get 未知说话人).
    """
    from src.runtime.microphone import Microphone

    vad = manager.load("vad")
    asr = manager.load("asr")
    print("engines loaded: vad, asr (real Silero VAD + SenseVoice)")

    speaker_engine = None
    db: Dict[str, List[List[float]]] = {}
    if speaker or enroll or enroll_mic:
        speaker_engine = manager.load("speaker")
        db = _load_speaker_db(speaker_engine, SPEAKER_DB_PATH)
        if db:
            print(f"speaker engine loaded; enrolled: {', '.join(sorted(db))}")
        else:
            print("speaker engine loaded; no enrolled speakers yet")

        # Enroll from wav files first (no microphone interaction needed).
        for spec in enroll or []:
            if ":" not in spec:
                raise EngineError(
                    f"--enroll expects NAME:WAV, got '{spec}'"
                )
            name, wav_path = spec.split(":", 1)
            emb = speaker_engine.enroll(name.strip(), _read_wav_16k(wav_path))
            db.setdefault(name.strip(), []).append(emb.tolist())
            print(f"enrolled '{name.strip()}' from {wav_path}")

    segments = 0
    voice: List[np.ndarray] = []
    try:
        with Microphone() as mic:
            # Microphone enrollment: one VAD-detected sentence per name.
            for name in enroll_mic or []:
                name = name.strip()
                input(f"按回车，然后请 '{name}' 对麦克风说一句话（约 3 秒）...")
                emb = _capture_one_segment(mic, vad, speaker_engine, name)
                db.setdefault(name, []).append(emb.tolist())
                print(f"enrolled '{name}' from microphone")
            if enroll_mic:
                _save_speaker_db(db, SPEAKER_DB_PATH)
                print(f"speaker database saved to {SPEAKER_DB_PATH}")
            elif enroll:
                _save_speaker_db(db, SPEAKER_DB_PATH)
                print(f"speaker database saved to {SPEAKER_DB_PATH}")

            print("listening on default microphone (Ctrl+C to stop)")
            while True:
                chunk = mic.read(timeout=1.0)
                if chunk is None:
                    continue
                result = vad.process(chunk)
                if result.data is not None:
                    voice.append(result.data)
                if result.metadata["segment_ended"] and voice:
                    segments += int(_transcribe_and_print(
                        asr, speaker_engine, np.concatenate(voice)
                    ))
                    voice = []
                    if max_segments and segments >= max_segments:
                        break
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        if voice:  # flush trailing speech
            _transcribe_and_print(asr, speaker_engine, np.concatenate(voice))
        manager.unload("vad")
        manager.unload("asr")
        if speaker_engine is not None:
            manager.unload("speaker")

    print(f"transcribed {segments} segment(s).")
    return 0


def _capture_one_segment(mic: Any, vad: Any, speaker_engine: Any, name: str) -> np.ndarray:
    """Capture one VAD-terminated speech segment and enroll it.

    Returns the enrolled embedding. Raises EngineError on timeout (~30s
    of silence without any detected speech).
    """
    voice: List[np.ndarray] = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        chunk = mic.read(timeout=1.0)
        if chunk is None:
            continue
        result = vad.process(chunk)
        if result.data is not None:
            voice.append(result.data)
        if result.metadata["segment_ended"] and voice:
            audio = np.concatenate(voice)
            print(f"captured {len(audio) / 16000:.1f}s of speech")
            return speaker_engine.enroll(name, audio)
    raise EngineError(f"enrollment timed out — no speech heard for '{name}'")


def _transcribe_and_print(asr: Any, speaker_engine: Any, audio: np.ndarray) -> bool:
    """ASR one segment (plus optional speaker ID) and print one line."""
    text = asr.process(audio).data
    if not text:
        return False
    prefix = ""
    if speaker_engine is not None and speaker_engine.speakers:
        name, score = speaker_engine.identify(audio)
        label = name if name else UNKNOWN_SPEAKER
        prefix = f"{label} ({score:.2f}) 说："
    print(f"[{time.strftime('%H:%M:%S')}] {prefix}{text}")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    manager = EngineManager()

    try:
        if args.command == "list":
            return cmd_list(manager)
        if args.command == "install":
            return cmd_install(manager, args.engine, args.mirror)
        if args.command == "watch":
            return cmd_watch(
                manager, args.source, args.engines, args.interval, args.max_frames
            )
        if args.command == "mcp":
            return cmd_mcp(manager, args.source, args.port)
        if args.command == "listen":
            return cmd_listen(
                manager, args.max_segments,
                speaker=args.speaker,
                enroll=args.enroll,
                enroll_mic=args.enroll_mic,
            )
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
