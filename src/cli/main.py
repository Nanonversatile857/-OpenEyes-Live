"""OpenEyes-Live command-line interface.

Source: docs/API_REFERENCE.md — "CLI Reference" section;
        docs/QUICKSTART_MOBILE.md — usage examples.

Commands:
    openeyes list                                  List available engines
    openeyes install <engine> [--mirror <name>]    Download an engine (mock in v0.1.0)
    openeyes watch --source <src> --engines <e+e>  Start visual understanding
    openeyes --version                             Show version
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from src.core.engine_manager import EngineManager
from src.core.errors import EngineError
from src.runtime.camera import Camera

__version__ = "0.1.0"

# Engines with a working implementation in v0.1.x.
IMPLEMENTED_ENGINES = {"sampler", "filter", "encoder", "llm"}


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

    p_install = sub.add_parser("install", help="Download an engine")
    p_install.add_argument("engine", help="Engine name (see `openeyes list`)")
    p_install.add_argument(
        "--mirror",
        default=None,
        help="Download mirror, e.g. 'modelscope' (default: primary source)",
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
        elif name in IMPLEMENTED_ENGINES:
            status = "available"
        else:
            status = "planned (v0.2.0+)"
        print(f"{name:<12} {info['version']:<8} {info['size_mb']:>5}MB  {status}")
    return 0


def cmd_install(manager: EngineManager, engine: str, mirror: Optional[str]) -> int:
    """Download (mock) an engine."""
    path = manager.download(engine, mirror=mirror)
    info = manager.engine_info(engine)
    print(f"[mock] '{engine}' v{info['version']} ({info['size_mb']}MB) "
          f"registered at {path}")
    print("note: v0.1.0 does not download real model files yet.")
    return 0


def cmd_watch(
    manager: EngineManager,
    source: str,
    engines: str,
    interval: float,
    max_frames: int,
) -> int:
    """Run the watch loop: camera -> sampler -> filter -> encoder -> llm.

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
        """Push one batch through filter -> encoder -> llm and print."""
        if "filter" in engine_objs and len(frames) > 1:
            frames = engine_objs["filter"].process(frames).data
        visual_tokens = None
        if "encoder" in engine_objs:
            visual_tokens = engine_objs["encoder"].process(frames).data
        if "llm" in engine_objs and visual_tokens is not None:
            result = engine_objs["llm"].process({"visual_tokens": visual_tokens})
            print(f"[{time.strftime('%H:%M:%S')}] {result.data}")
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
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
