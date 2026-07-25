"""Language Engine — multi-modal language reasoning (real VLM).

Source: docs/ENGINE_SPEC.md — "Core Engine Specifications / 1. LanguageEngine";
        docs/API_REFERENCE.md — "Core Engines / LanguageEngine".

REAL implementation: Phi-3.5-vision-instruct (int4 ONNX, ~3.2GB) via
onnxruntime-genai, fully offline. The engine takes camera frames directly
and produces natural-language scene descriptions — this is the "real
看图说话" path. A CLIP-embedding -> LLM projector (visual_tokens input)
requires a trained projection layer and is planned for a later release.

Model: models/llm/phi-3.5-vision-int4/
  (https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx,
   cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError

DEFAULT_PROMPT = "Describe what you see in this scene, briefly and factually."


class LanguageEngine(BaseEngine):
    """Multi-modal language reasoning (real Phi-3.5-vision VLM).

    Input:  Dict with
        - ``frames``: List[np.ndarray] (RGB, HWC) — required for real output
        - ``prompt``: str (optional, overrides config prompt)
        - ``audio_text`` / ``speaker_id``: str (optional, appended to prompt)
        - ``visual_tokens``: accepted for interface compatibility but cannot
          drive the VLM (no trained projector); passing only visual_tokens
          raises EngineProcessError.
    Output: EngineResult whose ``data`` is the description text.

    Config:
        model_path: str (default: "./models/llm/phi-3.5-vision-int4")
        prompt: str (default: scene description prompt)
        max_tokens: int (default: 128)
        temperature: float (default: 0.7) — reserved; generation is greedy
        context_length: int (default: 4096)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/llm/phi-3.5-vision-int4",
        "prompt": DEFAULT_PROMPT,
        "max_tokens": 128,
        "temperature": 0.7,
        "context_length": 4096,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._model: Any = None  # og.Model
        self._processor: Any = None  # multimodal processor

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="llm",
            version="0.2.0",
            description="Multi-modal language reasoning (Phi-3.5-vision int4, offline).",
            author="OpenEyes-Live",
            input_type="multi_modal_features",
            output_type="text",
            input_schema={
                "type": "object",
                "properties": {
                    "frames": "List[np.ndarray] (RGB, HWC)",
                    "audio_text": "str, optional",
                    "speaker_id": "str, optional",
                    "prompt": "str, optional",
                },
                "required": ["frames"],
            },
            output_schema={"type": "string"},
            size_mb=3200,
            memory_mb=4000,
            tags=["core", "reasoning", "llm", "vlm"],
        )

    def load(self) -> None:
        """Load the VLM (3.2GB — first load takes a while). Idempotent."""
        if self._loaded:
            return
        model_dir = Path(str(self.config["model_path"]))
        if not (model_dir / "genai_config.json").exists():
            raise EngineLoadError(
                f"Phi-3.5-vision model not found: {model_dir} — "
                f"run `openeyes install llm` or see models/README"
            )
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise EngineLoadError(
                "onnxruntime-genai is required for the llm engine; "
                "install it with `pip install onnxruntime-genai`"
            ) from exc
        try:
            self._model = og.Model(str(model_dir))
            self._processor = self._model.create_multimodal_processor()
        except Exception as exc:
            raise EngineLoadError(f"failed to load Phi-3.5-vision: {exc}") from exc
        self._loaded = True

    def process(self, input_data: Dict[str, Any]) -> EngineResult:
        """Generate a scene description from camera frames.

        Args:
            input_data: Dict containing ``frames`` (list of RGB ndarrays)
                and optional ``prompt`` / ``audio_text`` / ``speaker_id``.

        Returns:
            EngineResult whose ``data`` is the generated text.
        """
        if not self._loaded or self._model is None:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, dict):
            raise EngineProcessError("llm expects a dict input")

        frames = input_data.get("frames")
        if not frames:
            if "visual_tokens" in input_data:
                raise EngineProcessError(
                    "visual_tokens cannot drive the VLM (no trained projector); "
                    "pass 'frames' (list of RGB ndarrays) instead"
                )
            raise EngineProcessError("llm expects a dict with a 'frames' key")

        prompt_text = self._build_prompt(input_data)
        start = time.perf_counter()
        text, n_tokens = self._generate(frames, prompt_text)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=text,
            metadata={
                "engine": "llm",
                "model": "phi-3.5-vision-int4",
                "num_frames": len(frames),
                "generated_tokens": n_tokens,
                "tokens_per_s": round(n_tokens / max(latency_ms / 1000.0, 0.001), 1),
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free the model. Idempotent."""
        self._processor = None
        self._model = None
        self._loaded = False

    # === Real VLM inference ===

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        parts: List[str] = []
        if input_data.get("speaker_id"):
            parts.append(f"The speaker is {input_data['speaker_id']}.")
        if input_data.get("audio_text"):
            parts.append(f"Someone said: \"{input_data['audio_text']}\".")
        parts.append(str(input_data.get("prompt") or self.config["prompt"]))
        return " ".join(parts)

    def _generate(self, frames: List[np.ndarray], prompt_text: str) -> tuple[str, int]:
        import onnxruntime_genai as og

        # Phi-3.5-vision chat template with one <|image_i|> tag per frame.
        tags = "\n".join(f"<|image_{i + 1}|>" for i in range(len(frames)))
        full_prompt = f"<|user|>\n{tags}\n{prompt_text}<|end|>\n<|assistant|>\n"

        with tempfile.TemporaryDirectory(prefix="openeyes_vlm_") as tmpdir:
            paths = self._dump_frames(frames, Path(tmpdir))
            images = og.Images.open(*paths)

            inputs = self._processor(full_prompt, images=images)
            params = og.GeneratorParams(self._model)
            params.set_search_options(max_length=int(self.config["context_length"]))

            generator = og.Generator(self._model, params)
            generator.set_inputs(inputs)
            stream = self._processor.create_stream()
            chunks: List[str] = []
            n_tokens = 0
            max_new = int(self.config["max_tokens"])
            while not generator.is_done() and n_tokens < max_new:
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                chunks.append(stream.decode(token))
                n_tokens += 1

        return "".join(chunks).strip(), n_tokens

    @staticmethod
    def _dump_frames(frames: List[np.ndarray], tmpdir: Path) -> List[str]:
        """Write RGB ndarrays to temporary PNGs for og.Images.open.

        Uses cv2.imencode + Python file I/O instead of cv2.imwrite, which
        fails on non-ASCII paths (e.g. Windows users with CJK usernames).
        """
        try:
            import cv2
        except ImportError as exc:
            raise EngineProcessError(
                "opencv-python is required to feed frames to the VLM"
            ) from exc
        paths = []
        for i, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                raise EngineProcessError("each frame must be an np.ndarray (H, W, C)")
            ok, buf = cv2.imencode(".png", frame[:, :, ::-1])  # RGB -> BGR
            if not ok:
                raise EngineProcessError(f"failed to encode frame {i} as PNG")
            path = tmpdir / f"frame_{i}.png"
            path.write_bytes(buf.tobytes())
            paths.append(str(path))
        return paths
