"""Visual Encoder Engine — Stage 3 of the video pipeline.

Source: docs/VIDEO_PIPELINE.md — "Stage 3: Visual Encoder Engine" section;
        docs/ENGINE_SPEC.md — "Video Engine Specifications / 3. VisualEncoderEngine".

REAL implementation: CLIP ViT-B/32 vision encoder (quantized ONNX, ~89MB)
running on onnxruntime, fully offline. Frames are preprocessed per the
bundled preprocessor_config.json (224x224 resize, CLIP mean/std) and
L2-normalized 512-d embeddings are returned.

Model: models/encoder/clip-vit-b32/vision_model_quantized.onnx
  (https://huggingface.co/Xenova/clip-vit-base-patch32)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_engine import BaseEngine, EngineMetadata, EngineResult
from src.core.errors import EngineLoadError, EngineProcessError

# CLIP normalization constants (preprocessor_config.json / HF defaults).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CLIP_SIZE = 224


class VisualEncoderEngine(BaseEngine):
    """Encodes frames into visual embeddings (real CLIP ViT-B/32).

    Input:  List[np.ndarray] (key frames, RGB, HWC)
    Output: EngineResult with L2-normalized embeddings, shape (T, 512)

    Config:
        model_path: str (default: quantized CLIP vision ONNX in models/)
        model_name: "clip" | "videomamba_t" | "videomamba_m"
            (default: "clip"; only clip is implemented)
        embedding_dim: int (default: 512)
        use_fp16: bool (default: True) — informational; the shipped model
            is int8-quantized, which is faster on CPU
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "./models/encoder/clip-vit-b32/vision_model_quantized.onnx",
        "model_name": "clip",
        "embedding_dim": 512,
        "use_fp16": True,
    }

    _SIZES = {"clip": 89, "videomamba_t": 150, "videomamba_m": 280}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
        self._session: Any = None  # onnxruntime.InferenceSession
        self._input_name: str = "pixel_values"
        self._output_name: str = "image_embeds"

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            name="encoder",
            version="0.1.0",
            description="Encodes frames into visual embeddings (CLIP ViT-B/32, quantized ONNX, offline).",
            author="OpenEyes-Live",
            input_type="frames_rgb",
            output_type="visual_embeddings",
            input_schema={"type": "array", "items": "np.ndarray (H, W, 3) RGB"},
            output_schema={"type": "np.ndarray", "shape": "(T, 512), L2-normalized"},
            size_mb=self.size_mb,
            memory_mb=self.memory_usage_mb,
            tags=["video", "encoding", "vision"],
        )

    @property
    def size_mb(self) -> int:
        return self._SIZES.get(str(self.config["model_name"]), 89)

    @property
    def memory_usage_mb(self) -> int:
        return self.size_mb + 50  # Model + batch memory

    def load(self) -> None:
        """Load the CLIP vision ONNX model. Idempotent."""
        if self._loaded:
            return
        if str(self.config["model_name"]) != "clip":
            raise EngineLoadError(
                f"model '{self.config['model_name']}' not implemented; "
                f"v0.2.0 ships 'clip' only"
            )
        model_path = Path(str(self.config["model_path"]))
        if not model_path.exists():
            raise EngineLoadError(
                f"CLIP vision model not found: {model_path} — "
                f"run `openeyes install encoder` or see models/README"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EngineLoadError(
                "onnxruntime is required for the encoder engine; "
                "install it with `pip install onnxruntime`"
            ) from exc
        try:
            opts = ort.SessionOptions()
            self._session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        except Exception as exc:
            raise EngineLoadError(f"failed to load CLIP vision model: {exc}") from exc
        self._loaded = True

    def process(self, input_data: List[np.ndarray]) -> EngineResult:
        """Encode frames into L2-normalized visual embeddings.

        Args:
            input_data: List of key frames (RGB, HWC).

        Returns:
            EngineResult whose ``data`` is an np.ndarray of shape
            (T, embedding_dim), L2-normalized per row.
        """
        if not self._loaded or self._session is None:
            raise EngineProcessError("Engine not loaded")
        if not isinstance(input_data, list) or len(input_data) == 0:
            raise EngineProcessError("encoder expects a non-empty list of frames")

        start = time.perf_counter()
        embeddings = [self._encode_frame(f) for f in input_data]
        data = np.stack(embeddings)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return EngineResult(
            data=data,
            metadata={
                "engine": "encoder",
                "model": "clip_vit_b32_quantized",
                "num_frames": len(input_data),
                "embedding_dim": int(data.shape[1]),
            },
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Free the ONNX session. Idempotent."""
        self._session = None
        self._loaded = False

    # === Real CLIP inference ===

    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        pixel_values = self._preprocess(frame)
        out = self._session.run(
            [self._output_name], {self._input_name: pixel_values}
        )[0]
        emb = out[0].astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Resize to 224x224 and normalize with CLIP mean/std.

        Uses OpenCV when available, falls back to nearest-neighbor NumPy
        sampling so the engine stays importable without cv2.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise EngineProcessError("encoder frames must be np.ndarray (H, W, 3) RGB")
        try:
            import cv2

            resized = cv2.resize(
                frame, (_CLIP_SIZE, _CLIP_SIZE), interpolation=cv2.INTER_LINEAR
            )
        except ImportError:
            ys = (np.linspace(0, frame.shape[0] - 1, _CLIP_SIZE)).astype(int)
            xs = (np.linspace(0, frame.shape[1] - 1, _CLIP_SIZE)).astype(int)
            resized = frame[np.ix_(ys, xs)]
        x = resized.astype(np.float32) / 255.0
        x = (x - _CLIP_MEAN) / _CLIP_STD
        return x.transpose(2, 0, 1)[None, ...]  # (1, 3, 224, 224)

    # Convenience for tooling/debugging.
    def similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Cosine similarity between two frames' embeddings."""
        result = self.process([frame_a, frame_b]).data
        return float(result[0] @ result[1])
