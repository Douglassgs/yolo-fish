from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Tuple

from PIL import Image

from ..lib.config import get_settings
from ..lib.overlay import draw_overlays
from ..lib.image_io import encode_image_to_jpeg_bytes


try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    _import_error = e
else:
    _import_error = None


@dataclass
class InferenceResult:
    image_id: str
    predictions: list[dict[str, Any]]
    model_version: str
    width: int
    height: int
    annotated_jpeg: bytes
    latency_ms: int


class InferenceService:
    def __init__(self) -> None:
        self._model = None
        self._model_version = "unknown"

    def warmup(self) -> None:
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        settings = get_settings()
        if settings.disable_model:
            # mock mode: no real model loaded
            self._model = None
            self._names = None
            self._model_version = "mock"
            return
        if YOLO is None:
            raise RuntimeError(
                f"ultralytics not available: {_import_error}. Please install dependencies or set DISABLE_MODEL=1."
            )
        model_path = settings.model_path
        self._model = YOLO(model_path)
        names = getattr(self._model, "names", None)
        self._names = names if isinstance(names, (list, dict)) else None
        self._model_version = getattr(self._model, "version", "best_final")

    def _postprocess(
        self, img: Image.Image, det_boxes: List[Tuple[float, float, float, float]], det_cls: List[int], det_scores: List[float]
    ) -> Tuple[list[dict[str, Any]], Image.Image]:
        labels = []
        for c in det_cls:
            if isinstance(self._names, dict):
                labels.append(str(self._names.get(int(c), int(c))))
            elif isinstance(self._names, list) and 0 <= int(c) < len(self._names):
                labels.append(str(self._names[int(c)]))
            else:
                labels.append(str(int(c)))
        overlay_img = draw_overlays(img, det_boxes, labels, det_scores)
        preds = []
        for (x, y, w, h), label, score in zip(det_boxes, labels, det_scores):
            preds.append(
                {
                    "label": label,
                    "confidence": float(score),
                    "bbox": {"x": float(x), "y": float(y), "w": float(w), "h": float(h)},
                }
            )
        return preds, overlay_img

    def predict_image(self, img: Image.Image) -> InferenceResult:
        if self._model is None and self._model_version != "mock":
            self._load_model()

        w, h = img.width, img.height
        t0 = time.time()
        boxes_xywh: list[tuple[float, float, float, float]] = []
        cls: list[int] = []
        scores: list[float] = []
        if self._model is not None:
            results = self._model(img, verbose=False)
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xywh = r.boxes.xywh.cpu().numpy().tolist()
                cls = [int(c) for c in r.boxes.cls.cpu().numpy().tolist()]
                scores = [float(s) for s in r.boxes.conf.cpu().numpy().tolist()]
                for cx, cy, bw, bh in xywh:
                    x = float(cx - bw / 2.0)
                    y = float(cy - bh / 2.0)
                    boxes_xywh.append((x, y, float(bw), float(bh)))
        else:
            # mock prediction: single centered box
            boxes_xywh.append((w * 0.25, h * 0.25, w * 0.5, h * 0.5))
            cls.append(0)
            scores.append(0.5)
        t1 = time.time()

        preds, overlay_img = self._postprocess(img, boxes_xywh, cls, scores)
        annotated_bytes = encode_image_to_jpeg_bytes(overlay_img)
        image_id = uuid.uuid4().hex
        latency_ms = int((t1 - t0) * 1000)
        return InferenceResult(
            image_id=image_id,
            predictions=preds,
            model_version=self._model_version,
            width=w,
            height=h,
            annotated_jpeg=annotated_bytes,
            latency_ms=latency_ms,
        )


_singleton: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _singleton
    if _singleton is None:
        _singleton = InferenceService()
    return _singleton
