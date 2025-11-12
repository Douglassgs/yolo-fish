from __future__ import annotations

from typing import List

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..lib.image_io import load_image_from_bytes
from ..models.schemas import ImageBatchResult, ImageResult, Prediction
from ..services.history import get_history_service
from ..services.inference import get_inference_service


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/images", response_model=ImageBatchResult)
async def predict_images(files: List[UploadFile] = File(...)) -> ImageBatchResult:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded")

    svc = get_inference_service()
    svc.warmup()

    request_id = uuid.uuid4().hex
    results: list[ImageResult] = []
    history_svc = get_history_service()
    for f in files:
        content = await f.read()
        img, w, h = load_image_from_bytes(content)
        inf = svc.predict_image(img)
        history_svc.write_upload_record(
            record_id=inf.image_id,
            image_ref=f.filename or inf.image_id,
            predictions=inf.predictions,
            model_version=inf.model_version,
            latency_ms=inf.latency_ms,
            request_id=request_id,
            width=inf.width,
            height=inf.height,
        )
        results.append(
            ImageResult(
                image_id=inf.image_id,
                width=inf.width,
                height=inf.height,
                predictions=[Prediction(**p) for p in inf.predictions],
                model_version=inf.model_version,
            )
        )

    return ImageBatchResult(request_id=request_id, results=results)
