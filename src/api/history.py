from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException, Query

from ..models.schemas import (
    ImageBatchSummary,
    ImageDetectionRecord,
    ImageHistoryResponse,
    VideoDetectionPage,
    VideoDetectionRecord,
)
from ..services.history import get_history_service


router = APIRouter(tags=["history"])


@router.get("/history/video", response_model=VideoDetectionPage)
async def get_video_history(
    client_id: str = Query(..., description="客户端标识"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> VideoDetectionPage:
    svc = get_history_service()
    total, items = svc.get_video_history(client_id, page, page_size)
    detections = [VideoDetectionRecord(**item) for item in items]
    return VideoDetectionPage(total=total, page=page, page_size=page_size, items=detections)


@router.delete("/history/video")
async def delete_video_history(client_id: str = Query(..., description="客户端标识")) -> dict[str, object]:
    svc = get_history_service()
    deleted = svc.clear_video_history(client_id)
    return {"client_id": client_id, "deleted": deleted}


@router.get("/history/images/{image_id}", response_model=ImageHistoryResponse)
async def get_image_history(image_id: str) -> ImageHistoryResponse:
    svc = get_history_service()
    raw = svc.get_image_history(image_id)
    if not raw:
        raise HTTPException(status_code=404, detail="未找到图片识别记录")
    detections = [ImageDetectionRecord(**item) for item in raw]
    counter: Counter[str] = Counter(d.label for d in detections)
    confidences = [d.confidence for d in detections]
    summary = ImageBatchSummary(
        total_images=1,
        total_detections=len(detections),
        category_counts=dict(counter),
        confidence_min=min(confidences) if confidences else None,
        confidence_max=max(confidences) if confidences else None,
        confidence_avg=sum(confidences) / len(confidences) if confidences else None,
    )
    return ImageHistoryResponse(image_id=image_id, detections=detections, summary=summary)


@router.delete("/history/images/{image_id}")
async def delete_image_history(image_id: str) -> dict[str, object]:
    svc = get_history_service()
    deleted = svc.clear_image_history(image_id)
    return {"image_id": image_id, "deleted": deleted}
