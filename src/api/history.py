from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..models.schemas import HistoryRecord, PagedRecords
from ..services.history import get_history_service


router = APIRouter(tags=["history"])


@router.get("/history", response_model=PagedRecords)
async def get_history(
    start_time: Optional[str] = Query(None, description="ISO datetime >= start"),
    end_time: Optional[str] = Query(None, description="ISO datetime <= end"),
    source: Optional[str] = Query(None, pattern="^(upload|stream)$"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    label: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> PagedRecords:
    svc = get_history_service()
    total, items = svc.query(
        start_time=start_time,
        end_time=end_time,
        source=source,
        min_confidence=min_confidence,
        label=label,
        page=page,
        page_size=page_size,
    )
    records = [HistoryRecord(**it) for it in items]
    return PagedRecords(total=total, page=page, page_size=page_size, items=records)
