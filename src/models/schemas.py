from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox


class ImageResult(BaseModel):
    image_id: str
    predictions: List[Prediction]
    model_version: str
    width: int
    height: int


class ImageBatchResult(BaseModel):
    request_id: str
    results: List[ImageResult]
    annotated_zip_ref: Optional[str] = None


class HistoryRecord(BaseModel):
    record_id: str
    timestamp: str
    source: str
    image_ref: str
    predictions: List[Prediction]
    model_version: str
    latency_ms: int
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    width: int
    height: int


class PagedRecords(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[HistoryRecord]
