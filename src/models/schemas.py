from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    detection_index: int
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
    summary: Optional["ImageBatchSummary"] = None


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


class ImageBatchSummary(BaseModel):
    total_images: int
    total_detections: int
    category_counts: Dict[str, int]
    confidence_min: Optional[float]
    confidence_max: Optional[float]
    confidence_avg: Optional[float]


class VideoDetectionRecord(BaseModel):
    session_id: str
    client_id: str
    image_id: str
    frame_index: int
    detection_index: int
    detection_seq: int
    label: str
    confidence: float
    timestamp: str


class VideoDetectionPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[VideoDetectionRecord]


class ImageDetectionRecord(BaseModel):
    request_id: str
    image_id: str
    detection_index: int
    label: str
    confidence: float
    width: int
    height: int
    timestamp: str


class ImageHistoryResponse(BaseModel):
    image_id: str
    detections: List[ImageDetectionRecord]
    summary: Optional[ImageBatchSummary] = None


ImageBatchResult.model_rebuild()
ImageHistoryResponse.model_rebuild()
