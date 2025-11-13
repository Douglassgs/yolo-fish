from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..repositories.sqlite import (
    cleanup_retention,
    delete_image_detections,
    delete_video_detections,
    insert_image_detection,
    insert_record,
    insert_video_detection,
    query_image_detections,
    query_records,
    query_video_detections,
)


@dataclass
class HistoryService:
    retention_days: int = 30
    max_records: int = 100_000

    def write_upload_record(
        self,
        record_id: str,
        image_ref: str,
        predictions: list[dict],
        model_version: str,
        latency_ms: int,
        request_id: str | None,
        width: int,
        height: int,
        session_id: str | None = None,
    ) -> None:
        insert_record(
            record_id=record_id,
            source="upload",
            image_ref=image_ref,
            predictions=predictions,
            model_version=model_version,
            latency_ms=latency_ms,
            request_id=request_id,
            session_id=session_id,
            width=width,
            height=height,
        )
    # 写入后顺便执行一次过期清理
        cleanup_retention(self.retention_days, self.max_records)

    def query(
        self,
        start_time: Optional[str],
        end_time: Optional[str],
        source: Optional[str],
        min_confidence: Optional[float],
        label: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[int, list[dict]]:
        return query_records(
            start_time=start_time,
            end_time=end_time,
            source=source,
            min_confidence=min_confidence,
            label=label,
            page=page,
            page_size=page_size,
        )

    def write_stream_detection(
        self,
        *,
        session_id: str,
        client_id: str,
        image_id: str,
        frame_index: int,
        detection_index: int,
        detection_seq: int,
        label: str,
        confidence: float,
    ) -> bool:
        """保存视频流中的单条鱼检测。"""
        return insert_video_detection(
            session_id=session_id,
            client_id=client_id,
            image_id=image_id,
            frame_index=frame_index,
            detection_index=detection_index,
            detection_seq=detection_seq,
            label=label,
            confidence=confidence,
        )

    def write_image_detection(
        self,
        *,
        request_id: str,
        image_id: str,
        detection_index: int,
        label: str,
        confidence: float,
        width: int,
        height: int,
    ) -> bool:
        """保存图片识别中的单条鱼检测。"""
        return insert_image_detection(
            request_id=request_id,
            image_id=image_id,
            detection_index=detection_index,
            label=label,
            confidence=confidence,
            width=width,
            height=height,
        )

    def get_video_history(self, client_id: str, page: int, page_size: int) -> tuple[int, list[dict]]:
        """按客户端分页查询视频识别历史。"""
        return query_video_detections(client_id, page, page_size)

    def clear_video_history(self, client_id: str) -> int:
        """删除指定客户端的视频识别记录。"""
        return delete_video_detections(client_id)

    def get_image_history(self, image_id: str) -> list[dict]:
        """查询指定图片的识别记录。"""
        return query_image_detections(image_id)

    def clear_image_history(self, image_id: str) -> int:
        """删除指定图片的识别记录。"""
        return delete_image_detections(image_id)


_history_singleton: HistoryService | None = None


def get_history_service() -> HistoryService:
    global _history_singleton
    if _history_singleton is None:
        _history_singleton = HistoryService()
    return _history_singleton
