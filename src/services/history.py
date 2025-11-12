from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..repositories.sqlite import insert_record, query_records, cleanup_retention


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
    ) -> None:
        insert_record(
            record_id=record_id,
            source="upload",
            image_ref=image_ref,
            predictions=predictions,
            model_version=model_version,
            latency_ms=latency_ms,
            request_id=request_id,
            session_id=None,
            width=width,
            height=height,
        )
        # opportunistic retention cleanup
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


_history_singleton: HistoryService | None = None


def get_history_service() -> HistoryService:
    global _history_singleton
    if _history_singleton is None:
        _history_singleton = HistoryService()
    return _history_singleton
