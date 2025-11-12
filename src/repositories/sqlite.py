from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from ..lib.config import get_settings


@dataclass
class DB:
    path: str

    @contextmanager
    def connect(self):  # type: ignore[no-untyped-def]
        con = sqlite3.connect(self.path)
        try:
            yield con
        finally:
            con.close()


def get_db() -> DB:
    return DB(get_settings().db_path)


SCHEMA_RECORDS = """
CREATE TABLE IF NOT EXISTS records (
  record_id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  source TEXT NOT NULL,
  image_ref TEXT NOT NULL,
  predictions_json TEXT NOT NULL,
  model_version TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  request_id TEXT,
  session_id TEXT,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);
CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_model ON records(model_version);
"""

SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  start_time TEXT NOT NULL,
  end_time TEXT,
  frame_count INTEGER NOT NULL,
  error_count INTEGER NOT NULL
);
"""


def init_db() -> None:
    db = get_db()
    with db.connect() as con:
        con.executescript(SCHEMA_RECORDS)
        con.executescript(SCHEMA_SESSIONS)
        con.commit()


def insert_record(
    record_id: str,
    source: str,
    image_ref: str,
    predictions: Iterable[dict[str, Any]],
    model_version: str,
    latency_ms: int,
    request_id: str | None,
    session_id: str | None,
    width: int,
    height: int,
) -> None:
    db = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(list(predictions), ensure_ascii=False)
    with db.connect() as con:
        con.execute(
            """
            INSERT INTO records (
              record_id, timestamp, source, image_ref, predictions_json,
              model_version, latency_ms, request_id, session_id, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                ts,
                source,
                image_ref,
                payload,
                model_version,
                latency_ms,
                request_id,
                session_id,
                width,
                height,
            ),
        )
        con.commit()


def cleanup_retention(days: int = 30, max_records: int = 100_000) -> None:
    db = get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with db.connect() as con:
        # delete older than cutoff
        con.execute("DELETE FROM records WHERE timestamp < ?", (cutoff,))
        # enforce max count by deleting oldest
        cur = con.execute("SELECT COUNT(*) FROM records")
        (count,) = cur.fetchone() or (0,)
        if count > max_records:
            to_delete = count - max_records
            con.execute(
                "DELETE FROM records WHERE rowid IN (SELECT rowid FROM records ORDER BY timestamp ASC LIMIT ?)",
                (to_delete,),
            )
        con.commit()


def query_records(
    start_time: str | None,
    end_time: str | None,
    source: str | None,
    min_confidence: float | None,
    label: str | None,
    page: int,
    page_size: int,
) -> tuple[int, list[dict[str, object]]]:
    where = []
    params: list[object] = []
    if start_time:
        where.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        where.append("timestamp <= ?")
        params.append(end_time)
    if source:
        where.append("source = ?")
        params.append(source)
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""

    db = get_db()
    with db.connect() as con:
        total_sql = f"SELECT COUNT(*) FROM records {sql_where}"
        total = con.execute(total_sql, params).fetchone()[0]

        offset = (page - 1) * page_size
        rows = con.execute(
            f"SELECT record_id, timestamp, source, image_ref, predictions_json, model_version, latency_ms, request_id, session_id, width, height FROM records {sql_where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (*params, page_size, offset),
        ).fetchall()

    items: list[dict[str, object]] = []
    for (
        record_id,
        ts,
        src,
        image_ref,
        predictions_json,
        model_version,
        latency_ms,
        request_id,
        session_id,
        width,
        height,
    ) in rows:
        preds = json.loads(predictions_json)
        # optional label/confidence filter at application level
        if label is not None or min_confidence is not None:
            filtered = []
            for p in preds:
                if label is not None and p.get("label") != label:
                    continue
                if min_confidence is not None and float(p.get("confidence", 0.0)) < min_confidence:
                    continue
                filtered.append(p)
            preds = filtered
        items.append(
            {
                "record_id": record_id,
                "timestamp": ts,
                "source": src,
                "image_ref": image_ref,
                "predictions": preds,
                "model_version": model_version,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "session_id": session_id,
                "width": width,
                "height": height,
            }
        )

    return total, items
