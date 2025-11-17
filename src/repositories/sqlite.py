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
    error_count INTEGER NOT NULL,
    client_id TEXT
);
"""

SCHEMA_VIDEO_DETECTIONS = """
CREATE TABLE IF NOT EXISTS video_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    detection_index INTEGER NOT NULL,
    detection_seq INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    UNIQUE(session_id, detection_seq)
);
CREATE INDEX IF NOT EXISTS idx_video_client ON video_detections(client_id);
CREATE INDEX IF NOT EXISTS idx_video_session ON video_detections(session_id);
"""

SCHEMA_IMAGE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS image_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    detection_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    UNIQUE(image_id, detection_index)
);
CREATE INDEX IF NOT EXISTS idx_image_image_id ON image_detections(image_id);
CREATE INDEX IF NOT EXISTS idx_image_request_id ON image_detections(request_id);
"""


def init_db() -> None:
    db = get_db()
    with db.connect() as con:
        con.executescript(SCHEMA_RECORDS)
        con.executescript(SCHEMA_SESSIONS)
        con.executescript(SCHEMA_VIDEO_DETECTIONS)
        con.executescript(SCHEMA_IMAGE_DETECTIONS)
        try:
            con.execute("ALTER TABLE sessions ADD COLUMN client_id TEXT")
        except sqlite3.OperationalError:
            pass
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
        # 若记录已存在，则覆盖其内容，保证同一 record_id 的数据始终为最新
        con.execute(
            """
            INSERT INTO records (
              record_id, timestamp, source, image_ref, predictions_json,
              model_version, latency_ms, request_id, session_id, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
              timestamp=excluded.timestamp,
              source=excluded.source,
              image_ref=excluded.image_ref,
              predictions_json=excluded.predictions_json,
              model_version=excluded.model_version,
              latency_ms=excluded.latency_ms,
              request_id=excluded.request_id,
              session_id=excluded.session_id,
              width=excluded.width,
              height=excluded.height
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
        # 删除早于保留时间的记录
        con.execute("DELETE FROM records WHERE timestamp < ?", (cutoff,))
        # 若超出数量上限则删除最早的记录
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
    # 在应用层按需要过滤类别与置信度
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


def insert_video_detection(
    session_id: str,
    client_id: str,
    image_id: str,
    frame_index: int,
    detection_index: int,
    detection_seq: int,
    label: str,
    confidence: float,
) -> bool:
    """将视频流中识别出的单条鱼记录写入数据库，重复数据会被忽略。"""
    db = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    with db.connect() as con:
        cur = con.execute(
            """
            INSERT OR IGNORE INTO video_detections (
              session_id, client_id, image_id, frame_index, detection_index,
              detection_seq, label, confidence, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                client_id,
                image_id,
                frame_index,
                detection_index,
                detection_seq,
                label,
                confidence,
                ts,
            ),
        )
        con.commit()
        return cur.rowcount == 1


def query_video_detections(client_id: str, page: int, page_size: int) -> tuple[int, list[dict[str, object]]]:
    """按客户端查询视频流识别记录，按时间倒序分页。"""
    db = get_db()
    with db.connect() as con:
        total = (
            con.execute("SELECT COUNT(*) FROM video_detections WHERE client_id = ?", (client_id,))
            .fetchone()[0]
        )
        offset = (page - 1) * page_size
        rows = con.execute(
            """
            SELECT session_id, client_id, image_id, frame_index, detection_index,
                   detection_seq, label, confidence, timestamp
            FROM video_detections
            WHERE client_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (client_id, page_size, offset),
        ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        (
            session_id,
            client,
            image_id,
            frame_index,
            detection_index,
            detection_seq,
            label,
            confidence,
            ts,
        ) = row
        items.append(  # type: ignore[arg-type]
            {
                "session_id": session_id,
                "client_id": client,
                "image_id": image_id,
                "frame_index": frame_index,
                "detection_index": detection_index,
                "detection_seq": detection_seq,
                "label": label,
                "confidence": confidence,
                "timestamp": ts,
            }
        )
    return total, items


def delete_video_detections(client_id: str) -> int:
    """删除指定客户端的视频识别记录，返回删除条数。"""
    db = get_db()
    with db.connect() as con:
        cur = con.execute("DELETE FROM video_detections WHERE client_id = ?", (client_id,))
        con.commit()
        return cur.rowcount


def insert_image_detection(
    request_id: str,
    image_id: str,
    detection_index: int,
    label: str,
    confidence: float,
    width: int,
    height: int,
) -> bool:
    """写入图片批量识别中的单条结果，重复数据会被忽略。"""
    db = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    with db.connect() as con:
        # 保证同一 image_id + detection_index 下只有一条记录，新的结果覆盖旧的结果
        cur = con.execute(
            """
            INSERT INTO image_detections (
              request_id, image_id, detection_index, label, confidence, width, height, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id, detection_index) DO UPDATE SET
              request_id=excluded.request_id,
              label=excluded.label,
              confidence=excluded.confidence,
              width=excluded.width,
              height=excluded.height,
              timestamp=excluded.timestamp
            """,
            (request_id, image_id, detection_index, label, confidence, width, height, ts),
        )
        con.commit()
        # 无论是插入还是更新，都视为一次有效写入
        return cur.rowcount >= 0


def query_image_detections(image_id: str) -> list[dict[str, object]]:
    """按图片编号查询识别结果，按检测序号升序返回。"""
    db = get_db()
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT request_id, image_id, detection_index, label, confidence, width, height, timestamp
            FROM image_detections
            WHERE image_id = ?
            ORDER BY detection_index ASC
            """,
            (image_id,),
        ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        (
            request_id,
            img_id,
            detection_index,
            label,
            confidence,
            width,
            height,
            ts,
        ) = row
        items.append(  # type: ignore[arg-type]
            {
                "request_id": request_id,
                "image_id": img_id,
                "detection_index": detection_index,
                "label": label,
                "confidence": confidence,
                "width": width,
                "height": height,
                "timestamp": ts,
            }
        )
    return items


def delete_image_detections(image_id: str) -> int:
    """删除指定图片的识别记录，返回删除条数。"""
    db = get_db()
    with db.connect() as con:
        cur = con.execute("DELETE FROM image_detections WHERE image_id = ?", (image_id,))
        con.commit()
        return cur.rowcount
