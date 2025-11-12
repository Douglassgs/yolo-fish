from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Set

from ..lib.config import get_settings
from ..repositories.sqlite import get_db


@dataclass
class StreamSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    error_count: int = 0
    closed: bool = False

    def inc_frame(self) -> None:
        self.frame_count += 1

    def inc_error(self) -> None:
        self.error_count += 1

    def should_sample(self, every_n: int = 10) -> bool:
        return self.frame_count % every_n == 0


class StreamSessionService:
    def __init__(self) -> None:
        self._active: Set[str] = set()
        self._max = get_settings().max_ws_sessions

    def _start_session_db(self, session_id: str) -> None:
        db = get_db()
        with db.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, status, start_time, end_time, frame_count, error_count)
                VALUES (?, 'active', ?, NULL, 0, 0)
                """,
                (session_id, datetime.now(timezone.utc).isoformat()),
            )
            con.commit()

    def _end_session_db(self, session_id: str, status: str, frame_count: int, error_count: int) -> None:
        db = get_db()
        with db.connect() as con:
            con.execute(
                """
                UPDATE sessions
                SET status = ?, end_time = ?, frame_count = ?, error_count = ?
                WHERE session_id = ?
                """,
                (status, datetime.now(timezone.utc).isoformat(), frame_count, error_count, session_id),
            )
            con.commit()

    def open(self) -> StreamSession:
        if len(self._active) >= self._max:
            raise RuntimeError("Max WebSocket sessions reached")
        session = StreamSession()
        self._active.add(session.session_id)
        self._start_session_db(session.session_id)
        return session

    def close(self, session: StreamSession, status: str = "closed") -> None:
        if session.closed:
            return
        session.closed = True
        self._active.discard(session.session_id)
        self._end_session_db(session.session_id, status, session.frame_count, session.error_count)


_stream_service_singleton: StreamSessionService | None = None


def get_stream_session_service() -> StreamSessionService:
    global _stream_service_singleton
    if _stream_service_singleton is None:
        _stream_service_singleton = StreamSessionService()
    return _stream_service_singleton
