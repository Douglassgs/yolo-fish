from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "yolo_src/best_final.pt")
    db_path: str = os.getenv("DB_PATH", "./fish_history.sqlite3")
    max_batch_files: int = int(os.getenv("MAX_BATCH_FILES", "16"))
    max_file_mb: int = int(os.getenv("MAX_FILE_MB", "20"))
    max_ws_sessions: int = int(os.getenv("MAX_WS_SESSIONS", "4"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))
    disable_model: bool = os.getenv("DISABLE_MODEL", "0") in ("1", "true", "True")


def get_settings() -> Settings:
    return Settings()
