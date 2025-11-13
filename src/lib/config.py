from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "yolo_src/best_final.pt")
    db_path: str = os.getenv("DB_PATH", "./fish_history.sqlite3")
    max_batch_files: int = int(os.getenv("MAX_BATCH_FILES", "16"))
    max_file_mb: int = int(os.getenv("MAX_FILE_MB", "20"))
    max_ws_sessions: int = int(os.getenv("MAX_WS_SESSIONS", "4"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))
    disable_model: bool = os.getenv("DISABLE_MODEL", "0") in ("1", "true", "True")
    # 允许列表（显式放行）。当设置了这些时，优先生效。
    allowed_labels: str = os.getenv("ALLOWED_LABELS", "")
    allowed_label_keywords: str = os.getenv("ALLOWED_LABEL_KEYWORDS", "")
    # 禁止列表（默认过滤明显非鱼类）：若未设置允许列表，则按禁止列表排除
    disallowed_labels: str = os.getenv("DISALLOWED_LABELS", "human,person,water,unknown,no fish,人,水,未知,无鱼")
    disallowed_label_keywords: str = os.getenv(
        "DISALLOWED_LABEL_KEYWORDS", "human,person,water,unknown,no fish,人,水,未知,无鱼"
    )
    ws_tokens: str = os.getenv("WS_TOKENS", "")


def get_settings() -> Settings:
    return Settings()


def _parse_csv(raw: str) -> List[str]:
    """将逗号分隔的字符串转换成去重后的列表。"""
    items = []
    for part in raw.split(","):
        piece = part.strip()
        if piece:
            items.append(piece)
    return items


@lru_cache(maxsize=1)
def get_allowed_labels() -> List[str]:
    """获取允许的精确类别列表（全部转为小写）。"""
    settings = get_settings()
    return [item.lower() for item in _parse_csv(settings.allowed_labels)]


@lru_cache(maxsize=1)
def get_allowed_label_keywords() -> List[str]:
    """获取允许的类别关键词列表（全部转为小写）。"""
    settings = get_settings()
    return [item.lower() for item in _parse_csv(settings.allowed_label_keywords)]


@lru_cache(maxsize=1)
def get_disallowed_labels() -> List[str]:
    """获取禁止的精确类别列表（全部转为小写）。"""
    settings = get_settings()
    return [item.lower() for item in _parse_csv(settings.disallowed_labels)]


@lru_cache(maxsize=1)
def get_disallowed_label_keywords() -> List[str]:
    """获取禁止的类别关键词列表（全部转为小写）。"""
    settings = get_settings()
    return [item.lower() for item in _parse_csv(settings.disallowed_label_keywords)]


@lru_cache(maxsize=1)
def get_ws_token_map() -> Dict[str, str]:
    """返回 token 与客户端 ID 的映射，格式为 token->client_id。"""
    raw = get_settings().ws_tokens
    mapping: Dict[str, str] = {}
    for pair in _parse_csv(raw):
        if ":" in pair:
            client_id, token = pair.split(":", 1)
            mapping[token.strip()] = client_id.strip()
    return mapping


def resolve_client_id_by_token(token: str | None) -> str | None:
    """根据 token 查找客户端 ID，若未配置或未命中则返回 None。"""
    if not token:
        return None
    return get_ws_token_map().get(token.strip())
