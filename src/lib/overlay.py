from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """以兼容不同 Pillow 版本的方式返回文本宽高。"""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    # 回退策略：利用 textlength 与字体度量估算宽高
    width = int(draw.textlength(text, font=font)) if hasattr(draw, "textlength") else len(text) * 6
    try:
        ascent, descent = font.getmetrics()  # type: ignore[attr-defined]
        height = ascent + descent
    except Exception:
        height = 12
    return width, height


def draw_overlays(
    img: Image.Image,
    boxes: Iterable[tuple[float, float, float, float]],
    labels: Iterable[str],
    scores: Iterable[float],
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """在图片副本上绘制矩形框与标签后返回结果。"""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        # 兜底保证一定有字体，load_default 失败概率极低
        font = ImageFont.load_default()

    for (x, y, w, h), label, score in zip(boxes, labels, scores):
        x2, y2 = x + w, y + h
        draw.rectangle([(x, y), (x2, y2)], outline=color, width=2)
        text = f"{label} {score:.2f}"

        tw, th = _measure_text(draw, text, font)
        pad = 2
        bg_top = max(0, y - th - pad)
        bg_right = x + tw + pad * 2
        draw.rectangle([(x, bg_top), (bg_right, y)], fill=color)
        draw.text((x + pad, bg_top), text, fill=(0, 0, 0), font=font)

    return out