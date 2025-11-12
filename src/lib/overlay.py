from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Return (width, height) for text in a Pillow-version-safe way."""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    # Fallback: estimate width via textlength and height via font metrics
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
    """Draw rectangles and labels onto a copy of the image and return it."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        # Ensure we always have a font; load_default rarely fails
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