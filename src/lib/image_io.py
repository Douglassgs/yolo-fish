from __future__ import annotations

import io
from typing import Tuple

from PIL import Image


def load_image_from_bytes(data: bytes) -> Tuple[Image.Image, int, int]:
    """使用 Pillow 从字节流读取图片并返回图像对象及尺寸，遇到异常会抛出 OSError。"""
    bio = io.BytesIO(data)
    img = Image.open(bio)
    img.load()
    return img.convert("RGB"), img.width, img.height


def encode_image_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()
