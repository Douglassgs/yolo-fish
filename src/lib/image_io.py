from __future__ import annotations

import io
from typing import Tuple

from PIL import Image


def load_image_from_bytes(data: bytes) -> Tuple[Image.Image, int, int]:
    """Load image from raw bytes using Pillow; returns PIL Image and (width, height).

    Raises OSError if data is not a supported image.
    """
    bio = io.BytesIO(data)
    img = Image.open(bio)
    img.load()
    return img.convert("RGB"), img.width, img.height


def encode_image_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()
