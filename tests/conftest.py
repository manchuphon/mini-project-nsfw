"""
conftest.py – shared pytest fixtures for NSFW API tests
"""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App import (adjust if your project uses a different module path)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    """
    Create a single TestClient for the whole test session.
    TestClient handles lifespan (startup / shutdown) automatically.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _make_png_bytes(width: int = 224, height: int = 224, color=(128, 64, 200)) -> bytes:
    """Return raw PNG bytes of a solid-color image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    import random
    import numpy as np
    arr = np.random.randint(0, 256, (height, width, 3), dtype="uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def valid_png_file():
    """A well-formed PNG image as (filename, bytes, content_type)."""
    return ("test_image.png", _make_png_bytes(), "image/png")


@pytest.fixture(scope="session")
def valid_jpeg_file():
    """A well-formed JPEG image."""
    return ("test_image.jpg", _make_jpeg_bytes(), "image/jpeg")


@pytest.fixture
def corrupted_file():
    """Bytes that look like an image but are actually garbage."""
    return ("corrupted.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")


@pytest.fixture
def empty_file():
    """Zero-byte file."""
    return ("empty.png", b"", "image/png")


@pytest.fixture
def text_file_as_image():
    """A plain-text file sent with image/jpeg content type (wrong type)."""
    return ("note.txt", b"Hello, this is not an image.", "text/plain")


@pytest.fixture
def oversized_file():
    """File that exceeds the 10 MB limit."""
    # 11 MB of zeros
    big_bytes = b"\x00" * (11 * 1024 * 1024)
    return ("huge.png", big_bytes, "image/png")