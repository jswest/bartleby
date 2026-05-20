"""Unit tests for bartleby.ingest.images."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest import images as img_pipeline
from bartleby.providers.base import ImageAnalysis


def _png_bytes(width: int, height: int, color=(255, 0, 0)) -> bytes:
    im = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _rgba_png(width: int, height: int) -> bytes:
    im = Image.new("RGBA", (width, height), color=(0, 255, 0, 128))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _fake_embed_texts(texts):
    # Deterministic, correctly-sized fake.
    return [[0.001 * (i + 1)] * EMBEDDING_DIM for i in range(len(texts))]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr(img_pipeline, "embed_texts", _fake_embed_texts)


def test_hash_bytes_is_deterministic():
    assert img_pipeline.hash_bytes(b"x") == img_pipeline.hash_bytes(b"x")
    assert img_pipeline.hash_bytes(b"x") != img_pipeline.hash_bytes(b"y")


def test_prepare_image_downscales_long_edge():
    png = _png_bytes(2000, 1000)
    prepared = img_pipeline.prepare_image(png, max_dimension=1024)
    assert prepared.width == 1024
    assert prepared.height == 512
    # Hash is on the *original* bytes, not the scaled JPEG.
    assert prepared.hash == img_pipeline.hash_bytes(png)
    # JPEG always re-encoded.
    assert prepared.jpeg_bytes.startswith(b"\xff\xd8\xff")


def test_prepare_image_leaves_small_images_alone():
    png = _png_bytes(300, 200)
    prepared = img_pipeline.prepare_image(png, max_dimension=1024)
    assert prepared.width == 300
    assert prepared.height == 200


def test_prepare_image_handles_rgba_by_flattening():
    rgba = _rgba_png(400, 400)
    prepared = img_pipeline.prepare_image(rgba, max_dimension=1024)
    # Decoding the JPEG output should succeed (no alpha → no error).
    out = Image.open(io.BytesIO(prepared.jpeg_bytes))
    assert out.mode == "RGB"


def test_prepare_image_rejects_zero_max_dimension():
    with pytest.raises(ValueError, match="max_dimension must be positive"):
        img_pipeline.prepare_image(_png_bytes(10, 10), max_dimension=0)


def test_archive_image_writes_then_is_idempotent(tmp_path):
    prepared = img_pipeline.prepare_image(_png_bytes(50, 50), max_dimension=1024)
    p1 = img_pipeline.archive_image(prepared, tmp_path)
    assert p1.exists()
    assert p1 == tmp_path / "images" / f"{prepared.hash}.jpg"
    # Touch + verify second call doesn't rewrite.
    mtime_before = p1.stat().st_mtime_ns
    p2 = img_pipeline.archive_image(prepared, tmp_path)
    assert p2 == p1
    assert p2.stat().st_mtime_ns == mtime_before


def test_analysis_to_chunk_inputs_yields_one_per_non_empty_field():
    analysis = ImageAnalysis(
        kind="scene", text="WELCOME", description="A cat sits.", notes="",
    )
    rows = img_pipeline.analysis_to_chunk_inputs(analysis)
    assert [r.content_type for r in rows] == ["image_ocr", "image_description"]
    assert [r.text for r in rows] == ["WELCOME", "A cat sits."]
    assert [r.chunk_index for r in rows] == [0, 1]
    assert all(len(r.embedding) == EMBEDDING_DIM for r in rows)


def test_analysis_to_chunk_inputs_skips_empty_text():
    analysis = ImageAnalysis(
        kind="scene", text="", description="A red square.", notes="",
    )
    rows = img_pipeline.analysis_to_chunk_inputs(analysis)
    assert len(rows) == 1
    assert rows[0].content_type == "image_description"
    assert rows[0].chunk_index == 0


def test_analysis_to_chunk_inputs_yields_nothing_when_all_empty():
    analysis = ImageAnalysis(kind="scene", text="", description="", notes="")
    assert img_pipeline.analysis_to_chunk_inputs(analysis) == []


def test_analyze_dispatches_to_provider_with_jpeg_bytes():
    seen = {}

    class _FakeProvider:
        name = "fake"

        def analyze_image(self, image_bytes, *, model, media_type="image/jpeg"):
            seen["bytes"] = image_bytes
            seen["model"] = model
            seen["media_type"] = media_type
            return ImageAnalysis(
                kind="scene", text="", description="X", notes="",
            )

        def summarize(self, *a, **k):  # protocol completeness
            raise NotImplementedError

    prepared = img_pipeline.prepare_image(_png_bytes(50, 50), max_dimension=1024)
    out = img_pipeline.analyze(_FakeProvider(), prepared, model="fake-vl:1")
    assert out.description == "X"
    assert seen["bytes"] == prepared.jpeg_bytes
    assert seen["model"] == "fake-vl:1"
    assert seen["media_type"] == "image/jpeg"
