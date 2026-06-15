"""Unit tests for bartleby.ingest.images."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest import images as img_pipeline
from bartleby.ingest import ocr as ocr_module
from bartleby.providers.base import ImageAnalysis, VlmDescription


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


def test_is_below_vlm_minimum_flags_thin_strips():
    # A 512x24 strip (the qwen3-vl crash case) is below a 32px minimum.
    strip = img_pipeline.prepare_image(_png_bytes(512, 24), max_dimension=1024)
    assert img_pipeline.is_below_vlm_minimum(strip, min_dimension=32)
    # A square comfortably above the minimum is not flagged.
    square = img_pipeline.prepare_image(_png_bytes(100, 100), max_dimension=1024)
    assert not img_pipeline.is_below_vlm_minimum(square, min_dimension=32)
    # Exactly at the minimum on both edges is allowed (strictly-below check).
    edge = img_pipeline.prepare_image(_png_bytes(32, 32), max_dimension=1024)
    assert not img_pipeline.is_below_vlm_minimum(edge, min_dimension=32)


def test_is_below_vlm_minimum_uses_post_scale_dimensions():
    # 5000x100 downscaled to a 1024 long edge becomes 1024x20 — the short edge
    # drops below 32 only *after* scaling, so the post-scale check must catch it.
    prepared = img_pipeline.prepare_image(_png_bytes(5000, 100), max_dimension=1024)
    assert prepared.height < 32
    assert img_pipeline.is_below_vlm_minimum(prepared, min_dimension=32)


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


def test_analysis_to_chunk_inputs_text_image_yields_image_ocr_only():
    analysis = ImageAnalysis(
        kind="text", text="WELCOME SIGN AHEAD", description="", notes="",
    )
    rows = img_pipeline.analysis_to_chunk_inputs(analysis)
    assert [r.content_type for r in rows] == ["image_ocr"]
    assert rows[0].text == "WELCOME SIGN AHEAD"
    assert rows[0].chunk_index == 0
    assert len(rows[0].embedding) == EMBEDDING_DIM


def test_analysis_to_chunk_inputs_scene_image_yields_image_description_only():
    analysis = ImageAnalysis(
        kind="scene", text="", description="A red square.", notes="",
    )
    rows = img_pipeline.analysis_to_chunk_inputs(analysis)
    assert [r.content_type for r in rows] == ["image_description"]
    assert rows[0].text == "A red square."
    assert rows[0].chunk_index == 0


def test_analysis_to_chunk_inputs_empty_payload_yields_nothing():
    # Scene image whose VLM call returned only notes — no searchable content.
    analysis = ImageAnalysis(kind="scene", text="", description="", notes="x")
    assert img_pipeline.analysis_to_chunk_inputs(analysis) == []


def test_analyze_routes_text_image_via_tesseract_only(monkeypatch):
    """Tesseract clears the threshold → VLM not called, kind='text'."""
    vlm_called = {"n": 0}

    class _FakeProvider:
        name = "fake"
        def analyze_image(self, *a, **k):
            vlm_called["n"] += 1
            return VlmDescription(description="should not happen", notes="")
        def summarize(self, *a, **k):
            raise NotImplementedError

    # Tesseract returns lots of high-confidence text.
    monkeypatch.setattr(
        ocr_module, "run",
        lambda b: ocr_module.OcrResult(text="X" * 500, avg_confidence=80.0),
    )
    prepared = img_pipeline.prepare_image(_png_bytes(50, 50), max_dimension=1024)
    out = img_pipeline.analyze(
        _FakeProvider(), prepared, model="fake-vl:1", temperature=0.0,
    )
    assert out.kind == "text"
    assert out.text == "X" * 500
    assert out.description == ""
    assert vlm_called["n"] == 0


def test_analyze_routes_scene_image_via_vlm(monkeypatch):
    """Tesseract returns little → VLM called, kind='scene'."""
    seen = {}

    class _FakeProvider:
        name = "fake"
        def analyze_image(self, image_bytes, *, model, temperature, media_type="image/jpeg"):
            seen["bytes"] = image_bytes
            seen["model"] = model
            seen["temperature"] = temperature
            return VlmDescription(description="A red square.", notes="")
        def summarize(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr(
        ocr_module, "run",
        lambda b: ocr_module.OcrResult(text="hi", avg_confidence=10.0),
    )
    prepared = img_pipeline.prepare_image(_png_bytes(50, 50), max_dimension=1024)
    out = img_pipeline.analyze(
        _FakeProvider(), prepared, model="fake-vl:1", temperature=0.4,
    )
    assert out.kind == "scene"
    assert out.text == ""
    assert out.description == "A red square."
    assert seen["bytes"] == prepared.jpeg_bytes
    assert seen["model"] == "fake-vl:1"
    # analyze() threads the configured vision temperature to the provider.
    assert seen["temperature"] == 0.4


def test_analyze_falls_back_to_vlm_when_ocr_raises(monkeypatch):
    """A crashing Tesseract (e.g. unwritable TMPDIR) must not drop the image:
    OCR is only a classifier, so the VLM still runs and produces a caption."""
    seen = {}

    class _FakeProvider:
        name = "fake"
        def analyze_image(self, image_bytes, *, model, temperature, media_type="image/jpeg"):
            seen["called"] = True
            return VlmDescription(description="A red square.", notes="")
        def summarize(self, *a, **k):
            raise NotImplementedError

    def _boom(_bytes):
        raise RuntimeError("Tesseract OCR failed (unwritable TMPDIR).")

    monkeypatch.setattr(ocr_module, "run", _boom)
    prepared = img_pipeline.prepare_image(_png_bytes(50, 50), max_dimension=1024)
    out = img_pipeline.analyze(
        _FakeProvider(), prepared, model="fake-vl:1", temperature=0.0,
    )
    assert seen["called"]   # VLM ran despite OCR crashing
    assert out.kind == "scene"
    assert out.description == "A red square."
