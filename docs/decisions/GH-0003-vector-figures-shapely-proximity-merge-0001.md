# Shapely proximity-merge for vector figure detection (issue #3)

> Source: [#3](https://github.com/jswest/bartleby/issues/3)

Vector figures (matplotlib/Illustrator charts drawn as PDF path operators) never appear in `page.images`, so the VLM never sees them under the existing raster-embed path. The fix crops each vector-figure region from the **existing** page render — no second render needed.

## Settled design: shapely proximity-merge

The issue body explored several approaches (whole-page render, `len(page.images)==0` gating, single-bbox heuristic, PyMuPDF). The final accepted design is:

1. Collect `page.curves + page.lines + page.rects` (vector ink only — never `page.chars`).
2. Buffer each primitive by `tol/2` points and `unary_union` into merged clusters.
3. Subtract clusters whose centroid falls inside a `page.find_tables()` bbox — table gridlines are rects/lines and would otherwise produce false-positive figure crops. `find_tables()` is pdfplumber's purpose-built detector; no heuristic is needed.
4. Drop clusters below `min_area` (slivers and hairlines).
5. Crop each surviving cluster from the already-rendered raster.

## Runs alongside raster-embed cropping, never gated on it

A page with both a vector chart and a raster photo produces both a vector-region crop and a raster crop. The `vector_ink_threshold` config key (default 0, meaning "never skip") gates whether a page has enough vector primitives to bother running the shapely pass. This is cheap: the primitive count is already available from pdfplumber before any render.

## Index space for vector crops

Vector crop `EmbeddedImage` entries use `image_index_on_page = 1000 + cluster_index` to avoid colliding with raster-embed indices (which are 1-indexed from `page.images`). Downstream — `_ImageRoute`, `ParsedImage`, chunk source naming, and page-render-hash dedup — requires no changes because the index field is already an opaque integer at every layer.

## Table exclusion method

We use centroid-in-bbox rather than intersection-over-union: a cluster is excluded when its centroid falls inside any table bbox returned by `find_tables()`. This is correct for the common case (compact grid tables whose gridlines form a cluster fully inside the table bbox) and cheaper than IOU. A vector annotation that straddles a table boundary might escape exclusion, but that is an acceptable v1 trade-off.

## Non-schema, additive change

`vector_ink_threshold` is read at ingest time and never persisted. No `SCHEMA_VERSION` bump, no re-ingest required. Threaded through `ParseConfig` (alongside `vision_min_dimension` and `vision_max_dimension`) → `pdfplumber_pipeline.convert()`.
