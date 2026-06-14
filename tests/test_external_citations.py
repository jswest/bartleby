"""External-citation body markers ([^url:…] / [^doc:…]) — issue #563.

External citations ride *alongside* the mandatory ≥1 corpus-chunk citation,
never as a substitute. They live in the finding body as alpha-prefixed footnote
markers, parsed and rendered on read — no DB row, no schema change. These tests
pin:

- the alpha-prefixed markers never count toward NO_INLINE_CITATIONS (the digit-
  only chunk extractor ignores them);
- an external-only finding is still rejected;
- the save-time well-formedness check rejects unknown schemes / blank refs;
- save_finding / read_finding surface external citations as a distinct kind.
"""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import read_finding, save_finding
from bartleby.skill_scripts._common import (
    SkillError,
    extract_citations,
    extract_external_citations,
    reject_malformed_citations,
)
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
)


# --- pure helpers (no DB) ----------------------------------------------------

def test_extract_external_citations_url_and_doc():
    body = (
        "Corpus claim[^42]. Web claim[^url:https://example.gov/report]. "
        "Dataset claim[^doc:H-2024-0148]."
    )
    assert extract_external_citations(body) == [
        {"scheme": "url", "ref": "https://example.gov/report"},
        {"scheme": "doc", "ref": "H-2024-0148"},
    ]


def test_extract_external_citations_dedupes_and_orders():
    body = (
        "[^doc:X] then [^url:https://a.test] then [^doc:X] again "
        "and [^url:https://a.test]."
    )
    assert extract_external_citations(body) == [
        {"scheme": "doc", "ref": "X"},
        {"scheme": "url", "ref": "https://a.test"},
    ]


def test_extract_external_citations_normalizes_scheme_case_and_trims_ref():
    assert extract_external_citations("[^URL: https://a.test ]") == [
        {"scheme": "url", "ref": "https://a.test"},
    ]


def test_external_markers_do_not_count_as_chunk_citations():
    """The digit-only chunk extractor must ignore alpha-prefixed external
    markers — that is what keeps the ≥1-chunk invariant unchanged for free."""
    body = "Only external[^url:https://a.test] and[^doc:Y]."
    assert extract_citations(body) == []
    # A real chunk marker alongside is still counted, externals still ignored.
    assert extract_citations("Claim[^7] web[^url:https://a.test].") == [7]


def test_extract_external_citations_ignores_malformed_markers():
    """extract_* silently skips ill-formed markers; the save-time check is what
    rejects them. Unknown scheme and blank ref both drop out here."""
    assert extract_external_citations("[^ftp:nope] [^url:]") == []


def test_malformed_check_ignores_bracketed_digits_inside_external_ref():
    """A ``[N]``-shaped substring inside an external ref (e.g. a URL ending
    ``…/doc[3]``) must not false-trip the caret-less-marker guard when a valid
    chunk citation is present. Regression for the #563×malformed-check seam."""
    body = "Corpus[^42]. Web[^url:https://example.com/doc[3]]."
    reject_malformed_citations(body)  # must not raise
    # A genuine caret-less [N] outside any external marker is still rejected.
    with pytest.raises(SkillError) as exc:
        reject_malformed_citations("Bad cite [5] here.")
    assert exc.value.code == "MALFORMED_CITATION"


# --- integration: save / read ------------------------------------------------

def _first_chunk_id(seeded_project) -> int:
    conn = open_db(seeded_project["project"])
    try:
        return conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()


def test_save_finding_surfaces_external_citations(seeded_project, tmp_path, capsys):
    cid = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Corpus[^{cid}]. Web[^url:https://example.gov/x]. "
        f"Filing[^doc:H-001].",
        encoding="utf-8",
    )
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "with externals",
        "--description", "x",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    # The mandatory chunk citation still lands ...
    assert [c["chunk_id"] for c in out["citations"]] == [cid]
    # ... and the external citations surface as a distinct kind.
    assert out["external_citations"] == [
        {"scheme": "url", "ref": "https://example.gov/x"},
        {"scheme": "doc", "ref": "H-001"},
    ]


def test_save_finding_external_only_still_rejected(seeded_project, tmp_path, capsys):
    """A body carrying only external markers has zero [^N] chunk citations, so
    NO_INLINE_CITATIONS still fires — external citations never substitute."""
    body_file = tmp_path / "f.md"
    body_file.write_text(
        "Web only[^url:https://a.test] and a filing[^doc:H-2].",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "external only",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NO_INLINE_CITATIONS"


def test_save_finding_rejects_unknown_external_scheme(
    seeded_project, tmp_path, capsys
):
    cid = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Corpus[^{cid}] but bad scheme[^ftp:host/file].",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad scheme",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_EXTERNAL_CITATION"
    assert out["malformed_markers"] == ["[^ftp:host/file]"]


def test_save_finding_rejects_blank_external_ref(seeded_project, tmp_path, capsys):
    cid = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(f"Corpus[^{cid}] empty[^url:   ].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "blank ref",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_EXTERNAL_CITATION"


def test_read_finding_surfaces_external_citations(seeded_project, tmp_path, capsys):
    cid = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_text = f"Corpus[^{cid}]. Web[^url:https://example.gov/y]."
    body_file.write_text(body_text, encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "round trip",
        "--description", "x",
        "--body-file", str(body_file),
    ])
    saved = json.loads(capsys.readouterr().out)

    read_finding.main([
        "--project", seeded_project["project"],
        "--finding", str(saved["finding_id"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["body"] == body_text
    assert out["external_citations"] == [
        {"scheme": "url", "ref": "https://example.gov/y"},
    ]
    # The chunk citation is unaffected and external markers don't dangle (the
    # digit-only dangling check never sees them).
    assert [c["chunk_id"] for c in out["citations"]] == [cid]
    assert out["dangling_citations"] == []
