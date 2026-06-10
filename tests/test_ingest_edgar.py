"""Tests for the EDGAR full-submission envelope parser (bartleby.ingest.edgar)
and the anchor-based HTML splitting of EDGAR/iXBRL filings (#254).

The envelope-parser tests are pure-Python (no sec2md, no DB). The anchor-split
tests below drive the real sec2md converter and the Writer's container+section
persistence against an isolated SQLite project.
"""

from __future__ import annotations

import hashlib

import pytest

import bartleby.config
import bartleby.db.connection
import bartleby.project
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest import edgar
from bartleby.ingest import parsers
from bartleby.ingest import sec2md
from bartleby.ingest.edgar import InnerDocument
from bartleby.ingest.writer import ParsedSection, Writer


_SUBMISSION = """\
<SEC-DOCUMENT>0001234567-24-000001.txt : 20240215
<SEC-HEADER>0001234567-24-000001.hdr.sgml : 20240215
ACCESSION NUMBER:\t\t0001234567-24-000001
CONFORMED SUBMISSION TYPE:\t10-K
COMPANY CONFORMED NAME:\t\tACME CORP
CENTRAL INDEX KEY:\t\t0001234567
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>acme-10k.htm
<DESCRIPTION>FORM 10-K
<TEXT>
<html><body><p>Risk factors and revenue figures.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>ex99-1.txt
<TEXT>
Plain text press release body.
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<SEQUENCE>3
<FILENAME>logo.jpg
<TEXT>
begin 644 logo.jpg
M_]C_X``02D9)
end
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-101.INS
<SEQUENCE>4
<FILENAME>acme-20231231.xml
<TEXT>
<XBRL>
<xbrli:context id="c1"/>
</XBRL>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_detect_positive(tmp_path):
    p = tmp_path / "0001234567-24-000001.txt"
    p.write_text(_SUBMISSION, encoding="utf-8")
    assert edgar.detect(p) is True


def test_detect_negative_plain_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Just an ordinary text file mentioning the SEC in passing.")
    assert edgar.detect(p) is False


def test_detect_negative_plain_html(tmp_path):
    p = tmp_path / "page.htm"
    p.write_text("<html><body><h1>Hello</h1></body></html>")
    assert edgar.detect(p) is False


def test_detect_missing_file(tmp_path):
    assert edgar.detect(tmp_path / "nope.txt") is False


def test_parse_splits_inner_documents():
    docs = edgar.parse(_SUBMISSION.encode("utf-8"))
    assert len(docs) == 4
    types = [d.type for d in docs]
    assert types == ["10-K", "EX-99.1", "GRAPHIC", "EX-101.INS"]
    first = docs[0]
    assert first.filename == "acme-10k.htm"
    assert first.sequence == "1"
    assert first.description == "FORM 10-K"
    assert "Risk factors and revenue figures." in first.text


def test_parse_unwraps_xbrl_shell():
    docs = edgar.parse(_SUBMISSION.encode("utf-8"))
    xbrl = docs[3]
    # The <XBRL>...</XBRL> shell inside <TEXT> is stripped to its payload.
    assert xbrl.text.startswith("<xbrli:context")
    assert "<XBRL>" not in xbrl.text


def test_parse_metadata_not_polluted_by_body():
    block = (
        "<SEC-DOCUMENT>\n<DOCUMENT>\n<TYPE>10-K\n<FILENAME>a.htm\n"
        "<TEXT>\n<p>Body says &lt;TYPE&gt;FAKE and <FILENAME>fake.htm</p>\n"
        "</TEXT>\n</DOCUMENT>\n</SEC-DOCUMENT>\n"
    )
    docs = edgar.parse(block.encode("utf-8"))
    assert len(docs) == 1
    # Tags appearing inside the body must not override real metadata.
    assert docs[0].type == "10-K"
    assert docs[0].filename == "a.htm"


def test_classify_routes_each_inner_document():
    docs = edgar.parse(_SUBMISSION.encode("utf-8"))
    kinds = [edgar.classify(d) for d in docs]
    assert kinds == ["html", "text", "skip", "skip"]


def test_classify_skips_empty():
    assert edgar.classify(
        InnerDocument(type="10-K", sequence="1", filename="a.htm",
                      description=None, text="")
    ) == "skip"


def test_classify_html_by_body_sniff_when_filename_uninformative():
    doc = InnerDocument(
        type="10-K", sequence="1", filename=None, description=None,
        text="<?xml version='1.0'?>\n<html xmlns:ix='...'>...</html>",
    )
    assert edgar.classify(doc) == "html"


def test_classify_skips_uuencoded_without_filename_hint():
    doc = InnerDocument(
        type="GRAPHIC", sequence="1", filename=None, description=None,
        text="begin 644 image\nM_]C_X``02D9)\nend",
    )
    assert edgar.classify(doc) == "skip"


# -------------------- #254: anchor / TOC splitting of iXBRL filings -----------

# A synthetic iXBRL filing with a clickable TOC and three anchored sections —
# the shape #254 splits (no live SEC fetch, no network).
_ANCHORED_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<div>
<a href="#sec_business">Business</a>
<a href="#sec_risk">Risk Factors</a>
<a href="#sec_glossary">Glossary of Terms</a>
</div>
<h2 id="sec_business">Business</h2>
<p>We build rockets and launch them into orbit for customers worldwide today.</p>
<p>Our revenue comes from launch services and satellite internet broadband.</p>
<h2 id="sec_risk">Risk Factors</h2>
<p>Rockets are dangerous and may explode on the launch pad without any warning.</p>
<p>Competition in the launch market is intense and is growing larger every year.</p>
<h2 id="sec_glossary">Glossary of Terms</h2>
<p>Apogee is the highest point in an orbit reached by a spacecraft during flight.</p>
</body>
</html>
"""

# Same iXBRL marker, no table of contents — must ingest whole.
_UNANCHORED_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<p>A short filing with no clickable table of contents and no internal anchors.</p>
<p>It carries the iXBRL marker but nothing for the splitter to fracture along.</p>
</body>
</html>
"""

# A filing whose cover page (registrant name, CIK, period) precedes the TOC —
# the pre-TOC body content that the #254 rework must capture as a preamble
# section rather than silently drop (BUG 1).
_PREAMBLE_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<p>Registrant name STARLIGHT AEROSPACE INCORPORATED, Central Index Key 0001999888.</p>
<p>Annual report for the fiscal period ended December 31, 2025, ticker STAR outstanding.</p>
<div>
<a href="#sec_business">Business</a>
<a href="#sec_risk">Risk Factors</a>
</div>
<h2 id="sec_business">Business</h2>
<p>We build rockets and launch them into orbit for customers worldwide today.</p>
<h2 id="sec_risk">Risk Factors</h2>
<p>Rockets are dangerous and may explode on the launch pad without any warning.</p>
</body>
</html>
"""

# A TOC whose links are listed in the WRONG document order: the link to the
# later section comes first. Slicing by link order would run the first slice to
# end-of-body and duplicate the later section's content (BUG 2a).
_OUT_OF_ORDER_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<div>
<a href="#sec_risk">Risk Factors</a>
<a href="#sec_business">Business</a>
</div>
<h2 id="sec_business">Business</h2>
<p>We build rockets and launch them into orbit for customers worldwide today.</p>
<h2 id="sec_risk">Risk Factors</h2>
<p>Rockets are dangerous and may explode on the launch pad without any warning.</p>
</body>
</html>
"""

# A real TOC plus in-text cross-reference links ("see Item 1A") and a "back to
# top" link sprinkled through the body. Only the TOC cluster may create sections
# (BUG 2b) — the stray links must not spawn spurious / out-of-order sections.
_CROSSREF_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<div id="top">
<a href="#sec_business">Business</a>
<a href="#sec_risk">Risk Factors</a>
<a href="#sec_glossary">Glossary of Terms</a>
</div>
<h2 id="sec_business">Business</h2>
<p>We build rockets and launch them into orbit for customers worldwide today.</p>
<p>For competitive pressures, <a href="#sec_risk">see Risk Factors</a> below please.</p>
<h2 id="sec_risk">Risk Factors</h2>
<p>Rockets are dangerous and may explode on the launch pad without any warning.</p>
<p>As noted in <a href="#sec_business">Business</a>, <a href="#top">back to top</a>.</p>
<h2 id="sec_glossary">Glossary of Terms</h2>
<p>Apogee is the highest point in an orbit reached by a spacecraft during flight.</p>
</body>
</html>
"""


@pytest.fixture
def edgar_project(tmp_path, monkeypatch):
    """An isolated SQLite project plus a stub embedder, for the persist tests."""
    projects = tmp_path / "projects"
    projects.mkdir()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)

    def fake_embed(texts):
        return [[0.01 * (i + 1)] * EMBEDDING_DIM for i in range(len(texts))]
    monkeypatch.setattr("bartleby.ingest.embed.embed_texts", fake_embed)

    bartleby.project.create_project("edgar_proj")
    return "edgar_proj"


def _write(tmp_path, name, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_convert_sections_splits_toc_anchored_filing():
    sections = sec2md._convert_sections_bytes(_ANCHORED_FILING)
    assert [s.anchor_id for s in sections] == [
        "sec_business", "sec_risk", "sec_glossary",
    ]
    assert [s.title for s in sections] == [
        "Business", "Risk Factors", "Glossary of Terms",
    ]
    assert [s.order for s in sections] == [0, 1, 2]
    # Each slice converted independently and carries the right content.
    assert any("rockets" in c.text.lower() for c in sections[0].result.chunks)
    assert any("explode" in c.text.lower() for c in sections[1].result.chunks)
    assert any("apogee" in c.text.lower() for c in sections[2].result.chunks)


def test_convert_sections_returns_empty_without_toc():
    assert sec2md._convert_sections_bytes(_UNANCHORED_FILING) == []


def test_convert_sections_ignores_dangling_anchors():
    """A TOC whose links resolve to fewer than two in-document targets is not a
    split boundary — the file ingests whole."""
    html = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<a href="#real">Real</a>
<a href="#ghost">Ghost</a>
<h2 id="real">Real</h2>
<p>This is the only anchor that actually resolves to a target in the body here.</p>
</body></html>"""
    assert sec2md._convert_sections_bytes(html) == []


def test_convert_sections_emits_preamble_for_pre_toc_content():
    """Body content before the first TOC target (the EDGAR cover page) lands in a
    synthetic order-0 preamble section rather than being silently dropped (BUG 1).
    The TOC nav block itself is not re-indexed as preamble."""
    sections = sec2md._convert_sections_bytes(_PREAMBLE_FILING)
    assert sections[0].anchor_id == sec2md._PREAMBLE_ANCHOR_ID
    assert sections[0].order == 0
    # The cover-page facts are searchable in the preamble section's chunks.
    preamble_text = " ".join(c.text for c in sections[0].result.chunks)
    assert "STARLIGHT AEROSPACE" in preamble_text
    assert "0001999888" in preamble_text
    # The real sections follow, renumbered after the preamble.
    assert [s.anchor_id for s in sections[1:]] == ["sec_business", "sec_risk"]
    assert [s.order for s in sections[1:]] == [1, 2]
    # The TOC link list is navigation, not content — it is not re-emitted.
    assert "Risk Factors\nBusiness" not in preamble_text


def test_convert_sections_does_not_duplicate_out_of_document_order_anchors():
    """A TOC that lists anchors out of document order is sliced in DOCUMENT order,
    so no section's slice runs to end-of-body and duplicates a later section's
    content (BUG 2a)."""
    sections = sec2md._convert_sections_bytes(_OUT_OF_ORDER_FILING)
    # Sliced in document order regardless of TOC link order.
    assert [s.anchor_id for s in sections] == ["sec_business", "sec_risk"]
    # The risk paragraph ("explode") appears in exactly one section, once.
    occurrences = sum(
        c.text.lower().count("explode")
        for s in sections for c in s.result.chunks
    )
    assert occurrences == 1
    # And it belongs to the risk section, not bled into business.
    business_text = " ".join(c.text.lower() for c in sections[0].result.chunks)
    assert "explode" not in business_text


def test_convert_sections_ignores_in_text_cross_reference_links():
    """In-text cross-references ("see Risk Factors") and "back to top" links do
    not create spurious sections — only the contiguous TOC link cluster does
    (BUG 2b)."""
    sections = sec2md._convert_sections_bytes(_CROSSREF_FILING)
    assert [s.anchor_id for s in sections] == [
        "sec_business", "sec_risk", "sec_glossary",
    ]
    # No content duplicated by the stray forward/backward cross-reference links.
    occurrences = sum(
        c.text.lower().count("apogee")
        for s in sections for c in s.result.chunks
    )
    assert occurrences == 1


def test_persist_parse_preamble_section_is_searchable(edgar_project, tmp_path):
    """The preamble section persists as a real, FTS-indexed section row, so the
    cover-page content is searchable rather than lost with the zero-chunk
    container (BUG 1, end to end)."""
    src = _write(tmp_path, "filing.htm", _PREAMBLE_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="preamble-container", file_name="filing.htm",
    )
    # The container carries a preamble section child with the synthetic anchor.
    assert parsed.sections[0].anchor_id == sec2md._PREAMBLE_ANCHOR_ID
    conn = open_db(edgar_project)
    try:
        writer = Writer(conn)
        container_id = writer.persist_parse(parsed)
        cur = conn.cursor()
        preamble = cur.execute(
            "SELECT document_id FROM documents "
            "WHERE anchor_id = ? AND parent_document_id = ?",
            (sec2md._PREAMBLE_ANCHOR_ID, container_id),
        ).fetchone()
        assert preamble is not None
        # Its chunks are indexed in FTS — the cover-page facts are findable.
        hit = cur.execute(
            "SELECT COUNT(*) FROM chunks c JOIN chunks_fts f ON f.rowid = c.chunk_id "
            "WHERE c.source_kind = 'document' AND c.source_id = ? "
            "AND chunks_fts MATCH 'STARLIGHT'",
            (preamble[0],),
        ).fetchone()[0]
        assert hit >= 1
    finally:
        conn.close()


def test_parse_html_sec2md_builds_container_and_sections(tmp_path, monkeypatch):
    """The sec2md parse path returns a zero-chunk container ParsedDocument with
    N section children, each with its own chunks and TOC metadata."""
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts],
    )
    src = _write(tmp_path, "filing.htm", _ANCHORED_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="container-hash", file_name="filing.htm",
    )

    # Container: original hash, zero chunks of its own, N sections.
    assert parsed.file_hash == "container-hash"
    assert parsed.document_chunks == []
    assert len(parsed.sections) == 3

    s0 = parsed.sections[0]
    assert s0.anchor_id == "sec_business"
    assert s0.section_title == "Business"
    assert s0.section_order == 0
    assert s0.document_chunks  # own chunks
    # Derived hash = sha256(file_bytes + anchor_id), unique per section.
    expected = hashlib.sha256(_ANCHORED_FILING + b"sec_business").hexdigest()
    assert s0.file_hash == expected
    assert len({s.file_hash for s in parsed.sections}) == 3
    assert parsed.file_hash not in {s.file_hash for s in parsed.sections}


def test_parse_html_sec2md_ingests_unanchored_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts],
    )
    src = _write(tmp_path, "plain.htm", _UNANCHORED_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="whole-hash", file_name="plain.htm",
    )
    # No split: one document, its own chunks, no section children.
    assert parsed.sections == []
    assert parsed.document_chunks
    assert parsed.file_hash == "whole-hash"


def test_persist_parse_writes_container_and_sections(edgar_project, tmp_path):
    """The Writer persists the container (zero chunks) plus N section rows, each
    with parent_document_id wired and the anchor/title/order columns set."""
    src = _write(tmp_path, "filing.htm", _ANCHORED_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="container-hash", file_name="filing.htm",
    )
    conn = open_db(edgar_project)
    try:
        writer = Writer(conn)
        container_id = writer.persist_parse(parsed)
        cur = conn.cursor()

        # Four documents: one container + three sections.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 4

        # The container owns zero document chunks; its file_hash is the original.
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document' "
            "AND source_id = ?", (container_id,),
        ).fetchone()[0] == 0
        assert cur.execute(
            "SELECT file_hash, parent_document_id, anchor_id FROM documents "
            "WHERE document_id = ?", (container_id,),
        ).fetchone() == ("container-hash", None, None)

        # Three section rows, all parented to the container, in TOC order, each
        # with chunks of its own.
        rows = cur.execute(
            "SELECT anchor_id, section_title, section_order, parent_document_id "
            "FROM documents WHERE parent_document_id = ? "
            "ORDER BY section_order", (container_id,),
        ).fetchall()
        assert [r[0] for r in rows] == ["sec_business", "sec_risk", "sec_glossary"]
        assert [r[1] for r in rows] == [
            "Business", "Risk Factors", "Glossary of Terms",
        ]
        assert [r[2] for r in rows] == [0, 1, 2]
        assert all(r[3] == container_id for r in rows)

        # Every section carries its own indexed chunks (FTS + vec in lockstep).
        section_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks c "
            "JOIN documents d ON d.document_id = c.source_id "
            "WHERE c.source_kind='document' AND d.parent_document_id = ?",
            (container_id,),
        ).fetchone()[0]
        assert section_chunks >= 3
        total_doc_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert total_doc_chunks == section_chunks  # container added none
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks_fts"
        ).fetchone()[0] == total_doc_chunks
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks_vec"
        ).fetchone()[0] == total_doc_chunks
    finally:
        conn.close()


def test_persist_parse_split_is_atomic(edgar_project, tmp_path):
    """A failure mid-split rolls back the whole unit — no container, no section
    rows, no chunks — so resume (keyed on the container's file_hash) re-parses
    cleanly. The split writes N+1 rows in one transaction (#254/#358)."""
    from bartleby.db.chunks import ChunkInput

    src = _write(tmp_path, "filing.htm", _ANCHORED_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="container-hash", file_name="filing.htm",
    )
    # Poison the last section's chunks with an over-long embedding so the chunk
    # insert validation raises after earlier section rows already landed.
    bad = ChunkInput(
        text="poison", embedding=[0.1] * (EMBEDDING_DIM + 1), chunk_index=99,
    )
    parsed.sections[-1].document_chunks.append(bad)

    conn = open_db(edgar_project)
    try:
        writer = Writer(conn)
        with pytest.raises(ValueError, match="dims"):
            writer.persist_parse(parsed)
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    finally:
        conn.close()


def test_section_file_hash_is_deterministic_and_distinct():
    a = parsers._section_file_hash(b"filing-bytes", "anchor_a")
    b = parsers._section_file_hash(b"filing-bytes", "anchor_b")
    assert a == parsers._section_file_hash(b"filing-bytes", "anchor_a")  # stable
    assert a != b                                                        # per-anchor
    # Never collides with the container's plain byte-hash.
    assert a != hashlib.sha256(b"filing-bytes").hexdigest()
