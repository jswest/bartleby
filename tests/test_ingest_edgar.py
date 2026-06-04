"""Tests for the EDGAR full-submission envelope parser (bartleby.ingest.edgar).

Pure-Python: no sec2md, no DB. Covers detection, the SGML split, and the
per-inner-document routing decision.
"""

from __future__ import annotations

from bartleby.ingest import edgar
from bartleby.ingest.edgar import InnerDocument


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
