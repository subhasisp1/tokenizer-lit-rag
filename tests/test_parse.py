import json
import re

import pytest

from rag import config
from rag.parse import is_low, parse_html, parse_pdf

SAMPLES = config.RAW / "samples"
pytestmark = pytest.mark.skipif(not SAMPLES.exists(), reason="needs data/raw/samples")
REF_PATTERN = re.compile(r"In Proceedings of|arXiv preprint|Association for Computational Linguistics|pp\. \d+")


@pytest.fixture(scope="module")
def bpe():
    return parse_html(SAMPLES / "1508.07909v5.html", "1508.07909v5")


def test_html_headings_keep_numbers_and_nesting(bpe):
    paths = [s["path"] for s in bpe["sections"]]
    for path in ["1 Introduction", "3 Subword Translation > 3.2 Byte Pair Encoding (BPE)",
                 "4 Evaluation > 4.1 Subword statistics", "6 Conclusion"]:
        assert path in paths
    assert bpe["doc_id"] == "arxiv:1508.07909v5" and bpe["work_id"] == "1508.07909"
    assert bpe["title"] == "Neural Machine Translation of Rare Words with Subword Units"


def test_html_drops_bibliography_and_markup(bpe):
    assert bpe["n_refs_dropped"] > 0
    assert not any("In Proceedings" in p for s in bpe["sections"] for p in s["paragraphs"])
    assert "ltx_" not in json.dumps(bpe)
    assert not any(re.search(r"acknowledg", s["path"], re.I) for s in bpe["sections"])


def test_html_inline_math_is_latex(bpe):
    assert any(re.search(r"\$[^$]+\$", p) for s in bpe["sections"] for p in s["paragraphs"])


def test_html_tables_keep_header_row(bpe):
    table = bpe["sections"][6]["tables"][0]
    assert table["caption"].startswith("Table 1:")
    assert table["rows"][0] == ["segmentation", "# tokens", "# types", "# UNK"]


@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*v[0-9].html")), ids=lambda p: p.stem)
def test_every_html_sample_passes_quality_gate(path):
    assert not is_low(parse_html(path, path.stem))


def test_pdf_headings_without_references():
    doc = parse_pdf(SAMPLES / "1508.07909v5.pdf", "1508.07909v5")
    paths = [s["path"] for s in doc["sections"]]
    assert "1 Introduction" in paths and "6 Conclusion" in paths and doc["abstract"]
    assert not any(len(REF_PATTERN.findall(" ".join(s["paragraphs"]))) >= 3 for s in doc["sections"])
