import re

import pytest

from rag import config
from rag.chunk import chunk_doc, n_tokens
from rag.parse import parse_html

SAMPLES = config.RAW / "samples"
TITLE = "A Title"


def section(path, paragraphs=(), tables=()):
    return {"path": path, "level": 1, "paragraphs": list(paragraphs), "tables": list(tables), "captions": []}


def doc(*sections):
    return {"doc_id": "arxiv:0000.00000v1", "work_id": "0000.00000", "title": TITLE, "abstract": "", "sections": list(sections)}


def tokens(title, chunk):
    kind, path, text = chunk
    return n_tokens(f"{title} | {path}\n{text}")


@pytest.fixture(scope="module")
def bpe():
    if not SAMPLES.exists():
        pytest.skip("needs data/raw/samples")
    d = parse_html(SAMPLES / "1508.07909v5.html", "1508.07909v5")
    return d, chunk_doc(d, d["title"])


def test_text_chunks_never_cross_a_section(bpe):
    d, chunks = bpe
    for kind, path, text in chunks:
        if kind != "text":
            continue
        i = next(i for i, s in enumerate(d["sections"]) if s["path"] == path)
        allowed = list(d["sections"][i]["paragraphs"])
        while i > 0 and sum(len(p.split()) for p in d["sections"][i - 1]["paragraphs"]) < config.MIN_SECTION_WORDS:
            i -= 1
            allowed += d["sections"][i]["paragraphs"]  # short sections merged forward
        assert all(any(piece in p for p in allowed) for piece in text.split("\n\n"))


def test_no_chunk_looks_like_a_bibliography(bpe):
    pattern = re.compile(r"In Proceedings of|arXiv preprint|Association for Computational Linguistics")
    assert not [c for c in bpe[1] if len(pattern.findall(c[2])) >= 3]


def test_every_chunk_fits_512_tokens(bpe):
    d, chunks = bpe
    long_para = " ".join(f"Sentence {i} talks about byte pair encoding and $x_{{{i}}}$ merges." for i in range(400))
    synthetic = chunk_doc(doc(section("1 Long", [long_para])), TITLE)
    assert len(synthetic) > 2
    assert all(tokens(d["title"], c) <= 512 for c in chunks)
    assert all(tokens(TITLE, c) <= config.CHUNK_MAX_TOKENS for c in synthetic)


def test_split_table_repeats_caption_and_header():
    rows = [["language", "vocab", "fertility"]] + [[f"lang{i}", f"{i * 1000}", f"{i / 7:.3f}"] for i in range(300)]
    table = {"caption": "Table 9: Fertility per language.", "rows": rows}
    chunks = [c for c in chunk_doc(doc(section("1 Tables", tables=[table])), TITLE) if c[0] == "table"]
    assert len(chunks) > 1
    assert all(c[2].startswith("Table 9: Fertility per language.\nlanguage | vocab | fertility\n") for c in chunks)
    body = [line for c in chunks for line in c[2].split("\n")[2:]]
    assert body == [" | ".join(r) for r in rows[1:]]  # every row once, in order


def test_short_section_merges_into_next():
    short = "Only a few words here."
    long_text = " ".join(["Subword units make the vocabulary open."] * 20)
    chunks = chunk_doc(doc(section("1 Short", [short]), section("2 Long", [long_text])), TITLE)
    assert [c[1] for c in chunks] == ["2 Long"]
    assert chunks[0][2] == f"{short}\n\n{long_text}"
