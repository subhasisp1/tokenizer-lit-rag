import json
from pathlib import Path

import pytest

from rag import config, llm, questions, search

FIX = Path(__file__).parent / "fixtures"
CHUNKS = [json.loads(line) for line in open(FIX / "chunks_toy.jsonl")]
DOC = {"doc_id": "arxiv:1508.07909v5", "work_id": "1508.07909", "title": "Rare Words with Subword Units",
       "abstract": "We encode rare and unknown words as sequences of subword units.", "parser": "html",
       "sections": [{"path": "3 BPE", "level": 1, "tables": [], "captions": [], "paragraphs": [
           "Byte pair encoding iteratively merges the most frequent pair of symbols.",
           "We learn 59500 merge operations on the joint vocabulary of English and German."]}]}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Queue chat replies in stub["replies"]; calls are recorded. Retrieve returns the toy chunks."""
    rec = {"replies": [], "calls": []}

    def fake_chat(messages, model, schema=None, temperature=0.0, max_tokens=1024):
        rec["calls"].append(messages)
        return json.dumps(rec["replies"].pop(0)), 0.01

    def fake_retrieve(q, retriever=None, k=None, variant="canonical", collapse_mode="r1"):
        return [dict(c) for c in CHUNKS[:k]]

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "no_manifest.csv")
    monkeypatch.setattr(config, "PARSED", tmp_path / "parsed")
    monkeypatch.setattr(config, "INDEX", tmp_path / "index")
    config.PARSED.mkdir()
    return rec


def test_quote_verification():
    texts = questions.section_texts(DOC)
    assert questions.verify("iteratively merges the most frequent pair of symbols", texts, "3 BPE")[1] == 100
    edited = "Byte-pair encoding iteratively merges the most frequent pairs of symbols."
    assert questions.verify(edited, texts, "Abstract")[0] == "3 BPE"  # found in another section
    assert questions.verify(edited, texts, "Abstract")[1] >= 90
    assert questions.verify("The model is trained for ten days on eight GPUs.", texts, "3 BPE")[1] < 90


def test_lexical_overlap_by_hand():
    # {bpe, merge, symbols} vs {bpe, merges, most, frequent, pair, symbols}: 2 shared of 7
    assert questions.overlap("How does BPE merge symbols?", ["BPE merges the most frequent pair of symbols."]) == 0.2857
    # {fertility} vs {fertility, tokens, per, word, we, measure}: 1 of 6
    assert questions.overlap("What is fertility?", ["Fertility is tokens per word.", "We measure fertility."]) == 0.1667
    assert questions.overlap("the of", []) == 0.0


def test_propose_one_doc_keeps_verified_and_drops_unverifiable(stub, tmp_path):
    (config.PARSED / "arxiv_1508.07909v5.json").write_text(json.dumps(DOC))
    stub["replies"] = [{"questions": [
        {"question": "How many merges were learned?", "type": "lookup", "answer": "59500",
         "quote": "We learn 59500 merge operations on the joint vocabulary", "section": "3 BPE"},
        {"question": "Why use subwords?", "type": "conceptual", "answer": "Open vocabulary.",
         "quote": "Subwords let the decoder generate unseen compounds from known parts.", "section": "3 BPE"}]}]
    cands, dropped = questions.propose(1, 0, tmp_path / "c.jsonl", 13, "m")
    assert dropped == 1 and len(stub["calls"]) == 1
    assert [c["cand_id"] for c in cands] == ["1508.07909-0"]
    assert cands[0]["gold"] == [{"work_id": "1508.07909", "section": "3 BPE",
                                 "quote": "We learn 59500 merge operations on the joint vocabulary"}]
    assert questions.read_jsonl(tmp_path / "c.jsonl") == cands


def test_multi_theme_needs_two_works(stub, tmp_path):
    stub["replies"] = [
        {"question": "q1", "answer": "a1", "supports": [  # toy chunk 2 is w1, chunk 5 is w2
            {"n": 2, "quote": "iteratively merges the most frequent pair of adjacent symbols"},
            {"n": 5, "quote": "removes the pieces whose loss in corpus likelihood is smallest"}]},
        {"question": "q2", "answer": "a2", "supports": [  # w1 twice, and a w3 quote that is not in chunk 7
            {"n": 1, "quote": "We encode rare and unknown words as sequences of subword units"},
            {"n": 2, "quote": "iteratively merges the most frequent pair of adjacent symbols"},
            {"n": 7, "quote": "Bytes need a vocabulary of exactly 256 entries plus specials."}]}]
    cands, dropped = questions.propose(0, 2, tmp_path / "c.jsonl", 13, "m")
    assert dropped == 1 and len(cands) == 1
    assert cands[0]["type"] == "multi" and [g["work_id"] for g in cands[0]["gold"]] == ["w1", "w2"]
    assert cands[0]["titles"] == [CHUNKS[1]["title"], CHUNKS[4]["title"]]


def test_pool_skips_gold_and_resumes(stub, tmp_path):
    for r in ("bm25", "bge"):
        (config.INDEX / "canonical" / r).mkdir(parents=True)
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("".join(json.dumps(q) + "\n" for q in [
        {"id": "a", "type": "lookup", "question": "What does BPE merge?", "standalone_rewrite": "",
         "expected_answer": "pairs", "gold": [{"work_id": "w1", "section": "", "quote":
                                              "iteratively merges the most frequent pair of adjacent symbols"}]},
        {"id": "u", "type": "unanswerable", "question": "Klingon?", "gold": []}]))
    stub["replies"] = [{"grade": 3, "reason": "same sentence"}, {"grade": 0, "reason": "no"}, {"grade": 1, "reason": "bg"}]
    rows = questions.pool(qfile, ["bm25", "bge", "qwen-or"], 4, tmp_path / "p.jsonl", "m")
    # toy chunks 0-3: w1-c1 is the gold hit (no call); w1-c0, w1-c2, w2-c0 are graded once each
    assert len(stub["calls"]) == 3
    assert [(r["chunk_id"], r["grade"]) for r in rows] == [("w1-c0", 3), ("w1-c1", None), ("w1-c2", 0), ("w2-c0", 1)]
    questions.pool(qfile, ["bm25", "bge"], 4, tmp_path / "p.jsonl", "m")
    assert len(stub["calls"]) == 3 and len(questions.read_jsonl(tmp_path / "p.jsonl")) == 4


def test_stats_on_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHUNKS", FIX / "chunks_toy.jsonl")
    pool = tmp_path / "p.jsonl"
    pool.write_text("".join(json.dumps({"question_id": "t1", "chunk_id": c, "grade": g}) + "\n"
                            for c, g in [("w3-c1", None), ("w3-c0", 2), ("w4-c1", 1)]))
    scored, high = questions.stats(FIX / "questions_stats.jsonl", pool, tmp_path / "s.md")
    md = (tmp_path / "s.md").read_text()
    assert "| multi | 0 | 2 | 2 |" in md and "| lookup | 1 | 0 | 1 |" in md
    assert "- t4: parent yes, standalone_rewrite yes" in md and "- t5: parent MISSING, standalone_rewrite MISSING" in md
    assert "## Unanswerable: 1" in md and "- t6: no paper in the corpus covers Klingon" in md
    assert "min 2, median 2.5, max 3" in md
    # t1 copies its quote: 7 shared tokens of 10
    assert scored["t1"] == 0.7 and high == ["t1"]
    assert "median 0.139; above 0.30: 1/6 (17%): t1" in md
    assert "| t1 | 1/1 | 1 |" in md and "| d1 | 0/1 | 0 |" in md
