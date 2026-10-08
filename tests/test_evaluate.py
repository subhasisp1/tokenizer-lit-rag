import csv
import json
import sys
from pathlib import Path

import pytest

from rag import config, evaluate, search

FIX = Path(__file__).parent / "fixtures"
CHUNKS = {c["chunk_id"]: c for c in map(json.loads, open(FIX / "chunks_toy.jsonl"))}

# Fake retrievers: query text -> ranked chunk ids. Keyed on the text, so t2 only works via its rewrite.
RANKED = {
    "good": {"What does byte pair encoding merge?": ["w5-c1", "w5-c0", "w1-c0", "w1-c1", "w3-c0", "w2-c0"],
             "How is tokenizer fertility measured across languages?":
                 ["w2-c0", "w4-c2", "w3-c1", "w1-c2", "w5-c2", "w4-c0"]},
    "worse": {"What does byte pair encoding merge?": ["w5-c1", "w5-c0", "w1-c0", "w1-c1", "w3-c0", "w2-c0"],
              "How is tokenizer fertility measured across languages?": ["w2-c0", "w3-c1", "w1-c2"]},
}


def fake_retrieve(q, retriever, k, variant, collapse, **kw):
    return [dict(CHUNKS[c], score=0.0, folded=[]) for c in RANKED[retriever][q][:k]]


@pytest.fixture
def questions(monkeypatch):
    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    return evaluate.load_questions(FIX / "questions_toy.jsonl", "test")


def test_unanswerable_is_skipped(questions):
    assert [q["id"] for q in questions] == ["t1", "t2"]


def test_metrics_by_hand(questions):
    t1, t2 = evaluate.run(questions, "good", "canonical", "r1")
    # t1: the w5 chunk at rank 1 has the quote but the wrong work; first hit is w1-c1 at rank 4.
    # top-5 works w5 w5 w1 w1 w3: slots 2 and 4 repeat a work.
    assert (t1["recall@5"], t1["mrr@10"], t1["first_hit_rank"]) == (1.0, 0.25, 4)
    assert (t1["duprate@5"], t1["distinct_works@5"]) == (0.4, 3)
    # t2: table quote at rank 2 (hit), metric definition at rank 6 (outside top 5): recall 1/2.
    assert (t2["recall@5"], t2["mrr@10"], t2["first_hit_rank"]) == (0.5, 0.5, 2)
    assert (t2["duprate@5"], t2["distinct_works@5"]) == (0.0, 5)
    assert t2["top5"] == "w2-c0|w4-c2|w3-c1|w1-c2|w5-c2"


def test_summary_and_paired_comparison(questions):
    per_q = {name: evaluate.run(questions, name, "canonical", "r1") for name in ("worse", "good")}
    worse, good = evaluate.summarize(per_q, "canonical", "r1", 2000)
    assert (good["recall@5"], good["mrr@10"], good["duprate@5"], good["distinct_works@5"]) == (0.75, 0.375, 0.2, 4.0)
    assert (good["diff_vs_best"], good["wins"], good["losses"]) == (0.0, 0, 0)
    # worse: t2 recall 0 and no hit in the top 10.
    assert (worse["recall@5"], worse["mrr@10"]) == (0.5, 0.125)
    assert (worse["diff_vs_best"], worse["wins"], worse["losses"]) == (-0.25, 0, 1)
    assert -0.5 <= worse["diff_lo"] <= -0.25 <= worse["diff_hi"] <= 0
    assert worse["index_seconds"] == ""


def test_bootstrap_contains_mean_and_is_reproducible():
    values = [0, 1, 1, 0.5, 1, 0, 1, 0.25]
    lo, hi = evaluate.bootstrap_ci(values, 10000)
    assert lo < sum(values) / len(values) < hi
    assert (lo, hi) == evaluate.bootstrap_ci(values, 10000)


def test_gold_matcher_is_fuzzy_but_work_specific():
    gold = {"work_id": "w1", "quote": "iteratively merge the most frequent pairs of adjacent symbol"}
    assert evaluate.is_hit(gold, CHUNKS["w1-c1"])
    assert not evaluate.is_hit(gold, CHUNKS["w5-c1"])  # same sentence, different work
    assert not evaluate.is_hit(gold, CHUNKS["w1-c0"])  # right work, different passage


def test_main_writes_both_csvs(questions, monkeypatch, tmp_path):
    out = tmp_path / "retrieval.csv"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--questions", str(FIX / "questions_toy.jsonl"),
                                      "--retrievers", "good,worse", "--out", str(out), "--bootstrap", "200",
                                      "--index-root", str(tmp_path), "--chunks", str(FIX / "chunks_toy.jsonl")])
    for name in ("INDEX", "CHUNKS", "COLLAPSE_TAU"):  # main() sets these; restore after the test
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(search, "DEVICE", search.DEVICE)
    evaluate.main()
    rows = list(csv.DictReader(open(out)))
    assert [r["retriever"] for r in rows] == ["good", "worse"] and rows[0]["recall@5"] == "0.75"
    assert len(list(csv.DictReader(open(tmp_path / "retrieval_per_question.csv")))) == 4
