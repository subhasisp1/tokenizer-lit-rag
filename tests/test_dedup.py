import csv
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

from rag import collect, config, dedup, index, manifest

FIX = Path(__file__).parent / "fixtures"
RULE = (dedup.TITLE_RATIO, dedup.AUTHOR_JACCARD, dedup.ABSTRACT_COS)


def fake_encode(texts, model, device="cpu"):
    """Hashed bag of words, unit length: same words -> cosine 1, no model needed."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for k, t in enumerate(texts):
        for w in t.lower().split():
            out[k, zlib.crc32(w.encode()) % 256] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def tmp(tmp_path, monkeypatch):
    shutil.copy(FIX / "manifest_dedup.csv", tmp_path / "manifest.csv")
    for name, words in [("arxiv_2106.00006v1", 2000), ("arxiv_2207.00007v1", 6000)]:
        (tmp_path / f"{name}.json").write_text(json.dumps({"body_words": words}))
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "PARSED", tmp_path)
    monkeypatch.setattr(collect, "load_candidates", lambda: [json.loads(x) for x in open(FIX / "candidates_dedup.jsonl")])
    monkeypatch.setattr(index, "encode_texts", fake_encode)
    return tmp_path


def by_id():
    return {r["record_id"]: r for r in manifest.load()}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_group_work_ids_and_decisions(tmp):
    dedup.group(RULE, report=True)
    rows = by_id()
    for rid in ["arxiv:2101.00001", "arxiv:2101.00001v1", "acl:2021.acl-long.1", "oa:W100"]:
        assert rows[rid]["work_id"] == "2101.00001"
    assert rows["arxiv:2203.00003"]["work_id"] == "2102.00002"
    assert rows["arxiv:2105.00005"]["work_id"] == "2105.00005"
    assert rows["arxiv:2207.00007"]["work_id"] == "2207.00007"
    assert rows["arxiv:2109.00009"]["work_id"] == "old"  # out of scope: untouched
    decision = {(p["record_a"], p["record_b"]): p["decision"] for p in read_csv(tmp / "results/dedup_candidates.csv")}
    assert decision[("arxiv:2102.00002", "arxiv:2203.00003")] == "merge"
    assert decision[("arxiv:2104.00004", "arxiv:2105.00005")] == "separate"
    assert decision[("arxiv:2106.00006", "arxiv:2207.00007")] == "related"
    related = read_csv(tmp / "results/dedup_related.csv")
    assert [(r["work_a"], r["work_b"], r["body_ratio"]) for r in related] == [("2106.00006", "2207.00007", "0.333")]


def test_label_file_columns_and_no_overwrite(tmp, capsys):
    dedup.group(RULE, report=False)
    path = tmp / dedup.LABEL_FILE
    labels = read_csv(path)
    assert list(labels[0]) == dedup.LABEL_COLUMNS
    assert len(labels) == len(read_csv(tmp / "results/dedup_candidates.csv"))  # fewer than 20 pairs: all of them
    assert all(r["human_same_work"] == "" for r in labels)
    path.write_text("my labels\n")
    dedup.group(RULE, report=False)
    assert path.read_text() == "my labels\n"
    assert "not overwritten" in capsys.readouterr().out


def test_canonical(tmp):
    dedup.canonical(RULE)
    rows = by_id()
    canon = {rid for rid, r in rows.items() if r["canonical"] == "true"}
    assert canon == {"arxiv:2101.00001", "arxiv:2203.00003", "arxiv:2104.00004", "arxiv:2105.00005",
                     "arxiv:2106.00006", "arxiv:2207.00007", "arxiv:2108.00008", "arxiv:2110.00010",
                     "arxiv:2111.00011"}
    assert rows["oa:W100"]["duplicate_of"] == "arxiv:2101.00001"
    assert rows["oa:W100"]["not_indexed_reason"] == "duplicate_of:arxiv:2101.00001"
    assert rows["arxiv:2101.00001v1"]["duplicate_of"] == "arxiv:2101.00001"
    assert rows["arxiv:2102.00002"]["not_indexed_reason"] == "duplicate_of:arxiv:2203.00003"
    assert rows["arxiv:2101.00001"]["in_index"] == "true" and rows["arxiv:2101.00001"]["not_indexed_reason"] == ""
    assert (rows["arxiv:2110.00010"]["in_index"], rows["arxiv:2110.00010"]["not_indexed_reason"]) == ("false", "corpus_cap")
    assert (rows["arxiv:2111.00011"]["in_index"], rows["arxiv:2111.00011"]["not_indexed_reason"]) == ("false", "no_fulltext")
    assert rows["arxiv:2109.00009"]["canonical"] == "" and rows["arxiv:2109.00009"]["in_index"] == ""
    assert sum(r["in_index"] == "true" for r in rows.values()) == 7


def test_sweep(tmp, capsys):
    path = tmp / "labels.csv"
    rows = [  # a true pair the default rule misses (title 90), and a false pair with weak authors
        {"record_a": "x:1", "record_b": "x:2", "title_ratio": "90", "author_jaccard": "0.6", "abstract_cosine": "0.9",
         "body_ratio": "", "human_same_work": "true"},
        {"record_a": "x:3", "record_b": "x:4", "title_ratio": "96", "author_jaccard": "0.4", "abstract_cosine": "0.95",
         "body_ratio": "0.9", "human_same_work": "false"},
        {"record_a": "x:5", "record_b": "x:6", "title_ratio": "99", "author_jaccard": "1", "abstract_cosine": "",
         "body_ratio": "", "human_same_work": ""},  # unlabelled: ignored
    ]
    dedup.write_csv(path, dedup.LABEL_COLUMNS, rows)
    dedup.sweep(path)  # silver positive: arxiv:2101.00001 and oa:W100 share W100
    rules = {(float(r["title_ratio"]), float(r["author_jaccard"]), float(r["abstract_cos"])): r
             for r in read_csv(tmp / "results/dedup_rules.csv")}
    assert len(rules) == 105
    assert [rules[92, 0.5, 0.85][k] for k in ("tp", "fp", "fn", "precision", "recall")] == ["1", "0", "1", "1.0", "0.5"]
    assert [rules[80, 0.3, 0.8][k] for k in ("tp", "fp", "fn")] == ["2", "1", "0"]
    out = capsys.readouterr().out
    assert "positives 2 (1 silver)  negatives 1" in out
    assert "strictest of ties: {'title_ratio': 90, 'author_jaccard': 0.6, 'abstract_cos': 0.9, 'tp': 2, 'fp': 0" in out


def test_subset_title_is_not_a_match(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PARSED", tmp_path)
    short = {"record_id": "arxiv:1", "source": "arxiv", "arxiv_id": "1", "arxiv_latest_version": "1",
             "title": "Subword Units", "authors": "Rico Sennrich; Barry Haddow"}
    long = dict(short, record_id="arxiv:2", arxiv_id="2",
                title="Neural Machine Translation of Rare Words with Subword Units")
    f = dedup.features(short, long, {})
    assert f["title_ratio"] < dedup.TITLE_RATIO and dedup.decide(f) == "separate"
