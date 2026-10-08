import zlib
from pathlib import Path

import numpy as np
import pytest

from rag import config, index, search

FIX = Path(__file__).parent / "fixtures"
CHUNKS = FIX / "chunks_toy.jsonl"


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("index")
    try:
        index.load_model("bge", "cpu")
    except Exception as e:
        pytest.skip(f"bge model could not be loaded: {e}")
    chunks = index.load_chunks(CHUNKS, "canonical")
    for model in ("bge", "bm25"):
        index.build(chunks, model, root / "canonical" / model, "cpu")
    return root


@pytest.fixture(autouse=True)
def toy_config(monkeypatch):
    monkeypatch.setattr(config, "CHUNKS", CHUNKS)
    monkeypatch.setattr(config, "MANIFEST", FIX / "manifest_toy.csv")
    monkeypatch.setattr(search, "DEVICE", "cpu")


@pytest.fixture
def toy_index(monkeypatch, toy_root):
    monkeypatch.setattr(config, "INDEX", toy_root)


def test_bge_index_is_sane(toy_index):
    ids, emb = search.load_index("bge")
    assert len(ids) == len(index.load_chunks(CHUNKS, "canonical")) == 13  # one chunk is canonical=false
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
    assert ((emb @ emb.T).argmax(axis=1) == np.arange(len(ids))).all()


def test_rrf_matches_hand_computation():
    fused = dict(search.rrf([[("a", 9), ("b", 8)], [("b", 1), ("c", 0)]], k=60))
    assert fused == pytest.approx({"a": 1 / 61, "b": 1 / 62 + 1 / 61, "c": 1 / 62})
    assert [cid for cid, _ in search.rrf([[("a", 9), ("b", 8)], [("b", 1), ("c", 0)]])][0] == "b"


def test_r1_caps_chunks_per_work():
    chunks = search.load_chunks()
    ranked = [("w1-c0", 3), ("w1-c1", 2), ("w4-c0", 1.5), ("w1-c2", 1)]
    kept, folded = search.collapse(ranked, chunks, max_per_work=2)
    assert kept == ranked[:3]
    assert folded == {"w1-c0": [("more from this paper", "w1-c2")]}


def test_r2_folds_near_duplicate_from_another_work(toy_index):
    chunks = search.load_chunks()
    ids, emb = search.load_index("bge")
    lookup = dict(zip(ids, emb))
    ranked = [(cid, 0) for cid in ids]
    kept, folded = search.collapse(ranked, chunks, max_per_work=9, tau=0.9, emb_lookup=lookup)
    assert folded == {"w1-c1": [("also stated in", "w5-c1")]}
    assert len(kept) == len(ids) - 1


def test_r3_moves_second_survey_chunk_out_of_top5():
    chunks = search.load_chunks()
    types = {"w5": "survey"}
    ranked = [("w5-c0", 7), ("w1-c1", 6), ("w5-c2", 5), ("w2-c0", 4), ("w3-c0", 3), ("w4-c0", 2), ("w4-c1", 1)]
    kept, _ = search.collapse(ranked, chunks, paper_types=types)
    assert [c for c, _ in kept] == ["w5-c0", "w1-c1", "w2-c0", "w3-c0", "w4-c0", "w5-c2", "w4-c1"]
    kept, _ = search.collapse(ranked, chunks, paper_types=types, want_survey=True)
    assert kept == ranked


def test_hybrid_unites_dense_and_bm25(toy_index, monkeypatch):
    monkeypatch.setattr(config, "DENSE_CANDIDATES", 4)
    q = "tokenizer fertility"
    d = {c for c, _ in search.dense(q, "bge", 4)}
    b = {c for c, _ in search.bm25(q, 4)}
    hybrid = {c["chunk_id"] for c in search.retrieve(q, "hybrid-bge", k=20, collapse_mode="off")}
    assert hybrid == d | b
    assert d - b and b - d  # each list contributes a chunk the other missed


def test_retrieve_r1r2r3_uses_manifest_and_embeddings(toy_index):
    out = search.retrieve("byte pair encoding merges", "hybrid-bge", k=8, collapse_mode="r1r2r3")
    top = {c["chunk_id"]: c for c in out}
    assert "w5-c1" not in top and ("also stated in", "w5-c1") in top["w1-c1"]["folded"]
    assert sum(c["work_id"] == "w5" for c in out[:5]) <= 1
    assert all("score" in c for c in out)


def test_query_prefix_is_applied(toy_index):
    q = search.encode_query("x", "bge")
    doc = index.encode_texts(["x"], "bge")[0]
    assert float(q @ doc) < 0.99


def fake_embed(texts, model, provider=None):
    vecs = []
    for t in texts:
        v = np.random.default_rng(zlib.crc32(t.encode())).normal(size=4096)
        vecs.append((v / np.linalg.norm(v)).tolist())
    fake_embed.calls.append(list(texts))
    return vecs, 0.001


def test_qwen_index_resumes_and_query_gets_instruction(monkeypatch, tmp_path):
    fake_embed.calls = []
    monkeypatch.setattr(index.llm, "embed", fake_embed)
    monkeypatch.setattr(index, "QWEN_BATCH", 4)
    chunks = index.load_chunks(CHUNKS, "canonical")
    out = tmp_path / "canonical" / "qwen-or"
    out.mkdir(parents=True)
    done, _ = index.qwen_embed([c["embed_text"] for c in chunks[:4]])
    np.save(out / "partial.npy", done)
    (out / "partial_ids.txt").write_text("\n".join(c["chunk_id"] for c in chunks[:4]))
    fake_embed.calls = []
    meta = index.build(chunks, "qwen-or", out)
    sent = {t for call in fake_embed.calls for t in call}  # batches run concurrently, so check the set
    assert chunks[4]["embed_text"] in sent and chunks[3]["embed_text"] not in sent  # resumed after the 4 saved
    assert meta["dims"] == config.QWEN_DIMS and meta["n"] == 13 and meta["cost_usd"] == pytest.approx(0.003)
    emb = np.load(out / "emb.npy")
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
    assert ((emb @ emb.T).argmax(axis=1) == np.arange(13)).all()
    assert not (out / "partial.npy").exists()
    monkeypatch.setattr(config, "INDEX", tmp_path)
    assert search.dense("x", "qwen-or", 3)
    assert fake_embed.calls[-1] == [config.QWEN_QUERY_INSTRUCTION + "x"]
