"""Retrieve chunks: dense, BM25 or hybrid (reciprocal rank fusion), then collapse duplicates.

Decisions:
- Indexes and chunks load once and stay cached in module dicts, keyed by path. Paths are read from
  config at call time, so tests and evaluate.py can point config.INDEX / CHUNKS / MANIFEST elsewhere.
- Dense search is a dot product over unit vectors (exact cosine); at ~10^5 chunks that takes
  milliseconds, so there is no ANN index.
- BM25 drops zero-score results: a chunk with no query term is not a keyword match.
- Hybrid = RRF of the dense top-50 and the BM25 top-50. RRF uses ranks only, so the two score scales
  never need calibrating.
- Collapse runs at retrieval time on the ranked candidates, in one pass that preserves order:
  R1 at most MAX_PER_WORK chunks per work; R2 fold a chunk that restates a kept chunk from another
  work (cosine >= tau); R3 at most SURVEY_CAP survey chunks in the top 5 unless the user asks for a
  survey (extras move down, not out). Folded chunks stay attached to the kept one, for citations.
"""
import argparse
import json
from functools import cache
from pathlib import Path

import bm25s
import numpy as np

from rag import config, index, manifest

DEVICE = "auto"  # for query encoding; evaluate.py may set it
_CHUNKS, _INDEX = {}, {}


def load_chunks(path=None):
    path = Path(path or config.CHUNKS)
    if path not in _CHUNKS:
        with open(path) as f:
            _CHUNKS[path] = {c["chunk_id"]: c for c in map(json.loads, f)}
    return _CHUNKS[path]


def index_dir(model, variant, root=None):
    return Path(root or config.INDEX) / variant / model


def load_index(model, variant="canonical", root=None):
    """(ids, emb matrix) for a dense model, (ids, bm25s retriever) for bm25."""
    d = index_dir(model, variant, root)
    if d not in _INDEX:
        ids = (d / "ids.txt").read_text().splitlines()
        _INDEX[d] = (ids, bm25s.BM25.load(d) if model == "bm25" else np.load(d / "emb.npy"))
    return _INDEX[d]


@cache
def _lookup(d):
    ids, emb = _INDEX[d]
    return dict(zip(ids, emb))


@cache
def _paper_types(manifest_path):  # the path is the cache key; manifest.load() reads config.MANIFEST
    rows = manifest.load()
    return {r["work_id"]: r["paper_type"] for r in rows} or None


def encode_query(q, model):
    return index.encode_texts([index.QUERY_PREFIX[model] + q], model, index.pick_device(DEVICE))[0]


def dense(q, model, k=config.DENSE_CANDIDATES, variant="canonical"):
    ids, emb = load_index(model, variant)
    scores = emb @ encode_query(q, model)
    return [(ids[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


def bm25(q, k=config.DENSE_CANDIDATES, variant="canonical"):
    ids, retriever = load_index("bm25", variant)
    docs, scores = retriever.retrieve(index.bm25_tokens([q]), k=min(k, len(ids)), show_progress=False)
    return [(ids[d], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]


def rrf(lists, k=config.RRF_K):
    fused = {}
    for ranked in lists:
        for rank, (cid, _) in enumerate(ranked, 1):
            fused[cid] = fused.get(cid, 0.0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def cap_surveys(kept, chunks, paper_types, cap, top=5):
    """Move survey chunks beyond `cap` out of the top `top` positions, right after them."""
    head, moved, rest, n = [], [], [], 0
    for item in kept:
        survey = paper_types.get(chunks[item[0]]["work_id"]) == "survey"
        if len(head) >= top:
            rest.append(item)
        elif survey and n >= cap:
            moved.append(item)
        else:
            head.append(item)
            n += survey
    return head + moved + rest


def collapse(ranked, chunks, max_per_work=config.MAX_PER_WORK, tau=None, emb_lookup=None,
             survey_cap=config.SURVEY_CAP, paper_types=None, want_survey=False):
    """Return (kept [(chunk_id, score)], folded {kept_id: [(reason, chunk_id)]})."""
    kept, folded, by_work = [], {}, {}
    for cid, score in ranked:
        work = chunks[cid]["work_id"]
        same = by_work.setdefault(work, [])
        if len(same) >= max_per_work:
            folded.setdefault(same[0], []).append(("more from this paper", cid))
            continue
        if tau is not None and emb_lookup is not None:
            twin = next((k for k, _ in kept if chunks[k]["work_id"] != work
                         and float(emb_lookup[k] @ emb_lookup[cid]) >= tau), None)
            if twin:
                folded.setdefault(twin, []).append(("also stated in", cid))
                continue
        kept.append((cid, score))
        same.append(cid)
    if paper_types is not None and not want_survey:
        kept = cap_surveys(kept, chunks, paper_types, survey_cap)
    return kept, folded


def retrieve(q, retriever=config.RETRIEVER, k=config.TOP_K, variant="canonical", collapse_mode="r1",
             want_survey=False, max_per_work=None):
    """Top k chunk dicts (copies) with "score" and "folded" added."""
    chunks, n = load_chunks(), config.DENSE_CANDIDATES
    model = retriever.removeprefix("hybrid-")
    if retriever == "bm25":
        ranked = bm25(q, n, variant)
    elif retriever.startswith("hybrid-"):
        ranked = rrf([dense(q, model, n, variant), bm25(q, n, variant)])
    else:
        ranked = dense(q, model, n, variant)
    kept, folded = ranked, {}
    if collapse_mode != "off":
        tau = lookup = types = None
        if collapse_mode == "r1r2r3":
            d = index_dir("bge" if model == "bm25" else model, variant)
            if (d / "emb.npy").exists():
                load_index(d.name, variant)
                tau, lookup = config.COLLAPSE_TAU, _lookup(d)
            types = _paper_types(config.MANIFEST)
        kept, folded = collapse(ranked, chunks, max_per_work=max_per_work or config.MAX_PER_WORK, tau=tau,
                                emb_lookup=lookup, paper_types=types, want_survey=want_survey)
    return [dict(chunks[cid], score=s, folded=folded.get(cid, [])) for cid, s in kept[:k]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print the top 5 chunks for a query.")
    p.add_argument("query")
    p.add_argument("--retriever", default=config.RETRIEVER)
    p.add_argument("--collapse", default="r1", choices=["off", "r1", "r1r2r3"])
    a = p.parse_args()
    for i, c in enumerate(retrieve(a.query, a.retriever, 5, collapse_mode=a.collapse), 1):
        print(f"{i}. {c['score']:.4f}  {c['title'][:70]} | {c['section'][:40]} | {c['chunk_id']}")
        print(f"   {c['text'][:160]!r}  folded={len(c['folded'])}")
