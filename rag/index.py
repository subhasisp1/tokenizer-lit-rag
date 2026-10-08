"""Build one retrieval index over the chunks: dense (bge, scincl, qwen-or) or keyword (bm25).

Decisions:
- One folder per variant and model: <out-root>/<variant>/<model>/ with ids.txt (row order) and meta.json.
  "canonical" keeps only canonical chunks (one version per work); "raw" keeps every version, so the
  duplicate problem can be measured.
- Documents are embedded from `embed_text` with no prefix. Queries get the model's own prefix
  (QUERY_PREFIX, applied in search.py): bge's retrieval instruction, qwen's task instruction, nothing
  for scincl, which was trained without one.
- Every vector is unit length, so cosine similarity is a plain dot product.
- qwen-or vectors are cut to config.QWEN_DIMS (Matryoshka truncation) and renormalised: 4x less memory
  for little loss. The hosted run sends QWEN_WORKERS batches at a time, saves partial progress every 20 batches and resumes from it; cost_usd
  covers this run only (logs/llm_calls.jsonl has every call).
- bm25 uses bm25s defaults, English stopwords and a Snowball stemmer, over the same `embed_text`.
- --limit N keeps the first N chunks, for quick checks.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from rag import config, llm, manifest

QUERY_PREFIX = {"bge": config.BGE_QUERY_PREFIX, "scincl": "", "qwen-or": config.QWEN_QUERY_INSTRUCTION}
QWEN_BATCH = 32
QWEN_WORKERS = 8  # concurrent requests; one batch takes 5-18 s at the provider
STEMMER = Stemmer.Stemmer("english")


def pick_device(device):
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def load_model(model, device="cpu"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(config.EMBED_MODELS[model], device=device)
    m.max_seq_length = 512
    return m


def qwen_embed(texts):
    """Return (unit vectors cut to QWEN_DIMS, cost_usd)."""
    vectors, cost = llm.embed(texts, config.EMBED_MODELS["qwen-or"], provider=config.QWEN_PROVIDER)
    v = np.asarray(vectors, dtype=np.float32)[:, :config.QWEN_DIMS]
    return v / np.linalg.norm(v, axis=1, keepdims=True), cost


def encode_texts(texts, model, device="cpu", batch_size=32):
    """Unit float32 vectors, one row per text. Callers add any query prefix."""
    if model == "qwen-or":
        return qwen_embed(texts)[0]
    emb = load_model(model, device).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                           show_progress_bar=len(texts) > 1000)
    return np.asarray(emb, dtype=np.float32)


def bm25_tokens(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)


def load_chunks(path, variant):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [c for c in rows if variant == "raw" or manifest.is_true(c["canonical"])]


def embed_qwen(ids, texts, out):
    """Embed in batches of QWEN_BATCH, saving partial.npy every 20 batches and resuming from it."""
    parts, cost, start = [], 0.0, 0
    if (out / "partial.npy").exists():
        done = (out / "partial_ids.txt").read_text().splitlines()
        if done == ids[:len(done)]:
            parts, start = [np.load(out / "partial.npy")], len(done)
            print(f"resuming after {start} chunks")
    batches = [texts[i:i + QWEN_BATCH] for i in range(start, len(texts), QWEN_BATCH)]
    with ThreadPoolExecutor(QWEN_WORKERS) as pool:
        for b, (v, c) in enumerate(pool.map(qwen_embed, batches), 1):  # map keeps the batch order
            parts.append(v)
            cost += c
            if b % 20 == 0:
                np.save(out / "partial.npy", np.vstack(parts))
                (out / "partial_ids.txt").write_text("\n".join(ids[:start + b * QWEN_BATCH]))
    for name in ("partial.npy", "partial_ids.txt"):
        (out / name).unlink(missing_ok=True)
    return np.vstack(parts), cost


def build(chunks, model, out, device="cpu"):
    """Write the index for `chunks` to `out` and return its meta dict."""
    out.mkdir(parents=True, exist_ok=True)
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["embed_text"] for c in chunks]
    t0, cost, dims, batch = time.time(), 0.0, None, None
    if model == "bm25":
        retriever = bm25s.BM25()
        retriever.index(bm25_tokens(texts), show_progress=False)
        retriever.save(out)
    else:
        if model == "qwen-or":
            emb, cost = embed_qwen(ids, texts, out)
            batch = QWEN_BATCH
        else:
            batch = 64 if device == "cuda" else 32
            emb = encode_texts(texts, model, device, batch)
        np.save(out / "emb.npy", emb)
        dims = emb.shape[1]
    (out / "ids.txt").write_text("\n".join(ids))
    meta = {"model": model, "model_name": config.EMBED_MODELS.get(model, "bm25s"), "variant": out.parent.name,
            "n": len(ids), "dims": dims, "seconds": round(time.time() - t0, 2), "device": device,
            "cost_usd": round(cost, 6), "batch_size": batch, "query_prefix_or_instruction": QUERY_PREFIX.get(model, "")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=["bge", "scincl", "qwen-or", "bm25"])
    p.add_argument("--variant", default="canonical", choices=["canonical", "raw"])
    p.add_argument("--chunks", default=config.CHUNKS)
    p.add_argument("--out-root", default=config.INDEX)
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    chunks = load_chunks(a.chunks, a.variant)[:a.limit]
    if not chunks:
        raise SystemExit(f"no {a.variant} chunks in {a.chunks}: run rag.download, rag.parse and rag.chunk first")
    device = {"bm25": "cpu", "qwen-or": "openrouter"}.get(a.model) or pick_device(a.device)
    meta = build(chunks, a.model, Path(a.out_root) / a.variant / a.model, device)
    print(f"{a.model}/{a.variant}: n={meta['n']} dims={meta['dims']} seconds={meta['seconds']} "
          f"device={meta['device']} cost_usd={meta['cost_usd']}")


if __name__ == "__main__":
    main()
