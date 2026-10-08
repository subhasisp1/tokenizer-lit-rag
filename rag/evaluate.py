"""Score retrievers against the labelled questions, with bootstrap confidence intervals.

Decisions:
- The query is `standalone_rewrite` when present (follow-ups are scored on their rewrite, so this
  measures retrieval, not query rewriting), else `question`. Unanswerable questions are skipped: they
  have no gold passages to find.
- A gold item is hit when a chunk comes from the gold work and fuzz.partial_ratio(quote, text) >= 85.
  Matching on the work stops a survey that restates the quote from counting; the fuzzy match survives
  PDF-vs-HTML differences in whitespace, hyphens and ligatures.
- Metrics per question on the collapsed top 10: recall@5 (share of gold items hit in the top 5),
  mrr@10 (1 / rank of the first hitting chunk), duprate@5 (top-5 slots whose work already appeared
  higher), distinct_works@5. Latency is the wall time of one retrieve() call, after one warm-up query
  that loads the model and index.
- 95% CIs by bootstrap over questions (numpy Generator seeded with config.SEED). The same seed gives
  the same resamples for every retriever, so the paired difference to the best row (highest recall@5)
  is resampled per question, as the comparison needs.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from rag import config, search

DEPTH = 10
ALL = "bm25,bge,scincl,qwen-or,hybrid-bge,hybrid-scincl,hybrid-qwen-or"


def load_questions(path, split):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [q for q in rows if q["type"] != "unanswerable" and q["gold"] and split in ("all", q["split"])]


def query_text(q):
    return q.get("standalone_rewrite") or q["question"]


def is_hit(gold, chunk):
    return chunk["work_id"] == gold["work_id"] and fuzz.partial_ratio(gold["quote"], chunk["text"]) >= 85


def score(gold, ranked):
    top5 = ranked[:5]
    recall = sum(any(is_hit(g, c) for c in top5) for g in gold) / len(gold)
    first = next((i for i, c in enumerate(ranked[:DEPTH], 1) if any(is_hit(g, c) for g in gold)), 0)
    works = [c["work_id"] for c in top5]
    dup = sum(w in works[:i] for i, w in enumerate(works)) / len(works) if works else 0.0
    return {"recall@5": recall, "mrr@10": 1 / first if first else 0.0, "first_hit_rank": first or "",
            "duprate@5": dup, "distinct_works@5": len(set(works))}


def run(questions, retriever, variant, collapse, max_per_work=None):
    """One row per question: metrics, latency and the top-5 chunk ids."""
    search.retrieve(query_text(questions[0]), retriever, DEPTH, variant, collapse)  # warm-up
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        ranked = search.retrieve(query_text(q), retriever, DEPTH, variant, collapse, max_per_work=max_per_work)
        rows.append({"id": q["id"], "retriever": retriever, **score(q["gold"], ranked),
                     "latency_s": round(time.perf_counter() - t0, 4),
                     "top5": "|".join(c["chunk_id"] for c in ranked[:5])})
    return rows


def bootstrap_ci(values, b):
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(config.SEED).integers(0, len(v), (b, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def index_meta(retriever, variant):
    path = Path(config.INDEX) / variant / retriever.removeprefix("hybrid-") / "meta.json"
    if retriever == "bm25" or not path.exists():
        return {}
    return json.loads(path.read_text())


def summarize(per_question, variant, collapse, b):
    """One row per retriever; per_question maps retriever -> rows from run(), same question order."""
    col = {name: {m: np.array([r[m] for r in rows], dtype=float) for m in ("recall@5", "mrr@10")}
           for name, rows in per_question.items()}
    best = max(col, key=lambda n: col[n]["recall@5"].mean())
    out = []
    for name, rows in per_question.items():
        rec, mrr = col[name]["recall@5"], col[name]["mrr@10"]
        diff = rec - col[best]["recall@5"]
        lat = [r["latency_s"] for r in rows]
        meta = index_meta(name, variant)
        out.append({"retriever": name, "variant": variant, "collapse": collapse, "n": len(rows),
                    "recall@5": rec.mean(), **dict(zip(("recall@5_lo", "recall@5_hi"), bootstrap_ci(rec, b))),
                    "mrr@10": mrr.mean(), **dict(zip(("mrr@10_lo", "mrr@10_hi"), bootstrap_ci(mrr, b))),
                    "duprate@5": np.mean([r["duprate@5"] for r in rows]),
                    "distinct_works@5": np.mean([r["distinct_works@5"] for r in rows]),
                    "p50_s": np.percentile(lat, 50), "p95_s": np.percentile(lat, 95),
                    "diff_vs_best": diff.mean(), **dict(zip(("diff_lo", "diff_hi"), bootstrap_ci(diff, b))),
                    "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                    "index_seconds": meta.get("seconds", ""), "index_device": meta.get("device", ""),
                    "index_cost_usd": meta.get("cost_usd", "")})
    return [{k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in r.items()} for r in out]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--questions", default=config.QUESTIONS)
    p.add_argument("--chunks", default=config.CHUNKS)
    p.add_argument("--index-root", default=config.INDEX)
    p.add_argument("--split", default="test", choices=["test", "dev", "all"])
    p.add_argument("--retrievers", default=ALL)
    p.add_argument("--variant", default="canonical", choices=["canonical", "raw"])
    p.add_argument("--collapse", default="r1", choices=["off", "r1", "r1r2r3"])
    p.add_argument("--tau", type=float, default=config.COLLAPSE_TAU)
    p.add_argument("--max-per-work", type=int, default=config.MAX_PER_WORK, help="R1 cap on passages per work")
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--out", default=config.RESULTS / "retrieval.csv")
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    config.CHUNKS, config.INDEX, config.COLLAPSE_TAU = Path(a.chunks), Path(a.index_root), a.tau
    search.DEVICE = a.device
    questions = load_questions(a.questions, a.split)
    print(f"{len(questions)} answerable questions ({a.split})")
    per_question = {r: run(questions, r, a.variant, a.collapse, a.max_per_work) for r in a.retrievers.split(",")}
    table = summarize(per_question, a.variant, a.collapse, a.bootstrap)
    out = Path(a.out)
    write_csv(out, table)
    write_csv(out.with_name(out.stem + "_per_question.csv"), [r for rows in per_question.values() for r in rows])
    cols = ["retriever", "n", "recall@5", "recall@5_lo", "recall@5_hi", "mrr@10", "duprate@5",
            "distinct_works@5", "p50_s", "diff_vs_best", "diff_lo", "diff_hi", "wins", "losses"]
    print("  ".join(f"{c:>14}" for c in cols))
    for r in table:
        print("  ".join(f"{r[c]!s:>14}" for c in cols))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
