"""Tune the abstention score gate on the dev split.

The gate abstains before any LLM call when the best bge cosine for the query is below tau. tau is the
highest value that wrongly abstains on no answerable dev question (the minimum best-cosine over the
answerable dev questions, rounded down) minus a safety margin of MARGIN, because four answerable dev
questions make the minimum fragile; so the gate can only help on unanswerable questions. Writes
results/abstention_dev.csv (one row per dev question with its best scores) and prints tau with the
dev confusion matrix. The chosen tau is then written into rag/config.py by hand.
Usage: UV_NO_SYNC=1 uv run python -m scripts.tune_gate
"""
import csv
import json
import math

from rag import config, search
from rag.evaluate import query_text

rows = []
for line in open(config.QUESTIONS):
    q = json.loads(line)
    if q["split"] != "dev":
        continue
    text = query_text(q)
    bm25 = search.bm25(text, 1)
    rows.append({"id": q["id"], "type": q["type"], "answerable": q["type"] != "unanswerable",
                 "best_bge_cosine": round(search.dense(text, "bge", 1)[0][1], 4),
                 "best_bm25": round(bm25[0][1], 2) if bm25 else 0.0, "question": q["question"][:80]})
answerable = [r for r in rows if r["answerable"]]
MARGIN = 0.02
tau = math.floor((min(r["best_bge_cosine"] for r in answerable) - MARGIN) * 1000) / 1000
gated = [r for r in rows if r["best_bge_cosine"] < tau]
config.RESULTS.mkdir(exist_ok=True)
with open(config.RESULTS / "abstention_dev.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["gated_at_tau"])
    w.writeheader()
    w.writerows({**r, "gated_at_tau": r in gated} for r in rows)
for r in rows:
    print(f"{r['id']} {r['type']:13} cosine={r['best_bge_cosine']:.4f} bm25={r['best_bm25']:6.2f}  {r['question']}")
print(f"tau = {tau}: unanswerable gated {sum(not r['answerable'] for r in gated)}/{len(rows) - len(answerable)}, "
      f"answerable wrongly gated {sum(r['answerable'] for r in gated)}/{len(answerable)}")
