"""Score the five follow-up questions under four query forms, to show what the rewrite buys.

Forms: raw (the follow-up as typed, pronouns and all), concat (parent question + follow-up), rewrite (the
chatbot's own standalone rewrite recorded in results/answers.jsonl) and standalone (the gold rewrite from
the question file). Recall@5 and MRR@10 per question and per form -> results/followups.csv.
Usage: UV_NO_SYNC=1 uv run python -m scripts.followups
"""
import csv
import json

from rag import config, search
from rag.evaluate import score

qs = {q["id"]: q for q in map(json.loads, open(config.QUESTIONS))}
answers = {a["id"]: a for a in map(json.loads, open(config.RESULTS / "answers.jsonl"))}
rows = []
for q in qs.values():
    if q["type"] != "followup":
        continue
    parent = qs[q["parent_id"]]
    forms = {"raw": q["question"], "concat": parent["question"] + " " + q["question"],
             "rewrite": answers[q["id"]]["rewrite"], "standalone": q["standalone_rewrite"]}
    for form, text in forms.items():
        s = score(q["gold"], search.retrieve(text, config.RETRIEVER, 10, collapse_mode="r1"))
        rows.append({"id": q["id"], "form": form, "recall@5": s["recall@5"], "mrr@10": s["mrr@10"], "query": text})
for form in ("raw", "concat", "rewrite", "standalone"):
    part = [r for r in rows if r["form"] == form]
    rows.append({"id": "mean", "form": form, "recall@5": round(sum(r["recall@5"] for r in part) / len(part), 3),
                 "mrr@10": round(sum(r["mrr@10"] for r in part) / len(part), 3), "query": ""})
with open(config.RESULTS / "followups.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['id']:5} {r['form']:10} recall@5={r['recall@5']:.2f} mrr@10={r['mrr@10']:.2f}  {r['query'][:90]}")
