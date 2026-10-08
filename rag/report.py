"""Print every results table as Markdown, so no number in WRITEUP.md is typed by hand.

Each table comes straight from a file under results/ (written by evaluate.py, dedup.py, judge.py,
collect.py or scripts/compare_parsers.py) or from data/manifest.csv (the corpus funnel). A table whose
file does not exist yet is reported as missing rather than invented. Two runs give identical output.
"""
import csv
from collections import Counter

from rag import config, manifest

TABLES = [  # (heading, file under results/)
    ("Parser comparison: pymupdf4llm PDF parse against the LaTeXML HTML parse", "parser_comparison.csv"),
    ("Duplicate merge rules swept against the labelled pairs", "dedup_rules.csv"),
    ("Retrieval: Recall@5 and MRR@10 with 95% bootstrap intervals", "retrieval.csv"),
    ("Duplicates: raw against canonical index, with and without collapsing", "dedup.csv"),
    ("Follow-up questions: raw, concatenated history, rewritten query", "followups.csv"),
    ("Abstention gate tuned on the dev split", "abstention_dev.csv"),
    ("Chat quality judged sentence by sentence", "chat_quality.csv"),
]
TEXTS = [("Classifier agreement with the 60 labelled papers", "relevance_agreement.txt"),
         ("Judge agreement with hand-labelled sentences", "judge_agreement.txt")]


def md_table(rows):
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def funnel():
    """Corpus counts from the manifest: candidates, rejections by reason, works, indexed works, chunks."""
    rows = [r for r in manifest.load() if r["source"] in ("arxiv", "acl")]  # collected documents, not copies
    s1 = Counter(r["reject_reason"] for r in rows if r["stage1_result"] != "include")
    s2 = Counter(r["reject_reason"] or "unlabelled" for r in rows if r["stage1_result"] == "include" and not manifest.is_true(r["in_scope"]))
    scope = [r for r in rows if manifest.is_true(r["in_scope"])]
    indexed = [r for r in scope if manifest.is_true(r["in_index"])]
    out = [{"stage": "candidates collected (arXiv + ACL discovery)", "count": len(rows)}]
    out += [{"stage": f"rejected by rules: {k}", "count": v} for k, v in s1.most_common()]
    out += [{"stage": f"rejected by the classifier: {k}", "count": v} for k, v in s2.most_common()]
    out += [{"stage": "in scope (documents)", "count": len(scope)},
            {"stage": "outside the corpus cap", "count": sum(r["not_indexed_reason"] == "corpus_cap" for r in scope)},
            {"stage": "works (after merging versions and copies)", "count": len({r["work_id"] for r in scope})},
            {"stage": "works indexed (canonical document with full text)", "count": len(indexed)},
            {"stage": "chunks in the canonical index", "count": sum(int(r["n_chunks"] or 0) for r in indexed)}]
    return out


def main():
    print("## Corpus funnel\n")
    print(md_table(funnel()))
    for heading, name in TEXTS:
        path = config.RESULTS / name
        print(f"\n## {heading}\n")
        print("```\n" + path.read_text().strip() + "\n```" if path.exists() else f"(missing: results/{name})")
    for heading, name in TABLES:
        path = config.RESULTS / name
        print(f"\n## {heading}\n")
        if not path.exists():
            print(f"(missing: results/{name})")
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(md_table(rows) if rows else "(empty)")


if __name__ == "__main__":
    main()
