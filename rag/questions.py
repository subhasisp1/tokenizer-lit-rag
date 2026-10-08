"""Build the evaluation set: propose candidate questions, pool and pre-grade extra gold, report statistics.

Decisions:
- propose: one call per sampled work (temperature 0.7 for variety) for a lookup and a conceptual
  question with a verbatim quote. Input is paragraphs only (parsed tables do not quote verbatim), cut
  at about 3,000 words. Multi-paper questions come from themes over the top 10 r1-collapsed chunks and
  are kept only if >= 2 works keep a verified quote. The author selects and edits; nothing is final.
- A quote counts when fuzz.partial_ratio >= 90 against the named section, else the best section (gold
  stores the verified one); otherwise it is dropped, since a quote not in the corpus can never be hit.
- lexical_overlap (token Jaccard, question vs quotes) flags questions that copy their quote: the eval
  must not be a keyword-match test. stats scores the retrieval query (standalone_rewrite if present).
- pool: union of each retriever's top-depth chunks, no collapse (it pools chunks, not works), graded
  0-3 by the judge. Gold hits are written too (grade null, no call) so stats can measure gold coverage.
  Rows are appended per (question_id, chunk_id), so a rerun only grades new pairs. Grades >= 2 are
  candidate gold for the author to confirm.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from rag import config, llm, manifest, search
from rag.evaluate import is_hit, query_text

WORD_BUDGET = 3000
MIN_QUOTE_SCORE = 90
STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "by", "with",
        "as", "it", "this", "that", "what", "which", "how", "why", "does", "do"}
THEMES = [
    "BPE versus unigram language model tokenization", "effect of vocabulary size on model quality",
    "byte-level or character-level models versus subword tokenization",
    "tokenization fertility and fairness across languages", "tokenization and arithmetic or numerical reasoning",
    "transferring or extending a tokenizer to new languages",
    "compression rate of tokenizers and downstream performance", "glitch tokens and undertrained tokens",
    "morphological alignment of subword segmentation", "theoretical analysis of tokenization",
]


def obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


STR = {"type": "string"}
PAPER_SCHEMA = obj({"questions": {"type": "array", "items": obj({
    "question": STR, "type": {"type": "string", "enum": ["lookup", "conceptual"]},
    "answer": STR, "quote": STR, "section": STR})}})
THEME_SCHEMA = obj({"question": STR, "answer": STR,
                    "supports": {"type": "array", "items": obj({"n": {"type": "integer"}, "quote": STR})}})
GRADE_SCHEMA = obj({"grade": {"type": "string", "enum": ["0", "1", "2", "3"]}, "reason": STR})  # string enum: every provider

PAPER_PROMPT = """Write exactly 2 evaluation questions about the research paper below:
1. type "lookup": asks for a specific number, setting, dataset or named result.
2. type "conceptual": asks about a mechanism, definition or reason.
Each must be answerable from this paper alone. Give a short answer, `quote` = one sentence copied
verbatim from the text that supports the answer, and `section` = the section path (after "## ") it
comes from. Word the question in your own words: do not copy phrases from the quote.
The text is cut at about {budget} words; use only what is shown."""
THEME_PROMPT = """Below are numbered passages from several papers on: {theme}.
Write one question that can only be answered by combining at least 2 of the papers, and a short answer.
In `supports`, give one entry per passage the answer needs: its number `n` and one sentence copied
verbatim from that passage. Word the question in your own words: do not copy phrases from the quotes."""
GRADE_PROMPT = """Grade how well the passage answers the question.
0 = unrelated, 1 = related background, 2 = partially answers, 3 = fully answers the question."""


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def overlap(question, quotes):
    a, b = tokens(question), set().union(*map(tokens, quotes))
    return round(len(a & b) / len(a | b), 4) if a | b else 0.0


def section_texts(doc):
    texts = {"Abstract": doc.get("abstract", "")}
    texts.update({s["path"]: " ".join(s["paragraphs"]) for s in doc["sections"]})
    return texts


def verify(quote, texts, section):
    """(section, score): the named section if the quote is in it, else the best-scoring one."""
    if section in texts and (score := fuzz.partial_ratio(quote, texts[section])) >= MIN_QUOTE_SCORE:
        return section, score
    return max(((s, fuzz.partial_ratio(quote, t)) for s, t in texts.items()), key=lambda x: x[1])


def paper_text(doc):
    parts, words = [f"Title: {doc['title']}\n\n## Abstract\n{doc.get('abstract', '')}"], 0
    for s in doc["sections"]:
        body = " ".join(s["paragraphs"]).split()
        if not body:
            continue
        parts.append(f"## {s['path']}\n" + " ".join(body[:WORD_BUDGET - words]))
        words += len(body)
        if words >= WORD_BUDGET:
            break
    return "\n\n".join(parts)


def sample_docs(n, seed):
    rows = [r for r in manifest.load() if manifest.is_true(r["in_index"]) and manifest.is_true(r["canonical"])]
    if rows:
        ids = [f"arxiv:{r['arxiv_id']}v{r['arxiv_latest_version']}" if r["arxiv_id"] else r["record_id"]
               for r in rows]
        paths = [config.parsed_path(i) for i in sorted(ids)]
    else:
        paths = sorted(config.PARSED.glob("*.json"))
    paths = [p for p in paths if p.exists()]
    return [json.loads(p.read_text()) for p in random.Random(seed).sample(paths, min(n, len(paths)))]


def propose_paper(doc, model):
    """(candidates, dropped, cost) for one work."""
    messages = [{"role": "system", "content": PAPER_PROMPT.format(budget=WORD_BUDGET)},
                {"role": "user", "content": paper_text(doc)}]
    text, cost = llm.chat(messages, model, PAPER_SCHEMA, temperature=0.7, max_tokens=900)
    texts, cands, dropped = section_texts(doc), [], 0
    for i, q in enumerate(llm.parse_json(text)["questions"]):
        section, score = verify(q["quote"], texts, q["section"])
        if score < MIN_QUOTE_SCORE:
            dropped += 1
            print(f"  drop {doc['work_id']}: quote not found (best {score:.0f} in {section!r})")
            continue
        if section != q["section"]:
            print(f"  {doc['work_id']}: quote found in {section!r}, not {q['section']!r}")
        cands.append({"cand_id": f"{doc['work_id']}-{i}", "type": q["type"], "question": q["question"],
                      "answer": q["answer"], "gold": [{"work_id": doc["work_id"], "section": section,
                                                       "quote": q["quote"]}],
                      "titles": [doc["title"]], "lexical_overlap": overlap(q["question"], [q["quote"]])})
    return cands, dropped, cost


def propose_theme(j, theme, model):
    """(candidate or None, cost) for one theme."""
    chunks = search.retrieve(theme, k=10, collapse_mode="r1")
    shown = "\n\n".join(f"[{n}] {c['title']} | {c['section']}\n{c['text']}" for n, c in enumerate(chunks, 1))
    messages = [{"role": "system", "content": THEME_PROMPT.format(theme=theme)}, {"role": "user", "content": shown}]
    text, cost = llm.chat(messages, model, THEME_SCHEMA, temperature=0.7, max_tokens=900)
    out = llm.parse_json(text)
    gold, titles = [], []
    for s in out["supports"]:
        c = chunks[s["n"] - 1] if 1 <= s["n"] <= len(chunks) else None
        if c and fuzz.partial_ratio(s["quote"], c["text"]) >= MIN_QUOTE_SCORE:
            gold.append({"work_id": c["work_id"], "section": c["section"], "quote": s["quote"]})
            titles += [c["title"]] if c["title"] not in titles else []
    if len({g["work_id"] for g in gold}) < 2:
        print(f"  drop theme {theme!r}: fewer than 2 works with a verified quote")
        return None, cost
    return {"cand_id": f"theme{j}", "type": "multi", "question": out["question"], "answer": out["answer"],
            "gold": gold, "titles": titles,
            "lexical_overlap": overlap(out["question"], [g["quote"] for g in gold])}, cost


def propose(n_works, n_themes, out, seed, model):
    cands, dropped, total = [], 0, 0.0
    for doc in sample_docs(n_works, seed):
        c, d, cost = propose_paper(doc, model)
        cands, dropped, total = cands + c, dropped + d, total + cost
    for j, theme in enumerate(THEMES[:n_themes]):
        c, cost = propose_theme(j, theme, model)
        cands, dropped, total = cands + ([c] if c else []), dropped + (c is None), total + cost
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(c) + "\n" for c in cands))
    med = statistics.median(c["lexical_overlap"] for c in cands) if cands else 0.0
    print(f"candidates: {dict(Counter(c['type'] for c in cands))}  dropped unverifiable: {dropped}")
    print(f"median lexical overlap: {med:.3f}  cost: ${total:.4f}  wrote {out}")
    return cands, dropped


def grade(q, c, model):
    user = (f"Question: {query_text(q)}\nExpected answer: {q.get('expected_answer', '')}\n\n"
            f"Passage ({c['title']} | {c['section']}):\n{c['text']}")
    text, cost = llm.chat([{"role": "system", "content": GRADE_PROMPT}, {"role": "user", "content": user}],
                          model, GRADE_SCHEMA, temperature=0.0)
    out = llm.parse_json(text)
    return {"grade": int(out["grade"]), "reason": out["reason"]}, cost


def pool(questions_path, retrievers, depth, out, model):
    usable = [r for r in retrievers if (config.INDEX / "canonical" / r.removeprefix("hybrid-")).exists()]
    for r in sorted(set(retrievers) - set(usable)):
        print(f"skip {r}: no index at {config.INDEX / 'canonical' / r.removeprefix('hybrid-')}")
    rows = read_jsonl(out) if Path(out).exists() else []
    done, total = {(r["question_id"], r["chunk_id"]) for r in rows}, 0.0
    with open(out, "a") as f:
        for q in (q for q in read_jsonl(questions_path) if q["type"] != "unanswerable"):
            pooled = {c["chunk_id"]: c for r in usable
                      for c in search.retrieve(query_text(q), r, k=depth, collapse_mode="off")}
            todo = [c for cid, c in pooled.items() if (q["id"], cid) not in done]
            gold_hit = [any(is_hit(g, c) for g in q["gold"]) for c in todo]
            with ThreadPoolExecutor(8) as ex:  # one judge call per non-gold chunk, 8 at a time
                graded = list(ex.map(lambda c: grade(q, c, model), [c for c, h in zip(todo, gold_hit) if not h]))
            graded.reverse()
            for c, hit in zip(todo, gold_hit):
                if hit:
                    g = {"grade": None, "reason": "gold"}
                else:
                    g, cost = graded.pop()
                    total += cost
                row = {"question_id": q["id"], "chunk_id": c["chunk_id"], "work_id": c["work_id"], "section": c["section"],
                       "grade": g["grade"], "reason": g["reason"], "quote": c["text"][:300]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
    for qid, n in Counter(r["question_id"] for r in rows if (r["grade"] or 0) >= 2).items():
        print(f"{qid}: {n} chunks graded >= 2")
    print(f"cost: ${total:.4f}  wrote {out}")
    return rows


def stats(questions_path, pool_path, out):
    qs = read_jsonl(questions_path)
    ids = {q["id"] for q in qs}
    splits = sorted({q["split"] for q in qs})
    lines = ["# Evaluation set statistics", "", f"{len(qs)} questions.", "",
             "| type | " + " | ".join(splits) + " | total |", "|---" * (len(splits) + 2) + "|"]
    for t, n in sorted(Counter(q["type"] for q in qs).items()):
        lines.append(f"| {t} | " + " | ".join(str(sum(q["type"] == t and q["split"] == s for q in qs))
                                              for s in splits) + f" | {n} |")
    lines += ["", "## Follow-ups", ""]
    for q in (q for q in qs if q["type"] == "followup"):
        lines.append(f"- {q['id']}: parent {'yes' if q.get('parent_id') in ids else 'MISSING'}, "
                     f"standalone_rewrite {'yes' if q.get('standalone_rewrite') else 'MISSING'}")
    unans = [q for q in qs if q["type"] == "unanswerable"]
    lines += ["", f"## Unanswerable: {len(unans)}", ""] + [f"- {q['id']}: {q.get('why_unanswerable', '')}"
                                                           for q in unans]
    works = [len({g["work_id"] for g in q["gold"]}) for q in qs if q["type"] == "multi"]
    if works:
        lines += ["", f"## Gold works per multi question: min {min(works)}, median "
                      f"{statistics.median(works)}, max {max(works)}"]
    scored = {q["id"]: overlap(query_text(q), [g["quote"] for g in q["gold"]]) for q in qs if q["gold"]}
    high = [i for i, v in scored.items() if v > 0.30]
    if scored:
        lines += ["", "## Lexical overlap (query vs gold quotes)", "",
                  f"median {statistics.median(scored.values()):.3f}; above 0.30: {len(high)}/{len(scored)} "
                  f"({len(high) / len(scored):.0%}): {', '.join(high) or 'none'}"]
    if Path(pool_path).exists():
        rows, chunks = read_jsonl(pool_path), search.load_chunks()
        lines += ["", "## Pooling completeness", "", "| question | gold items in pool | candidate gold to confirm |",
                  "|---|---|---|"]
        for q in (q for q in qs if q["gold"]):
            pooled = [chunks[r["chunk_id"]] for r in rows if r["question_id"] == q["id"] and r["chunk_id"] in chunks]
            found = sum(any(is_hit(g, c) for c in pooled) for g in q["gold"])
            extra = sum(r["question_id"] == q["id"] and (r["grade"] or 0) >= 2 for r in rows)
            lines.append(f"| {q['id']} | {found}/{len(q['gold'])} | {extra} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return scored, high


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("propose")
    a.add_argument("--n-works", type=int, default=40)
    a.add_argument("--themes", type=int, default=len(THEMES))
    a.add_argument("--out", default=config.EVAL / "candidates.jsonl")
    a.add_argument("--seed", type=int, default=config.SEED)
    a.add_argument("--model", default=config.ANSWER_MODEL)
    b = sub.add_parser("pool")
    b.add_argument("--questions", default=config.QUESTIONS)
    b.add_argument("--retrievers", default="bm25,bge,qwen-or")
    b.add_argument("--depth", type=int, default=20)
    b.add_argument("--out", default=config.EVAL / "pool_judgments.jsonl")
    b.add_argument("--model", default=config.JUDGE_MODEL)
    c = sub.add_parser("stats")
    c.add_argument("--questions", default=config.QUESTIONS)
    c.add_argument("--pool", default=config.EVAL / "pool_judgments.jsonl")
    c.add_argument("--out", default=config.RESULTS / "eval_set_stats.md")
    x = p.parse_args()
    if x.cmd == "propose":
        propose(x.n_works, x.themes, x.out, x.seed, x.model)
    elif x.cmd == "pool":
        pool(x.questions, x.retrievers.split(","), x.depth, x.out, x.model)
    else:
        stats(x.questions, x.pool, x.out)


if __name__ == "__main__":
    main()
