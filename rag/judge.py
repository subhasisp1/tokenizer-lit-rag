"""Answer the frozen question set, judge every answer sentence by sentence, and score the chatbot.

Subcommands: run, judge, metrics, label-sheet, agreement.

Decisions:
- Records are keyed by (id, model, retriever), so one answers file can hold several configurations
  and metrics writes one row per configuration. run and judge append and skip keys already done, so
  an interrupted run resumes. An unjudged answer is retried by the next judge run (the last line for
  a key wins).
- A follow-up gets its parent's answer from the same configuration as history. A parent in the
  split is answered and recorded first; a parent outside the split is answered but not recorded
  (its cost shows only in logs/llm_calls.jsonl).
- The judge (config.JUDGE_MODEL, another model family than the answerer) sees only the sources shown
  to the answerer and the expected answer, and replies in strict JSON. Invalid JSON is retried once,
  then the answer is recorded as unjudged. Abstentions are not sent to the judge.
- Metrics count factual sentences of judged, non-abstained answers.
  fully_grounded_rate = answers with >= 1 factual sentence, all "supported" / n (all answers).
  citation_precision = correct citations / citations on factual sentences; a citation [m] is correct
  when its sentence is "supported" and m == the judge's support_source. The judge names one source,
  so a second valid citation on a sentence counts as wrong: this is a lower bound.
  correctness shares are over answerable, non-abstained, judged answers. Abstention is scored as a
  classifier whose positive class is "unanswerable", so tp + fp + fn + tn == n.
  cost_per_answer is the answering cost only; judge prints its own cost.
- Agreement: human rows pair with judge sentences by order when the counts match, else each human
  sentence takes the judge sentence with the best fuzz.ratio >= 80. The judge's verdict is used even
  when it calls the sentence non-factual. Rows labelled "-" (non-factual) or left empty are skipped.
"""
import argparse
import csv
import datetime
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

from rag import config, llm

SENTENCE = {"type": "object", "additionalProperties": False,
            "required": ["text", "factual", "cited", "verdict", "support_source", "note"],
            "properties": {"text": {"type": "string"}, "factual": {"type": "boolean"},
                           "cited": {"type": "array", "items": {"type": "integer"}},
                           "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                           "support_source": {"type": "integer"}, "note": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sentences", "correctness", "reasoning"],
          "properties": {"sentences": {"type": "array", "items": SENTENCE},
                         "correctness": {"type": "string", "enum": ["correct", "partial", "wrong", "not_applicable"]},
                         "reasoning": {"type": "string"}}}
SYSTEM = """You check answers from a research chatbot against the numbered sources it was shown.
You see only those sources and the expected answer.
Split the answer into sentences (a sentence ends at . ! or ? followed by whitespace) and copy each one as written. For each sentence:
- factual: true when it states something checkable; false for questions and meta-talk such as "the sources show".
- cited: the numbers in square brackets in the sentence.
- verdict: "supported" only when a shown source states it, cited or not; "partial" when only part of the sentence is supported; "unsupported" otherwise. A claim that may be true but is not in the sources is unsupported.
- support_source: the number of the source that supports the sentence, 0 when none.
- note: one short reason.
correctness compares the answer as a whole with the expected answer: "correct", "partial" or "wrong"; "not_applicable" when the question is unanswerable.
reasoning: one or two sentences on correctness."""
FAILURE_FIELDS = ["model", "retriever", "id", "category", "evidence", "question"]
LABEL_FIELDS = ["answer_id", "model", "retriever", "sentence_no", "sentence", "cited", "sources_shown", "human_verdict"]


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def key(r):
    return r["id"], r["model"], r["retriever"]


def body(answer):
    """The answer text without its rendered Sources list."""
    return answer.split("\n\nSources:")[0]


def run(questions, split, retriever, k, model, out):
    from rag import chat  # imported here so this module loads and tests run without rag/chat.py
    questions = read_jsonl(questions)
    by_id = {q["id"]: q for q in questions}
    texts = {r["id"]: r["answer"] for r in read_jsonl(out) if (r["model"], r["retriever"]) == (model, retriever)}
    new = []

    def answer(q, record):
        history = []
        if q["type"] == "followup":
            parent = by_id[q["parent_id"]]
            if parent["id"] not in texts:
                answer(parent, split in ("all", parent["split"]))
            history = [{"role": "user", "content": parent["question"]},
                       {"role": "assistant", "content": texts[parent["id"]]}]
        r = chat.answer(history, q["question"], retriever=retriever, k=k, model=model)
        texts[q["id"]] = r["text"]
        if record:
            new.append({"id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
                        "rewrite": r["rewrite"], "answer": r["text"], "raw": r["raw"], "sources": r["sources"],
                        "cited": r["cited"], "invalid_citations": r["invalid_citations"],
                        "uncited_sentences": r["uncited_sentences"], "abstained": r["abstained"],
                        "gate_abstained": r["gate_abstained"], "expected_answer": q["expected_answer"],
                        "gold": q["gold"], "why_unanswerable": q["why_unanswerable"], "cost_usd": r["cost_usd"],
                        "latency_s": round(sum(r["timings"].values()), 3), "model": model, "retriever": retriever})
            append_jsonl(out, new[-1])

    for q in questions:
        if split in ("all", q["split"]) and q["id"] not in texts:
            answer(q, True)
    latency = [r["latency_s"] for r in new]
    print(f"answered {len(new)}; cost ${sum(r['cost_usd'] for r in new):.4f}; "
          f"p50 latency {statistics.median(latency) if latency else 0:.2f}s; "
          f"abstentions {sum(r['abstained'] for r in new)}")
    return new


def judge_one(a, model):
    sources = "\n\n".join(f"[{s['n']}] {s['title']} ({s['year']}, {s['venue']}), § {s['section']}\n{s['text']}"
                          for s in a["sources"])
    expected = f"unanswerable: {a['why_unanswerable']}" if a["type"] == "unanswerable" else a["expected_answer"]
    user = (f"Sources:\n{sources}\n\nQuestion: {a['question']}\n\nExpected answer: {expected}\n\n"
            f"Answer to check:\n{body(a['answer'])}")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    cost = 0.0
    for _ in range(2):
        text, c = llm.chat(messages, model, schema=SCHEMA, temperature=0.0, max_tokens=1500)
        cost += c
        try:
            out = llm.parse_json(text)
            return {"sentences": out["sentences"], "correctness": out["correctness"],
                    "reasoning": out["reasoning"], "cost_usd": cost}
        except (ValueError, KeyError, TypeError):
            pass
    return {"unjudged": True, "cost_usd": cost}


def judge(answers, out, model):
    done = {key(j) for j in read_jsonl(out) if not j.get("unjudged")}
    counts, cost = Counter(), 0.0
    for a in read_jsonl(answers):
        if key(a) in done:
            continue
        row = {"id": a["id"], "model": a["model"], "retriever": a["retriever"], "judge_model": model}
        if a["abstained"]:
            row |= {"abstained": True, "sentences": [], "correctness": "not_applicable"}
        else:
            row |= judge_one(a, model)
            cost += row["cost_usd"]
        counts["unjudged" if row.get("unjudged") else "judged"] += 1
        append_jsonl(out, row)
    print(f"judged {counts['judged']}; unjudged {counts['unjudged']}; cost ${cost:.4f}")
    return counts


def div(a, b):
    return round(a / b, 4) if b else ""


def factual(j):
    return [s for s in j["sentences"] if s["factual"]]


def config_metrics(answers, judgments):
    n = len(answers)
    spoken = [a for a in answers if not a["abstained"]]
    judged = [(a, judgments[key(a)]) for a in spoken if "sentences" in judgments.get(key(a), {})]
    facts = [s for _, j in judged for s in factual(j)]
    verdicts = Counter(s["verdict"] for s in facts)
    cites = [s["verdict"] == "supported" and m == s["support_source"] for s in facts for m in s["cited"]]
    grounded = sum(bool(factual(j)) and all(s["verdict"] == "supported" for s in factual(j)) for _, j in judged)
    correctness = Counter(j["correctness"] for a, j in judged if a["type"] != "unanswerable")
    n_unans = sum(a["type"] == "unanswerable" for a in answers)
    tp = sum(a["abstained"] and a["type"] == "unanswerable" for a in answers)
    fp = n - len(spoken) - tp
    fn, tn = n_unans - tp, n - n_unans - fp
    return {"model": answers[0]["model"], "retriever": answers[0]["retriever"], "n": n,
            "fully_grounded_rate": div(grounded, n),
            "sentence_support_rate": div(verdicts["supported"], len(facts)),
            "partial_rate": div(verdicts["partial"], len(facts)),
            "citation_precision": div(sum(cites), len(cites)),
            "citation_coverage": div(sum(bool(s["cited"]) for s in facts), len(facts)),
            "correctness_correct": div(correctness["correct"], correctness.total()),
            "correctness_partial": div(correctness["partial"], correctness.total()),
            "unanswerable_n": n_unans, "abstained_on_unanswerable": tp, "answered_unanswerable": fn,
            "answerable_n": n - n_unans, "abstained_on_answerable": fp, "answered_answerable": tn,
            "abstention_recall": div(tp, n_unans), "false_abstention_rate": div(fp, n - n_unans),
            "abstention_f1": div(2 * tp, 2 * tp + fp + fn),
            "gate_abstentions": sum(a["gate_abstained"] for a in answers),
            "invalid_citations_total": sum(len(a["invalid_citations"]) for a in answers),
            "unjudged": len(spoken) - len(judged),
            "cost_per_answer": div(sum(a["cost_usd"] for a in answers), n),
            "p50_latency_s": round(statistics.median(a["latency_s"] for a in answers), 4)}


def find_failures(a, j):
    """(category, evidence) pairs for one answer and its judgment ({} when not judged)."""
    answerable = a["type"] != "unanswerable"
    gold = {g["work_id"] for g in a["gold"]}
    found = [("unsupported_sentence", s["text"]) for s in j.get("sentences", [])
             if s["factual"] and s["verdict"] == "unsupported"]
    found += [("invalid_citation", f"[{m}]") for m in a["invalid_citations"]]
    if not answerable and not a["abstained"]:
        found.append(("missed_abstention", body(a["answer"])[:200]))
    if answerable and a["abstained"]:
        found.append(("false_abstention", "score gate" if a["gate_abstained"] else "model"))
    if answerable and gold and not gold & {s["work_id"] for s in a["sources"]}:
        found.append(("retrieval_miss", " ".join(sorted(gold))))
    if j.get("correctness") == "wrong":
        found.append(("wrong_answer", j["reasoning"]))
    return found


def metrics(answers, judgments, out, failures):
    by_key = {key(j): j for j in read_jsonl(judgments)}
    groups = {}
    for a in read_jsonl(answers):
        groups.setdefault((a["model"], a["retriever"]), []).append(a)
    if not groups:
        raise SystemExit(f"no answers in {answers}")
    table = [config_metrics(rows, by_key) for rows in groups.values()]
    found = [{"model": a["model"], "retriever": a["retriever"], "id": a["id"], "category": c,
              "evidence": e, "question": a["question"]}
             for rows in groups.values() for a in rows for c, e in find_failures(a, by_key.get(key(a), {}))]
    write_csv(out, table, list(table[0]))
    write_csv(failures, found, FAILURE_FIELDS)
    for r in table:
        print("\n".join(f"{k:>26}  {v}" for k, v in r.items()) + "\n")
    print(f"wrote {out} and {failures} ({len(found)} failure candidates)")
    return table, found


def label_sheet(answers, n, out):
    if Path(out).exists():
        raise SystemExit(f"{out} exists; refusing to overwrite hand labels")
    pool = [a for a in read_jsonl(answers) if not a["abstained"]]
    rows = []
    for a in random.Random(config.SEED).sample(pool, min(n, len(pool))):
        shown = "; ".join(f"[{s['n']}] {s['title']}" for s in a["sources"])
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", body(a["answer"]).strip()) if s]
        for i, s in enumerate(sentences, 1):
            rows.append({"answer_id": a["id"], "model": a["model"], "retriever": a["retriever"], "sentence_no": i,
                         "sentence": s, "cited": " ".join(re.findall(r"\[\d+\]", s)), "sources_shown": shown,
                         "human_verdict": ""})
    write_csv(out, rows, LABEL_FIELDS)
    print(f"wrote {len(rows)} sentences from {min(n, len(pool))} answers to {out}")


def align(humans, sentences):
    """(human row, judge sentence or None) pairs: by order when the counts match, else by fuzzy text."""
    if len(humans) == len(sentences):
        return list(zip(humans, sentences))
    texts = [s["text"] for s in sentences]
    best = [process.extractOne(h["sentence"], texts, scorer=fuzz.ratio, score_cutoff=80) for h in humans]
    return [(h, sentences[b[2]] if b else None) for h, b in zip(humans, best)]


def kappa(pairs):
    """(accuracy, Cohen's kappa) over (human, judge) label pairs."""
    n = len(pairs)
    if not n:
        return "", ""
    po = sum(h == j for h, j in pairs) / n
    hc, jc = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
    pe = sum(hc[c] * jc[c] for c in hc) / n / n
    return round(po, 4), round((po - pe) / (1 - pe), 4) if pe < 1 else ""


def agreement(labels, judgments, out):
    by_key = {key(j): j for j in read_jsonl(judgments)}
    by_answer = {}
    with open(labels) as f:
        for row in csv.DictReader(f):
            by_answer.setdefault((row["answer_id"], row["model"], row["retriever"]), []).append(row)
    pairs, unaligned = [], 0
    for k, humans in by_answer.items():
        for h, s in align(humans, by_key.get(k, {}).get("sentences", [])):
            human = h["human_verdict"].strip().lower()
            if human in ("", "-"):
                continue
            if s is None:
                unaligned += 1
            else:
                pairs.append((human, s["verdict"]))
    acc3, k3 = kappa(pairs)
    acc2, k2 = kappa([(h == "supported", j == "supported") for h, j in pairs])
    line = (f"{datetime.date.today()} labels={labels} n={len(pairs)} unaligned={unaligned} "
            f"accuracy3={acc3} kappa3={k3} accuracy2={acc2} kappa2={k2}")
    print(line)
    with open(out, "a") as f:
        f.write(line + "\n")
    return {"n": len(pairs), "unaligned": unaligned, "accuracy3": acc3, "kappa3": k3, "accuracy2": acc2, "kappa2": k2}


def main():
    answers, judgments = config.RESULTS / "answers.jsonl", config.RESULTS / "judgments.jsonl"
    labels = config.DATA / "judge_labels.csv"
    commands = {  # subcommand -> (function, its keyword arguments as --flags with defaults)
        "run": (run, {"questions": config.QUESTIONS, "split": "test", "retriever": config.RETRIEVER,
                      "k": config.TOP_K, "model": config.ANSWER_MODEL, "out": answers}),
        "judge": (judge, {"answers": answers, "out": judgments, "model": config.JUDGE_MODEL}),
        "metrics": (metrics, {"answers": answers, "judgments": judgments, "out": config.RESULTS / "chat_quality.csv",
                              "failures": config.RESULTS / "failures_candidates.csv"}),
        "label-sheet": (label_sheet, {"answers": answers, "n": 20, "out": labels}),
        "agreement": (agreement, {"labels": labels, "judgments": judgments,
                                  "out": config.RESULTS / "judge_agreement.txt"}),
    }
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (_, defaults) in commands.items():
        s = sub.add_parser(name)
        for flag, value in defaults.items():
            s.add_argument("--" + flag, default=value, type=int if isinstance(value, int) else None)
    args = vars(p.parse_args())
    commands[args.pop("cmd")][0](**args)


if __name__ == "__main__":
    main()
