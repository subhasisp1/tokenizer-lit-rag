"""CLI chatbot: rewrite the question, retrieve passages, answer from them with checked [n] citations.

Decisions:
- Follow-ups are condensed into a standalone query by one short LLM call (last 6 turns). The first
  turn skips it: the message is already the query. The answer model still sees the user's own words
  plus the last 4 turns, so the rewrite only steers retrieval.
- Score gate: when config.ABSTAIN_DENSE_TAU (or ABSTAIN_BM25_TAU for bm25) is > 0 and the best
  score is below it, abstain without calling the answer model. Hybrid retrievers are gated on their
  dense cosine, since RRF scores carry no absolute meaning. No sources also means abstain.
- Citations are checked in code, not trusted: [n] outside 1..len(sources) is removed and the sentence
  is marked " [citation removed]". "[1, 3]" and "[1][3]" both parse. Factual sentences (> 4 words,
  not questions, not the abstain sentence) with no [n] are listed so the judge can count them.
- The Sources list shows only cited passages, labelled "Title (Year, Venue), § Section".
- On a gate abstention `sources` still holds the retrieved passages, so :sources can show what came
  back; abstained answers get no Sources list.
- A turn = one history message. Every LLM call goes through rag.llm.chat at temperature 0.
"""
import argparse
import datetime
import re
import time

from rag import config, llm, search

REWRITE_PROMPT = (
    "Rewrite the user's last message as a standalone search query about the tokenizer literature. "
    "Resolve pronouns and words like that/it/those from the conversation. Keep exact technical terms. "
    "Do not add topics. If it is already standalone, return it unchanged. Output only the query."
)
ANSWER_PROMPT = (
    "You answer questions about research on tokenization for language models, using only the numbered "
    "sources provided. Rules: put a citation like [2] after every factual sentence, using only the "
    "source numbers given; copy numbers, names and settings exactly as the sources state them; when "
    "sources disagree, say so and cite both; prefer the primary paper over a survey for the same "
    "finding; answer in at most 180 words; if the sources do not contain the answer, reply exactly: "
    f"\"{config.ABSTAIN_SENTENCE}\" and you may add one sentence on what the closest sources cover instead."
)
CITE = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")
SOURCE_KEYS = ("chunk_id", "work_id", "title", "year", "venue", "section", "score", "text")


def label(s):
    meta = ", ".join(str(x) for x in (s["year"], s["venue"]) if x)
    return s["title"] + (f" ({meta})" if meta else "") + (f", § {s['section']}" if s["section"] else "")


def dialogue(history):
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)


def rewrite(history, user_msg, model):
    if not history:
        return user_msg, 0.0
    msg = f"{dialogue(history[-6:])}\nLast message: {user_msg}"
    text, cost = llm.chat([{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": msg}],
                          model, temperature=0.0, max_tokens=100)
    return text.strip().strip("\"'").strip() or user_msg, cost


def gated(query, retriever):
    if retriever == "bm25":
        tau, hits = config.ABSTAIN_BM25_TAU, (search.bm25(query, 1) if config.ABSTAIN_BM25_TAU > 0 else [])
    else:
        tau = config.ABSTAIN_DENSE_TAU
        hits = search.dense(query, retriever.removeprefix("hybrid-"), 1) if tau > 0 else []
    return tau > 0 and (not hits or hits[0][1] < tau)


def check_citations(raw, n_sources):
    """Return (text, cited, invalid, uncited) with invalid [n] removed and flagged per sentence."""
    cited, invalid, uncited, out = set(), [], [], []
    raw = re.sub(r"([.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)", r"\2\1", raw)  # "x. [1]" -> "x [1]."
    for part in re.split(r"((?<=[.!?])\s+)", raw):
        if not part.strip():
            out.append(part)
            continue
        bad = []

        def fix(m):
            nums = [int(x) for x in m.group(2).split(",")]
            good = [n for n in nums if 1 <= n <= n_sources]
            bad.extend(n for n in nums if n not in good)
            cited.update(good)
            return f"{m.group(1)}[{', '.join(map(str, good))}]" if good else ""

        sent = CITE.sub(fix, part)
        if bad:
            invalid.extend(bad)
            sent = re.sub(r"([.!?]*)$", r" [citation removed]\1", sent, count=1)
        s = sent.strip()
        factual = config.ABSTAIN_SENTENCE not in s and not s.endswith("?") and len(s.split()) > 4
        if factual and not re.search(r"\[\d", s):
            uncited.append(s)
        out.append(sent)
    return "".join(out), sorted(cited), invalid, uncited


def answer(history, user_msg, retriever=config.RETRIEVER, k=config.TOP_K, model=config.ANSWER_MODEL):
    t0 = time.time()
    query, cost = rewrite(history, user_msg, model)
    t1 = time.time()
    hits = search.retrieve(query, retriever, k, collapse_mode="r1")
    sources = [{"n": i, **{key: c[key] for key in SOURCE_KEYS}} for i, c in enumerate(hits, 1)]
    stop = not sources or gated(query, retriever)
    t2 = time.time()
    result = {"rewrite": query, "sources": sources, "cited": [], "invalid_citations": [],
              "uncited_sentences": [], "cost_usd": cost}
    if stop:
        return {**result, "text": config.ABSTAIN_SENTENCE, "raw": "", "abstained": True, "gate_abstained": True,
                "timings": {"rewrite_s": t1 - t0, "retrieve_s": t2 - t1, "answer_s": 0.0}}
    context = "\n\n".join(f"[{s['n']}] {label(s)}\n{s['text']}" for s in sources)
    if history:
        context += f"\n\nConversation so far:\n{dialogue(history[-4:])}"
    msgs = [{"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {user_msg}"}]
    raw, answer_cost = llm.chat(msgs, model, temperature=0.0, max_tokens=400)
    t3 = time.time()
    text, cited, invalid, uncited = check_citations(raw.strip(), len(sources))
    abstained = raw.lstrip(" \t\n\"'").startswith(config.ABSTAIN_SENTENCE)
    if cited and not abstained:
        text += "\n\nSources:\n" + "\n".join(f"[{n}] {label(sources[n - 1])}" for n in cited)
    return {**result, "text": text, "raw": raw, "cited": cited, "invalid_citations": invalid,
            "uncited_sentences": uncited, "abstained": abstained, "gate_abstained": False,
            "cost_usd": cost + answer_cost,
            "timings": {"rewrite_s": t1 - t0, "retrieve_s": t2 - t1, "answer_s": t3 - t2}}


def show_sources(sources):
    for s in sources:
        print(f"[{s['n']}] {s['score']:.4f}  {s['chunk_id']}  {s['title']} | § {s['section']}")
        print(f"    {s['text'][:200]!r}")


def save_turn(path, user_msg, r):
    lines = [f"**You:** {user_msg}", ""]
    if r["rewrite"] != user_msg:
        lines += [f"(searching: {r['rewrite']})", ""]
    lines += [f"**Assistant:** {r['text']}", "",
              f"<small>chunks: {', '.join(s['chunk_id'] for s in r['sources'])}</small>", "", ""]
    with open(path, "a") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Chat with the tokenizer literature. Commands: :sources :reset :quit")
    p.add_argument("--retriever", default=config.RETRIEVER)
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--model", default=config.ANSWER_MODEL)
    p.add_argument("--save", help="append a Markdown transcript to this path")
    a = p.parse_args()
    if a.save:
        with open(a.save, "a") as f:
            f.write(f"## Chat {datetime.date.today()} | retriever {a.retriever} | model {a.model}\n\n")
    history, last = [], None
    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if msg == ":quit":
            break
        if msg == ":reset":
            history, last = [], None
        elif msg == ":sources":
            show_sources(last["sources"] if last else [])
        elif msg:
            last = answer(history, msg, a.retriever, a.k, a.model)
            if last["rewrite"] != msg:
                print(f"(searching: {last['rewrite']})")
            print(last["text"] + "\n")
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": last["text"]}]
            if a.save:
                save_turn(a.save, msg, last)


if __name__ == "__main__":
    main()
