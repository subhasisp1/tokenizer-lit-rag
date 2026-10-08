"""Cut the parsed papers (data/parsed/*.json) into retrieval chunks: data/chunks.jsonl.

The four chunking decisions:
1. Reference lists are dropped (in parse.py): they are other papers' titles and match exact-term
   queries like "SentencePiece" in almost every paper, crowding the top 5.
2. Figure captions are their own chunks: they are dense summaries of results.
3. Tables are their own chunks, caption + rows as "cell | cell" lines, with the caption and header row
   repeated when a table is split, because vocabulary sizes and fertility numbers live in tables.
4. Inline math is kept as LaTeX and appendices are kept (hyperparameters live there); acknowledgements,
   footnotes and broken macros are dropped (in parse.py).

Packing: within one section, whole paragraphs are appended until the next would push the chunk past
CHUNK_TARGET_TOKENS; a paragraph over CHUNK_MAX_TOKENS is split on sentence boundaries; no overlap;
a chunk never crosses a section, except that a section under MIN_SECTION_WORDS words is merged into the
first chunk of the next section (which keeps the next section's path). The abstract is one chunk.
Token counts use the bge tokenizer on embed_text ("<title> | <section path>\\n<text>"), no special tokens.
"""
import json
import re
import statistics
from collections import Counter
from functools import cache

from transformers import AutoTokenizer

from rag import config, manifest

HARD_LIMIT = 512


@cache
def tokenizer():
    tok = AutoTokenizer.from_pretrained(config.EMBED_MODELS["bge"])
    tok.model_max_length = 10**9  # we count long texts on purpose; silence the length warning
    return tok


def n_tokens(text):
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def halves(text):
    words = text.split()
    if len(words) > 1:
        return [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
    return [text[:len(text) // 2], text[len(text) // 2:]]


def fit(prefix, text):
    """Split `text` in halves until every piece fits CHUNK_MAX_TOKENS with the prefix (last resort)."""
    if n_tokens(prefix + text) <= config.CHUNK_MAX_TOKENS:
        return [text]
    return [piece for half in halves(text) for piece in fit(prefix, half)]


def pack(prefix, units, sep):
    """Greedily join units with `sep` while the chunk stays within CHUNK_TARGET_TOKENS."""
    chunks, current = [], []
    for unit in units:
        if current and n_tokens(prefix + sep.join(current + [unit])) > config.CHUNK_TARGET_TOKENS:
            chunks.append(sep.join(current))
            current = []
        current.append(unit)
    return chunks + [sep.join(current)] if current else chunks


def split_paragraph(prefix, para):
    if n_tokens(prefix + para) <= config.CHUNK_MAX_TOKENS:
        return [para]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$(\[])", para)
    return [piece for part in pack(prefix, sentences, " ") for piece in fit(prefix, part)]


def table_texts(prefix, table):
    rows = [" | ".join(r) for r in table["rows"]]
    head = "\n".join([table["caption"]] + rows[:1]).strip()
    if len(rows) < 2:
        return [head] if head else []
    return [f"{head}\n{part}" for part in pack(prefix + head + "\n", rows[1:], "\n")]


def chunk_doc(doc, title):
    """Return [(kind, section path, text)] for one parsed doc, in document order."""
    out = [("abstract", "Abstract", t) for t in fit(f"{title} | Abstract\n", doc["abstract"])] if doc["abstract"] else []
    carry = []  # paragraphs of short sections waiting for the next section
    sections = doc["sections"]
    for i, s in enumerate(sections):
        prefix = f"{title} | {s['path']}\n"
        paras = carry + s["paragraphs"]
        if sum(len(p.split()) for p in s["paragraphs"]) < config.MIN_SECTION_WORDS and i + 1 < len(sections):
            carry, paras = paras, []
        else:
            carry = []
        units = [piece for p in paras for piece in split_paragraph(prefix, p)]
        out += [("text", s["path"], t) for t in pack(prefix, units, "\n\n")]
        for table in s["tables"]:
            out += [("table", s["path"], t) for text in table_texts(prefix, table) for t in fit(prefix, text)]
        out += [("caption", s["path"], t) for c in s["captions"] for t in fit(prefix, c)]
    return out


def year_from_id(work_id):
    m = re.match(r"(\d\d)\d\d\.", work_id)
    return 2000 + int(m.group(1)) if m else None


def metadata(doc, rows_by_id):
    """title, year, venue, canonical and the manifest row (or None) for one parsed doc."""
    row = rows_by_id.get(doc["doc_id"]) or rows_by_id.get(f"arxiv:{doc['work_id']}")
    if row is None:
        return doc["title"], year_from_id(doc["work_id"]), "", True, None
    year = int(row["year"]) if str(row.get("year", "")).isdigit() else year_from_id(doc["work_id"])
    canonical = manifest.is_true(row["canonical"]) or not row.get("canonical", "").strip()
    return row["title"] or doc["title"], year, row.get("venue", ""), canonical, row


def main():
    rows = manifest.load()
    rows_by_id = {r["record_id"]: r for r in rows}
    chunks, per_doc = [], Counter()
    for path in sorted(config.PARSED.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        title, year, venue, canonical, row = metadata(doc, rows_by_id)
        for n, (kind, section, text) in enumerate(chunk_doc(doc, title)):
            embed_text = f"{title} | {section}\n{text}"
            c = {"chunk_id": f"{doc['doc_id']}#{n}", "doc_id": doc["doc_id"],
                 "work_id": (row or {}).get("work_id") or doc["work_id"],  # the manifest merges versions/copies
                 "title": title, "year": year, "venue": venue, "section": section, "kind": kind, "text": text,
                 "embed_text": embed_text, "n_tokens": n_tokens(embed_text), "canonical": canonical}
            assert c["n_tokens"] <= HARD_LIMIT, c["chunk_id"]
            chunks.append(c)
            per_doc[doc["doc_id"]] += 1
        if row is not None:
            row["n_chunks"] = per_doc[doc["doc_id"]]
    with open(config.CHUNKS, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    if rows:
        manifest.save(rows)
    tokens = [c["n_tokens"] for c in chunks]
    print(f"docs: {len(per_doc)}  chunks: {len(chunks)}  chunks per doc: {len(chunks) / max(len(per_doc), 1):.1f}")
    print(f"n_tokens median: {statistics.median(tokens):.0f}  max: {max(tokens)}")
    print("kinds:", dict(Counter(c["kind"] for c in chunks)))
    for doc_id, n in per_doc.items():
        print(f"  {doc_id}: {n}")


if __name__ == "__main__":
    main()
