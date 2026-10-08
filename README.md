# tokenizer-lit-rag

Talk to the literature: a retrieval-augmented chatbot over research papers about how text is tokenized
for language models (subword algorithms, byte- and character-level models, vocabulary size, multilingual
fertility and fairness, tokenizer transfer, tokenization effects on reasoning, glitch tokens, theory).
The corpus is 933 works collected from arXiv and the ACL Anthology, deduplicated across versions and venue
copies, chunked by section, indexed four ways and evaluated on a frozen, labelled question set. The design,
numbers and limitations are in [`WRITEUP.md`](WRITEUP.md); every table there is printed by `rag/report.py`
from the files under `results/`.

## Quickstart (fresh clone, CPU, about 15 minutes)

```bash
git clone https://github.com/subhasisp1/tokenizer-lit-rag.git && cd tokenizer-lit-rag
uv sync                                   # Python 3.12, CPU torch; the venv is about 1.4 GB
cp .env.example .env                      # then put your OpenRouter key in OPENROUTER_API_KEY=
uv run python -m rag.download --limit 150 # 150 works from the committed manifest, 1 request/s
uv run python -m rag.parse                # arXiv HTML first, PDF fallback -> data/parsed/
uv run python -m rag.chunk                # section-aware chunks -> data/chunks.jsonl
uv run python -m rag.index --model bm25   # the chatbot's default retriever; no model download
uv run python -m rag.chat                 # commands: :sources  :reset  :quit
```

The corpus collection and classification need not be rerun: the manifest (`data/manifest.csv`) and the
cached classifier labels (`data/relevance_labels.jsonl`) are committed. Only the chat turns call the API
(answer model `anthropic/claude-haiku-4.5` through OpenRouter, well under a cent per turn). Dense
retrievers are optional: `uv run python -m rag.index --model bge` then `uv run python -m rag.chat
--retriever hybrid-bge` (first run downloads the 440 MB model; indexing 150 works takes a few minutes on CPU).

## Full reproduction, in order

| Step | Command | What it does | Time and cost on this machine |
|---|---|---|---|
| 1 | `rag.collect arxiv` | 4 query families in yearly windows against the arXiv API (1 request / 3.1 s) | 84 requests, 5 min, free |
| 2 | `rag.collect stage1 --report` | deterministic include and exclude rules on title + abstract | seconds |
| 3 | `rag.enrich discover-acl` | ACL Anthology papers without an arXiv version, via OpenAlex | 10 requests |
| 4 | `rag.collect classify` | one strict-JSON call per rule-passing candidate (`google/gemini-2.5-flash`), cached | 2,369 calls, 15 min, $1.35 |
| 5 | `rag.collect manifest --summary` then `rag.collect cap --n 1000` | manifest with rejection reasons; cap to 1,000 works | seconds |
| 6 | `rag.enrich link --report` | venue, DOI, licence and published-version link from OpenAlex | ~1,000 requests, 15 min (free key: $1/day budget) |
| 7 | `rag.download` then `rag.download --extras` | HTML -> ar5iv -> PDF per work; v1 and venue copies for the duplicate study | 1,714 requests, 58 min, 1.1 GB |
| 8 | `rag.parse --report` | LaTeXML HTML parser, pymupdf4llm fallback; resumable | 1,564 documents, 2 h (PDFs dominate) |
| 9 | `rag.chunk` | section-bounded chunks, 350-token target | 100k chunks, 5 min |
| 10 | `rag.dedup group --report` then `rag.dedup canonical`, then `rag.chunk` again | works, canonical documents, in_index flags | 1 min |
| 11 | `rag.index --model {bm25,bge,scincl,qwen-or} [--variant raw]` | one index per model and variant | bge 2 min on GPU (about 1 h on CPU); qwen-or 23 min, $0.14 |
| 12 | `rag.questions propose`, `rag.questions pool`, `rag.questions stats` | candidate questions with verified quotes; pooled gold; statistics | $0.34 + $0.43 |
| 13 | `rag.evaluate --split test --collapse r1 --device cpu` | Recall@5, MRR@10, bootstrap intervals, latency | 10 min |
| 14 | `rag.judge run --split test`, `rag.judge judge`, `rag.judge metrics` | answers, sentence-level judge (`google/gemini-2.5-flash`), quality table | $0.15 |
| 15 | `scripts.followups`, `scripts.tune_gate`, `rag.report` | follow-up rewrite table, gate tuning on dev, Markdown tables | seconds |

Every command is `uv run python -m <module> ...`. Steps 4, 6, 11 (qwen-or), 12 and 14 need `.env`;
`OPENALEX_API_KEY` (free) is needed for step 6. The frozen question set is `eval/questions.jsonl`; the
dedup labelling file is `data/dup_pairs_to_label.csv`; the 60-paper relevance labels are
`data/relevance_sample.csv`. The GPU is optional everywhere: the venv installs CPU torch, and a local
`uv pip install torch --index-url https://download.pytorch.org/whl/cu128` (then `UV_NO_SYNC=1 uv run ...`)
only speeds up step 11.

## Where things are

- `rag/`: one module per stage (`collect`, `enrich`, `download`, `parse`, `chunk`, `dedup`, `index`, `search`,
  `evaluate`, `questions`, `chat`, `judge`, `report`), plus `config.py` (every path, model and threshold),
  `llm.py` (the OpenRouter client) and `manifest.py`.
- `data/`: the committed manifest, labels and labelling sheets; raw files, parsed text, chunks and indexes
  are gitignored.
- `eval/`: the frozen questions, the proposals they were chosen from, the pooled gold extension.
- `results/`: every table the write-up uses (`retrieval.csv`, `dedup.csv`, `dedup_rules.csv`,
  `chat_quality.csv`, `failures_candidates.csv`, `parser_comparison.csv`, `RECOMMENDATION.md`, ...).
- `docs/`: the corpus boundary, the chat transcript, the failure analysis.
- `tests/`: `uv run pytest -q` (86 tests; the first run downloads the bge model for the toy index).

## Rates, licences and what is not redistributed

The arXiv API is called at most once per 3.1 seconds on one connection; full texts come from
`export.arxiv.org` (and `ar5iv.labs.arxiv.org`, `aclanthology.org`) at one request per second per host,
with `Retry-After` honoured and a hard stop on any 403. `arxiv.org` itself is never crawled. OpenAlex is
called at most five times per second with a free key. arXiv metadata is CC0 and is committed in the
manifest; the full texts stay under their own licences (recorded per document in the manifest's `license`
column: arXiv's non-exclusive licence, Creative Commons variants, ACL Anthology CC BY 4.0) and are never
committed, so a reviewer's own run fetches them. OpenAlex data is CC0.

## Hardware used

Ubuntu 22.04, 24 cores, 62 GB RAM, one RTX 5090 used only for the local embedding indexes. All reported
query latencies were measured with the local models forced to CPU.
