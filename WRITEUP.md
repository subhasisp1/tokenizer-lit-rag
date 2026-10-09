# Talk to the literature: a RAG chatbot over LLM tokenizer research

Every number below is printed by `uv run python -m rag.report` from the files under `results/`; the
commands that produced them are in `README.md`. 
## 1. Corpus: what counts as a tokenizer paper

**Boundary.** A paper is in scope when its main contribution or main analysis is how text is split into
tokens for a neural language model or translation system, or the measured consequences of that choice:
subword algorithms, byte- and character-level models (when the choice of input unit is the point),
vocabulary size and scaling, tokenizer transfer and adaptation, multilingual fertility and fairness,
tokenization effects on arithmetic, code and reasoning, glitch tokens, theory and evaluation, and surveys of
these. Out: image, speech, molecule and DNA "tokenizers"; token pruning, merging and KV-cache work; prompt
compression; papers that merely use a tokenizer or a character-level model; corpus statistics such as
Zipf and Heaps laws. The full text, with a decision rule ("would a reader describe the main topic as
tokenization? using tokens as a tool is not enough; in doubt, out"), is `docs/boundary.md` and is also
the classifier's system prompt.

**Collection.** Four arXiv API query families (title terms, specific abstract phrases, generic abstract
terms, byte and character terms) in yearly `submittedDate` windows, one request per 3.1 seconds; the
yearly windows summed exactly to the unwindowed totals for every family. OpenAlex added ACL Anthology
papers that never went to arXiv (10 phrase searches) and, for every in-scope arXiv paper, the venue,
DOI, licence and published-version link (946 of 993 resolved by title, the rest left unresolved when the
free key's daily budget ran out). Two filters follow: deterministic rules (include patterns such as
`byte[- ]pair` and `fertility`; title-only exclusions for other modalities and for pruning or
compression; a single "tokenizer" in the abstract is not enough), then one strict-JSON call per
survivor to `google/gemini-2.5-flash`, cached and committed in `data/relevance_labels.jsonl` so a rerun
makes no calls. The cheaper Flash-Lite was tried first on 40 hard candidates and kept 13 to 16 of them
including Zipf-law and keyword-spotting papers; Flash kept 7, the ones a reader would keep; a
routing model kept 3 and rejected a seed paper, so Flash stayed ($1.35 for 2,369 calls).

| stage | count |
|---|---|
| candidates collected (arXiv + ACL discovery) | 17419 |
| rejected by rules: incidental_mention | 12223 |
| rejected by rules: other_modality | 2499 |
| rejected by rules: pruning_or_compression | 328 |
| rejected by the classifier: other | 464 |
| rejected by the classifier: incidental_mention | 432 |
| rejected by the classifier: other_modality | 284 |
| rejected by the classifier: pruning_or_compression | 54 |
| rejected by the classifier: non_ml_tokenization | 14 |
| rejected by the classifier: incomplete_metadata | 1 |
| in scope (documents) | 1120 |
| outside the corpus cap | 113 |
| works (after merging versions and copies) | 1047 |
| works indexed (canonical document with full text) | 933 |
| chunks in the canonical index | 58032 |

All 26 seed papers named while designing the boundary were found by the queries and kept. The corpus
was capped at 1,000 works (seeds first, then papers whose title matches an include rule, then a seeded
random fill) to keep a CPU rebuild under two hours; the manifest lists every candidate with the reason
it was rejected, capped or merged.

**How good is the classifier?** On a seeded sample of 60 rule-passing candidates, 30 were labelled by
the author and reviewed by an LLM judge (Claude Fable 5.1), which overrode 7 labels on explicit boundary
rules (a mesh tokenizer, a user-behaviour tokenizer, CANINE left out by oversight); the other 30 were
labelled by the judge alone. The source of every label is in `data/relevance_sample.csv`.

```
dev:  n=30 precision=0.688 recall=1.000 F1=0.815 accuracy=0.833 (95% Wilson 0.664-0.927)
test: n=30 precision=0.923 recall=0.857 F1=0.889 accuracy=0.900 (95% Wilson 0.744-0.965)
```

The classifier misses nothing on the author-labelled half and lets in borderline papers that use
tokenization inside a larger contribution (a released pool of byte-level models, an OCR paper that
picks ByT5). For a retrieval corpus that is the cheaper error, so the corpus was kept as is.

## 2. Parsing and chunking

Full text is fetched politely (1,157 requests for 1,000 works, one per second, zero 403s): arXiv's own
LaTeXML HTML first (810 works), ar5iv as a fallback (8), the PDF last (192, including the 116 ACL-only
papers). HTML keeps headings, tables and the LaTeX of every formula, so it is parsed with a dedicated
parser; PDFs go through pymupdf4llm. On 11 works with both formats the PDF parse reaches a heading F1
of 0.943 and a body-word recall of 0.937 against the HTML parse, excludes the reference list in all 11,
and costs 11.3 s per paper against 0.18 s (`results/parser_comparison.csv`), which is why GROBID was not
needed. All 1,564 documents parsed (818 HTML, 746 PDF), none failed the quality gate.

Chunking decisions, each visible in `rag/chunk.py`: reference lists are dropped (they are other
papers' titles and match exact-term queries in almost every paper); figure captions and tables are
their own chunks (vocabulary sizes and fertility numbers live in tables; the header row is repeated
when a table is split); inline math is kept as LaTeX; appendices are kept, acknowledgements and
footnotes dropped. Text is packed by whole paragraphs to a 350-token target (hard maximum 480, bge
tokenizer), never across a section, with sections under 40 words merged forward. Each chunk embeds as
`title | section path` plus text. Result: 99,839 chunks (58,032 canonical), median 249 tokens, 59% text,
27% table, 12% caption.

## 3. Versions and duplicates

Three different problems, three mechanisms. **Same work:** arXiv versions, the venue copy and the
preprint are documents of one work. Documents are grouped by identifiers first (arXiv id, OpenAlex
work id, non-arXiv DOI, and the copy links the downloader records), then by a guarded fuzzy rule between
groups: title token-set ratio at least 92, author-surname Jaccard at least 0.5, abstract cosine (bge)
at least 0.85, and a body-length guard that marks a short workshop paper against a much longer journal
version as related rather than merged. The canonical document is the latest arXiv version with full
text, else the venue copy. On 1,674 in-scope documents this gives 1,047 works; the fuzzy rule added
5 merges, all of them an arXiv paper and the same paper under a second OpenAlex record.

Against 25 hand-checked pairs (5 same work, 20 hard negatives with similar titles) plus 70 pairs known
to be the same work through a shared OpenAlex id, the default rule has precision 1.00 and recall 0.93;
the best rule in the sweep (`results/dedup_rules.csv`) reaches 0.96 at the same precision, but every
pair it adds was already joined by an identifier, so the default stayed. What it might wrongly merge: an
extended journal version with new experiments (the length guard catches the clear cases only), and
OpenAlex's own mis-links. What it might wrongly keep apart: a camera-ready copy with a changed title
and no OpenAlex link; three of the 74 known pairs show this.

**Before and after.** The "raw" index deliberately holds every version and venue copy (the situation
the brief describes); the canonical index holds one document per work. Retrieval-time collapsing is a
separate, measured choice: a cap on passages per work (R1), a fold of near-duplicate passages across
works (R2, cosine above τ), and a survey cap (R3). DupRate@5 is the share of top-5 slots whose work
already appeared higher in the list.

| retriever | index | collapse | Recall@5 | 95% CI | DupRate@5 | distinct works@5 |
|---|---|---|---|---|---|---|
| hybrid-bge | raw | off | 0.808 | 0.69-0.91 | 0.68 | 1.6 |
| hybrid-bge | canonical | off | 0.825 | 0.71-0.92 | 0.64 | 1.8 |
| hybrid-bge | canonical | cap 4 per work | 0.758 | 0.63-0.88 | 0.52 | 2.4 |
| hybrid-bge | canonical | cap 2 per work | 0.646 | 0.49-0.79 | 0.28 | 3.5 |
| bm25 | raw | off | 0.743 | 0.61-0.87 | 0.68 | 1.6 |
| bm25 | canonical | off | 0.776 | 0.66-0.89 | 0.61 | 1.9 |
| bm25 | canonical | cap 4 per work | 0.776 | 0.66-0.89 | 0.50 | 2.5 |
| bm25 | canonical | cap 2 per work | 0.693 | 0.55-0.83 | 0.25 | 3.6 |

Removing copies helps every retriever (the full grid is `results/dedup.csv`). Capping passages per work
is a trade: a cap of 2 halves DupRate@5 but costs lookup recall, because single-paper questions often
need the third-best passage of the right paper; a cap of 4 costs BM25 nothing, so the chatbot uses 4.
The cross-work fold never fired between τ 0.90 and 0.97 and hurt at 0.88, so it is off. Surveys are too
few in this corpus (3) for the survey cap to matter.

## 4. Embedding models, keyword search and the hybrid

Three "ways" as the brief asks, plus BM25 and reciprocal-rank fusion (k = 60) of each dense top-50 with
the BM25 top-50: `BAAI/bge-base-en-v1.5` (general, local, query prefix), `malteos/scincl` (scientific,
trained on title-plus-abstract citation pairs, local) and `qwen/qwen3-embedding-8b` through OpenRouter
(hosted, 1024-d truncation, provider pinned). The decision rule was fixed before running: highest Recall@5 on the 30 answerable test
questions wins; if its interval overlaps a cheaper, faster or local row, that row is taken.

| retriever | Recall@5 | 95% CI | MRR@10 | p95 query (CPU) | index build |
|---|---|---|---|---|---|
| bm25 | 0.776 | 0.66-0.89 | 0.722 | 2 ms | 3 s |
| hybrid-qwen-or | 0.775 | 0.64-0.89 | 0.751 | 8.6 s | 23 min, $0.14 |
| hybrid-bge | 0.758 | 0.63-0.88 | 0.718 | 1.4 s | 123 s (GPU) |
| qwen-or | 0.743 | 0.60-0.87 | 0.744 | 3.8 s | 23 min, $0.14 |
| bge | 0.660 | 0.49-0.82 | 0.630 | 1.4 s | 123 s (GPU) |
| hybrid-scincl | 0.526 | 0.37-0.68 | 0.484 | 1.0 s | 119 s (GPU) |
| scincl | 0.250 | 0.10-0.40 | 0.171 | 1.5 s | 119 s (GPU) |

BM25 wins and is itself the cheapest and fastest row, so it is the chatbot's retriever; the two hybrids
are within its interval (4 wins, 3 losses each on the 30 questions) and remain one flag away. Two
findings worth stating plainly. First, a paper-level scientific model is the wrong tool for passage
retrieval: SciNCL finds a quarter of what BM25 finds. Second, the question set is lookup-heavy (12 of
30 ask for a number or a setting) and the questions name their setting, which suits exact-term
matching; the per-question comparison shows the dense rows winning more of the conceptual and
multi-paper questions. Gold here includes 71 passages the pooling step added (section 5), so the
dense models are not penalised for finding equally good passages the first gold missed.

## 5. The evaluation set and how it can mislead

44 questions: 36 test (12 lookup, 6 conceptual, 7 multi-paper, 5 follow-ups with a parent and a gold
standalone rewrite, 6 unanswerable) and 8 dev (4 answerable, 4 unanswerable). Candidates were proposed
by an LLM from 40 sampled papers and 10 multi-paper themes, each with a verbatim supporting sentence
that was verified by string match against the parsed text (8 of 90 proposals dropped as unverifiable);
36 plus 8 were selected and edited from the 82 survivors so that each question names its setting
without copying the quote (median word overlap between question and quote 0.15), and the follow-ups
and unanswerables were written to the quotas: a fictional tokenizer, a number no paper reports, a topic
outside the boundary, a claim no paper makes, a model newer than the corpus, a paper not in the corpus.
The set was frozen in commit `521ab39` before any retrieval or chat result was computed; thresholds were
tuned on the 8 dev items only. Gold is a (work, section, verbatim quote) triple; a chunk hits when it
belongs to the work and matches the quote at partial-ratio 85, so the labels survive re-chunking.
After the retrievers ran, the top-20 of BM25, bge and Qwen were pooled and pre-graded 0-3 by
`google/gemini-2.5-flash` (1,327 chunks, $0.43); the 71 passages graded "fully answers" were added as
gold, and a gold work counts as found when any of its passages is in the top 5.

Biases to keep in mind: 30 answerable questions give wide intervals, so differences under 0.1 are
noise; LLM-proposed questions lean towards what papers state crisply; pooling only extends gold for
retrievers in the pool.

## 6. The chatbot

A plain Python loop: the latest message is condensed into a standalone query with one short LLM call
when there is history (shown to the user as "searching: …"), 8 passages are retrieved (BM25, canonical
index, cap 4 per work), and `anthropic/claude-haiku-4.5` answers from the numbered passages with a
citation after every factual sentence, or with the exact sentence "The corpus does not support an
answer to this question." Citations are validated in code (a `[n]` outside the shown range is removed
and marked), then rendered as "Title (Year, Venue), § Section". Answer calls disable the model's hidden reasoning, because a model that thinks by default can spend the whole answer budget before writing a visible word. The first version abstained on 16 of
36 test questions, 10 of them wrongly: 7 times the model refused although the right paper was shown
and 3 times a score gate tuned on four dev questions fired. Version 2 answers from partial evidence and
abstains only when nothing bears on the question, uses the cap of 4, and drops the gate (on the dev
split the prompt alone abstains on all four unanswerable questions and none of the answerable). Both runs are kept under `results/`.

A sentence-level judge from a different model family (`google/gemini-2.5-flash`) sees only the shown
passages and the expected answer. A second, independent labelling pass over 78 sentences from 20 answers
gives the judge's agreement: on the 62 sentences the two passes could be aligned (11 differed in sentence splitting), they agree on 56 (accuracy 0.90). Four disagreements are sentences the second pass calls partial because the model added an inference to a quoted fact ("performance implications differ by task") while the judge called them supported; in the other two the judge was the stricter one (one partial, one unsupported). Cohen's kappa is near zero because almost every sentence is "supported" in both passes (chance agreement 0.9), so the counts are reported instead; on balance the judge is slightly the more lenient, so the grounded rates below are an upper bound by a few percent. A second judge run at temperature 0 over the same answers moved one verdict: fully grounded 0.722 to 0.694, every other metric within 0.02 (`results/chat_quality_rerun.csv`)

| fully grounded answers | sentences supported | citation precision | citation coverage | correct / partial / wrong | abstained on 6 unanswerable | false abstentions on 30 | invalid citations | cost per answer | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| 0.72 | 0.985 | 0.82 | 0.84 | 0.75 / 0.25 / 0.00 | 6 | 2 | 0 | $0.0034 | 3.4 s |

Follow-ups are the weak spot. Scoring the five follow-ups under four query forms gives Recall@5 0.40
for the raw follow-up, 0.20 for parent-plus-follow-up concatenation, 0.40 for the chatbot's rewrite and
0.80 for the gold standalone wording (`results/followups.csv`): the rewrite beats concatenation but
sometimes degrades into a keyword list that drops the question, which caused one of the two false
abstentions. `docs/transcript.md` is a six-turn session with two pronoun follow-ups resolved correctly
and one abstention on a fictional tokenizer; `docs/failures.md` works through five failures with
evidence and a fix each.

## 7. Limitations

The corpus is capped and the 47 unlinked arXiv works lack venue data; the boundary is a judgement and
the classifier's precision against the author's labels is 0.69; the evaluation set is small and partly
machine-made; BM25's win may not transfer to paraphrased or conversational queries, which is where the
hybrids won their questions; the judge is itself a model, validated on 78 sentences only; latency was
measured on one machine with local models on CPU. A fresh clone reproduces the quickstart on CPU in
about 5 minutes (150 works, a BM25 index and one cited answer) with the committed manifest; the full pipeline needs about four hours and $3.
