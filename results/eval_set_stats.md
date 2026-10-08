# Evaluation set statistics

44 questions.

| type | dev | test | total |
|---|---|---|---|
| conceptual | 1 | 6 | 7 |
| followup | 0 | 5 | 5 |
| lookup | 2 | 12 | 14 |
| multi | 1 | 7 | 8 |
| unanswerable | 4 | 6 | 10 |

## Follow-ups

- t26: parent yes, standalone_rewrite yes
- t27: parent yes, standalone_rewrite yes
- t28: parent yes, standalone_rewrite yes
- t29: parent yes, standalone_rewrite yes
- t30: parent yes, standalone_rewrite yes

## Unanswerable: 10

- t31: MorphoPiece is a fictional tokenizer; no paper in the corpus describes a tokenizer by that name.
- t32: No paper in the corpus reports the compute spent on training the Llama 3 tokenizer; papers only use or modify that tokenizer.
- t33: Token merging for vision transformers is outside the corpus boundary (token pruning and merging, image tokenization); ToMe is only name-dropped in one related-work sentence.
- t34: No paper makes this claim; the corpus reports persistent token premiums under every subword algorithm and attributes them to training-data coverage, not to BPE versus WordPiece.
- t35: GPT-6 is newer than the corpus; no paper mentions it (the latest OpenAI tokenizer discussed is o200k for GPT-4o/GPT-5).
- t36: The RETRO paper (Borgeaud et al., 2022) is not in the corpus and is not about tokenization; no chunk mentions it.
- d05: SyllaBPE is a fictional tokenizer; no paper in the corpus describes it (syllable-based segmentation papers exist but none by that name or with that merge rule).
- d06: KV-cache eviction is outside the corpus boundary; H2O appears only in one related-work sentence with no numbers.
- d07: No paper reports emissions for training OpenAI's o200k tokenizer; the only carbon-footprint numbers in the corpus are for pretraining small models with different vocabularies.
- d08: Llama 5 is newer than the corpus; the only mention is advice to watch for its vocabulary announcement, with no details.

## Gold works per multi question: min 2, median 2.0, max 4

## Lexical overlap (query vs gold quotes)

median 0.147; above 0.30: 1/34 (3%): t26

## Pooling completeness

| question | gold items in pool | candidate gold to confirm |
|---|---|---|
| t01 | 1/1 | 2 |
| t02 | 1/1 | 6 |
| t03 | 1/1 | 3 |
| t04 | 1/1 | 7 |
| t05 | 1/1 | 0 |
| t06 | 1/1 | 8 |
| t07 | 1/1 | 4 |
| t08 | 1/1 | 6 |
| t09 | 1/1 | 5 |
| t10 | 1/1 | 8 |
| t11 | 1/1 | 6 |
| t12 | 1/1 | 1 |
| t13 | 1/1 | 1 |
| t14 | 1/1 | 8 |
| t15 | 1/1 | 6 |
| t16 | 1/1 | 4 |
| t17 | 1/1 | 5 |
| t18 | 1/1 | 1 |
| t19 | 2/2 | 32 |
| t20 | 3/4 | 34 |
| t21 | 3/4 | 23 |
| t22 | 2/2 | 5 |
| t23 | 3/4 | 11 |
| t24 | 2/2 | 9 |
| t25 | 3/3 | 28 |
| t26 | 1/1 | 8 |
| t27 | 1/1 | 3 |
| t28 | 1/1 | 1 |
| t29 | 1/1 | 6 |
| t30 | 1/1 | 6 |
| d01 | 1/1 | 3 |
| d02 | 1/1 | 8 |
| d03 | 1/1 | 14 |
| d04 | 2/3 | 15 |
