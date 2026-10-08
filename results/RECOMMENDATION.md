# Retriever recommendation

Decision rule, fixed before any run: the row with the highest Recall@5 on the 30 answerable test questions wins;
if its 95% bootstrap interval overlaps a cheaper, faster or local row, that row is taken instead.

| retriever | Recall@5 | 95% CI | MRR@10 | wins/losses vs best | p50 query (CPU) | index build | index $ |
|---|---|---|---|---|---|---|---|
| bm25 | 0.776 | 0.66-0.89 | 0.722 | 0/0 | 0 ms | 6 s on cpu | 0 |
| hybrid-qwen-or | 0.775 | 0.64-0.89 | 0.751 | 4/3 | 1290 ms | 1364 s on openrouter | 0.1417 |
| hybrid-bge | 0.758 | 0.63-0.88 | 0.718 | 4/3 | 723 ms | 123 s on cuda | 0.0 |
| qwen-or | 0.743 | 0.60-0.87 | 0.744 | 3/4 | 1204 ms | 1364 s on openrouter | 0.1417 |
| bge | 0.660 | 0.49-0.82 | 0.630 | 5/9 | 822 ms | 123 s on cuda | 0.0 |
| hybrid-scincl | 0.526 | 0.37-0.68 | 0.484 | 0/8 | 757 ms | 119 s on cuda | 0.0 |
| scincl | 0.250 | 0.10-0.40 | 0.171 | 0/20 | 837 ms | 119 s on cuda | 0.0 |

**Chosen: bm25.** It has the highest Recall@5 and is itself the cheapest, fastest and fully local row, so the
tie-break never applies. hybrid-qwen-or and hybrid-bge are within its interval (4 wins, 3 losses each on the
30 questions) and remain available through `--retriever`. bge alone trails by 0.12, and SciNCL, trained on
title-plus-abstract pairs rather than passages, is far behind (0.25). Query latency is measured with the local
models forced to CPU; the Qwen rows pay an API round trip per query.

Gold = the frozen set plus 71 pooled passages graded 3 by the pre-judge; a gold work counts as found when any
of its passages is in the top 5. Collapse: canonical index, at most 4 passages per work.
