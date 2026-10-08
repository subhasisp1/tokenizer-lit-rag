# Failure analysis

Five failures from the final run on the frozen test set (`results/answers.jsonl`, `results/judgments.jsonl`,
`results/retrieval_per_question.csv`, `results/followups.csv`), each with the evidence, a root cause and a
concrete fix. Categories: retrieval ranking, query rewriting, citation attribution, derived numbers.

## 1. t11, lookup: the right paper, the wrong passages (false abstention)

- Question: in the study of tokenization bypassing knowledge editing and unlearning, what F1 score does the Toketive attack reach for detecting edits?
- Expected: 84.2% F1, a 26.2% relative gain over the strongest baseline.
- System: "The corpus does not support an answer to this question", followed by a correct description of the attack's two phases.
- Evidence: all 8 shown passages come from the gold paper (2609.29045), but none of them holds the number: BM25 ranks the passage with the F1 score outside its top 10 (`first_hit_rank` empty). The answer model refused correctly given what it saw.
- Root cause: retrieval ranking. The question's words ("F1 score", "detecting edits") match the method sections lexically; the number sits in the introduction's results sentence, which shares fewer terms.
- Fix: when the top passages all come from one work, add that work's abstract and introduction passages to the shown set (a same-work expansion), or switch to hybrid-qwen-or for number lookups; the retrieval table shows the hybrids find such passages more often (MRR@10 0.75 vs 0.72).

## 2. t28, follow-up: the rewrite lost the question (false abstention)

- Question (after "In the systematic study of vocabulary and BPE settings for fine-tuning NMT on in-domain data, how many BLEU points did the best model gain?"): "Which combination of BPE model and vocabulary source produced that best result?"
- Expected: combination C3, a BPE model from the original training data with a vocabulary from the in-domain data.
- System: abstained, restating the 6 BLEU figure from the parent turn.
- Evidence: the recorded rewrite is "systematic study of vocabulary and BPE settings for fine-tuning NMT model on in-domain data best fine-tuned model BLEU i…": a bag of search terms with no interrogative, so the retrieved passages were the parent's, not the results section (gold passage at BM25 rank 4 for the ideal standalone wording, see `results/followups.csv`: the standalone form scores 1.0 where the rewrite scores 0).
- Root cause: query rewriting. The rewrite prompt allows any "standalone search query"; the model sometimes produces a keyword list that drops the ask.
- Fix: require the rewrite to be a full question (reject rewrites without a question word or a verb, fall back to parent + follow-up concatenation), and show the rewrite to the user as the chatbot already does. On the five follow-ups the rewrite scores Recall@5 0.40 against 0.80 for the gold standalone wording, so this is the largest single lever for follow-up quality.

## 3. t23, multi-paper: the right evidence from the wrong papers (retrieval miss, partial answer)

- Question: what does the evidence say about the link between a tokenizer's compression rate and downstream performance?
- Expected: findings from three works (2605.01188, 2608.00582, 2608.18062) showing the link is inconsistent.
- System: an answer the judge rated partially correct, built from other papers that also discuss compression and downstream quality.
- Evidence: none of the three gold works is among the shown sources; the pooled gold extension confirms other papers answer the question (grade 3), so the answer is grounded but not in the intended works.
- Root cause: retrieval ranking on a theme shared by many papers; BM25 prefers passages that repeat the query's exact terms.
- Fix: for multi-paper questions, retrieve more candidates (k = 20) and apply the per-work cap before truncation, so more distinct works reach the answer model; dense retrievers rank conceptual matches higher here (bge-hybrid wins 4 of these questions).

## 4. t29, follow-up: right claim, wrong bracket (citation attribution)

- Question: does the Toketive attack also recover the original answers the edit was supposed to remove?
- Expected: yes, with a top-5 accuracy of 74.5%, 21.7% above the best baseline.
- System: "Yes, Toketive recovers the original answers…", citing [2] and [4], without the number.
- Evidence: the judge marks both sentences supported, but by passages [3] and [2], not the cited ones. Three of the shown passages come from the same paper and read alike, and the number lives in a fourth that was not shown (gold at BM25 rank 2 for the question wording, outside the 8 passages for the rewrite).
- Root cause: citation attribution among near-interchangeable passages of one work. Citation precision in the quality table is 0.82 for this reason; the citations are to the right paper but the wrong passage.
- Fix: after generation, re-attach each sentence's citation to the shown passage with the highest overlap with the sentence (a deterministic post-pass), and show passage section titles in bold so the model can tell them apart.

## 5. t12, lookup: a number the sources never state (derived figure)

- Question: by how much did the hierarchical character model beat the deep LSTM baseline in word perplexity?
- System: "the 4x512 HLSTM-B beat the 4x1024 deep LSTM by approximately 10.7% in word perplexity [1, 2]".
- Evidence: the judge finds no supporting passage (`support_source` 0): the two perplexities are in the sources, the percentage is the model's own arithmetic.
- Root cause: the answer rules say "copy numbers exactly" but do not forbid computing new ones.
- Fix: add "do not compute or convert numbers; quote them as the source states them" to the answer rules, and have the judge flag derived figures explicitly.
