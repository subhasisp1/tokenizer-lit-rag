## Chat 2026-10-08 | retriever bm25 | model anthropic/claude-haiku-4.5

**You:** What evidence is there that byte-level tokenization beats BPE for low-resource languages?

**Assistant:** The sources provide mixed evidence on this question.

**Evidence supporting byte-level tokenization:**

Source [3] reports that in a multilingual many-to-English translation task with 58 languages, "byte model and character model perform significantly better than all BPE and BBPE models in this multilingual setting." However, the authors note this advantage may be specific to multilingual scenarios with imbalanced source-target pairs, and they conclude that "BBPE is still the most practical solution since it makes a good balance between performance (better BLEU than BPE) and speed." [3]

**Evidence against byte-level superiority:**

Source [4] found that for Indic NER tasks, "SentencePiece performed better than BPE in fine-tuned and zero-shot cross-lingual settings," but does not report byte-level tokenization outperforming BPE [4].

Source [1] shows byte-level BPE achieves faithful decoding comparable to morphology-aware tokenizers for Turkish [1].

**What the sources don't cover:**

The sources do not provide direct comparative evidence of byte-level tokenization (as distinct from byte-level BPE) systematically beating standard BPE across low-resource language tasks. The strongest evidence comes from one multilingual translation study with specific caveats about its applicability.

Sources:
[1] Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish (2026, arXiv (Cornell University)), § 5 Discussion > 5.1 Morpheus as a tokenizer: what it gains and what it costs
[3] Neural Machine Translation with Byte-Level Subwords (2019, Proceedings of the AAAI Conference on Artificial Intelligence), § Results and Analysis > BBPE on Many-to-En Translation
[4] Tokenization Matters: Improving Zero-Shot NER for Indic Languages (2025, IEEE International Conference on Electro Information Technology), § V Conclusion

<small>chunks: arxiv:2606.18717v2#64, arxiv:2506.01535v1#10, arxiv:1909.03341v2#22, arxiv:2504.16977v1#39, arxiv:2608.27658v1#10, arxiv:2608.27658v1#1, arxiv:2608.27658v1#12, arxiv:2608.26449v1#9</small>

**You:** How does that compare with WordPiece?

(searching: What evidence is there that byte-level tokenization compares to WordPiece for low-resource languages?)

**Assistant:** The sources do not directly compare byte-level tokenization performance against WordPiece for low-resource languages.

However, the sources provide some relevant comparisons:

**WordPiece vs. BPE for low-resource languages:**

Source [2] shows that for Dzongkha, WordPiece produced more continued words (0.27 proportion) compared to Byte-Pair Encoding (0.13 proportion), suggesting WordPiece fragments words more [2].

Source [3] evaluated WordPiece, SentencePiece, and Byte-Pair Encoding for Dzongkha and concluded that "SentencePiece is the most suitable tokenizer for Dzongkha, achieving the best performance in terms of Normalized Sequence Length (NSL), subword fertility, proportion of continued words and execution time," with WordPiece not identified as optimal [3].

Source [7] notes that WordPiece "optimizes [BPE] further with probability-driven subword selection" but does not provide empirical comparisons showing WordPiece outperforming byte-level approaches for low-resource languages [7].

**What the sources don't cover:**

The sources do not provide direct empirical evidence comparing byte-level tokenization specifically against WordPiece performance on low-resource language tasks.

Sources:
[2] Comparative Analysis of Tokenization Algorithms for Low-Resource Language Dzongkha (2025), § 5 Results and Analysis:
[3] Comparative Analysis of Tokenization Algorithms for Low-Resource Language Dzongkha (2025), § 7 Conclusion
[7] Tokenization Matters: Improving Zero-Shot NER for Indic Languages (2025, IEEE International Conference on Electro Information Technology), § II Related Work > II-A Tokenization Strategies for Low-Resource NLP

<small>chunks: arxiv:2406.15358v1#16, arxiv:2509.15255v1#13, arxiv:2509.15255v1#19, arxiv:2204.08832v1#2, arxiv:2406.15358v1#7, arxiv:2509.15255v1#12, arxiv:2504.16977v1#4, arxiv:2608.27658v1#12</small>

**You:** What vocabulary sizes did those studies use?

(searching: What vocabulary sizes did the studies comparing byte-level tokenization, BPE, and WordPiece for low-resource languages use?)

**Assistant:** **Vocabulary sizes in the studies:**

For Dzongkha, source [4] used uniform vocabulary sizes of 10,000 and 30,000 across WordPiece, BPE, and SentencePiece comparisons [4].

For Indic NER, source [3] does not specify the vocabulary sizes used in their tokenization comparisons [3].

For low-resource language tokenization more broadly, source [5] used a target vocabulary size of 8,500 for subword units (resulting in 7,952 for Fon and 8,116 for French), and 18,759 and 19,785 tokens for phrase-based tokenization [5].

Source [1] mentions that GPT-2 has a vocabulary size of 50,257 using byte-level BPE [1].

Source [8] trained byte-level BPE vocabularies with sizes of 50k and 100k for multilingual experiments, comparing against Google's multilingual BERT [8].

**What the sources don't cover:**

The sources do not provide a systematic comparison of how vocabulary size choices affect byte-level versus WordPiece performance specifically for low-resource languages.

Sources:
[1] Introducing Syllable Tokenization for Low-resource Languages: A Case Study with Swahili (2024, arXiv (Cornell University)), § 2 Related Work
[3] Tokenization Matters: Improving Zero-Shot NER for Indic Languages (2025, IEEE International Conference on Electro Information Technology)
[4] Comparative Analysis of Tokenization Algorithms for Low-Resource Language Dzongkha (2025), § 5 Results and Analysis
[5] Crowdsour

Sources:
[1] Introducing Syllable Tokenization for Low-resource Languages: A Case Study with Swahili (2024, arXiv (Cornell University)), § 2 Related Work
[3] Tokenization Matters: Improving Zero-Shot NER for Indic Languages (2025, IEEE International Conference on Electro Information Technology), § II Related Work > II-A Tokenization Strategies for Low-Resource NLP
[4] Comparative Analysis of Tokenization Algorithms for Low-Resource Language Dzongkha (2025), § 5 Results and Analysis:
[5] Crowdsourced Phrase-Based Tokenization for Low-Resourced Neural Machine Translation: The Case of Fon Language (2021, African NLP, EACL 2021), § 6 Methodology, Results and Conclusion > 6.1 Creation of vocabularies for WB, PhB, SU and WEB
[8] Training Multilingual Pre-trained Language Model with Byte-level Subwords (2021, arXiv (Cornell University)), § 4 Experiments > 4.3 Experimental Results

<small>chunks: arxiv:2406.15358v1#7, arxiv:2406.15358v1#16, arxiv:2504.16977v1#4, arxiv:2509.15255v1#8, arxiv:2103.08052v2#17, arxiv:2509.15255v1#3, arxiv:2506.01535v1#10, arxiv:2101.09469v2#38</small>

**You:** What is BPE-dropout and what problem does it address?

**Assistant:** **What is BPE-dropout:**

BPE-dropout is a subword regularization technique applied during training [6].

**Problem it addresses:**

BPE-dropout addresses the problem that rare tokens learned by standard BPE models lack semantically related neighbors in their embedding space [6]. In standard BPE models, "a rare word usually does not" have semantically related neighbors; instead, "a vast majority of closest neighbors of rare words are rare words." [6]

**How it works:**

The mechanism involves variational dropout applied to the tokenization process [1]. Figure 2 in source [1] illustrates that tokens like "New" and "What" can be safely removed without harming classification performance due to large dropout probability, suggesting BPE-dropout randomly drops or varies token selections during training.

**Results:**

When models are trained with BPE-dropout, rare tokens are no longer separated from the rest of the embedding space as they are in standard BPE models [6]. This contrasts with prior work by Gong et al. (2018) that proposed adversarial training for embedding layers to address the same issue—BPE-dropout achieves this improvement without requiring adversarial training [6].

Sources:
[1] How Large a Vocabulary Does Text Classification Need? A Variational Approach to Vocabulary Selection (2019, Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)), § 3 Our Method
[6] BPE-Dropout: Simple and Effective Subword Regularization (2019, Annual Meeting of the Association for Computational Linguistics (ACL)), § 6 Analysis > 6.2 Properties of the learned embeddings

<small>chunks: arxiv:1902.10339v4#12, arxiv:2606.28325v1#69, arxiv:2308.05585v1#8, arxiv:2511.15709v2#33, arxiv:2609.17553v1#10, arxiv:1910.13267v2#35, arxiv:2609.17553v1#118, arxiv:2606.14122v2#6</small>

**You:** How does the MorphoPiece tokenizer handle Finnish compounds?

**Assistant:** The corpus does not support an answer to this question.

The closest sources discuss Finnish compound handling more generally: source [7] notes that Finnish has "productive compounding" and that "morphological segmentation improves Finnish NLP," suggesting compounds are a challenge [7]. Source [2] mentions that optimal segmentation for Finnish shows improvements in handling compound words across tasks [2]. However, neither source describes how a specific tokenizer called "MorphoPiece" handles Finnish compounds, as MorphoPiece is not discussed in any of the provided sources.

<small>chunks: arxiv:2608.12278v1#15, arxiv:2412.06926v5#33, arxiv:2109.01100v1#26, arxiv:2305.14214v2#2, arxiv:2304.10158v1#28, arxiv:2607.23319v1#18, arxiv:2603.05883v1#30, arxiv:1809.01301v2#21</small>

**You:** Does a larger vocabulary always help a language model, according to the papers?

(searching: vocabulary size effects on language model performance according to the papers)

**Assistant:** No, a larger vocabulary does not always help according to the papers, though findings differ.

Source [6] reports that "the larger vocabulary size is, the better performance the language model achieves in both languages" (English and Japanese) [6].

However, source [2] found that "a larger source language vocabulary set does not necessarily lead to better performance," suggesting that vocabulary quality matters as much as size [2].

Source [5] found that vocabulary size is "not a crucial decision over this range of sizes" (32,768 to 49,152 tokens), with high correlation between downstream performance across different vocabulary sizes, and concluded they could "focus our analysis on the overall average accuracy" by averaging across vocabulary sizes [5].

Source [3] examines how F1 performance changes with vocabulary size on downstream tasks but does not explicitly state whether larger is always better [3].

So the evidence is mixed: while one study shows consistent improvements with larger vocabularies, another indicates vocabulary quality matters more than size alone, and a third suggests vocabulary size has limited impact within certain ranges.

Sources:
[2] Effective vocabulary expansion of multilingual language models for extremely low-resource languages (2026, arXiv (Cornell University)), § 5 Results and analysis > 5.2 Analysis > 5.2.1 The impact of vocabulary size of source language
[3] Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic Language Models (2024, arXiv (Cornell University)), § Impact of Vocabulary Size on Model Performance
[5] Tokenization Is More Than Compression (2024, Conference on Empirical Methods in Natural Language Processing (EMNLP)), § 5 Results > 5.1 Vocabulary Size
[6] Large Vocabulary Size Improves Large Language Models (2024), § 6 Conclusion

<small>chunks: arxiv:2402.18376v2#25, arxiv:2602.09388v2#38, arxiv:2403.11130v2#38, arxiv:2509.15255v1#4, arxiv:2402.18376v2#24, arxiv:2406.16508v2#20, arxiv:2011.13220v1#0, arxiv:2305.05480v3#9</small>

