# Corpus boundary

## In scope

The main contribution or main analysis is how text is split into tokens for neural language models or machine translation, or the measured consequences of that choice:

- subword algorithms (BPE, WordPiece, Unigram, SentencePiece, BPE-dropout, subword regularization);
- byte-level, character-level and tokenizer-free models (ByT5, CANINE, MegaByte, Byte Latent Transformer);
- vocabulary size and its scaling;
- tokenizer transfer, extension and adaptation to new languages or domains;
- multilingual fertility, fairness, parity and cost of tokenization;
- tokenization effects on arithmetic, code, reasoning or robustness;
- glitch tokens and tokenizer security;
- theory and evaluation of tokenization (compression, morphology, intrinsic metrics);
- surveys of these topics.

## Out of scope

- image, visual, video, speech, audio and codec "tokenizers" (VQ-VAE and similar);
- molecule, protein, DNA or other non-text tokenization;
- token pruning, merging, dropping or KV-cache eviction;
- prompt or context compression;
- language-model papers that merely state which tokenizer they used;
- tokenization outside machine learning (lexers, parsers, security tokens, blockchain tokens, asset tokenization).

## Borderline

- classical word segmentation only if it is aimed at neural LM or MT input;
- multimodal models only if the text tokenizer itself is studied;
- NMT-era subword papers are in scope.
