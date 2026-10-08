# Corpus boundary

## In scope

The main contribution or main analysis is how text is split into tokens for neural language models or machine translation, or the measured consequences of that choice:

- subword algorithms (BPE, WordPiece, Unigram, SentencePiece, BPE-dropout, subword regularization);
- byte-level, character-level and tokenizer-free models (ByT5, CANINE, MegaByte, Byte Latent Transformer), when the paper's point is the choice of input unit (bytes or characters instead of words or subwords) for a language model or translation system; a paper that merely uses a character-level model for some task is out;
- the size of a neural model's token vocabulary and how it scales with model or data size;
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
- tokenization outside machine learning (lexers, parsers, security tokens, blockchain tokens, asset tokenization);
- statistical laws of word frequency or vocabulary growth in corpora (Zipf, Heaps) and other corpus linguistics that does not build, choose or evaluate a tokenizer for a neural model;
- word-level vocabularies of n-gram, embedding or classical models, and vocabulary in the sense of lexical knowledge or terminology.

## Decision rule

A paper is in scope only if a reader would describe its main topic as tokenization, or the tokenizer or token vocabulary of a neural language model or machine translation system, or the measured consequences of that choice. Using tokens, subwords or a vocabulary as a tool is not enough. When in doubt, mark it out of scope.

## Borderline

- classical word segmentation only if it is aimed at neural LM or MT input;
- multimodal models only if the text tokenizer itself is studied;
- NMT-era subword papers are in scope.
