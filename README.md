# tokenizer-lit-rag

Talk to the literature: a retrieval-augmented chatbot over arXiv and ACL papers about how text is
tokenized for language models. Under construction; the write-up will live in `WRITEUP.md`.

## Keys

Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (all hosted LLM and embedding calls go
through OpenRouter). `.env` is gitignored and a pre-commit hook rejects staged keys.
