"""Every path, model id and threshold in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"                      # downloaded HTML and PDF (not committed)
PARSED = DATA / "parsed"                # one JSON per document (not committed)
CACHE = DATA / "cache"                  # API responses (not committed)
INDEX = DATA / "index"                  # one folder per model (not committed)
MANIFEST = DATA / "manifest.csv"        # committed
LABELS = DATA / "relevance_labels.jsonl"  # committed: cached classifier labels
CHUNKS = DATA / "chunks.jsonl"          # not committed
EVAL = ROOT / "eval"
QUESTIONS = EVAL / "questions.jsonl"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"


def parsed_path(doc_id):
    """data/parsed file for a document id; ':' and '/' (old arXiv ids like cmp-lg/9702003) are not file-safe."""
    return PARSED / (doc_id.replace(":", "_").replace("/", "_") + ".json")

# Models (all hosted ones go through OpenRouter)
CLASSIFIER_MODEL = "google/gemini-2.5-flash"  # flash-lite over-included on a 40-paper dry run (2026-10-08)
ANSWER_MODEL = "anthropic/claude-haiku-4.5"
JUDGE_MODEL = "google/gemini-2.5-flash"
EMBED_MODELS = {
    "bge": "BAAI/bge-base-en-v1.5",        # general, local
    "scincl": "malteos/scincl",            # scientific, local
    "qwen-or": "qwen/qwen3-embedding-8b",  # modern, hosted via OpenRouter
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a question about LLM tokenization research, retrieve passages that answer it\nQuery: "
)
QWEN_DIMS = 1024
QWEN_PROVIDER = "nebius"  # pin one OpenRouter provider so index and query vectors come from the same weights

# Collection
SEED = 13
CORPUS_CAP = 1000          # max works indexed; the manifest still lists every work
ARXIV_API_DELAY = 3.1      # seconds between arXiv API requests (terms of use: 1 per 3 s)
FILE_DELAY = 1.0           # seconds between full-text fetches per host

# Chunking
CHUNK_TARGET_TOKENS = 350
CHUNK_MAX_TOKENS = 480
MIN_SECTION_WORDS = 40

# Retrieval
RETRIEVER = "hybrid-bge"   # set from results/RECOMMENDATION.md after the comparison
DENSE_CANDIDATES = 50
RRF_K = 60
TOP_K = 8                  # passages shown to the answer model
MAX_PER_WORK = 4           # R1 cap on passages from one work in the shown set (2 cost lookup recall: results/dedup.csv)
COLLAPSE_TAU = 0.92        # R2: fold a passage whose cosine with a shown one is >= tau
SURVEY_CAP = 1             # R3: surveys allowed in the top 5 unless the question asks for one

# Chat
ABSTAIN_SENTENCE = "The corpus does not support an answer to this question."
ABSTAIN_DENSE_TAU = 0.0    # score gate off: on the dev split the prompt alone abstains on 4/4 unanswerable (scripts/tune_gate.py measured tau 0.777; results/answers_dev_gate*.jsonl)
ABSTAIN_BM25_TAU = 0.0
