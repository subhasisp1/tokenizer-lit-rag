"""Read and write data/manifest.csv: one row per collected document, every stage fills its columns."""
import csv

from rag import config

COLUMNS = [
    "record_id", "work_id", "source", "arxiv_id", "arxiv_latest_version", "openalex_id", "doi",
    "title", "authors", "year", "venue", "published_version_url", "source_url", "fulltext_url",
    "license", "matched_queries", "stage1_result", "in_scope", "topic", "paper_type",
    "reject_reason", "fulltext_status", "duplicate_of", "canonical", "in_index",
    "not_indexed_reason", "n_chunks",
]


def load():
    if not config.MANIFEST.exists():
        return []
    with open(config.MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    config.DATA.mkdir(exist_ok=True)
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def is_true(v):
    return str(v).strip().lower() == "true"
