import csv
import json
from pathlib import Path

from rag import config, enrich, manifest

FIX = Path(__file__).parent / "fixtures"


def results(arxiv_id):
    return json.loads((FIX / f"openalex_{arxiv_id}.json").read_text())["results"]


def test_norm_and_title_match():
    assert enrich.norm("Byte  Latent Transformer: Patches,  Scale!") == "byte latent transformer patches scale"
    work = {"display_name": "Neural machine translation of rare words with subword units", "publication_year": 2016}
    assert enrich.title_matches("Neural Machine Translation of Rare Words with Subword Units", "2015", work)
    assert enrich.title_matches("Neural Machine Translation of Rare Word with Subword Units", "2015", work)  # fuzzy
    assert not enrich.title_matches("Neural Machine Translation of Rare Words with Subword Units", "2013", work)
    assert not enrich.title_matches("Statistical Machine Translation of Rare Phrases", "2015", work)
    assert not enrich.title_matches("Subword Units", "2015", work)  # a subset of the words is not a match


def test_abstract_from_inverted_index():
    assert enrich.abstract({"tokens": [1, 4], "Byte": [0], "are": [2], "not": [3]}) == "Byte tokens are not tokens"
    assert enrich.abstract(None) == ""


def test_fields_from_real_responses():
    bpe, ieee, aaai = results("1508.07909")
    f = enrich.fields(bpe)
    assert f["openalex_id"] == "W1816313093"
    assert f["doi"] == "10.18653/v1/p16-1162"  # not the 10.48550/arxiv DOI the work also lists
    assert f["venue"] == "Annual Meeting of the Association for Computational Linguistics (ACL)"
    assert f["published_version_url"] == "https://www.aclweb.org/anthology/P16-1162.pdf"  # ACL PDF wins
    assert f["license"] == "cc-by"
    assert enrich.fields(aaai)["published_version_url"] == "https://doi.org/10.1609/aaai.v34i05.6451"  # landing
    blt_acl, blt_arxiv, _ = results("2412.09871")
    assert enrich.fields(blt_acl)["published_version_url"] == "https://aclanthology.org/2025.acl-long.453.pdf"
    f = enrich.fields(blt_arxiv)  # preprint-only record: no DOI, no publishedVersion
    assert (f["doi"], f["venue"], f["published_version_url"]) == ("", "arXiv (Cornell University)", "")
    _, gqa_arxiv, _ = results("2305.13245")
    assert enrich.fields(gqa_arxiv)["doi"] == "10.48550/arxiv.2305.13245"  # only DOI there is


def test_by_title_prefers_published_version(monkeypatch):
    monkeypatch.setattr(enrich, "get", lambda params, key: json.loads((FIX / f"openalex_{key}.json").read_text()))
    row = {"arxiv_id": "2305.13245", "year": "2023",
           "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"}
    assert enrich.by_title(row)["id"].endswith("W4389518760")  # the EMNLP work, ranked first; the arXiv one also matches


def test_acl_record_skips_arxiv_and_builds_urls():
    work = {"id": "https://openalex.org/W1", "display_name": "T", "publication_year": 2020,
            "authorships": [{"author": {"display_name": "A B"}}],
            "abstract_inverted_index": {"x": [0]}, "primary_location": {"source": None},
            "locations": [{"landing_page_url": "https://www.aclweb.org/anthology/2020.udw-1.9.pdf", "pdf_url": None}]}
    r = enrich.acl_record(work)
    assert (r["published_version_url"], r["fulltext_url"]) == ("https://www.aclweb.org/anthology/2020.udw-1.9",
                                                               "https://www.aclweb.org/anthology/2020.udw-1.9.pdf")
    assert (r["record_id"], r["published"], r["authors"], r["abstract"]) == ("oa:W1", "2020-01-01", ["A B"], "x")
    work["locations"].append({"landing_page_url": "http://arxiv.org/abs/2001.00001", "pdf_url": None})
    assert enrich.acl_record(work) is None


def test_link_updates_manifest_and_keeps_other_columns(monkeypatch, tmp_path):
    def fake_get(params, key):
        path = FIX / f"openalex_{key}.json"
        return json.loads(path.read_text()) if path.exists() else {"results": []}
    monkeypatch.setattr(enrich, "get", fake_get)
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "manifest.csv")
    manifest.save([
        {"record_id": "arxiv:1508.07909", "arxiv_id": "1508.07909", "year": "2015", "in_scope": "true",
         "title": "Neural Machine Translation of Rare Words with Subword Units", "topic": "bpe", "n_chunks": "7",
         "license": "arxiv-nonexclusive", "fulltext_status": "ok"},
        {"record_id": "arxiv:2001.00001", "arxiv_id": "2001.00001", "year": "2020", "in_scope": "true",
         "title": "Nothing matches this", "topic": "other"},
        {"record_id": "arxiv:2001.00002", "arxiv_id": "2001.00002", "year": "2020", "in_scope": "false",
         "title": "Out of scope"},
    ])
    enrich.link()
    with open(config.MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    bpe, miss, out = rows
    assert (bpe["openalex_id"], bpe["doi"], bpe["license"]) == ("W1816313093", "10.18653/v1/p16-1162", "cc-by")
    assert (bpe["topic"], bpe["n_chunks"], bpe["fulltext_status"], bpe["year"]) == ("bpe", "7", "ok", "2015")
    assert (miss["openalex_id"], miss["topic"]) == ("", "other")
    assert out["openalex_id"] == ""
