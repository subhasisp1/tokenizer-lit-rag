import csv
import json

import pytest

from rag import collect, config, manifest

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>1234</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/1508.07909v5</id>
    <title>Neural Machine Translation of Rare Words
      with Subword Units</title>
    <updated>2016-06-10T14:45:08Z</updated>
    <summary>  Neural machine translation models
 typically operate with a fixed vocabulary.</summary>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <published>2015-08-31T16:37:31Z</published>
    <arxiv:doi>10.18653/v1/P16-1162</arxiv:doi>
    <arxiv:journal_ref>ACL 2016</arxiv:journal_ref>
    <arxiv:primary_category term="cs.CL"/>
    <author><name>Rico Sennrich</name></author>
    <author><name>Barry Haddow</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0501001v1</id>
    <title>An old-style identifier</title>
    <updated>2005-01-01T03:14:14Z</updated>
    <summary>Abstract.</summary>
    <category term="cs.CR" scheme="http://arxiv.org/schemas/atom"/>
    <published>2005-01-01T03:14:14Z</published>
    <arxiv:primary_category term="cs.CR"/>
    <author><name>Michael Treaster</name></author>
  </entry>
</feed>"""


def test_parse_feed():
    total, (new, old) = collect.parse_feed(FEED)
    assert total == 1234
    assert new["record_id"] == "arxiv:1508.07909" and new["arxiv_latest_version"] == 5
    assert new["title"] == "Neural Machine Translation of Rare Words with Subword Units"
    assert new["abstract"] == "Neural machine translation models typically operate with a fixed vocabulary."
    assert new["authors"] == ["Rico Sennrich", "Barry Haddow"]
    assert (new["year"], new["published"], new["updated"]) == (2015, "2015-08-31", "2016-06-10")
    assert new["categories"] == ["cs.CL", "cs.LG"]
    assert (new["doi"], new["venue"]) == ("10.18653/v1/P16-1162", "ACL 2016")
    assert new["source_url"] == "https://arxiv.org/abs/1508.07909"
    assert old["arxiv_id"] == "cs/0501001" and old["arxiv_latest_version"] == 1
    assert old["source_url"] == "https://arxiv.org/abs/cs/0501001"


@pytest.mark.parametrize("title, abstract, result", [
    ("Neural Machine Translation of Rare Words with Subword Units", "We translate.", "include"),
    ("Language Models are Multilingual", "We study the fertility of languages.", "include"),
    ("A Better Language Model", "We use a tokenizer. The tokenizer is BPE.", "include"),
    ("A Better Language Model", "Text is tokenized and then tokenization is analysed.", "include"),
    ("A Better Language Model", "We train with a standard tokenizer.", "reject:incidental_mention"),
    ("Image Tokenizers for Diffusion", "A visual tokenizer with BPE.", "reject:other_modality"),
    ("SMILES Tokenization for Molecules", "Tokenization of molecules.", "reject:other_modality"),
    ("Token Merging for Fast Transformers", "Subword tokens are merged.", "reject:pruning_or_compression"),
    ("Efficient KV-Cache Eviction", "We study the tokenizer.", "reject:pruning_or_compression"),
    ("Speeding up inference", "We prune the KV cache of a speech model.", "reject:incidental_mention"),
])
def test_stage1_rule(title, abstract, result):
    assert collect.stage1_rule(title, abstract)[0] == result


def candidate(i, stage1="include", **extra):
    return {"record_id": f"arxiv:2401.0000{i}", "source": "arxiv", "arxiv_id": f"2401.0000{i}",
            "arxiv_latest_version": 1, "title": f"Paper {i}", "abstract": "About BPE.", "authors": ["A", "B"],
            "year": 2024, "categories": ["cs.CL"], "doi": "", "venue": "", "published_version_url": "",
            "fulltext_url": "", "license": "", "source_url": f"https://arxiv.org/abs/2401.0000{i}",
            "matched_queries": ["title", "seed"], "stage1_result": stage1, **extra}


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "CANDIDATE_FILES", [tmp_path / "arxiv.jsonl", tmp_path / "acl.jsonl"])
    monkeypatch.setattr(collect, "SAMPLE", tmp_path / "sample.csv")
    monkeypatch.setattr(config, "LABELS", tmp_path / "labels.jsonl")
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(config, "DATA", tmp_path)
    return tmp_path


REPLY = json.dumps({"in_scope": True, "topic": "subword_algorithms", "paper_type": "primary",
                    "out_of_scope_reason": "none", "reason": "It is about BPE."})


def test_classify_cache_and_errors(tmp_data, monkeypatch):
    collect.write_jsonl(collect.CANDIDATE_FILES[0], [candidate(i) for i in range(5)] + [candidate(9, "reject:other")])
    calls = []

    def stub(messages, model, schema=None, temperature=0.0, max_tokens=1024):
        calls.append(messages)
        assert schema is collect.SCHEMA and max_tokens == 300 and "In scope" in messages[0]["content"]
        return REPLY, 0.0001

    monkeypatch.setattr(collect.llm, "chat", stub)
    n, failures, cost = collect.classify()
    assert (n, failures, round(cost, 6)) == (5, 0, 0.0005)
    labels = collect.load_labels()
    assert len(labels) == 5 and labels["arxiv:2401.00000"]["prompt_version"] == collect.PROMPT_VERSION
    assert collect.classify() == (0, 0, 0.0)  # cached: no calls

    collect.write_jsonl(collect.CANDIDATE_FILES[1], [candidate(7) | {"record_id": "oa:W7", "source": "acl"}])
    monkeypatch.setattr(collect.llm, "chat", lambda *a, **k: ("not json", 0.0))
    assert collect.classify() == (2, 1, 0.0)  # retried once, then recorded as an error
    assert collect.load_labels()["oa:W7"]["out_of_scope_reason"] == "error"


def test_manifest_build(tmp_data):
    collect.write_jsonl(collect.CANDIDATE_FILES[0], [candidate(1), candidate(2), candidate(3, "reject:other_modality")])
    collect.write_jsonl(collect.CANDIDATE_FILES[1], [candidate(4) | {"record_id": "oa:W4", "source": "acl", "arxiv_id": ""}])
    collect.write_jsonl(config.LABELS, [
        {"record_id": "arxiv:2401.00001", **json.loads(REPLY)},
        {"record_id": "arxiv:2401.00002", "in_scope": False, "topic": "out_of_scope", "paper_type": "other",
         "out_of_scope_reason": "incidental_mention", "reason": "x"}])
    manifest.save([{"record_id": "arxiv:2401.00001", "fulltext_status": "ok", "doi": "10.1/x", "n_chunks": "7"}])
    collect.build_manifest()
    rows = {r["record_id"]: r for r in manifest.load()}
    assert len(rows) == 4
    one = rows["arxiv:2401.00001"]
    assert (one["in_scope"], one["topic"], one["reject_reason"]) == ("true", "subword_algorithms", "")
    assert (one["fulltext_status"], one["doi"], one["n_chunks"]) == ("ok", "10.1/x", "7")  # preserved
    assert (one["authors"], one["matched_queries"], one["work_id"]) == ("A; B", "title|seed", "2401.00001")
    assert (rows["arxiv:2401.00002"]["in_scope"], rows["arxiv:2401.00002"]["reject_reason"]) == ("false", "incidental_mention")
    assert (rows["arxiv:2401.00003"]["in_scope"], rows["arxiv:2401.00003"]["reject_reason"]) == ("false", "other_modality")
    four = rows["oa:W4"]
    assert (four["in_scope"], four["reject_reason"], four["work_id"]) == ("false", "", "oa:W4")  # unlabelled
    collect.summary()


def test_agreement(tmp_data, capsys):
    with open(collect.SAMPLE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "split", "title", "abstract", "llm_in_scope", "human_in_scope"])
        for i, (llm_v, human) in enumerate([("true", "true")] * 6 + [("true", "false"), ("false", "true"), ("false", "false")]):
            w.writerow([f"r{i}", "dev", f"T{i}", "", llm_v, human])
        w.writerow(["r9", "test", "T9", "", "true", ""])  # not yet labelled
    collect.agreement()
    out = capsys.readouterr().out
    assert "dev: n=9 precision=0.857 recall=0.857 F1=0.857 accuracy=0.778" in out
    assert "disagree: r6" in out and "disagree: r7" in out
    assert "test: no labelled rows" in out
    lo, hi = collect.wilson(7, 9)
    assert 0.44 < lo < 0.46 and 0.93 < hi < 0.95


def test_sample(tmp_data):
    collect.write_jsonl(collect.CANDIDATE_FILES[0], [candidate(i) for i in range(6)] + [candidate(8, "reject:other")])
    collect.write_jsonl(config.LABELS, [{"record_id": "arxiv:2401.00001", **json.loads(REPLY)}])
    collect.sample(4, 13)
    rows = list(csv.DictReader(open(collect.SAMPLE, newline="")))
    assert [r["split"] for r in rows] == ["dev", "dev", "test", "test"]
    assert "arxiv:2401.00008" not in {r["record_id"] for r in rows}
    assert all(r["llm_in_scope"] == ("true" if r["record_id"] == "arxiv:2401.00001" else "") for r in rows)
    assert all(r["human_in_scope"] == "" for r in rows)
    first = collect.SAMPLE.read_text()
    collect.sample(4, 99)  # refuses to overwrite hand labels
    assert collect.SAMPLE.read_text() == first
