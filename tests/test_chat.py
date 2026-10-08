import pytest

from rag import chat, config, llm, search

PASSAGES = [
    {"chunk_id": "a#0", "work_id": "a", "title": "Paper A", "year": 2016, "venue": "ACL",
     "section": "3.2 Byte Pair Encoding (BPE)", "score": 0.9, "text": "BPE merges frequent pairs."},
    {"chunk_id": "b#0", "work_id": "b", "title": "Paper B", "year": 2018, "venue": "",
     "section": "Method", "score": 0.8, "text": "Unigram LM samples segmentations."},
    {"chunk_id": "c#0", "work_id": "c", "title": "Paper C", "year": 2020, "venue": "EMNLP",
     "section": "Results", "score": 0.7, "text": "BPE-dropout improves BLEU."},
]


@pytest.fixture
def stub(monkeypatch):
    """Queue replies with stub.replies; calls and retrieve queries are recorded."""
    rec = {"replies": [], "calls": [], "queries": []}

    def fake_chat(messages, model, schema=None, temperature=0.0, max_tokens=1024):
        rec["calls"].append({"messages": messages, "max_tokens": max_tokens})
        return rec["replies"].pop(0), 0.001

    def fake_retrieve(q, retriever=None, k=None, variant="canonical", collapse_mode="r1", want_survey=False):
        rec["queries"].append(q)
        return [dict(p) for p in PASSAGES]

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    return rec


def test_sources_list_has_only_cited_entries(stub):
    stub["replies"] = ["BPE merges frequent symbol pairs [1]. Dropout helps translation quality a lot [3]."]
    r = chat.answer([], "What is BPE?")
    assert r["cited"] == [1, 3]
    sources = r["text"].split("Sources:\n")[1].splitlines()
    assert sources == ["[1] Paper A (2016, ACL), § 3.2 Byte Pair Encoding (BPE)",
                       "[3] Paper C (2020, EMNLP), § Results"]


def test_invalid_citation_is_removed(stub):
    stub["replies"] = ["BPE merges frequent symbol pairs together [9]. It is greedy and fast overall [1]."]
    r = chat.answer([], "What is BPE?")
    assert r["invalid_citations"] == [9]
    assert "[9]" not in r["text"]
    assert r["text"].count("[citation removed]") == 1
    assert r["cited"] == [1]


def test_abstain_sentence_gives_no_sources(stub):
    stub["replies"] = [config.ABSTAIN_SENTENCE]
    r = chat.answer([], "What is the airspeed of a swallow?")
    assert r["abstained"] and not r["gate_abstained"]
    assert "Sources:" not in r["text"]


def test_rewrite_only_on_follow_up(stub):
    stub["replies"] = ["BPE merges frequent symbol pairs [1]."]
    first = chat.answer([], "What is BPE?")
    assert len(stub["calls"]) == 1 and stub["queries"] == ["What is BPE?"]
    history = [{"role": "user", "content": "What is BPE?"}, {"role": "assistant", "content": first["text"]}]
    stub["replies"] = ['"BPE-dropout regularization"', "BPE-dropout drops merges at random [3]."]
    r = chat.answer(history, "What about dropout for it?")
    assert len(stub["calls"]) == 3
    assert stub["queries"][-1] == r["rewrite"] == "BPE-dropout regularization"
    assert "Last message: What about dropout for it?" in stub["calls"][1]["messages"][1]["content"]
    assert stub["calls"][2]["messages"][1]["content"].endswith("Question: What about dropout for it?")


def test_score_gate_abstains_without_llm_call(stub, monkeypatch):
    monkeypatch.setattr(config, "ABSTAIN_DENSE_TAU", 0.9)
    monkeypatch.setattr(search, "dense", lambda q, model, k=50, variant="canonical": [("x", 0.2)])
    r = chat.answer([], "Unrelated question about cooking pasta?", retriever="hybrid-bge")
    assert r["gate_abstained"] and r["abstained"]
    assert r["text"] == config.ABSTAIN_SENTENCE
    assert stub["calls"] == []


def test_uncited_sentences(stub):
    stub["replies"] = ["BPE merges frequent symbol pairs [1]. Unigram LM samples many segmentations of words. "
                       "Do you want to know more about it?"]
    r = chat.answer([], "Compare BPE and Unigram.")
    assert r["uncited_sentences"] == ["Unigram LM samples many segmentations of words."]


def test_citation_after_full_stop_belongs_to_its_sentence():
    text, cited, invalid, uncited = chat.check_citations("BPE merges frequent pairs. [1] It is a greedy method. [2][3]", 3)
    assert text == "BPE merges frequent pairs [1]. It is a greedy method [2][3]."
    assert cited == [1, 2, 3] and invalid == [] and uncited == []
