import csv
import json
import sys
import types
from pathlib import Path

import pytest

import rag
from rag import judge, llm

FIX = Path(__file__).parent / "fixtures"
ANSWERS, JUDGMENTS, QUESTIONS = FIX / "answers_toy.jsonl", FIX / "judgments_toy.jsonl", FIX / "questions_judge.jsonl"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture
def fake_chat(monkeypatch):
    """Stub rag.chat (it may not exist yet); records (history, user_msg) per call."""
    calls = []

    def answer(history, user_msg, retriever, k, model):
        calls.append((history, user_msg))
        abstain = "GPT-7" in user_msg
        return {"text": f"Answer to: {user_msg} [1].\n\nSources:\n[1] T (2016, ACL)", "raw": "x", "rewrite": user_msg,
                "sources": [], "cited": [1], "invalid_citations": [], "uncited_sentences": [], "abstained": abstain,
                "gate_abstained": False, "timings": {"rewrite_s": 0.1, "retrieve_s": 0.2, "answer_s": 0.7},
                "cost_usd": 0.001}

    fake = types.ModuleType("rag.chat")
    fake.answer = answer
    monkeypatch.setitem(sys.modules, "rag.chat", fake)
    monkeypatch.setattr(rag, "chat", fake, raising=False)
    return calls


def test_run_passes_parent_answer_and_resumes(fake_chat, tmp_path):
    out = tmp_path / "answers.jsonl"
    rows = judge.run(QUESTIONS, "all", "r", 8, "m", out)
    assert [r["id"] for r in rows] == ["j1", "j2", "j3"]
    assert fake_chat[1][0] == [{"role": "user", "content": "What does byte pair encoding merge?"},
                               {"role": "assistant", "content": rows[0]["answer"]}]
    assert fake_chat[0][0] == fake_chat[2][0] == []
    assert rows[0]["latency_s"] == 1.0 and rows[2]["abstained"] and rows[1]["gold"][0]["work_id"] == "w9"
    fake_chat.clear()
    assert judge.run(QUESTIONS, "all", "r", 8, "m", out) == [] and fake_chat == []


def test_run_answers_parent_outside_split_without_recording_it(fake_chat, tmp_path):
    rows = judge.run(QUESTIONS, "test", "r", 8, "m", tmp_path / "answers.jsonl")
    assert [r["id"] for r in rows] == ["j2", "j3"]
    assert [msg for _, msg in fake_chat] == ["What does byte pair encoding merge?", "Who introduced it?",
                                              "Which tokenizer does GPT-7 use?"]
    assert fake_chat[1][0][1]["content"].startswith("Answer to: What does byte pair encoding merge?")


def test_judge_retries_once_then_marks_unjudged(monkeypatch, tmp_path):
    valid = (JUDGMENTS.read_text().splitlines()[0])  # any object with the judgment keys
    replies = iter(["not json", valid, "nope", "{bad"])
    calls = []

    def fake(messages, model, schema=None, temperature=0.0, max_tokens=1024):
        calls.append((messages, model, schema, temperature, max_tokens))
        return next(replies), 0.01

    monkeypatch.setattr(llm, "chat", fake)
    out = tmp_path / "judgments.jsonl"
    assert judge.judge(ANSWERS, out, "jm") == {"judged": 2, "unjudged": 1}  # j1 judged, j3 abstained, j2 unjudged
    rows = {r["id"]: r for r in judge.read_jsonl(out)}
    assert len(calls) == 4  # 2 for j1 (retry worked), 2 for j2, none for the abstention j3
    assert rows["j1"]["correctness"] == "correct" and rows["j1"]["cost_usd"] == 0.02
    assert rows["j2"] == {"id": "j2", "model": "m", "retriever": "r", "judge_model": "jm", "unjudged": True,
                          "cost_usd": 0.02}
    assert rows["j3"]["abstained"] and rows["j3"]["sentences"] == [] and rows["j3"]["correctness"] == "not_applicable"
    messages, model, schema, temperature, max_tokens = calls[0]
    assert (model, schema, temperature, max_tokens) == ("jm", judge.SCHEMA, 0.0, 1500)
    user = messages[1]["content"]
    assert "[1] Neural MT of Rare Words (2016, ACL), § 3 Method\nBPE iteratively merges" in user
    assert "Expected answer: The most frequent pair of adjacent symbols." in user
    assert user.endswith("It repeats until the vocabulary is full [2].")  # the answer's Sources list is cut
    # Rerun: only the unjudged answer is tried again.
    replies = iter([valid])
    calls.clear()
    judge.judge(ANSWERS, out, "jm")
    assert len(calls) == 1 and "Who introduced it?" in calls[0][0][1]["content"]


def test_metrics_by_hand(tmp_path):
    table, failures = judge.metrics(ANSWERS, JUDGMENTS, tmp_path / "q.csv", tmp_path / "f.csv")
    (row,) = table
    # Factual sentences: j1 S[1]->1, S[2]->2; j2 S[2]->1 (mis-cited), P (uncited)->2, U[7]->0.
    # The meta sentence "Let me know..." is not factual.
    assert row["n"] == 3
    assert row["fully_grounded_rate"] == 0.3333     # only j1: 1/3 answers
    assert row["sentence_support_rate"] == 0.6      # 3 supported / 5 factual
    assert row["partial_rate"] == 0.2               # 1 / 5
    assert row["citation_precision"] == 0.5         # citations [1] [2] [2] [7]: j1's two correct / 4
    assert row["citation_coverage"] == 0.8          # 4 of 5 factual sentences cite something
    assert (row["correctness_correct"], row["correctness_partial"]) == (0.5, 0.5)  # j1 correct, j2 partial
    assert (row["unanswerable_n"], row["abstained_on_unanswerable"], row["answered_unanswerable"]) == (1, 1, 0)
    assert (row["answerable_n"], row["abstained_on_answerable"], row["answered_answerable"]) == (2, 0, 2)
    assert (row["abstention_recall"], row["false_abstention_rate"], row["abstention_f1"]) == (1.0, 0.0, 1.0)  # 2*1/(2*1+0+0)
    assert (row["gate_abstentions"], row["invalid_citations_total"], row["unjudged"]) == (1, 1, 0)
    assert row["cost_per_answer"] == 0.0018         # (0.002 + 0.003 + 0.0005) / 3 = 0.00183
    assert row["p50_latency_s"] == 1.0              # median of 1.0, 2.0, 0.5
    assert [(f["id"], f["category"], f["evidence"]) for f in read_csv(tmp_path / "f.csv")] == [
        ("j2", "unsupported_sentence", "It was introduced in 1994 [7]."),
        ("j2", "invalid_citation", "[7]"),
        ("j2", "retrieval_miss", "w9")]
    assert read_csv(tmp_path / "q.csv")[0]["citation_precision"] == "0.5"


def test_abstention_confusion_adds_up(tmp_path):
    answers = judge.read_jsonl(ANSWERS)
    answers[0]["abstained"] = True    # answerable j1 abstains: fp
    answers[2]["abstained"] = False   # unanswerable j3 answers: fn
    path = tmp_path / "a.jsonl"
    path.write_text("".join(json.dumps(a) + "\n" for a in answers))
    table, failures = judge.metrics(path, JUDGMENTS, tmp_path / "q.csv", tmp_path / "f.csv")
    r = table[0]
    tp, fp = r["abstained_on_unanswerable"], r["abstained_on_answerable"]
    fn, tn = r["answered_unanswerable"], r["answered_answerable"]
    assert (tp, fp, fn, tn) == (0, 1, 1, 1) and tp + fp + fn + tn == r["n"]
    assert {f["category"] for f in failures} >= {"false_abstention", "missed_abstention"}


def test_agreement_kappa_and_shifted_alignment(tmp_path):
    def sent(text, verdict):
        return {"text": text, "factual": True, "cited": [], "verdict": verdict, "support_source": 0, "note": ""}

    judgments = [
        {"id": "a", "model": "m", "retriever": "r", "sentences": [
            sent("BPE merges pairs [1].", "supported"), sent("It stops at a vocabulary size [2].", "supported"),
            sent("The sources show this.", "unsupported")]},
        {"id": "b", "model": "m", "retriever": "r", "sentences": [  # an extra first sentence shifts the order
            sent("In short:", "unsupported"), sent("Fertility is tokens per word [1].", "partial"),
            sent("Finnish has higher fertility [2].", "unsupported"), sent("English is lowest [1].", "unsupported")]}]
    jpath = tmp_path / "j.jsonl"
    jpath.write_text("".join(json.dumps(j) + "\n" for j in judgments))
    labels = [("a", 1, "BPE merges pairs [1].", "supported"), ("a", 2, "It stops at a vocabulary size [2].", "supported"),
              ("a", 3, "The sources show this.", "-"), ("b", 1, "Fertility is tokens per word [1].", "partial"),
              ("b", 2, "Finnish has higher fertility [2].", "unsupported"), ("b", 3, "English is lowest [1].", "supported"),
              ("c", 1, "No judgment exists for this answer.", "supported")]
    lpath = tmp_path / "labels.csv"
    judge.write_csv(lpath, [{"answer_id": i, "model": "m", "retriever": "r", "sentence_no": n, "sentence": s,
                             "cited": "", "sources_shown": "", "human_verdict": v} for i, n, s, v in labels],
                    judge.LABEL_FIELDS)
    out = tmp_path / "agreement.txt"
    res = judge.agreement(lpath, jpath, out)
    # human S S P U S vs judge S S P U U: 4/5 agree. 3-class: pe = (3*2 + 1*1 + 1*2) / 25 = 0.36,
    # kappa = (0.8 - 0.36) / 0.64 = 0.6875. Binary: human S S N N S, judge S S N N N: pe = (3*2 + 2*3) / 25 = 0.48,
    # kappa = (0.8 - 0.48) / 0.52 = 0.6154.
    assert res == {"n": 5, "unaligned": 1, "accuracy3": 0.8, "kappa3": 0.6875, "accuracy2": 0.8, "kappa2": 0.6154}
    assert "kappa3=0.6875" in out.read_text()


def test_label_sheet_writes_sentences_and_refuses_overwrite(tmp_path):
    out = tmp_path / "labels.csv"
    judge.label_sheet(ANSWERS, 20, out)
    rows = read_csv(out)
    assert [(r["answer_id"], r["sentence_no"]) for r in rows if r["answer_id"] == "j2"] == [("j2", str(i)) for i in range(1, 5)]
    assert len(rows) == 6 and {r["human_verdict"] for r in rows} == {""}  # j1: 2 sentences, j2: 4, j3 abstained
    j2 = [r for r in rows if r["answer_id"] == "j2"]
    assert (j2[2]["sentence"], j2[2]["cited"]) == ("It was introduced in 1994 [7].", "[7]")
    assert j2[0]["sources_shown"] == "[1] Neural MT of Rare Words; [2] Subword Survey"
    with pytest.raises(SystemExit):
        judge.label_sheet(ANSWERS, 20, out)
