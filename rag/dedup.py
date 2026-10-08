"""Decide which manifest rows are the same work and pick one canonical document per work.

Decisions and why:
- Only in-scope rows take part; out-of-scope rows are never touched.
- Identifiers first (union-find): same arxiv_id, openalex_id, non-arXiv DOI (10.48550/arxiv.* DOIs are
  arXiv's own) or a duplicate_of link. These are exact, so they always merge.
- Fuzzy second, between main documents (source arxiv/acl) of different groups: normalised titles with
  token_set_ratio >= CANDIDATE_RATIO and years at most YEAR_GAP apart make a candidate pair. A pair
  merges when title_ratio >= TITLE_RATIO, surname Jaccard >= AUTHOR_JACCARD and the bge cosine of
  "title. abstract" >= ABSTRACT_COS (or no abstract). If both parsed bodies exist and the shorter is
  under BODY_RELATED of the longer, the pair is "related", not merged: an extended version with new
  experiments is kept as its own work. title_ratio is the token_set_ratio of the candidate step, except
  that when one title is under 60% of the other's length the token_sort_ratio is used (a subset
  of the words scores 100 on the set ratio).
- work_id: the (smallest) arXiv id of an arXiv main row in the group, else the smallest record_id.
- 20 hard pairs (title_ratio 70-95 or author Jaccard 0.3-0.8, sampled with config.SEED, topped up
  with the highest title ratios) go to dup_pairs_to_label.csv next to the manifest, so a temp-dir
  manifest never touches data/. Never overwritten: it may hold hand labels. `sweep` scores the
  threshold grid on those labels plus silver positives (rows sharing an openalex_id).
- Canonical: arXiv main with full text (html/ar5iv over pdf, then latest version), else the ACL-only
  row with full text, else the arXiv main without full text. Only canonical rows with full text are
  indexed; "corpus_cap" is kept.
"""
import argparse
import csv
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from rag import collect, config, index, manifest

TITLE_RATIO = 92
AUTHOR_JACCARD = 0.5
ABSTRACT_COS = 0.85
BODY_RELATED = 0.6
CANDIDATE_RATIO = 80
YEAR_GAP = 2
N_LABEL = 20
LABEL_FILE = "dup_pairs_to_label.csv"  # next to the manifest: data/ for the real one
FULLTEXT = ("html", "ar5iv", "pdf")
FEATURES = ["title_ratio", "author_jaccard", "abstract_cosine", "body_ratio"]
LABEL_COLUMNS = ["pair_id", "record_a", "title_a", "year_a", "authors_a", "record_b", "title_b", "year_b",
                 "authors_b", *FEATURES, "rule_says", "human_same_work"]
CANDIDATE_COLUMNS = ["record_a", "record_b", "work_a", "work_b", "title_a", "title_b", *FEATURES, "decision"]
RELATED_COLUMNS = ["work_a", "work_b", "title_a", "title_b", "body_ratio"]
GRID = list(itertools.product([80, 85, 88, 90, 92, 95, 98], [0.3, 0.4, 0.5, 0.6, 0.7], [0.80, 0.85, 0.90]))


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(parent, i, j):
    parent[find(parent, i)] = find(parent, j)


def id_groups(rows):
    """Union-find parents over rows joined by shared identifiers and duplicate_of links."""
    parent, first = list(range(len(rows))), {}
    for i, r in enumerate(rows):
        doi = r["doi"].lower()
        keys = [("arxiv", r["arxiv_id"]), ("oa", r["openalex_id"]), ("rec", r["record_id"]),
                ("rec", r["duplicate_of"]), ("doi", "" if doi.startswith("10.48550/arxiv.") else doi)]
        for key in keys:
            if key[1]:
                union(parent, i, first.setdefault(key, i))
    return parent


def norm_title(title):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def surnames(row):
    return {a.split()[-1].lower() for a in row["authors"].split(";") if a.strip()}


def body_words(row):
    """body_words of the row's parsed document, 0 if it was not parsed."""
    doc_id = f"arxiv:{row['arxiv_id']}v{row['arxiv_latest_version']}" if row["source"] == "arxiv" else row["record_id"]
    path = config.parsed_path(doc_id)
    return json.loads(path.read_text())["body_words"] if path.exists() else 0


def embed(rows):
    """record_id -> unit bge vector of "title. abstract", for the rows whose abstract we have."""
    abstracts = {c["record_id"]: c.get("abstract", "") for c in collect.load_candidates()}
    have = list({r["record_id"]: r for r in rows if abstracts.get(r["record_id"])}.values())
    if not have:
        return {}
    vectors = index.encode_texts([f"{r['title']}. {abstracts[r['record_id']]}" for r in have], "bge", device="cpu")
    return {r["record_id"]: v for r, v in zip(have, vectors)}


def features(a, b, vectors):
    sa, sb = surnames(a), surnames(b)
    wa, wb = body_words(a), body_words(b)
    va, vb = vectors.get(a["record_id"]), vectors.get(b["record_id"])
    ta, tb = norm_title(a["title"]), norm_title(b["title"])
    ratio = fuzz.token_set_ratio(ta, tb)
    if min(len(ta), len(tb)) < 0.6 * max(len(ta), len(tb)):  # a subset scores 100 on the set ratio
        ratio = min(ratio, fuzz.token_sort_ratio(ta, tb))
    return {"title_ratio": round(ratio, 1),
            "author_jaccard": round(len(sa & sb) / len(sa | sb), 3) if sa and sb else 0.0,
            "abstract_cosine": "" if va is None or vb is None else round(float(va @ vb), 4),
            "body_ratio": round(min(wa, wb) / max(wa, wb), 3) if wa and wb else ""}


def decide(f, title_ratio=TITLE_RATIO, author_jaccard=AUTHOR_JACCARD, abstract_cos=ABSTRACT_COS):
    cos = f["abstract_cosine"]
    if f["title_ratio"] < title_ratio or f["author_jaccard"] < author_jaccard or (cos != "" and cos < abstract_cos):
        return "separate"
    return "related" if f["body_ratio"] != "" and f["body_ratio"] < BODY_RELATED else "merge"


def candidate_pairs(rows, parent):
    """(i, j) of main documents in different groups with similar titles and close years."""
    mains = [i for i, r in enumerate(rows) if r["source"] in ("arxiv", "acl")]
    titles = [norm_title(rows[i]["title"]) for i in mains]
    scores = process.cdist(titles, titles, scorer=fuzz.token_set_ratio, score_cutoff=CANDIDATE_RATIO, workers=-1)
    pairs = [(mains[a], mains[b]) for a, b in zip(*np.nonzero(np.triu(scores, 1)))]
    return [(i, j) for i, j in pairs
            if find(parent, i) != find(parent, j) and abs(int(rows[i]["year"] or 0) - int(rows[j]["year"] or 0)) <= YEAR_GAP]


def build(rows, thresholds):
    """Group the in-scope rows, write work_id. Returns (groups, scored pairs, number of identifier groups)."""
    scope = [r for r in rows if manifest.is_true(r["in_scope"])]
    parent = id_groups(scope)
    n_id = len({find(parent, i) for i in range(len(scope))})
    cands = candidate_pairs(scope, parent)
    vectors = embed([scope[k] for pair in cands for k in pair])
    pairs = []
    for i, j in cands:
        f = features(scope[i], scope[j], vectors)
        f["decision"] = decide(f, *thresholds)
        if f["decision"] == "merge":
            union(parent, i, j)
        pairs.append({"a": scope[i], "b": scope[j], **f})
    members = defaultdict(list)
    for i, r in enumerate(scope):
        members[find(parent, i)].append(r)
    for group in members.values():
        arxiv = sorted(r["arxiv_id"] for r in group if r["source"] == "arxiv" and r["arxiv_id"])
        for r in group:
            r["work_id"] = arxiv[0] if arxiv else min(g["record_id"] for g in group)
    return list(members.values()), pairs, n_id


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def pick_label_pairs(pairs):
    hard = [k for k, p in enumerate(pairs) if 70 <= p["title_ratio"] <= 95 or 0.3 <= p["author_jaccard"] <= 0.8]
    picked = random.Random(config.SEED).sample(hard, min(N_LABEL, len(hard)))
    rest = sorted(set(range(len(pairs))) - set(picked), key=lambda k: -pairs[k]["title_ratio"])
    return [pairs[k] for k in picked + rest[:N_LABEL - len(picked)]]


def label_row(n, p):
    side = {f"{c}_{s}": p[s][c] for s in "ab" for c in ("title", "year", "authors")}
    return {"pair_id": f"p{n:02d}", "record_a": p["a"]["record_id"], "record_b": p["b"]["record_id"], **side,
            **{c: p[c] for c in FEATURES}, "rule_says": p["decision"], "human_same_work": ""}


def group(thresholds, report):
    rows = manifest.load()
    groups, pairs, n_id = build(rows, thresholds)
    flat = [{"record_a": p["a"]["record_id"], "record_b": p["b"]["record_id"], "work_a": p["a"]["work_id"],
             "work_b": p["b"]["work_id"], "title_a": p["a"]["title"], "title_b": p["b"]["title"],
             **{c: p[c] for c in FEATURES}, "decision": p["decision"]} for p in pairs]
    write_csv(config.RESULTS / "dedup_candidates.csv", CANDIDATE_COLUMNS, flat)
    write_csv(config.RESULTS / "dedup_related.csv", RELATED_COLUMNS,
              [{c: p[c] for c in RELATED_COLUMNS} for p in flat if p["decision"] == "related"])
    decisions = Counter(p["decision"] for p in pairs)
    print(f"in-scope rows {sum(len(g) for g in groups)}  groups by identifier {n_id}  candidate pairs "
          f"{len(pairs)}  merges {decisions['merge']}  related {decisions['related']}  works {len(groups)}")
    label_path, to_label = config.MANIFEST.parent / LABEL_FILE, pick_label_pairs(pairs)
    if label_path.exists():
        print(f"{label_path} exists (it may hold labels): not overwritten")
    else:
        write_csv(label_path, LABEL_COLUMNS, [label_row(n, p) for n, p in enumerate(to_label, 1)])
        print(f"wrote {len(to_label)} pairs to {label_path}")
    if report:
        for p in to_label:
            print(f"  {p['decision']:8} t={p['title_ratio']} a={p['author_jaccard']} c={p['abstract_cosine']} "
                  f"b={p['body_ratio']}  {p['a']['title'][:50]} | {p['b']['title'][:50]}")
    manifest.save(rows)


def labelled_examples(path):
    """{record pair: (features, same work)} from the hand-labelled rows of the labelling file."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_same_work"].strip().lower() in ("true", "false")]
    return {frozenset((r["record_a"], r["record_b"])): ({c: float(r[c]) if r[c] != "" else "" for c in FEATURES},
                                                       manifest.is_true(r["human_same_work"])) for r in rows}


def sweep(labels):
    by_oa = defaultdict(list)
    for r in manifest.load():
        if manifest.is_true(r["in_scope"]) and r["openalex_id"]:
            by_oa[r["openalex_id"]].append(r)
    silver = [(a, b) for g in by_oa.values() for a, b in itertools.combinations(g, 2)]
    vectors = embed([r for pair in silver for r in pair])
    examples = {frozenset((a["record_id"], b["record_id"])): (features(a, b, vectors), True) for a, b in silver}
    examples.update(labelled_examples(labels))
    n_pos = sum(same for _, same in examples.values())
    print(f"positives {n_pos} ({len(silver)} silver)  negatives {len(examples) - n_pos}")
    out = []
    for t, a, c in GRID:
        hits = Counter((decide(f, t, a, c) == "merge", same) for f, same in examples.values())
        tp, fp, fn = hits[True, True], hits[True, False], hits[False, True]
        out.append({"title_ratio": t, "author_jaccard": a, "abstract_cos": c, "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0})
    write_csv(config.RESULTS / "dedup_rules.csv", list(out[0]), out)
    best = max((r for r in out if r["precision"] == 1.0), default=None,
               key=lambda r: (r["recall"], r["title_ratio"], r["author_jaccard"], r["abstract_cos"]))
    print(f"precision 1.0 with the highest recall, strictest of ties: {best}")


def canon_key(r):
    full = r["fulltext_status"] in FULLTEXT
    tier = {("arxiv", True): 0, ("acl", True): 1, ("arxiv", False): 2}.get((r["source"], full), 3)
    return tier, r["fulltext_status"] not in ("html", "ar5iv"), -int(r["arxiv_latest_version"] or 0), r["record_id"]


def canonical(thresholds):
    rows = manifest.load()
    groups, _, _ = build(rows, thresholds)
    for g in groups:
        canon = min(g, key=canon_key)
        full = canon["fulltext_status"] in FULLTEXT
        for r in g:
            if r is canon:
                capped = r["not_indexed_reason"] == "corpus_cap"
                r.update(canonical="true", duplicate_of="", in_index=str(full and not capped).lower(),
                         not_indexed_reason="corpus_cap" if capped else "no_fulltext" if not full else "")
            else:
                keep = r["source"] in ("arxiv_v1", "acl_copy") and r["duplicate_of"]
                r.update(canonical="false", in_index="false", duplicate_of=keep or canon["record_id"],
                         not_indexed_reason=f"duplicate_of:{canon['record_id']}")
    per_work = Counter(r["work_id"] for g in groups for r in g if r["canonical"] == "true")
    assert len(per_work) == len(groups) and set(per_work.values()) == {1}, "not one canonical row per work"
    reasons = Counter(r["not_indexed_reason"].split(":")[0] for g in groups for r in g)
    print(f"works {len(groups)}  indexed {sum(r['in_index'] == 'true' for g in groups for r in g)}  "
          f"not_indexed_reason: {dict(reasons)}")
    manifest.save(rows)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=config.MANIFEST)
    rule = argparse.ArgumentParser(add_help=False)
    rule.add_argument("--title-ratio", type=float, default=TITLE_RATIO)
    rule.add_argument("--author-jaccard", type=float, default=AUTHOR_JACCARD)
    rule.add_argument("--abstract-cos", type=float, default=ABSTRACT_COS)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("group", parents=[common, rule]).add_argument("--report", action="store_true")
    sub.add_parser("sweep", parents=[common]).add_argument("--labels", type=Path)
    sub.add_parser("canonical", parents=[common, rule])
    a = p.parse_args()
    config.MANIFEST = a.manifest  # manifest.load/save read this path at call time
    if a.cmd == "sweep":
        return sweep(a.labels or a.manifest.parent / LABEL_FILE)
    thresholds = (a.title_ratio, a.author_jaccard, a.abstract_cos)
    group(thresholds, a.report) if a.cmd == "group" else canonical(thresholds)


if __name__ == "__main__":
    main()
