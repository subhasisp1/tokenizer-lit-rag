"""Build the candidate pool from the arXiv API, filter it in two stages, and write data/manifest.csv.

Decisions and why:
- Plain requests + feedparser, one request per ARXIV_API_DELAY s on one session (arXiv terms: 1 per 3 s).
- Four query families x one submittedDate window per year keep every result set small enough to page
  through; each (family, window) is cached as one JSON file, so a rerun fetches only what is missing.
  The unwindowed totalResults per family checks that the windows add up.
- An empty page while totalResults says more exist is a known transient API fault: retried with
  5..80 s backoff, then the run fails rather than caching a short window.
- Stage 1 is a deterministic regex filter (title exclusions first; one "tokenizer" in the abstract is
  not enough). Stage 2 is one strict-JSON LLM call per stage-1 include with docs/boundary.md as the
  system prompt, cached in data/relevance_labels.jsonl; a failed call is retried once, then stored as
  out of scope with reason "error".
- The manifest keeps the columns later stages fill, and never blanks doi, venue, urls or license that
  a later stage found.
"""
import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import feedparser
import requests

from rag import config, llm, manifest

API = "https://export.arxiv.org/api/query"
UA = "tokenizer-lit-rag/0.1 (+https://github.com/subhasisp1/tokenizer-lit-rag)"
ARXIV_DIR = config.CACHE / "arxiv"
CANDIDATE_FILES = [ARXIV_DIR / "candidates.jsonl", config.CACHE / "acl" / "candidates.jsonl"]
SAMPLE = config.DATA / "relevance_sample.csv"
BOUNDARY = config.DOCS / "boundary.md"
PROMPT_VERSION = "v3"  # v1 accepted corpus-statistics papers; v2 added a decision rule; v3 qualified "character-level models"

SEEDS = ["1508.07909", "1609.08144", "1804.10959", "1808.06226", "1910.13267", "2004.03720",
         "2012.15613", "2103.06874", "2105.13626", "2106.12672", "2112.10508", "2305.07185",
         "2305.13707", "2305.15425", "2306.16842", "2310.08754", "2312.02598", "2402.01035",
         "2402.14903", "2402.18376", "2404.14408", "2405.05417", "2405.07883", "2407.13623",
         "2412.09871", "2503.13423"]
CATS = "(cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.SE OR cat:cs.IR OR cat:stat.ML)"
FAMILIES = {  # key: (name used in matched_queries and cache files, query)
    "F1": ("title", '(ti:tokenizer OR ti:tokenizers OR ti:tokenization OR ti:tokenisation OR ti:tokenizing'
           ' OR ti:subword OR ti:subwords OR ti:BPE OR ti:WordPiece OR ti:SentencePiece OR ti:"byte pair"'
           ' OR ti:"byte level" OR ti:"character level" OR ti:"tokenizer free" OR ti:"token free"'
           ' OR ti:vocabulary OR ti:vocabularies) AND ' + CATS),
    "F2": ("abs_specific", '(abs:"byte pair encoding" OR abs:SentencePiece OR abs:WordPiece'
           ' OR abs:"subword tokenization" OR abs:"subword segmentation" OR abs:"subword units"'
           ' OR abs:"subword vocabulary" OR abs:"vocabulary size" OR abs:fertility OR abs:"byte level"'
           ' OR abs:"character level" OR abs:"tokenizer free" OR abs:"token free"'
           ' OR abs:"unigram language model" OR abs:"glitch tokens") AND ' + CATS),
    "F3": ("abs_generic", "(abs:tokenizer OR abs:tokenizers OR abs:tokenization OR abs:tokenisation)"
           " AND cat:cs.CL"),
    "F4": ("byte_char", '(abs:byte OR abs:bytes OR abs:"byte level" OR abs:"character level") AND'
           ' (abs:"language model" OR abs:"language models" OR abs:"language modeling"'
           ' OR abs:"machine translation" OR abs:transformer) AND cat:cs.CL'),
}

INCLUDE = {name: re.compile(p, re.I) for name, p in {
    "tokeniz": r"\btokeni[sz](?:er|ers|ation|ations|e|ed|es|ing)?\b",
    "bpe": r"\bbpe\b|byte[- ]pair",
    "sentencepiece": r"sentence[- ]?piece",
    "wordpiece": r"word[- ]?piece",
    "unigram_lm": r"unigram (?:language model|lm)\b",
    "subword": r"\bsub-?words?\b",
    "byte_level": r"\bbyte[- ]level\b",
    "char_level": r"\bcharacter[- ]level\b",
    "tokenizer_free": r"\btoken(?:izer|ization)?[- ]free\b",
    "vocab_size": r"vocabulary size|vocab size|size of the vocabulary",
    "fertility": r"\bfertility\b",
    "vocab_adapt": r"vocab(?:ulary)? (?:expansion|extension|adaptation|transfer|pruning|trimming|replacement)",
    "glitch": r"glitch tokens?",
}.items()}
TITLE_EXCLUDE = {name: re.compile(p, re.I) for name, p in {
    "other_modality": r"\b(image|images|visual|vision|video|videos|speech|audio|acoustic|vq|vector[- ]quanti\w*"
                      r"|molecul\w*|smiles|protein\w*|genom\w*|dna|rna|music|graph|time[- ]series|point cloud"
                      r"|action|blockchain|crypto\w*|asset|3d|motion|robot\w*|remote sensing|medical imag\w*)\b",
    "pruning_or_compression": r"\b(token (?:pruning|merging|reduction|dropping|compression)|kv[- ]cache"
                              r"|prompt compression)\b",
}.items()}

TOPICS = ["subword_algorithms", "byte_or_character_level", "vocabulary_size_and_scaling",
          "tokenizer_transfer_or_adaptation", "multilingual_fairness_fertility",
          "tokenization_and_model_capabilities", "glitch_tokens_and_security", "theory_and_evaluation",
          "survey", "other_in_scope", "out_of_scope"]
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["in_scope", "topic", "paper_type", "out_of_scope_reason", "reason"],
          "properties": {
              "in_scope": {"type": "boolean"},
              "topic": {"type": "string", "enum": TOPICS},
              "paper_type": {"type": "string", "enum": ["primary", "survey", "position", "resource", "other"]},
              "out_of_scope_reason": {"type": "string", "enum": [
                  "none", "other_modality", "pruning_or_compression", "non_ml_tokenization",
                  "incidental_mention", "other"]},
              "reason": {"type": "string"}}}
INSTRUCTION = ("Decide whether the paper's main contribution or main analysis is within this boundary. "
               "Judge from the title and abstract only. Reply with JSON only.")


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_candidates():
    return [c for path in CANDIDATE_FILES for c in read_jsonl(path)]


def load_labels():
    return {r["record_id"]: r for r in read_jsonl(config.LABELS)}


def parse_feed(text):
    """Return (totalResults, candidate records) for one Atom page."""
    feed = feedparser.parse(text)
    records = []
    for e in feed.entries:
        if "/api/errors" in e.id:
            raise RuntimeError(f"arXiv API error: {e.get('summary', '')}")
        arxiv_id, version = re.search(r"abs/(.+)v(\d+)$", e.id).groups()
        primary = e.get("arxiv_primary_category", {}).get("term", "")
        doi = e.get("arxiv_doi", "")
        records.append({
            "record_id": f"arxiv:{arxiv_id}", "source": "arxiv", "arxiv_id": arxiv_id,
            "arxiv_latest_version": int(version), "title": " ".join(e.title.split()),
            "abstract": " ".join(e.summary.split()), "authors": [a["name"] for a in e.get("authors", [])],
            "year": int(e.published[:4]), "published": e.published[:10], "updated": e.updated[:10],
            "categories": [primary] + [t["term"] for t in e.get("tags", []) if t["term"] != primary],
            "doi": doi, "venue": e.get("arxiv_journal_ref", ""),
            "published_version_url": f"https://doi.org/{doi}" if doi else "", "fulltext_url": "",
            "license": "", "source_url": f"https://arxiv.org/abs/{arxiv_id}", "matched_queries": []})
    return int(feed.feed.get("opensearch_totalresults", 0)), records


SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA
STATE = {"last": 0.0, "requests": 0}


def api_get(params, label):
    """One API page, paced to one request per ARXIV_API_DELAY, retried on errors and empty pages."""
    start = params.get("start", 0)
    for wait in [5, 10, 20, 40, 80, None]:
        time.sleep(max(0.0, STATE["last"] + config.ARXIV_API_DELAY - time.time()))
        STATE["last"] = time.time()
        STATE["requests"] += 1
        print(f"{datetime.now():%H:%M:%S.%f}"[:-3], f"GET {label} start={start}", flush=True)
        try:
            r = SESSION.get(API, params=params, timeout=60)
            if r.status_code == 200:
                total, records = parse_feed(r.text)
                if records or start >= total:
                    return total, records
                problem = f"empty page, totalResults={total}"
            else:
                problem = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            problem = repr(e)
        if wait is None:
            raise RuntimeError(f"{label} start={start}: {problem} after 5 retries")
        print(f"  {problem}; retrying in {wait} s", flush=True)
        time.sleep(wait)


def windows():
    today = date.today()
    out = [("pre2015", "199101010000", "201412312359")]
    out += [(str(y), f"{y}01010000", f"{y}12312359") for y in range(2015, today.year + 1)]
    out[-1] = (out[-1][0], out[-1][1], f"{today:%Y%m%d}2359")
    return out


def fetch(name, window, params):
    """Fetch every page of one query into ARXIV_DIR/<name>_<window>.json; skip if already cached."""
    path = ARXIV_DIR / f"{name}_{window}.json"
    if path.exists():
        return json.loads(path.read_text())
    records, total = [], None
    while total is None or len(records) < total:
        total, page = api_get({**params, "start": len(records)}, f"{name}_{window}")
        records += page
    data = {"family": name, "window": window, "params": params, "total": total, "records": records}
    path.write_text(json.dumps(data))
    return data


def merge():
    """Merge every cached result set by base id: union of matched_queries, highest version wins."""
    by_id = {}
    for path in sorted(ARXIV_DIR.glob("*_*.json")):
        data = json.loads(path.read_text())
        for r in data["records"]:
            old = by_id.get(r["arxiv_id"])
            queries = set(old["matched_queries"]) if old else set()
            if old is None or r["arxiv_latest_version"] > old["arxiv_latest_version"]:
                by_id[r["arxiv_id"]] = dict(r)
            by_id[r["arxiv_id"]]["matched_queries"] = sorted(queries | {data["family"]})
    return by_id


def collect_arxiv(families, years):
    began = time.time()
    ARXIV_DIR.mkdir(parents=True, exist_ok=True)
    if "seed" in families:
        fetch("seed", "all", {"id_list": ",".join(SEEDS), "max_results": len(SEEDS)})
    chosen = [w for w in windows() if not years or w[0] in years]
    for key in [f for f in families if f in FAMILIES]:
        name, query = FAMILIES[key]
        totals = [fetch(name, w, {"search_query": f"({query}) AND submittedDate:[{lo} TO {hi}]",
                                  "max_results": 500, "sortBy": "submittedDate", "sortOrder": "ascending"})["total"]
                  for w, lo, hi in chosen]
        if not years:
            unwindowed, _ = api_get({"search_query": query, "max_results": 1}, f"{name}_unwindowed")
            warn = "  WARNING: differ by more than 1%" if abs(sum(totals) - unwindowed) > 0.01 * unwindowed else ""
            print(f"{key} {name}: windows sum = {sum(totals)}, unwindowed = {unwindowed}{warn}")
    by_id = merge()
    write_jsonl(CANDIDATE_FILES[0], [by_id[k] for k in sorted(by_id)])
    cached = {p.stem: json.loads(p.read_text())["total"] for p in sorted(ARXIV_DIR.glob("*_*.json"))}
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "upper_bound": windows()[-1][2],
            "queries": {k: list(v) for k, v in FAMILIES.items()}, "total_results": cached}
    (ARXIV_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    found = [s for s in SEEDS if set(by_id.get(s, {}).get("matched_queries", [])) - {"seed"}]
    print(f"unique base ids: {len(by_id)}")
    print(f"seeds found by the families: {len(found)}/{len(SEEDS)}; misses: {sorted(set(SEEDS) - set(found))}")
    print(f"requests this run: {STATE['requests']}, elapsed {time.time() - began:.0f} s")


def stage1_rule(title, abstract):
    """Return (stage1_result, matched include rule names) for one title and abstract."""
    t, a = title.lower(), abstract.lower()
    rules = sorted(n for n, p in INCLUDE.items() if p.search(t) or p.search(a))
    for reason, pattern in TITLE_EXCLUDE.items():
        if pattern.search(t):
            return f"reject:{reason}", rules
    if (any(p.search(t) for p in INCLUDE.values())
            or any(p.search(a) for n, p in INCLUDE.items() if n != "tokeniz")
            or len(INCLUDE["tokeniz"].findall(a)) >= 2):
        return "include", rules
    return "reject:incidental_mention", rules


def stage1(report):
    for path in [p for p in CANDIDATE_FILES if p.exists()]:
        rows = read_jsonl(path)
        for c in rows:
            c["stage1_result"], c["stage1_rules"] = stage1_rule(c["title"], c["abstract"])
        write_jsonl(path, rows)
    rows = load_candidates()
    results = Counter(c["stage1_result"] for c in rows)
    print(f"candidates: {len(rows)}, include: {results['include']}")
    if report:
        print("include rules (among includes):", dict(Counter(
            r for c in rows if c["stage1_result"] == "include" for r in c["stage1_rules"]).most_common()))
        print("reject reasons:", {k: v for k, v in results.most_common() if k != "include"})
        rejects = [c for c in rows if c["stage1_result"] != "include"]
        for c in random.Random(config.SEED).sample(rejects, min(20, len(rejects))):
            print(f"{c['record_id']} | {c['stage1_result'][7:]} | {c['title']}")


def classify_one(c, system):
    """Return (label row, calls made, cost, failed) for one candidate."""
    user = f"Title: {c['title']}\nCategories: {', '.join(c.get('categories', []))}\nAbstract: {c['abstract']}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cost, calls = 0.0, 0
    for _ in range(2):
        calls += 1
        try:
            text, call_cost = llm.chat(messages, config.CLASSIFIER_MODEL, schema=SCHEMA, temperature=0, max_tokens=300)
            cost += call_cost
            out = llm.parse_json(text)
            label, failed = {k: out[k] for k in SCHEMA["required"]}, False
            break
        except Exception as e:  # rag.llm has already retried transport errors
            error = e
    else:
        label, failed = {"in_scope": False, "topic": "out_of_scope", "paper_type": "other",
                         "out_of_scope_reason": "error", "reason": f"error: {error}"[:300]}, True
    row = {"record_id": c["record_id"], **label, "model": config.CLASSIFIER_MODEL, "prompt_version": PROMPT_VERSION}
    return row, calls, cost, failed


def classify(limit=None, workers=8):
    done = load_labels()
    todo = [c for c in load_candidates() if c.get("stage1_result") == "include" and c["record_id"] not in done]
    todo = todo[:limit] if limit else todo
    system = BOUNDARY.read_text() + "\n" + INSTRUCTION
    calls = failures = 0
    cost = 0.0
    with ThreadPoolExecutor(workers) as pool, open(config.LABELS, "a") as out:
        for future in as_completed([pool.submit(classify_one, c, system) for c in todo]):
            row, n, c_usd, failed = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            calls, cost, failures = calls + n, cost + c_usd, failures + failed
    print(f"labelled {len(todo)} (already cached {len(done)}): {calls} calls, {failures} failures, ${cost:.4f}")
    return calls, failures, cost


def sample(n, seed):
    if SAMPLE.exists():
        print(f"{SAMPLE} exists (it may hold hand labels); delete it to draw a new sample")
        return
    pool = sorted((c for c in load_candidates() if c.get("stage1_result") == "include"), key=lambda c: c["record_id"])
    labels = load_labels()
    with open(SAMPLE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "split", "title", "abstract", "llm_in_scope", "human_in_scope"])
        for i, c in enumerate(random.Random(seed).sample(pool, min(n, len(pool)))):
            llm_v = str(labels[c["record_id"]]["in_scope"]).lower() if c["record_id"] in labels else ""
            w.writerow([c["record_id"], "dev" if i < n // 2 else "test", c["title"], c["abstract"], llm_v, ""])
    print(f"wrote {SAMPLE}")


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def ratio(a, b):
    return a / b if b else 0.0


def agreement():
    labels = load_labels()
    rows = list(csv.DictReader(open(SAMPLE, newline="")))
    for r in rows:
        r["llm"] = r["llm_in_scope"] or str(labels.get(r["record_id"], {}).get("in_scope", "")).lower()
    for split in ("dev", "test"):
        pairs = [(r, manifest.is_true(r["llm"]), manifest.is_true(r["human_in_scope"])) for r in rows
                 if r["split"] == split and r["human_in_scope"].strip() and r["llm"]]
        if not pairs:
            print(f"{split}: no labelled rows")
            continue
        n, tp = len(pairs), sum(l and h for _, l, h in pairs)
        wrong = [(r, l, h) for r, l, h in pairs if l != h]
        precision, recall = ratio(tp, sum(l for _, l, _ in pairs)), ratio(tp, sum(h for _, _, h in pairs))
        lo, hi = wilson(n - len(wrong), n)
        print(f"{split}: n={n} precision={precision:.3f} recall={recall:.3f} "
              f"F1={ratio(2 * precision * recall, precision + recall):.3f} "
              f"accuracy={(n - len(wrong)) / n:.3f} (95% Wilson {lo:.3f}-{hi:.3f})")
        for r, l, h in wrong:
            print(f"  disagree: {r['record_id']} | llm={l} human={h} | {r['title']}")


COPIED = ["record_id", "source", "arxiv_id", "arxiv_latest_version", "doi", "title", "year", "venue",
          "published_version_url", "source_url", "fulltext_url", "license"]
KEEP_IF_EMPTY = {"doi", "venue", "published_version_url", "fulltext_url", "license"}


def manifest_row(c, label):
    """Collection columns for one candidate; `label` is its classifier row or None."""
    s1 = c.get("stage1_result", "")
    label = label if s1 == "include" else None
    reason = s1.removeprefix("reject:") if s1.startswith("reject:") else ""
    if label and not label["in_scope"]:
        reason = label["out_of_scope_reason"]
    if label and label["in_scope"] and not all(c.get(k) for k in ("title", "authors", "year")):
        label, reason = None, "incomplete_metadata"  # an OpenAlex record without authors or year
    return {k: c.get(k, "") for k in COPIED} | {
        "work_id": c["arxiv_id"] if c["source"] == "arxiv" else c["record_id"],
        "authors": "; ".join(c.get("authors", [])), "matched_queries": "|".join(c.get("matched_queries", [])),
        "stage1_result": s1, "in_scope": "true" if label and label["in_scope"] is True else "false",
        "topic": label["topic"] if label else "", "paper_type": label["paper_type"] if label else "",
        "reject_reason": reason}


def build_manifest():
    old = {r["record_id"]: r for r in manifest.load()}
    labels = load_labels()
    rows = []
    for c in load_candidates():
        row = old.pop(c["record_id"], {})
        row.update({k: v for k, v in manifest_row(c, labels.get(c["record_id"])).items()
                    if str(v) != "" or k not in KEEP_IF_EMPTY})
        rows.append(row)
    rows += old.values()  # rows other stages added (v1 and venue copies) are not candidates: keep them
    manifest.save(rows)
    print(f"wrote {config.MANIFEST}: {len(rows)} rows")


def cap(n, seed):
    """Keep at most n in-scope works for indexing: seeds, then title-rule hits, then a seeded random fill."""
    rows = manifest.load()
    main = [r for r in rows if manifest.is_true(r["in_scope"]) and not r.get("duplicate_of")]
    title_hit = lambda r: any(p.search(r["title"].lower()) for p in INCLUDE.values())
    keep = [r for r in main if r["arxiv_id"] in SEEDS]
    core = [r for r in main if r not in keep and title_hit(r)]
    rest = [r for r in main if r not in keep and not title_hit(r)]
    random.Random(seed).shuffle(core)
    random.Random(seed).shuffle(rest)
    kept = {r["record_id"] for r in keep + (core + rest)[:max(0, n - len(keep))]}
    for r in main:
        if r["record_id"] not in kept:
            r["in_index"], r["not_indexed_reason"] = "false", "corpus_cap"
        elif r.get("not_indexed_reason") == "corpus_cap":
            r["in_index"], r["not_indexed_reason"] = "", ""
    manifest.save(rows)
    print(f"in-scope works {len(main)}: kept {len(kept)} (seeds {len(keep)}, title hits {min(len(core), n - len(keep))}), "
          f"capped {len(main) - len(kept)}")


def summary():
    rows = [r for r in manifest.load() if not r.get("duplicate_of")]  # collected documents only
    included = [r for r in rows if manifest.is_true(r["in_scope"])]
    stage2 = [r for r in rows if r["stage1_result"] == "include" and not manifest.is_true(r["in_scope"])]
    print(f"candidates: {len(rows)}")
    print("stage-1 rejected:", Counter(r["reject_reason"] for r in rows if r["stage1_result"] != "include"))
    print("stage-2 rejected:", Counter(r["reject_reason"] or "unlabelled" for r in stage2))
    print(f"included: {len(included)}")
    misses = sorted(set(SEEDS) - {r["arxiv_id"] for r in included})
    print(f"seeds among included: {len(SEEDS) - len(misses)}/{len(SEEDS)}; misses: {misses}")
    bad = [r["record_id"] for r in included if not all(r[k] for k in ("title", "authors", "year", "source_url"))]
    assert not bad, f"included rows missing title, authors, year or source_url: {bad[:20]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arxiv")
    a.add_argument("--families", default="seed,F1,F2,F3,F4")
    a.add_argument("--years", default="", help="comma-separated window names, e.g. pre2015,2016")
    sub.add_parser("stage1").add_argument("--report", action="store_true")
    c = sub.add_parser("classify")
    c.add_argument("--limit", type=int)
    c.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=config.SEED)
    sub.add_parser("agreement")
    sub.add_parser("manifest").add_argument("--summary", action="store_true")
    k = sub.add_parser("cap")
    k.add_argument("--n", type=int, default=config.CORPUS_CAP)
    k.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.cmd == "arxiv":
        collect_arxiv(args.families.split(","), set(filter(None, args.years.split(","))))
    elif args.cmd == "stage1":
        stage1(args.report)
    elif args.cmd == "classify":
        classify(args.limit, args.workers)
    elif args.cmd == "sample":
        sample(args.n, args.seed)
    elif args.cmd == "agreement":
        agreement()
    elif args.cmd == "cap":
        cap(args.n, args.seed)
    elif args.cmd == "manifest":
        build_manifest()
        if args.summary:
            summary()


if __name__ == "__main__":
    main()
