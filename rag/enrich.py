"""Link in-scope arXiv papers to OpenAlex works, and find ACL-venue papers that never went to arXiv.

`link` fills openalex_id, doi, venue, published_version_url and license in the manifest:
  1. title search per paper: the query is the normalised title (no punctuation, so no OpenAlex
     operators). A hit matches when its normalised title is equal or token_set_ratio >= 95 (with the
     shorter title at least 60% as long as the longer, since a subset scores 100) and its year is within 2 of the arXiv year (preprint 2015, ACL paper 2016). Among the 3 hits the first
     match with a publishedVersion location wins, else the first match.
  2. batch DOI lookup on 10.48550/arxiv.<id> (50 per request) only for papers the title search missed.
     Tested 2026-10-08: arXiv DOIs resolve only where OpenAlex keeps the preprint as its own DataCite
     record (2207.04672, 2305.13245 did; 1508.07909, 1706.03762, 2412.09871 did not), and that record
     has no venue: 2305.13245 by DOI gave the arXiv-only work, by title the EMNLP 2023 work. So title
     search goes first and DOI is the fallback.
  Venue, URL and licence come from the first publishedVersion location, else the primary location. An
  ACL Anthology PDF wins as published_version_url because the downloader needs a PDF link; old records
  use www.aclweb.org/anthology/, which redirects to aclanthology.org, so both count as ACL. A non-arXiv
  DOI is preferred. A field the work leaves empty keeps the manifest's value.
`discover-acl` runs one title-and-abstract search per phrase (first page of 200 only) and keeps works
  with an ACL Anthology location and no arxiv.org location, in the shared candidate schema.

Every response is cached in data/cache/openalex/, so reruns make no requests. 0.2 s between requests
(OpenAlex allows 5/s); a 429 waits Retry-After (default 10 s), at most 5 retries.
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

from rag import config, manifest

load_dotenv(config.ROOT / ".env")
API = "https://api.openalex.org/works"
CACHE = config.CACHE / "openalex"
SELECT = "id,doi,title,display_name,publication_year,type,primary_location,locations,best_oa_location,ids"
ACL_SELECT = ("id,doi,title,display_name,publication_year,publication_date,authorships,primary_location,"
              "locations,best_oa_location,abstract_inverted_index")
PHRASES = ["byte pair encoding", "subword tokenization", "subword segmentation", "SentencePiece", "WordPiece",
           "vocabulary size tokenizer", "tokenizer-free", "byte-level language model",
           "character-level language model", "tokenization multilingual fertility"]
STATS = {"requests": 0, "429": 0}


def get(params, key):
    """GET /works with `params`, cached as <key>.json."""
    path = CACHE / f"{key.replace('/', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    params = dict(params)
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("CONTACT_EMAIL"):
        params["mailto"] = os.environ["CONTACT_EMAIL"]
    for attempt in range(6):
        time.sleep(0.2)
        STATS["requests"] += 1
        r = requests.get(API, params=params, timeout=60)
        if r.status_code != 429 or attempt == 5:
            break
        STATS["429"] += 1
        wait = float(r.headers.get("Retry-After") or 10)
        if wait > 600:  # the daily budget is spent; it resets at midnight UTC
            raise RuntimeError(f"OpenAlex budget exhausted; retry in {wait / 3600:.1f} h")
        time.sleep(wait)
    r.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.json()


def norm(title):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (title or "").lower()).split())


def title_matches(title, year, work):
    a, b = norm(title), norm(work.get("display_name") or work.get("title"))
    similar_length = min(len(a), len(b)) >= 0.6 * max(len(a), len(b))  # token_set_ratio is 100 for a subset
    close = bool(b) and (a == b or (fuzz.token_set_ratio(a, b) >= 95 and similar_length))
    return close and bool(year) and abs(int(year) - (work.get("publication_year") or 0)) <= 2


def is_acl(url):
    return "aclanthology.org" in (url or "") or "aclweb.org/anthology" in (url or "")


def bare_doi(url):
    return (url or "").lower().removeprefix("https://doi.org/")


def published(work):
    return next((l for l in work.get("locations") or [] if l.get("version") == "publishedVersion"), None)


def fields(work):
    """Manifest fields from one OpenAlex work."""
    locs = work.get("locations") or []
    pub = published(work)
    loc = pub or work.get("primary_location") or {}
    url = (pub.get("landing_page_url") or pub.get("pdf_url") or "") if pub else ""
    acl_pdf = next((l["pdf_url"] for l in locs if is_acl(l.get("pdf_url"))), "")
    dois = [bare_doi(u) for u in [work.get("doi")] + [l.get("landing_page_url") for l in locs]
            if (u or "").startswith("https://doi.org/")]
    doi = next((d for d in dois if not d.startswith("10.48550/arxiv.")), dois[0] if dois else "")
    return {"openalex_id": work["id"].rsplit("/", 1)[-1], "doi": doi,
            "venue": (loc.get("source") or {}).get("display_name") or "",
            "published_version_url": acl_pdf or url, "license": (pub or {}).get("license") or ""}


def fill(row, work):
    """Write the work's fields into the row; empty ones keep the row's value."""
    for k, v in fields(work).items():
        row[k] = v or row.get(k, "")


def by_title(row):
    data = get({"search": norm(row["title"]), "per-page": 3, "select": SELECT}, row["arxiv_id"])
    hits = [w for w in data["results"] if title_matches(row["title"], row["year"], w)]
    return next((w for w in hits if published(w)), hits[0] if hits else None)


def by_doi(rows):
    """{arxiv_id: work} from batch DOI lookups, 50 ids per request."""
    found = {}
    for i in range(0, len(rows), 50):
        ids = sorted(r["arxiv_id"].lower() for r in rows[i:i + 50])
        key = "doi-" + hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]
        data = get({"filter": "doi:" + "|".join(f"10.48550/arxiv.{a}" for a in ids),
                    "per-page": 50, "select": SELECT}, key)
        for work in data["results"]:
            found[bare_doi(work.get("doi")).removeprefix("10.48550/arxiv.")] = work
    return found


def link(limit=None, report=False):
    rows = manifest.load()
    todo = [r for r in rows if manifest.is_true(r["in_scope"]) and r["arxiv_id"] and not r["openalex_id"]]
    todo = todo[:limit] if limit else todo
    matched, skipped = {}, 0
    for r in todo:
        try:
            matched[r["record_id"]] = None if skipped else by_title(r)
        except RuntimeError as e:  # budget spent: leave the rest unresolved, keep what is cached
            print(f"title search stopped: {e}")
            matched[r["record_id"]] = None
        skipped += matched[r["record_id"]] is None and STATS["429"] > 0
    n_title = sum(w is not None for w in matched.values())
    try:
        hits = by_doi([r for r in todo if matched[r["record_id"]] is None])
    except RuntimeError as e:  # keep the title matches; the DOI fallback can run another day
        print(f"DOI fallback skipped: {e}")
        hits = {}
    pairs = []
    for r in todo:
        work = matched[r["record_id"]] or hits.get(r["arxiv_id"].lower())
        if work:
            fill(r, work)
            pairs.append(f"{r['title']} || {work['display_name']} | {r['venue']} | {work['publication_year']}")
    manifest.save(rows)
    print(f"considered {len(todo)}  by DOI {len(pairs) - n_title}  by title {n_title}  skipped (budget) {skipped}  unresolved "
          f"{len(todo) - len(pairs)}  429s {STATS['429']}  live requests {STATS['requests']}")
    if report:
        for p in random.Random(config.SEED).sample(pairs, min(20, len(pairs))):
            print(" ", p)


def abstract(inverted):
    words = {i: w for w, places in (inverted or {}).items() for i in places}
    return " ".join(words[i] for i in sorted(words))


def acl_record(work):
    locs = work.get("locations") or []
    urls = [u for l in locs for u in (l.get("landing_page_url"), l.get("pdf_url")) if u]
    if any("arxiv.org" in u for u in urls) or not any(is_acl(u) for u in urls):
        return None
    acl = next(l for l in locs if is_acl(l.get("landing_page_url")) or is_acl(l.get("pdf_url")))
    acl_urls = [u for u in urls if is_acl(u)]  # a landing_page_url can itself be the PDF
    pdf = next((u for u in acl_urls if u.endswith(".pdf")), acl_urls[0].rstrip("/") + ".pdf")
    page = next((u for u in acl_urls if not u.endswith(".pdf")), pdf.removesuffix(".pdf"))
    year = work.get("publication_year")
    return {"record_id": "oa:" + work["id"].rsplit("/", 1)[-1], "source": "acl", "arxiv_id": "",
            "arxiv_latest_version": "", "title": work.get("display_name") or "",
            "abstract": abstract(work.get("abstract_inverted_index")),
            "authors": [a["author"]["display_name"] for a in work.get("authorships") or []],
            "year": year, "published": work.get("publication_date") or f"{year}-01-01", "categories": [],
            "doi": bare_doi(work.get("doi")),
            "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            "published_version_url": page, "fulltext_url": pdf, "license": acl.get("license") or "",
            "source_url": page, "matched_queries": []}


def discover_acl(max_phrases, out):
    found = {}
    for phrase in PHRASES[:max_phrases]:
        data = get({"filter": f"title_and_abstract.search:{phrase}", "per-page": 200, "select": ACL_SELECT},
                   "acl-" + re.sub(r"[^a-z0-9]+", "-", phrase.lower()))
        recs = [r for r in map(acl_record, data["results"]) if r]
        for r in recs:
            found.setdefault(r["record_id"], r)["matched_queries"].append(f"acl:{phrase}")
        print(f"{phrase}: {len(data['results'])} works, {len(recs)} ACL-only")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in found.values()))
    print(f"total {len(found)} ACL-only candidates -> {out}  live requests {STATS['requests']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("link")
    a.add_argument("--manifest", type=Path, default=config.MANIFEST)
    a.add_argument("--limit", type=int)
    a.add_argument("--report", action="store_true")
    b = sub.add_parser("discover-acl")
    b.add_argument("--max-phrases", type=int, default=10)
    b.add_argument("--out", type=Path, default=config.CACHE / "acl" / "candidates.jsonl")
    args = p.parse_args()
    if args.cmd == "link":
        config.MANIFEST = args.manifest  # manifest.load/save read this path at call time
        link(args.limit, args.report)
    else:
        discover_acl(args.max_phrases, args.out)


if __name__ == "__main__":
    main()
