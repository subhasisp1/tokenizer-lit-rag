"""Fetch the full text of every in-scope work: arXiv HTML, then ar5iv HTML, then the PDF.

Decisions:
- arXiv's own LaTeXML HTML keeps section structure, so it goes first; ar5iv covers older
  papers; the PDF is the last resort. One HTML file per version, whichever host it came from.
- HTML is validated before saving. The invalid page we saw (ar5iv, 2305.15425v2) is 6 KB,
  titled "No content available", with an ar5iv-severity-fatal link and 33 words. Valid pages
  often carry ltx_ERROR spans and ltx_missing_label refs, so those do not reject a page.
- Polite and resumable: our User-Agent, >= config.FILE_DELAY s between requests to one host,
  Retry-After or 2-32 s back-off on 429/503, stop on a 403 from export.arxiv.org. Files on
  disk are never fetched again, rows with a fulltext_status are skipped, the manifest is saved
  every 50 rows, and a row that only met transient errors stays empty so the next run retries.
- Licence: the "License: ..." line arXiv prints in its HTML, else the manifest's, else a note
  that we hold the files under arXiv's default licence and do not redistribute them.
- --extras adds v1 PDFs and ACL PDFs as their own rows, so dedupe has real duplicates to find.
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from rag import config, manifest

HTML_DIR = config.RAW / "html"
PDF_DIR = config.RAW / "pdf"
LOG = config.DATA / "download_log.csv"
DEFAULT_LICENSE = "arXiv non-exclusive distribution licence; not redistributed"
EXTRA_LICENSE = {"arxiv_v1": DEFAULT_LICENSE, "acl": "ACL Anthology licence (see the paper page); not redistributed"}
last_request = {}  # host -> time.monotonic() of the last request
stats = Counter()


def make_session():
    load_dotenv(config.ROOT / ".env")
    email = os.getenv("CONTACT_EMAIL", "").strip()
    ua = "tokenizer-lit-rag/0.1 (+https://github.com/subhasisp1/tokenizer-lit-rag"
    s = requests.Session()
    s.headers["User-Agent"] = ua + (f"; mailto:{email})" if email else ")")
    return s


def wait_for(host):
    gap = config.FILE_DELAY - (time.monotonic() - last_request.get(host, float("-inf")))
    if gap > 0:
        time.sleep(gap)
    last_request[host] = time.monotonic()


def get(session, url):
    """Return the response, or None if every try hit 429/503 or a network error."""
    host = urlparse(url).netloc
    for i in range(5):
        wait_for(host)
        stats["requests"] += 1
        try:
            r = session.get(url, timeout=90)
        except requests.RequestException as e:
            print(f"  {url}: {e}")
            time.sleep(2 ** (i + 1))
            continue
        if r.status_code == 403 and host == "export.arxiv.org":
            sys.exit(f"403 from {url}: arXiv is refusing us. Stop and wait before running again.")
        if r.status_code not in (429, 503):
            return r
        after = r.headers.get("Retry-After", "")
        time.sleep(int(after) if after.isdigit() else 2 ** (i + 1))
    return None


def html_is_valid(text):
    soup = BeautifulSoup(text, "lxml")
    title = soup.title.get_text() if soup.title else ""
    if "No content available" in title:
        return False, "title says No content available"
    if "ar5iv-severity-fatal" in text:
        return False, "ar5iv reports a fatal conversion error"
    if soup.select_one(".ltx_missing"):
        return False, "has an ltx_missing element"
    if not soup.select_one(".ltx_page_main"):
        return False, "no ltx_page_main"
    words = len(soup.body.get_text(" ").split()) if soup.body else 0
    if words < 1500:
        return False, f"only {words} words"
    sections = len(soup.select(".ltx_section"))
    if sections < 2:
        return False, f"only {sections} ltx_section elements"
    return True, "ok"


def html_license(text):
    m = re.search(r"License:\s*([^<\n]+)", text)
    return m.group(1).strip() if m else ""


def log(tag, step, url, r, seconds, outcome):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "step", "url", "status", "bytes", "seconds", "outcome"])
        code, size = (r.status_code, len(r.content)) if r is not None else ("error", 0)
        w.writerow([tag, step, url, code, size, f"{seconds:.2f}", outcome])


def fetch(session, tag, step, url, path, is_html):
    """Fetch url into path. Returns True on success, False on a bad answer, None on no answer."""
    start = time.monotonic()
    r = get(session, url)
    ok, outcome = r is not None and r.status_code == 200, "no response"
    if ok and is_html:
        ok, outcome = html_is_valid(r.text)
    elif ok:
        ok, outcome = r.content.startswith(b"%PDF"), "not a PDF"
    elif r is not None:
        outcome = f"http {r.status_code}"
    if ok:
        path.write_bytes(r.content)
        outcome = "saved"
    log(tag, step, url, r, time.monotonic() - start, outcome)
    return ok if r is not None else None


def download_row(session, row):
    tag = f"{row['arxiv_id']}v{row['arxiv_latest_version']}"
    html, pdf = HTML_DIR / f"{tag}.html", PDF_DIR / f"{tag}.pdf"
    steps = [("html", f"https://export.arxiv.org/html/{tag}", html),
             ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{tag}", html),
             ("pdf", f"https://export.arxiv.org/pdf/{tag}", pdf)]
    if html.exists():  # a file from an earlier run: no request
        steps = [steps[1] if "ar5iv-footer" in html.read_text(encoding="utf-8") else steps[0]]
    elif pdf.exists():
        steps = [steps[2]]
    status, answers = "none", []
    for step, url, path in steps:
        answers.append(path.exists() or fetch(session, tag, step, url, path, step != "pdf"))
        if answers[-1]:
            status, row["fulltext_url"] = step, url
            break
    if status == "none":
        stats["failures"] += 1
        if None in answers:
            status = ""  # no answer from a host: retry next run
    row["fulltext_status"] = status
    found = html_license(html.read_text(encoding="utf-8")) if status in ("html", "ar5iv") else ""
    row["license"] = found or row.get("license") or DEFAULT_LICENSE


def is_main(r):
    return manifest.is_true(r["in_scope"]) and r.get("arxiv_id") and r.get("source") not in EXTRA_LICENSE


def extra_jobs(rows):
    """(record_id, source, url, path, main row) for every v1 and ACL PDF we could add."""
    for r in filter(is_main, rows):
        if int(r.get("arxiv_latest_version") or 1) > 1:
            i = r["arxiv_id"]
            yield f"arxiv:{i}v1", "arxiv_v1", f"https://export.arxiv.org/pdf/{i}v1", PDF_DIR / f"{i}v1.pdf", r
        url = r.get("published_version_url", "")
        if "aclanthology.org" in url and url.endswith(".pdf"):
            name = url.rsplit("/", 1)[1]
            yield f"acl:{name[:-4]}", "acl", url, PDF_DIR / f"acl_{name}", r


def download_extras(session, rows, limit, dry_run):
    have = {r["record_id"] for r in rows}
    jobs = [j for j in extra_jobs(rows) if j[0] not in have][:limit]
    for record_id, source, url, path, main in jobs:
        print(f"extra {record_id}: {url}")
        if dry_run or not (path.exists() or fetch(session, record_id, source, url, path, False)):
            continue
        new = {c: main.get(c, "") for c in ("work_id", "arxiv_id", "title", "authors", "year",
                                            "venue", "topic", "paper_type")}
        new.update(record_id=record_id, source=source, duplicate_of=main["record_id"], in_scope="true",
                   fulltext_status="pdf", fulltext_url=url, canonical="false", license=EXTRA_LICENSE[source])
        rows.append(new)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(config.MANIFEST))
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="comma-separated arXiv ids")
    p.add_argument("--extras", action="store_true")
    p.add_argument("--force", action="store_true", help="redo rows that have a status (files on disk are reused)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    config.MANIFEST = Path(a.manifest)
    start, rows = time.monotonic(), manifest.load()
    ids = set(a.ids.split(",")) if a.ids else None
    todo = [r for r in rows if is_main(r) and (ids is None or r["arxiv_id"] in ids)
            and (a.force or not r.get("fulltext_status"))][:a.limit]
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    for n, row in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {row['arxiv_id']}v{row['arxiv_latest_version']}: html, ar5iv, pdf")
        if a.dry_run:
            continue
        download_row(session, row)
        print(f"  -> {row['fulltext_status'] or 'retry later'} | {row['license']}")
        if n % 50 == 0:
            manifest.save(rows)
    if a.extras:
        download_extras(session, rows, a.limit, a.dry_run)
    if not a.dry_run:
        manifest.save(rows)
    counts = Counter(r.get("fulltext_status") or "-" for r in rows if manifest.is_true(r["in_scope"]))
    print(f"status: {dict(counts)} | failures: {stats['failures']} | requests: {stats['requests']}"
          f" | {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    main()
