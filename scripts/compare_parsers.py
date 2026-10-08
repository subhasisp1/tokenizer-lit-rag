"""Compare the PDF parser (pymupdf4llm) with the HTML parser (LaTeXML) on works that have both.

Writes results/parser_comparison.csv, one row per work plus a mean row:
- heading_f1: a PDF heading matches an HTML heading when rapidfuzz ratio >= 85 after stripping the
  leading number ("3.2", "A", "Appendix A") and lower-casing; precision over PDF headings, recall over
  HTML headings (last path component), F1 of the two.
- body_word_recall: share of HTML body words (lower-cased \\w+ tokens, as a multiset) found in the PDF body.
- pdf_refs_excluded: true when no PDF section has >= 3 matches of REF_PATTERN. Counted per section, not
  per paragraph: pymupdf4llm puts each reference in its own block, so no single block reaches 3. On
  the samples the raw reference sections have 4-113 matches and every kept section 0-2.
Usage: uv run python -m scripts.compare_parsers [--raw-dir data/raw/samples_layout]
"""
import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from rag import config
from rag.parse import body_text, parse_html, parse_pdf

REF_PATTERN = re.compile(r"In Proceedings of|arXiv preprint|Association for Computational Linguistics|pp\. \d+")
FIELDS = ["tag", "html_sections", "pdf_sections", "heading_f1", "body_word_recall", "pdf_refs_excluded",
          "html_seconds", "pdf_seconds"]


def norm(heading):
    heading = heading.split(" > ")[-1]
    return re.sub(r"^(appendix\s+)?([A-Z]|\d+)(\.\d+)*\.?\s+", "", heading, flags=re.I).lower()


def heading_f1(html_doc, pdf_doc):
    h = [norm(s["path"]) for s in html_doc["sections"]]
    p = [norm(s["path"]) for s in pdf_doc["sections"]]
    if not h or not p:
        return 0.0
    precision = sum(any(fuzz.ratio(x, y) >= 85 for y in h) for x in p) / len(p)
    recall = sum(any(fuzz.ratio(x, y) >= 85 for y in p) for x in h) / len(h)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def word_recall(html_doc, pdf_doc):
    h = Counter(re.findall(r"\w+", body_text(html_doc).lower()))
    p = Counter(re.findall(r"\w+", body_text(pdf_doc).lower()))
    return sum(min(n, p[w]) for w, n in h.items()) / sum(h.values())


def timed(parse, path, tag):
    start = time.perf_counter()
    doc = parse(path, tag)
    return doc, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", type=Path, default=config.RAW)
    a = p.parse_args()
    rows = []
    for html in sorted((a.raw_dir / "html").glob("*.html")):
        pdf = a.raw_dir / "pdf" / f"{html.stem}.pdf"
        if not pdf.exists():
            continue
        h, h_sec = timed(parse_html, html, html.stem)
        d, p_sec = timed(parse_pdf, pdf, html.stem)
        refs_excluded = not any(len(REF_PATTERN.findall(" ".join(s["paragraphs"]))) >= 3 for s in d["sections"])
        rows.append({"tag": html.stem, "html_sections": len(h["sections"]), "pdf_sections": len(d["sections"]),
                     "heading_f1": round(heading_f1(h, d), 3), "body_word_recall": round(word_recall(h, d), 3),
                     "pdf_refs_excluded": str(refs_excluded).lower(), "html_seconds": round(h_sec, 2),
                     "pdf_seconds": round(p_sec, 2)})
        print(rows[-1])
    mean = {"tag": "mean", "pdf_refs_excluded": f"{sum(r['pdf_refs_excluded'] == 'true' for r in rows)}/{len(rows)}"}
    for f in FIELDS[1:]:
        if f != "pdf_refs_excluded":
            mean[f] = round(sum(r[f] for r in rows) / len(rows), 3)
    rows.append(mean)
    print(mean)
    config.RESULTS.mkdir(exist_ok=True)
    with open(config.RESULTS / "parser_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
