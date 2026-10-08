"""Turn each downloaded paper into section-aware JSON: data/parsed/<doc_id with : as _>.json.

Decisions:
- arXiv's LaTeXML HTML is parsed first (bs4 + lxml): it keeps headings, tables and the LaTeX of every
  formula. The PDF (pymupdf4llm markdown) is the fallback.
- One section entry per heading (section, subsection, subsubsection, appendix) in document order; the
  path joins nested headings with " > " and keeps the printed numbers. Content before the first
  subsection belongs to the parent. \\paragraph run-in titles are prefixed to their first paragraph.
- Dropped: everything outside .ltx_page_main, the bibliography (counted in n_refs_dropped), broken
  macros (span.ltx_ERROR, counted in n_errors), footnotes and title notes (.ltx_note, .ltx_pubnotes)
  and acknowledgement sections.
- Math: inline as $latex$; display math as $$latex$$, or "[equation]" when longer than 300 chars.
- PDF: split the markdown on # headings (numbered-heading regex when there are none), drop picture
  text, cut from a References/Bibliography heading to the next appendix heading. An appendix heading is
  one matching /appendix|supplement/ or a lettered heading such as "A Background on Tokenization":
  many PDFs print appendices without the word, and cutting to the end would lose them.
- Quality gate: a parse with < 3 sections or < 5000 body characters is low. A low HTML parse is
  replaced by the PDF parse when that has more body characters; a doc whose best parse is still low is
  failed and not written.
- Body = paragraphs, captions, table captions and cells.
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf4llm
from bs4 import BeautifulSoup, NavigableString

from rag import config, manifest

SECTION_KINDS = {"ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix"}
MAX_DISPLAY_MATH = 300
MIN_SECTIONS, MIN_BODY_CHARS = 3, 5000
REFS = re.compile(r"^(\d+\s+)?(references|bibliography)$", re.I)
APPENDIX = re.compile(r"(?i:appendix|supplement)|^[A-Z](\.\d+)*\.?\s+[A-Z]")
CAPTION = re.compile(r"^(figure|table)\s+\d+[:.]", re.I)
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\s+[A-Z][^\n]{2,80}$")
PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)


def clean(text):
    return " ".join(text.split())


def ids_for(tag):
    """File tag -> (doc_id, work_id)."""
    if tag.startswith("acl_"):
        return f"acl:{tag[4:]}", tag[4:]
    return f"arxiv:{tag}", re.sub(r"v\d+$", "", tag)


def new_doc(tag, parser):
    doc_id, work_id = ids_for(tag)
    return {"doc_id": doc_id, "work_id": work_id, "title": "", "abstract": "", "parser": parser,
            "sections": [], "n_refs_dropped": 0, "n_errors": 0, "body_words": 0}


def body_text(doc):
    parts = []
    for s in doc["sections"]:
        parts += s["paragraphs"] + s["captions"]
        for t in s["tables"]:
            parts += [t["caption"]] + [c for row in t["rows"] for c in row]
    return "\n".join(parts)


def finish(doc):
    doc["body_words"] = len(body_text(doc).split())
    return doc


def is_low(doc):
    return len(doc["sections"]) < MIN_SECTIONS or len(body_text(doc)) < MIN_BODY_CHARS


def is_section(tag):
    return tag.name == "section" and bool(SECTION_KINDS & set(tag.get("class", [])))


def display_math(latex):
    latex = clean(latex.replace("\\displaystyle", ""))
    return f" $${latex}$$ " if len(latex) <= MAX_DISPLAY_MATH else " [equation] "


def replace_math(main):
    for eq in main.select("table.ltx_equation, table.ltx_equationgroup"):
        eq.replace_with(NavigableString(display_math(" ".join(m.get("alttext", "") for m in eq.find_all("math")))))
    for m in main.find_all("math"):
        latex = m.get("alttext", "")
        m.replace_with(NavigableString(display_math(latex) if m.get("display") == "block" else f"${latex}$"))


def heading(sec):
    h = sec.find(class_="ltx_title")
    return clean(h.get_text()) if h else ""


def parse_html(path, tag):
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "lxml")
    main, title = soup.select_one(".ltx_page_main"), soup.select_one("h1.ltx_title_document")
    doc = new_doc(tag, "latexml")
    if main is None:  # not a LaTeXML page: an empty parse, which the quality gate marks low
        return doc
    for bib in main.select("section.ltx_bibliography"):
        doc["n_refs_dropped"] += len(bib.select("li.ltx_bibitem"))
        bib.decompose()
    for err in main.select("span.ltx_ERROR"):
        doc["n_errors"] += 1
        err.decompose()
    for note in main.select(".ltx_note, .ltx_pubnotes, .ltx_acknowledgements"):
        note.decompose()
    replace_math(main)
    doc["title"] = clean(title.get_text()) if title else ""
    abstract = main.select_one("div.ltx_abstract")
    doc["abstract"] = clean(" ".join(p.get_text() for p in abstract.select("p"))) if abstract else ""

    entries = {}  # id(section tag) -> section entry, or None when the section is dropped
    for sec in main.find_all(is_section):
        parents = [p for p in sec.parents if is_section(p)][::-1]
        if re.search(r"acknowledg", heading(sec), re.I) or any(entries.get(id(p)) is None for p in parents):
            entries[id(sec)] = None
            continue
        path = " > ".join([heading(p) for p in parents] + [heading(sec)])
        entries[id(sec)] = {"path": path, "level": min(len(parents) + 1, 3),
                            "paragraphs": [], "tables": [], "captions": []}
        doc["sections"].append(entries[id(sec)])

    def owner(el):
        sec = el.find_parent(is_section)
        return entries.get(id(sec)) if sec else None

    for fig in main.select("figure.ltx_table, figure.ltx_figure"):
        entry, cap = owner(fig), fig.find("figcaption", recursive=False)
        if fig.find_parent("figure") or entry is None:
            continue
        caption = clean(cap.get_text()) if cap else ""
        if "ltx_table" in fig["class"]:
            rows = [[clean(c.get_text()) for c in tr.select("td.ltx_td, th.ltx_td")] for tr in fig.select("tr.ltx_tr")]
            entry["tables"].append({"caption": caption, "rows": rows})
        elif caption:
            entry["captions"].append(caption)
    for fig in main.select("figure"):
        fig.decompose()

    for para in main.select("div.ltx_para"):
        entry = owner(para)
        if entry is None or para.find_parent("div", class_="ltx_para"):
            continue
        text = clean(para.get_text())
        box = para.parent
        if "ltx_paragraph" in box.get("class", []) and para is box.find("div", class_="ltx_para", recursive=False):
            text = f"{heading(box)}. {text}"
        if text:
            entry["paragraphs"].append(text)
    return finish(doc)


def strip_md(line):
    return clean(line.lstrip("#").replace("**", ""))


def split_headings(lines):
    """[(line index, level, heading text)] from # lines, else from numbered headings."""
    found = [(i, len(l) - len(l.lstrip("#")), strip_md(l)) for i, l in enumerate(lines) if l.startswith("#")]
    if found:
        return found
    return [(i, l.strip().split()[0].count(".") + 2, l.strip()) for i, l in enumerate(lines)
            if NUMBERED_HEADING.match(l.strip())]


def parse_pdf(path, tag):
    md = PICTURE_TEXT.sub("", pymupdf4llm.to_markdown(str(path)))
    lines = md.splitlines()
    heads = split_headings(lines)
    titles = [h for h in heads if h[1] == 1]
    doc = new_doc(tag, "pymupdf4llm")
    doc["title"] = titles[0][2] if titles else strip_md(next((l for l in lines if l.strip()), ""))
    in_refs = False
    for n, (i, hashes, text) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        in_refs = bool(REFS.match(text)) or (in_refs and not APPENDIX.search(text))
        if in_refs or (titles and (i, hashes, text) == titles[0]) or re.search(r"acknowledg", text, re.I):
            continue
        entry = {"path": text, "level": min(max(hashes - 1, 1), 3), "paragraphs": [], "tables": [], "captions": []}
        for block in re.split(r"\n\s*\n", "\n".join(lines[i + 1:end])):
            block_lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not block_lines:
                continue
            if block_lines[0].startswith("|"):
                rows = [[clean(c.replace("<br>", " ")) for c in l.strip("|").split("|")] for l in block_lines
                        if l.startswith("|") and not re.fullmatch(r"[|\s:-]+", l)]
                entry["tables"].append({"caption": "", "rows": rows})
            elif CAPTION.match(strip_md(block_lines[0])):
                entry["captions"].append(strip_md(" ".join(block_lines)))
            elif re.search(r"[A-Za-z]{2}", block):  # skips page numbers
                entry["paragraphs"].append(clean(" ".join(block_lines)))
        if re.fullmatch(r"abstract", text, re.I):
            doc["abstract"] = " ".join(entry["paragraphs"])
        else:
            doc["sections"].append(entry)
    return finish(doc)


def tags_to_parse(raw_dir, manifest_path):
    """Tags from the manifest rows that have full text, else every file under html/ and pdf/."""
    if not Path(manifest_path).exists():
        return sorted({p.stem for p in (raw_dir / "html").glob("*.html")} | {p.stem for p in (raw_dir / "pdf").glob("*.pdf")})
    config.MANIFEST = Path(manifest_path)
    tags = []
    for r in manifest.load():
        if r.get("fulltext_status") not in ("html", "ar5iv", "pdf"):
            continue
        kind, rid = r["record_id"].split(":", 1)
        tags.append(f"acl_{rid}" if kind == "acl" else rid if re.search(r"v\d+$", rid)
                    else f"{r['arxiv_id']}v{r['arxiv_latest_version']}")
    return tags


def parse_tag(raw_dir, tag):
    """Return (doc, html_was_low): the best parse of one tag."""
    html, pdf = raw_dir / "html" / f"{tag}.html", raw_dir / "pdf" / f"{tag}.pdf"
    doc = parse_html(html, tag) if html.exists() else None
    low = doc is not None and is_low(doc)
    if (doc is None or low) and pdf.exists():
        alt = parse_pdf(pdf, tag)
        if doc is None or len(body_text(alt)) > len(body_text(doc)):
            doc = alt
    return doc, low


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", type=Path, default=config.RAW)
    p.add_argument("--manifest", default=str(config.MANIFEST))
    p.add_argument("--only", help="parse one tag, e.g. 1508.07909v5")
    p.add_argument("--report", action="store_true", help="print per-parser counts and means at the end")
    a = p.parse_args()
    tags = [a.only] if a.only else tags_to_parse(a.raw_dir, a.manifest)
    config.PARSED.mkdir(parents=True, exist_ok=True)
    done, failed, n_low = [], [], 0
    for tag in tags:
        doc, low = parse_tag(a.raw_dir, tag)
        n_low += low
        if doc is None or is_low(doc):
            failed.append(tag)
            print(f"{tag}: FAILED")
            continue
        out = config.PARSED / f"{doc['doc_id'].replace(':', '_')}.json"  # no colon: Windows-safe
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        done.append(doc)
        print(f"{tag}: {doc['parser']} sections={len(doc['sections'])} words={doc['body_words']} "
              f"errors={doc['n_errors']} refs={doc['n_refs_dropped']}")
    if a.report:
        for parser in ("latexml", "pymupdf4llm"):
            print(f"{parser}: {sum(d['parser'] == parser for d in done)}")
        n = max(len(done), 1)
        print(f"low html: {n_low}  failed: {len(failed)} {failed}")
        print(f"mean sections: {sum(len(d['sections']) for d in done) / n:.1f}  "
              f"mean body words: {sum(d['body_words'] for d in done) / n:.0f}")


if __name__ == "__main__":
    main()
