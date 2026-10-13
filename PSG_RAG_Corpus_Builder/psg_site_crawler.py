#!/usr/bin/env python3
"""
PSG Tech public-site corpus builder for RAG.

Fetches public pages under approved PSG domains, extracts readable text, records
source URLs and metadata, and writes JSONL + Markdown. It respects robots.txt,
uses a polite delay, avoids common account/session/search endpoints, and does
not bypass authentication, paywalls, CAPTCHAs, or access controls.

Install:
    python -m pip install requests beautifulsoup4 pypdf

Optional higher-quality extraction:
    python -m pip install trafilatura

Run:
    python psg_site_crawler.py --out ./psg_corpus --max-pages 5000 --delay 1.2

The official website and its linked subsites can change. Review robots.txt and
the website's terms before crawling, and adjust --domains if needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

START_URLS_DEFAULT = [
    "https://www.psgtech.edu/",
    "https://library.psgtech.ac.in/",
    "https://sports.psgtech.ac.in/",
    "https://alumni.psgtech.ac.in/",
    "https://iprcell.psgtech.ac.in/",
    "https://ir.psgtech.ac.in/",
    "https://journals.psgtech.ac.in/",
    "https://academicschedule.psgtech.ac.in/",
    "https://enquiry.psgtech.ac.in/",
    "https://iipc.psgtech.ac.in/",
    "https://cnce.psgtech.ac.in/",
    "https://csrc.psgtech.ac.in/",
    "https://aiconsortium.psgtech.ac.in/",
]
DEFAULT_DOMAINS = [
    "psgtech.edu", "psgtech.ac.in", "psgsonscharities.org",
    "psgstep.org", "psgtifac.co.in", "psgproducts.org",
]
SKIP_PATH_PARTS = [
    "/login", "/logout", "/signin", "/sign-in", "/auth", "/account",
    "/cart", "/checkout", "/wp-admin", "/admin", "/search?",
]
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z",
    ".exe", ".dmg", ".css", ".js", ".woff", ".woff2", ".ttf",
}
DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

def canonicalize(url: str) -> str | None:
    try:
        p = urllib.parse.urlsplit(url.strip())
        if p.scheme not in ("http", "https") or not p.netloc:
            return None
        host = p.hostname.lower() if p.hostname else ""
        if not host:
            return None
        path = re.sub(r"/{2,}", "/", p.path or "/")
        # Drop fragments; keep query parameters because some public pages use them.
        query = p.query
        return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), path, query, ""))
    except Exception:
        return None

def allowed_host(url: str, domains: list[str]) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)

def allowed_path(url: str) -> bool:
    p = urllib.parse.urlsplit(url)
    low = (p.path + "?" + p.query).lower()
    if any(part in low for part in SKIP_PATH_PARTS):
        return False
    ext = Path(p.path.lower()).suffix
    if ext in SKIP_EXTENSIONS:
        return False
    # Avoid highly dynamic query endpoints, but retain known public pages.
    if len(p.query) > 300:
        return False
    return True

def robots_allowed(url: str, session: requests.Session, cache: dict[str, urllib.robotparser.RobotFileParser]) -> bool:
    parts = urllib.parse.urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        try:
            r = session.get(robots_url, timeout=15)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            elif r.status_code in (401, 403):
                # If robots is explicitly inaccessible, fail closed.
                rp.parse(["User-agent: *", "Disallow: /"])
            else:
                rp.parse(["User-agent: *", "Allow: /"])
        except requests.RequestException:
            # Network errors should not be treated as permission to crawl.
            rp.parse(["User-agent: *", "Disallow: /"])
        cache[origin] = rp
    return cache[origin].can_fetch("PSGPublicRAGCorpusBot/1.0", url)

def clean_text(html: str, url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    # Keep navigation/footer text because the user requested broad site coverage.
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I)) or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    links = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(url, a["href"])
        href = canonicalize(href)
        if href:
            links.append(href)
    return title, text.strip(), links

def extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append(f"[PDF page {i}]\n{txt.strip()}")
    return "\n\n".join(pages).strip()

def extract_office_doc(data: bytes, ext: str) -> str:
    import io
    if ext == ".docx":
        from docx import Document
        d = Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip().replace("\n", " / ") for c in row.cells))
        return "\n".join(parts)
    if ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(io.BytesIO(data))
        parts = []
        for n, slide in enumerate(prs.slides, 1):
            parts.append(f"[Slide {n}]")
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    parts.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)
    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
            book = pd.ExcelFile(io.BytesIO(data))
            parts = []
            for sheet in book.sheet_names:
                df = pd.read_excel(book, sheet_name=sheet, header=None)
                parts.append(f"[Sheet: {sheet}]")
                parts.extend(" | ".join("" if v is None else str(v) for v in row) for row in df.fillna("").values.tolist())
            return "\n".join(parts)
        except Exception as e:
            return f"[Spreadsheet extraction error: {e}]"
    if ext == ".doc":
        return "[Legacy .doc file downloaded; convert to .docx or PDF for text extraction.]"
    if ext == ".ppt":
        return "[Legacy .ppt file downloaded; convert to .pptx or PDF for text extraction.]"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="psg_corpus")
    ap.add_argument("--max-pages", type=int, default=5000)
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    ap.add_argument("--start-urls", default=",".join(START_URLS_DEFAULT))
    ap.add_argument("--include-external-links", action="store_true",
                    help="Allow external hosts discovered on PSG pages only if they match --domains.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "downloads"
    raw_dir.mkdir(exist_ok=True)
    domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    starts = [canonicalize(u) for u in args.start_urls.split(",") if u.strip()]
    starts = [u for u in starts if u]
    session = requests.Session()
    session.headers.update({"User-Agent": "PSGPublicRAGCorpusBot/1.0 (polite public-page archival; contact site administrator if needed)"})
    robots_cache = {}
    queue = deque(starts)
    seen = set()
    records = []
    failures = []
    started = datetime.now(timezone.utc).isoformat()

    # Sitemap discovery is attempted for every start host, subject to robots.
    for url in starts:
        origin = urllib.parse.urlsplit(url)
        sm = f"{origin.scheme}://{origin.netloc}/sitemap.xml"
        try:
            if robots_allowed(sm, session, robots_cache):
                r = session.get(sm, timeout=args.timeout)
                if r.ok and "xml" in r.headers.get("Content-Type", "").lower():
                    root = ET.fromstring(r.content)
                    for loc in root.iter():
                        if loc.tag.endswith("loc") and loc.text:
                            u = canonicalize(loc.text)
                            if u and allowed_host(u, domains) and allowed_path(u):
                                queue.append(u)
        except Exception:
            pass

    while queue and len(seen) < args.max_pages:
        url = queue.popleft()
        url = canonicalize(url)
        if not url or url in seen or not allowed_host(url, domains) or not allowed_path(url):
            continue
        seen.add(url)
        if not robots_allowed(url, session, robots_cache):
            failures.append({"url": url, "reason": "blocked_by_robots_txt"})
            continue
        try:
            resp = session.get(url, timeout=args.timeout, allow_redirects=True)
            final_url = canonicalize(resp.url) or url
            if resp.status_code >= 400:
                failures.append({"url": url, "status": resp.status_code})
                time.sleep(args.delay)
                continue
            content_type = resp.headers.get("Content-Type", "").lower()
            path = urllib.parse.urlsplit(final_url).path
            ext = Path(path.lower()).suffix
            title, text, links = "", "", []
            kind = "html"
            if "text/html" in content_type or (not content_type and ext not in DOC_EXTENSIONS):
                title, text, links = clean_text(resp.text, final_url)
                if not text:
                    failures.append({"url": final_url, "reason": "no_text_extracted"})
            elif "application/pdf" in content_type or ext == ".pdf":
                kind = "pdf"
                text = extract_pdf(resp.content)
                title = Path(path).name
            elif ext in {".docx", ".pptx", ".xlsx", ".xls", ".doc", ".ppt"}:
                kind = ext.lstrip(".")
                title = Path(path).name
                try:
                    text = extract_office_doc(resp.content, ext)
                except Exception as e:
                    text = f"[Extraction failed: {e}]"
            else:
                failures.append({"url": final_url, "reason": f"unsupported_content_type:{content_type}"})
                time.sleep(args.delay)
                continue

            if text:
                digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                rec = {
                    "id": hashlib.sha1(final_url.encode()).hexdigest(),
                    "url": final_url,
                    "canonical_url": final_url,
                    "title": title or final_url,
                    "content_type": content_type,
                    "kind": kind,
                    "text": text,
                    "sha256_text": digest,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source_domain": urllib.parse.urlsplit(final_url).hostname,
                }
                records.append(rec)
                safe_name = rec["id"] + (".md")
                (raw_dir / safe_name).write_text(
                    f"# {rec['title']}\n\nSource: {final_url}\nCrawled: {rec['crawled_at']}\n\n{text}\n",
                    encoding="utf-8"
                )

            for link in links:
                if not allowed_host(link, domains) or not allowed_path(link):
                    continue
                # By default, keep crawl to PSG-owned/approved domains.
                if link not in seen:
                    queue.append(link)
        except Exception as e:
            failures.append({"url": url, "reason": repr(e)})
        time.sleep(max(args.delay, 0.2))

    jsonl = out / "psg_corpus.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "crawl_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "start_urls": starts,
        "allowed_domains": domains,
        "max_pages": args.max_pages,
        "pages_and_documents_extracted": len(records),
        "unique_urls_attempted": len(seen),
        "failures_or_skips": len(failures),
        "output_jsonl": str(jsonl),
        "note": "Text extraction is not proof of completeness. Scanned PDFs may need OCR; check crawl_failures.json and compare sitemap/navigation."
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
