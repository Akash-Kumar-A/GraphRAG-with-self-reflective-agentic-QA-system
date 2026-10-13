# PSG Tech public website corpus builder

This package creates a source-attributed text corpus suitable as a starting point for a RAG pipeline.

## Install

```bash
python -m pip install requests beautifulsoup4 pypdf
# Optional extraction support:
python -m pip install python-docx python-pptx pandas openpyxl
```

## Run

```bash
python psg_site_crawler.py --out ./psg_corpus --max-pages 5000 --delay 1.2
```

Outputs:
- `psg_corpus.jsonl`: one JSON record per extracted page/document, with `url`, `title`, `text`, timestamps and a text hash.
- `downloads/*.md`: readable page-level Markdown snapshots with source URLs.
- `manifest.json`: run metadata.
- `crawl_failures.json`: pages blocked by robots.txt, HTTP failures, and unsupported file types.

## Important notes for RAG ingestion

1. Review `robots.txt`, the site's terms, and any applicable institutional policies before crawling. The crawler obeys robots.txt and does not bypass access controls.
2. It follows public pages and documents only on the configured approved domains. Add or remove domains using `--domains` after reviewing each linked site's ownership and permissions.
3. Scanned/image-only PDFs need OCR (for example, a local OCRmyPDF/Tesseract workflow); the script does not bypass this limitation.
4. Faculty profiles may be hosted on separate departmental systems or dynamically generated pages. Review `crawl_failures.json`, compare `manifest.json` with site navigation, and add any official department subdomains that are approved but not listed.
5. The JSONL is an extraction corpus, not a guarantee that every word of every page was captured. Navigation, JavaScript-only content, inaccessible files, and content loaded after user interaction may require separate processing.
6. Keep the URL and timestamp fields attached to every chunk. When chunking for embeddings, split by headings/paragraphs and preserve `url`, `title`, and `crawled_at` as metadata. Deduplicate using `sha256_text` or URL.
7. For production, use a smaller chunking script (e.g. 500–1,000 tokens with 80–150 token overlap) and retain heading context. Do not remove source metadata.

## Suggested ingestion record

```json
{
  "id": "stable-page-id#chunk-0001",
  "text": "chunk text...",
  "metadata": {
    "url": "https://official-source.example/page",
    "title": "Page title",
    "heading_path": ["Academics", "Department", "Faculty"],
    "chunk_index": 1,
    "crawled_at": "ISO-8601 timestamp"
  }
}
```
