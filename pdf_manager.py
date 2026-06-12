#!/usr/bin/env python3
"""
pdf_manager.py — Automatically retrieve, organize, and maintain local PDF
collections.  Detects duplicates via content hash, extracts metadata from
filenames and PDF metadata (when pypdf is available), and maintains a
searchable library index.

Outputs
-------
outputs/pdf_library/library_index.csv
outputs/pdf_library/library_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("pdf_manager")

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

PDF_DIR = Path("outputs/pdf_library")


def _file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _guess_metadata_from_filename(fname: str) -> Dict[str, str]:
    """Heuristic: try to extract year, title, first author from filename."""
    meta: Dict[str, str] = {"title": fname, "year": "", "first_author": ""}
    # Pattern: Author_Year_Title.pdf
    m = re.match(r"([A-Za-z]+)_(\d{4})_(.+)\.pdf", fname)
    if m:
        meta["first_author"] = m.group(1)
        meta["year"] = m.group(2)
        meta["title"] = m.group(3).replace("_", " ")
        return meta
    # Pattern: Year_Title.pdf
    m = re.match(r"(\d{4})_(.+)\.pdf", fname)
    if m:
        meta["year"] = m.group(1)
        meta["title"] = m.group(2).replace("_", " ")
    return meta


def _extract_pdf_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from PDF using pypdf if available."""
    if not HAS_PYPDF:
        return {}
    try:
        reader = pypdf.PdfReader(str(path))
        info = reader.metadata or {}
        return {
            "title": str(info.title or ""),
            "author": str(info.author or ""),
            "subject": str(info.subject or ""),
            "pages": len(reader.pages),
            "producer": str(info.producer or ""),
        }
    except Exception:
        return {}


def manage_local_library(pdf_dir: str = "pdfs",
                         output_dir: str = "outputs/pdf_library",
                         copy_mode: bool = False) -> pd.DataFrame:
    """Scan a directory for PDFs, hash-deduplicate, build index."""
    src = Path(pdf_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.warning("PDF source dir not found: %s", src)
        return pd.DataFrame()

    pdf_files = sorted(src.rglob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", src)
        return pd.DataFrame()

    seen_hashes: Dict[str, Path] = {}
    rows = []
    for pdf_path in pdf_files:
        fhash = _file_hash(pdf_path)
        fname = pdf_path.name

        # Detect duplicate
        if fhash in seen_hashes:
            log.info("Duplicate detected: %s == %s", pdf_path.name, seen_hashes[fhash].name)
            continue
        seen_hashes[fhash] = pdf_path

        meta = _guess_metadata_from_filename(fname)
        pdf_meta = _extract_pdf_metadata(pdf_path)
        if pdf_meta.get("title"):
            meta["title"] = pdf_meta["title"]
        if pdf_meta.get("author"):
            meta["first_author"] = pdf_meta["author"]

        if copy_mode:
            dest = out / "pdfs" / fname
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(pdf_path, dest)

        rows.append({
            "filename": fname,
            "file_hash": fhash[:16],
            "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
            "title": meta.get("title", fname),
            "first_author": meta.get("first_author", ""),
            "year": meta.get("year", ""),
            "pages": pdf_meta.get("pages", 0),
            "source_path": str(pdf_path),
            "indexed_at": datetime.now().isoformat(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "library_index.csv", index=False)
    log.info("Library index: %d PDFs -> %s", len(df), out / "library_index.csv")

    metadata = {"library_path": str(out), "total_pdfs": len(df), "indexed_at": datetime.now().isoformat()}
    json.dump(metadata, open(out / "library_metadata.json", "w"), indent=2)

    return df


def download_open_access_pdfs(papers_path: str = "search_results.csv") -> int:
    """Check paper URLs and record open-access status (proxy for download)."""
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    if df.empty:
        return 0
    out = Path("outputs/pdf_library")
    out.mkdir(parents=True, exist_ok=True)
    oa_records = []
    count = 0
    for _, row in df.iterrows():
        url = str(row.get("url", ""))
        doi = str(row.get("doi", ""))
        source = str(row.get("source", ""))
        if url and url != "nan":
            # Flag as potentially open-access if from known OA sources
            is_oa = source in ("OpenAlex", "CORE", "PubMed") or "doi.org" in url
            oa_records.append({
                "doi": doi,
                "title": str(row.get("title", ""))[:120],
                "url": url,
                "open_access": is_oa,
                "source": source,
            })
            if is_oa:
                count += 1

    oa_df = pd.DataFrame(oa_records)
    oa_df.to_csv(out / "oa_records.csv", index=False)
    log.info("Open-access records: %d/%d", count, len(oa_df))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage PDF library.")
    parser.add_argument("--pdf-dir", type=str, default="pdfs", help="Directory with PDFs")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/pdf_library")
    parser.add_argument("--copy", action="store_true", help="Copy PDFs to library")
    args = parser.parse_args()

    df = manage_local_library(args.pdf_dir, args.output_dir, copy_mode=args.copy)
    oa_count = download_open_access_pdfs(args.papers)

    print(f"\n--- PDF Manager Complete ---")
    print(f"  PDFs indexed: {len(df)}")
    print(f"  Open-access flagged: {oa_count}")
    print()


if __name__ == "__main__":
    main()
