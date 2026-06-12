#!/usr/bin/env python3
"""
figure_table_extractor.py — Extract scientific figures, tables, captions,
and supplementary material references from paper metadata and available
PDFs.  Builds searchable catalogs.

Outputs
-------
outputs/figures/figure_catalog.csv
outputs/tables/table_catalog.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("figure_table_extractor")

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

FIG_PATTERN = re.compile(r"(?:Fig(?:ure)?\.?\s*\d+[A-Za-z]?[^.]*\.)", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"(?:Table\s+\d+[A-Za-z]?[^.]*\.)", re.IGNORECASE)
CAPTION_PATTERN = re.compile(r"(?:Fig(?:ure)?\.?\s*\d+[A-Za-z]?[. :].*?)(?=\.\s[A-Z]|$)", re.IGNORECASE)
SUPP_PATTERN = re.compile(r"(?:Supplementary|Supplemental|Additional|Extra)\s+(?:Figure|Table|Material|Data|File)[^.]*\.", re.IGNORECASE)


def extract_from_abstract(abstract: str, doi: str, title: str) -> Tuple[List[Dict], List[Dict]]:
    figures = []
    tables = []
    for m in FIG_PATTERN.finditer(abstract):
        figures.append({"doi": doi, "paper_title": title[:120], "caption": m.group(0).strip()[:200],
                        "type": "figure", "source": "abstract"})
    for m in TABLE_PATTERN.finditer(abstract):
        tables.append({"doi": doi, "paper_title": title[:120], "caption": m.group(0).strip()[:200],
                       "type": "table", "source": "abstract"})
    return figures, tables


def extract_from_pdf(pdf_path: Path, doi: str) -> Tuple[List[Dict], List[Dict]]:
    figures, tables = [], []
    if not HAS_PYPDF:
        return figures, tables
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            for m in FIG_PATTERN.finditer(text):
                figures.append({"doi": doi, "page": i + 1, "caption": m.group(0).strip()[:200],
                                "type": "figure", "source": "pdf"})
            for m in TABLE_PATTERN.finditer(text):
                tables.append({"doi": doi, "page": i + 1, "caption": m.group(0).strip()[:200],
                               "type": "table", "source": "pdf"})
            for m in SUPP_PATTERN.finditer(text):
                for tgt in [figures, tables]:
                    tgt.append({"doi": doi, "page": i + 1, "caption": m.group(0).strip()[:200],
                                "type": "supplementary", "source": "pdf"})
    except Exception:
        pass
    return figures, tables


def build_catalogs(papers_path: str = "search_results.csv",
                   pdf_library_path: str = "outputs/pdf_library",
                   output_dir: str = "outputs") -> None:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    all_figures, all_tables = [], []
    pdf_lib = Path(pdf_library_path)

    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        doi = str(row.get("doi", ""))
        abstract = str(row.get("abstract", ""))
        if not title or title.lower() in ("nan", ""):
            continue
        figs, tabs = extract_from_abstract(abstract, doi, title)
        all_figures.extend(figs)
        all_tables.extend(tabs)

        # Try PDF
        pdf_index = pdf_lib / "library_index.csv"
        if pdf_index.exists():
            lib_df = pd.read_csv(pdf_index)
            for _, lib_row in lib_df.iterrows():
                if doi in str(lib_row.get("filename", "")) or title[:40] in str(lib_row.get("filename", "")):
                    pdf_path = Path(str(lib_row.get("source_path", "")))
                    if pdf_path.exists():
                        figs2, tabs2 = extract_from_pdf(pdf_path, doi)
                        all_figures.extend(figs2)
                        all_tables.extend(tabs2)

    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    fig_df = pd.DataFrame(all_figures)
    if not fig_df.empty:
        fig_df.to_csv(fig_dir / "figure_catalog.csv", index=False)
    log.info("Figures: %d -> %s", len(fig_df), fig_dir / "figure_catalog.csv")

    tab_df = pd.DataFrame(all_tables)
    if not tab_df.empty:
        tab_df.to_csv(tab_dir / "table_catalog.csv", index=False)
    log.info("Tables: %d -> %s", len(tab_df), tab_dir / "table_catalog.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract figures and tables.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--pdf-library", type=str, default="outputs/pdf_library")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    build_catalogs(args.papers, args.pdf_library, args.output_dir)

    print(f"\n--- Figure & Table Extraction Complete ---")
    print()


if __name__ == "__main__":
    main()
