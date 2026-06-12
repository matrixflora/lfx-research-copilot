#!/usr/bin/env python3
"""
support_claims_with_references.py — Enrich every generated statement with
supporting APA references.  For each statement drawn from the pipeline's
report files, the module retrieves supporting papers, ranks them by
semantic similarity, citation strength, and evidence strength, and then
attaches APA inline citations and/or full references.

Configuration
-------------
citation_mode : str
    "apa_inline"        — only inline citations (Author, Year)
    "apa_full_reference" — only full references below each statement
    "both" (default)    — both inline citation and full reference

Usage
-----
    python support_claims_with_references.py
        [--papers search_results.csv]
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--evidence outputs/reports/evidence_strength.csv]
        [--citation-metrics outputs/reports/citation_metrics.csv]
        [--report outputs/reports/executive_summary.md]
            (repeatable — processes each report file)
        [--citation-mode both]
        [--top-k 3]

If no --report paths are supplied, the default report set is processed:
    outputs/reports/executive_summary.md
    outputs/reports/research_gaps.md
    outputs/reports/hypothesis_bank.md
    outputs/reports/contradictory_findings.md
    outputs/reports/research_brief.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("support_claims")

CITATION_MODES = ("apa_inline", "apa_full_reference", "both")
DEFAULT_TOP_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_REPORTS = [
    "outputs/reports/executive_summary.md",
    "outputs/reports/research_gaps.md",
    "outputs/reports/hypothesis_bank.md",
    "outputs/reports/contradictory_findings.md",
    "outputs/reports/research_brief.md",
]


# ---------------------------------------------------------------------------
# 1.  Paper loader with MiniLM embeddings
# ---------------------------------------------------------------------------

class PaperStore:
    """Load papers from search_results.csv / knowledge_base.json, compute
    MiniLM embeddings, and provide lookup by DOI / title."""

    def __init__(self, papers_csv: str = "search_results.csv",
                 kb_json: str = "outputs/knowledge_base/knowledge_base.json"):
        self._model: Optional[SentenceTransformer] = None
        self.papers: List[Dict[str, Any]] = []  # each with title,authors,year,doi,abstract,venue,citation_count,embedding
        self._doi_index: Dict[str, Dict] = {}
        self._title_index: Dict[str, Dict] = {}

        # Load from CSV first
        csv_path = Path(papers_csv)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                if title.lower() in ("", "nan", "none"):
                    continue
                doi = str(row.get("doi", "")).strip().lower()
                text = f"{title} {row.get('abstract', '')}"
                self.papers.append({
                    "title": title,
                    "authors": str(row.get("authors", "")).strip(),
                    "year": row.get("year"),
                    "doi": doi,
                    "abstract": str(row.get("abstract", "")),
                    "venue": str(row.get("venue", "")),
                    "source": str(row.get("source", "")),
                    "citation_count": float(row.get("citation_count", 0) or 0),
                    "url": str(row.get("url", "")),
                    "embedding": None,
                    "_text": text,
                })

        # If no papers from CSV, try knowledge base
        if not self.papers:
            kb_path = Path(kb_json)
            if kb_path.exists():
                with open(kb_path) as f:
                    kb = json.load(f)
                for p in kb.get("papers", []):
                    title = p.get("title", "").strip()
                    if not title:
                        continue
                    self.papers.append({
                        "title": title,
                        "authors": p.get("authors", ""),
                        "year": p.get("year"),
                        "doi": str(p.get("doi", "")).lower(),
                        "abstract": p.get("abstract", ""),
                        "venue": p.get("venue", ""),
                        "source": p.get("source", ""),
                        "citation_count": float(p.get("citation_count", 0) or 0),
                        "url": p.get("url", ""),
                        "embedding": None,
                        "_text": f"{title} {p.get('abstract', '')}",
                    })

        self._doi_index = {p["doi"]: p for p in self.papers if p["doi"]}
        self._title_index = {}
        for p in self.papers:
            key = p["title"].lower().strip().strip('"').strip("'")
            self._title_index[key] = p

        log.info("Loaded %d papers from %s", len(self.papers), papers_csv)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading MiniLM model ...")
            self._model = SentenceTransformer(MODEL_NAME, device="cpu")
        return self._model

    def compute_embeddings(self) -> None:
        """Compute sentence embeddings for all papers (title + abstract)."""
        texts = [p["_text"] for p in self.papers]
        if not texts:
            return
        log.info("Computing embeddings for %d papers ...", len(texts))
        batch_size = 32
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch, show_progress_bar=False)
            all_emb.append(emb)
        full = np.concatenate(all_emb, axis=0) if all_emb else np.array([])
        for p, e in zip(self.papers, full):
            p["embedding"] = e / np.linalg.norm(e)  # unit-normalize

    def get_paper(self, doi: str = "", title: str = "") -> Optional[Dict]:
        if doi:
            return self._doi_index.get(doi.lower())
        if title:
            key = title.lower().strip().strip('"').strip("'")
            return self._title_index.get(key)
        return None

    def search(self, query: str, top_k: int = 5,
               citation_weight: float = 0.3,
               evidence_lookup: Optional[Dict[str, float]] = None,
               ) -> List[Tuple[Dict, float]]:
        """Return top-k papers sorted by composite score.

        Composite = 0.5 * sim + 0.25 * citation_strength + 0.25 * evidence_score
        """
        if not self.papers or self.papers[0]["embedding"] is None:
            self.compute_embeddings()

        q_emb = self.model.encode([query], show_progress_bar=False)[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        scored: List[Tuple[Dict, float]] = []
        max_cites = max((p["citation_count"] for p in self.papers), default=1)
        for p in self.papers:
            if p["embedding"] is None:
                continue
            sim = float(np.dot(q_emb, p["embedding"]))
            cites = min(p["citation_count"] / max(max_cites, 1), 1.0)

            # Evidence strength — by theme if we can match
            ev = 0.5
            if evidence_lookup:
                for theme, score in evidence_lookup.items():
                    kw = theme.lower().split(", ")
                    if any(k in p["title"].lower() for k in kw):
                        ev = max(ev, score)

            composite = 0.50 * sim + 0.25 * cites + 0.25 * ev
            scored.append((p, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 2.  APA reference formatter
# ---------------------------------------------------------------------------

def _parse_authors(authors_str: str) -> List[str]:
    """Parse author string into list of APA-formatted names."""
    if not authors_str or authors_str.lower() == "nan":
        return ["Unknown"]
    # Try splitting by semicolon or comma–space
    parts = re.split(r"[;,]+\s*", authors_str)
    parts = [p.strip() for p in parts if p.strip()]
    formatted = []
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        last = tokens[-1].rstrip(",").rstrip(".")
        initials = " ".join(t[0].upper() + "." for t in tokens[:-1] if t[0].isupper() or t[0].islower())
        if not initials:
            # Single name (e.g. mononym)
            formatted.append(last)
        else:
            formatted.append(f"{last}, {initials}")
    return formatted if formatted else ["Unknown"]


def _format_apa(paper: Dict) -> str:
    """Build APA 7th-edition reference from paper metadata."""
    authors_str = paper.get("authors", "")
    year = paper.get("year")
    title = paper.get("title", "").strip()
    venue = paper.get("venue", "").strip()
    doi = paper.get("doi", "").lower().strip()

    if year and year != "nan" and year != 0:
        year_str = str(int(year)) if not np.isnan(float(year)) else "n.d."
    else:
        year_str = "n.d."

    authors = _parse_authors(authors_str)
    if len(authors) == 1:
        author_part = authors[0]
    elif len(authors) == 2:
        author_part = f"{authors[0]} & {authors[1]}"
    else:
        author_part = ", ".join(authors[:-1]) + f", & {authors[-1]}"

    ref = f"{author_part} ({year_str}). {title}."
    if venue and venue.lower() not in ("", "nan", "unknown"):
        ref += f" *{venue}*."
    if doi and doi != "nan":
        ref += f" https://doi.org/{doi}"

    return ref


def _inline_citation(paper: Dict) -> str:
    """Generate APA inline citation (Author, Year)."""
    year = paper.get("year")
    if year and year != "nan" and year != 0:
        year_str = str(int(float(year)))
    else:
        year_str = "n.d."
    authors_str = paper.get("authors", "")
    authors = _parse_authors(authors_str)
    if not authors or authors == ["Unknown"]:
        return f"({year_str})"
    # Use first author's last name + "et al." if 3+
    if len(authors) >= 3:
        last = authors[0].split(",")[0]
        return f"({last} et al., {year_str})"
    elif len(authors) == 2:
        last1 = authors[0].split(",")[0]
        last2 = authors[1].split(",")[0]
        return f"({last1} & {last2}, {year_str})"
    else:
        last = authors[0].split(",")[0]
        return f"({last}, {year_str})"


# ---------------------------------------------------------------------------
# 3.  Statement extraction from markdown reports
# ---------------------------------------------------------------------------

def extract_statements(md_text: str, source_file: str = "") -> List[Dict]:
    """Extract meaningful statements from markdown report text.

    Returns list of dicts: {statement, section, line_number, source_file}
    """
    statements: List[Dict] = []
    lines = md_text.split("\n")
    current_section = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track section headers
        h_match = re.match(r"^#{1,3}\s+(.*)", stripped)
        if h_match:
            current_section = h_match.group(1).strip()
            continue

        # Skip empty lines, separators, footers
        if not stripped or stripped.startswith("---") or stripped.startswith("*Generated"):
            continue
        if stripped.startswith("> "):
            continue  # blockquote notes

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            continue

        # Bullet points
        bm = re.match(r"^-\s+\*\*(.*?)\*\*:?\s*(.*)", stripped)
        if bm:
            label = bm.group(1).strip()
            detail = bm.group(2).strip()
            statement = f"{label}: {detail}" if detail else label
            if len(statement) > 20:
                statements.append({
                    "statement": statement,
                    "section": current_section,
                    "line_number": i + 1,
                    "source_file": source_file,
                })
            continue

        bm2 = re.match(r"^-\s+(.*)", stripped)
        if bm2:
            statement = bm2.group(1).strip()
            if len(statement) > 20:
                statements.append({
                    "statement": statement,
                    "section": current_section,
                    "line_number": i + 1,
                    "source_file": source_file,
                })
            continue

        # Bold-header lines (theme descriptions)
        bm3 = re.match(r"^\*\*(\d+\.\s+.*?)\*\*\s*——\s*(.*)", stripped)
        if bm3:
            label = bm3.group(1).strip()
            detail = bm3.group(2).strip()
            statement = f"{label}: {detail}" if detail else label
            statements.append({
                "statement": statement,
                "section": current_section,
                "line_number": i + 1,
                "source_file": source_file,
            })
            continue

        # Numbered list items
        nm = re.match(r"^\d+\.\s+(.*)", stripped)
        if nm:
            statement = nm.group(1).strip()
            if len(statement) > 20:
                statements.append({
                    "statement": statement,
                    "section": current_section,
                    "line_number": i + 1,
                    "source_file": source_file,
                })
            continue

        # Regular paragraph sentences (min 80 chars)
        if len(stripped) >= 80:
            # Split into sentences
            sents = re.split(r"(?<=[.!?])\s+", stripped)
            for s in sents:
                s = s.strip()
                if len(s) >= 60:
                    statements.append({
                        "statement": s,
                        "section": current_section,
                        "line_number": i + 1,
                        "source_file": source_file,
                    })

    return statements


# ---------------------------------------------------------------------------
# 4.  Evidence strength loader
# ---------------------------------------------------------------------------

def load_evidence_scores(path: str = "outputs/reports/evidence_strength.csv"
                         ) -> Dict[str, float]:
    """Return {theme: evidence_score} lookup."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    scores = {}
    for _, row in df.iterrows():
        theme = str(row.get("theme", "")).strip()
        score = row.get("evidence_score")
        if theme and score is not None:
            scores[theme] = float(score)
    log.info("Loaded %d evidence scores", len(scores))
    return scores


# ---------------------------------------------------------------------------
# 5.  Main orchestrator
# ---------------------------------------------------------------------------

def support_claims_with_references(
    paper_store: PaperStore,
    evidence_scores: Dict[str, float],
    report_paths: List[Path],
    citation_mode: str = "both",
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """For each statement in the given reports, find supporting papers and
    return a DataFrame with all claim–reference pairs."""
    all_rows: List[Dict] = []
    enriched_map: Dict[str, List[str]] = {}  # source_file -> lines to append

    for rp in report_paths:
        if not rp.exists():
            log.warning("Report not found: %s", rp)
            continue
        text = rp.read_text()
        source = str(rp)
        statements = extract_statements(text, source)
        log.info("  %s: %d statements extracted", rp.name, len(statements))
        enriched_lines: List[str] = []

        for stmt in statements:
            statement_text = stmt["statement"]
            matches = paper_store.search(
                statement_text,
                top_k=top_k,
                evidence_lookup=evidence_scores,
            )

            for paper, score in matches:
                apa_ref = _format_apa(paper)
                inline = _inline_citation(paper)
                theme = ""
                # Try to match statement to a theme
                for t_key in evidence_scores:
                    if t_key.lower()[:20] in statement_text.lower():
                        theme = t_key
                        break

                all_rows.append({
                    "Claim": statement_text[:200],
                    "Theme": theme,
                    "Supporting_Paper": paper["title"][:200],
                    "APA_Reference": apa_ref,
                    "DOI": paper.get("doi", ""),
                    "Confidence_Score": round(score, 3),
                    "Source_File": source,
                    "Section": stmt.get("section", ""),
                    "Line_Number": stmt.get("line_number", 0),
                })

            # Build enriched text for this statement
            if citation_mode in ("apa_inline", "both"):
                inline_str = "; ".join(_inline_citation(m[0]) for m in matches[:2])
                enriched_lines.append(f"{statement_text} ({inline_str})\n"
                                      if inline_str else f"{statement_text}\n")
            else:
                enriched_lines.append(f"{statement_text}\n")

            if citation_mode in ("apa_full_reference", "both") and matches:
                enriched_lines.append("\n**Supporting References:**\n")
                for i, (paper, score) in enumerate(matches, 1):
                    ref = _format_apa(paper)
                    enriched_lines.append(f"{i}. {ref}\n")
                enriched_lines.append("\n")

        enriched_map[source] = enriched_lines

    # Write enriched versions alongside originals
    for source, lines in enriched_map.items():
        src_path = Path(source)
        enriched_path = src_path.parent / f"{src_path.stem}_with_refs{src_path.suffix}"
        try:
            with open(enriched_path, "w") as f:
                f.writelines(lines)
            log.info("  Enriched -> %s", enriched_path)
        except Exception as e:
            log.warning("Could not write enriched %s: %s", enriched_path, e)

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Support claims with APA references.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--knowledge-base", type=str,
                        default="outputs/knowledge_base/knowledge_base.json")
    parser.add_argument("--evidence", type=str,
                        default="outputs/reports/evidence_strength.csv")
    parser.add_argument("--citation-metrics", type=str,
                        default="outputs/reports/citation_metrics.csv")
    parser.add_argument("--report", type=str, action="append", default=None,
                        help="Report file to process (repeatable)")
    parser.add_argument("--citation-mode", type=str, default="both",
                        choices=CITATION_MODES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=str,
                        default="outputs/references/claim_support_references.csv")
    args = parser.parse_args()

    # Determine report paths
    if args.report:
        report_paths = [Path(p) for p in args.report]
    else:
        report_paths = [Path(p) for p in DEFAULT_REPORTS]

    # Load data
    log.info("Loading paper store ...")
    paper_store = PaperStore(papers_csv=args.papers, kb_json=args.knowledge_base)
    paper_store.compute_embeddings()

    log.info("Loading evidence scores ...")
    evidence_scores = load_evidence_scores(args.evidence)

    log.info("Processing %d report(s) with citation_mode='%s', top_k=%d",
             len(report_paths), args.citation_mode, args.top_k)

    df = support_claims_with_references(
        paper_store, evidence_scores, report_paths,
        citation_mode=args.citation_mode, top_k=args.top_k,
    )

    # Write consolidated CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %d claim–reference pairs -> %s", len(df), out_path)

    print(f"\n--- Claim Support with References Complete ---")
    print(f"  Reports processed: {len(report_paths)}")
    print(f"  Claim–reference pairs: {len(df)}")
    print(f"  Citation mode: {args.citation_mode}")
    print()


if __name__ == "__main__":
    main()
