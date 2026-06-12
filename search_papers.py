#!/usr/bin/env python3
"""
search_papers.py — Multi-source scholarly literature search and deduplication.

Searches Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, and CORE
via their free/public API tiers.  Merges, deduplicates (DOI + semantic
title matching via MiniLM), and saves results as CSV and JSON.

Usage
-----
    python search_papers.py "query string"

Requirements (installed via pip)
--------------------------------
    requests, pandas, sentence-transformers, tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("search_papers")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_PER_SOURCE = 10
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 0.5  # seconds between requests
SIMILARITY_THRESHOLD = 0.85  # cosine similarity for duplicate detection

USER_AGENT = (
    "DigitalLitReview/1.0 (mailto:your-email@example.com) "
    "Using free scholarly APIs"
)

HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# Common schema field names
# ---------------------------------------------------------------------------
FIELDS = [
    "title",
    "authors",
    "year",
    "abstract",
    "doi",
    "venue",
    "source",
    "url",
    "citation_count",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _safe_get(d: dict, *keys, default: Any = "") -> Any:
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
    return d if d is not None else default


def _safe_request(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    method: str = "GET",
) -> Optional[requests.Response]:
    """Wrapper around *requests* with retry + rate-limit delay."""
    time.sleep(RATE_LIMIT_SLEEP)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method,
                url,
                params=params,
                headers=headers or HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                log.warning("Rate limited — sleeping %ds (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                log.warning("Request failed after %d attempts: %s", MAX_RETRIES, url)
                return None
            time.sleep(1)
    return None


def _build_record(
    title: str,
    authors: str,
    year: int,
    abstract: str,
    doi: str,
    venue: str,
    source: str,
    url: str,
    citation_count: int,
) -> Dict[str, Any]:
    return {
        "title": (title or "").strip(),
        "authors": (authors or "").strip(),
        "year": year if year else 0,
        "abstract": (abstract or "").strip(),
        "doi": (doi or "").strip().lower(),
        "venue": (venue or "").strip(),
        "source": source,
        "url": (url or "").strip(),
        "citation_count": citation_count if citation_count else 0,
    }


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------
def search_crossref(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search Crossref REST API."""
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": max_results,
        "sort": "relevance",
        "order": "desc",
    }
    resp = _safe_request(url, params=params)
    if resp is None:
        return []

    try:
        items = resp.json().get("message", {}).get("items", [])
    except (json.JSONDecodeError, KeyError):
        return []

    results = []
    for item in items:
        authors_list = item.get("author", [])
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors_list
        )
        results.append(
            _build_record(
                title=_safe_get(item, "title", default=[""])[0],
                authors=authors,
                year=_safe_get(item, "published-print", "date-parts", default=[[""]])[0][0]
                or _safe_get(item, "created", "date-parts", default=[[""]])[0][0],
                abstract=item.get("abstract", ""),
                doi=item.get("DOI", ""),
                venue=item.get("container-title", [""])[0],
                source="Crossref",
                url=item.get("URL", ""),
                citation_count=item.get("is-referenced-by-count", 0),
            )
        )
    return results


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def search_openalex(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search OpenAlex API."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": max_results,
        "sort": "relevance_score:desc",
    }
    resp = _safe_request(url, params=params)
    if resp is None:
        return []

    try:
        items = resp.json().get("results", [])
    except (json.JSONDecodeError, KeyError):
        return []

    results = []
    for item in items:
        authors_list = item.get("authorships", [])
        authors = ", ".join(
            a.get("author", {}).get("display_name", "") for a in authors_list
        )
        results.append(
            _build_record(
                title=item.get("title", ""),
                authors=authors,
                year=item.get("publication_year", 0),
                abstract=item.get("abstract_inverted_index", "")
                and _reconstruct_abstract(item.get("abstract_inverted_index", {})),
                doi=item.get("doi", "").replace("https://doi.org/", ""),
                venue=_safe_get(item, "primary_location", "source", "display_name"),
                source="OpenAlex",
                url=item.get("id", ""),
                citation_count=item.get("cited_by_count", 0),
            )
        )
    return results


def _reconstruct_abstract(inverted: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inverted:
        return ""
    word_positions = []
    for word, positions in inverted.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------
def search_semantic_scholar(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search Semantic Scholar free API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,abstract,externalIds,venue,url,citationCount",
    }
    resp = _safe_request(url, params=params)
    if resp is None:
        return []

    try:
        items = resp.json().get("data", [])
    except (json.JSONDecodeError, KeyError):
        return []

    results = []
    for item in items:
        authors_list = item.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list)
        results.append(
            _build_record(
                title=item.get("title", ""),
                authors=authors,
                year=item.get("year", 0),
                abstract=item.get("abstract", ""),
                doi=item.get("externalIds", {}).get("DOI", ""),
                venue=item.get("venue", ""),
                source="Semantic Scholar",
                url=item.get("url", ""),
                citation_count=item.get("citationCount", 0),
            )
        )
    return results


# ---------------------------------------------------------------------------
# PubMed (via NCBI E-utilities)
# ---------------------------------------------------------------------------
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search PubMed via NCBI E-utilities."""
    # Step 1 — get PMIDs
    search_url = f"{PUBMED_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = _safe_request(search_url, params=params)
    if resp is None:
        return []

    try:
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    except (json.JSONDecodeError, KeyError):
        return []
    if not id_list:
        return []

    # Step 2 — fetch details
    fetch_url = f"{PUBMED_BASE}/esummary.fcgi"
    params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
    resp = _safe_request(fetch_url, params=params)
    if resp is None:
        return []

    try:
        result_map = resp.json().get("result", {})
    except (json.JSONDecodeError, KeyError):
        return []

    results = []
    for pmid in id_list:
        item = result_map.get(pmid, {})
        authors_list = item.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list)
        results.append(
            _build_record(
                title=item.get("title", ""),
                authors=authors,
                year=item.get("pubdate", "")[:4],
                abstract="",
                doi="",
                venue=item.get("source", ""),
                source="PubMed",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                citation_count=0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
ARXIV_BASE = "http://export.arxiv.org/api/query"


def search_arxiv(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search arXiv API (Atom feed)."""
    params = {
        "search_query": f"all:{quote(query)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_BASE}?{urlencode(params)}"
    resp = _safe_request(url, headers={"User-Agent": USER_AGENT})
    if resp is None:
        return []

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", ).strip()
        summary = entry.findtext("atom:summary", default="").strip()
        published = entry.findtext("atom:published", default="")[:4]
        doi_tag = entry.find("arxiv:doi", ns)
        doi = doi_tag.text if doi_tag is not None else ""
        link_el = entry.find("atom:id", ns)
        url = link_el.text if link_el is not None else ""
        journal = entry.findtext("arxiv:journal_ref", ns)

        authors_list = entry.findall("atom:author", ns)
        authors = ", ".join(
            a.findtext("atom:name", default="") for a in authors_list
        )
        results.append(
            _build_record(
                title=title,
                authors=authors,
                year=int(published) if published.isdigit() else 0,
                abstract=summary,
                doi=doi,
                venue=journal or "arXiv",
                source="arXiv",
                url=url,
                citation_count=0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
CORE_BASE = "https://api.core.ac.uk/v3"


def search_core(query: str, max_results: int = DEFAULT_MAX_PER_SOURCE) -> List[Dict]:
    """Search CORE API (free tier)."""
    # CORE free tier requires an API key; try unauthenticated first (limited).
    url = f"{CORE_BASE}/search/works"
    params = {
        "q": query,
        "limit": min(max_results, 10),
        "offset": 0,
    }
    resp = _safe_request(
        url,
        params=params,
        headers={**HEADERS, "Accept": "application/json"},
    )
    if resp is None:
        return []

    try:
        items = resp.json().get("results", [])
    except (json.JSONDecodeError, KeyError):
        return []

    results = []
    for item in items:
        authors_list = item.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list)
        results.append(
            _build_record(
                title=item.get("title", ""),
                authors=authors,
                year=item.get("yearPublished", 0),
                abstract=item.get("abstract", ""),
                doi=item.get("doi", ""),
                venue=item.get("publisher", ""),
                source="CORE",
                url=item.get("sourceUrl", "") or item.get("downloadUrl", ""),
                citation_count=item.get("citationCount", 0),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_results(
    records: List[Dict],
    model: SentenceTransformer,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """Remove duplicate papers by DOI exact match and title semantic similarity."""

    def _doi_key(r: dict) -> str:
        return r.get("doi", "").strip().lower()

    def _title_key(r: dict) -> str:
        return r.get("title", "").strip().lower()

    # --- Phase 1: exact DOI dedup ---
    seen_dois: Dict[str, Dict] = {}
    doi_deduped: List[Dict] = []
    for rec in records:
        dk = _doi_key(rec)
        if dk and dk in seen_dois:
            # Keep record with more fields filled
            existing = seen_dois[dk]
            if sum(1 for v in rec.values() if v) > sum(1 for v in existing.values() if v):
                doi_deduped[-1] = rec
                seen_dois[dk] = rec
        else:
            seen_dois[dk] = rec
            doi_deduped.append(rec)

    # --- Phase 2: title semantic dedup ---
    titles = [_title_key(r) for r in doi_deduped]
    if not titles:
        return []

    embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(embeddings, embeddings).cpu()

    keep = [True] * len(doi_deduped)
    for i in range(len(doi_deduped)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(doi_deduped)):
            if not keep[j]:
                continue
            score = float(cosine_scores[i][j])
            if score >= threshold:
                # Keep the one with more populated fields
                if sum(1 for v in doi_deduped[j].values() if v) > sum(
                    1 for v in doi_deduped[i].values() if v
                ):
                    keep[i] = False
                else:
                    keep[j] = False

    deduped = [r for i, r in enumerate(doi_deduped) if keep[i]]
    return deduped


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(records: List[Dict], base_name: str = "search_results") -> None:
    """Save records to CSV and JSON."""
    csv_path = f"{base_name}.csv"
    json_path = f"{base_name}.json"

    if not records:
        log.warning("No records to save.")
        # Write empty files
        pd.DataFrame(columns=FIELDS).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump([], f)
        return

    df = pd.DataFrame(records)
    df = df[FIELDS]  # ensure column order

    df.to_csv(csv_path, index=False)
    log.info("Saved %d records → %s", len(df), csv_path)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    log.info("Saved %d records → %s", len(records), json_path)


# ---------------------------------------------------------------------------
# Display (tabular)
# ---------------------------------------------------------------------------
def display_results(records: List[Dict]) -> None:
    """Print a clean terminal table."""
    if not records:
        print("\nNo results to display.\n")
        return

    df = pd.DataFrame(records)
    print("\n" + "=" * 100)
    print(f"{'TITLE':<60} {'YEAR':<6} {'SOURCE':<18} {'CITATIONS':<10}")
    print("=" * 100)
    for _, row in df.iterrows():
        title = (row.get("title") or "")[:57]
        year = row.get("year", "")
        source = (row.get("source") or "")[:16]
        cites = row.get("citation_count", 0)
        print(f"{title:<60} {year:<6} {source:<18} {cites:<10}")
    print("=" * 100)
    print(f"Total: {len(df)} unique papers\n")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary(
    source_counts: Dict[str, int],
    duplicates_removed: int,
    final_count: int,
) -> None:
    """Print per-source retrieval stats."""
    print("\n--- Retrieval Summary ---")
    for src, count in sorted(source_counts.items()):
        print(f"  {src:<20} {count:>4} records")
    print(f"  {'Duplicates removed':<20} {duplicates_removed:>4}")
    print(f"  {'Final unique papers':<20} {final_count:>4}")
    print()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
SEARCHERS = {
    "Crossref": search_crossref,
    "OpenAlex": search_openalex,
    "Semantic Scholar": search_semantic_scholar,
    "PubMed": search_pubmed,
    "arXiv": search_arxiv,
    "CORE": search_core,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source scholarly literature search and deduplication."
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE,
        help=f"Max results per source (default: {DEFAULT_MAX_PER_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="search_results",
        help="Base name for output files (default: search_results)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=3,
        help="Number of parallel search threads (default: 3)",
    )
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        log.error("Query cannot be empty.")
        return

    print(f"\nSearching for: {query}\n")

    # --- Phase 1: parallel search across sources ---
    all_records: List[Dict] = []
    source_counts: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        fut_map = {
            pool.submit(fn, query, args.max): name
            for name, fn in SEARCHERS.items()
        }
        for fut in tqdm(
            as_completed(fut_map),
            total=len(fut_map),
            desc="Searching sources",
            unit="source",
        ):
            name = fut_map[fut]
            try:
                records = fut.result()
                if not isinstance(records, list):
                    records = []
            except Exception:
                log.error("Search '%s' failed:\n%s", name, traceback.format_exc())
                records = []
            source_counts[name] = len(records)
            all_records.extend(records)

    # --- Phase 2: deduplication ---
    log.info("Loading MiniLM model for semantic deduplication...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    total_before = len(all_records)

    deduped = deduplicate_results(all_records, model)
    duplicates_removed = total_before - len(deduped)
    log.info("Deduplication complete: %d → %d (-%d)", total_before, len(deduped), duplicates_removed)

    # --- Phase 3: save & display ---
    save_results(deduped, args.output)
    print_summary(source_counts, duplicates_removed, len(deduped))
    display_results(deduped)


if __name__ == "__main__":
    main()
