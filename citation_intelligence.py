#!/usr/bin/env python3
"""
citation_intelligence.py — Citation network analysis and field intelligence.

Fetches reference data from OpenAlex, builds a directed citation graph,
and identifies foundational papers, influential authors, citation clusters,
bottlenecks, and neglected (under-cited) papers.

Usage
-----
    python citation_intelligence.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
        [--skip-fetch]         # use cached data only
        [--max-refetch N]      # max OpenAlex lookups (default: all)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("citation_intelligence")

CURRENT_YEAR = datetime.now().year
CACHE_FILE = "outputs/citation_cache.json"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 0.3
USER_AGENT = (
    "DigitalLitReview/1.0 (mailto:your-email@example.com) "
    "Using free scholarly APIs"
)
HEADERS = {"User-Agent": USER_AGENT}
SEMANTIC_EDGE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_request(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    time.sleep(RATE_LIMIT_SLEEP)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                log.warning("Rate limited — sleeping %ds (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(1)
    return None


def _parse_authors(authors_str: str) -> List[str]:
    if not authors_str or authors_str == "nan":
        return []
    parts = [a.strip() for a in authors_str.replace(";", ",").split(",")]
    cleaned: List[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in ("", "nan", "none", "and"):
            cleaned.append(p)
    return cleaned


# ---------------------------------------------------------------------------
# OpenAlex reference fetching
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            log.warning("Cache corrupted; starting fresh.")
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    log.info("Cache saved (%d entries) -> %s", len(cache), CACHE_FILE)


def _fetch_openlex_work(doi: str) -> Optional[Dict]:
    url = f"https://api.openalex.org/works/doi:{doi}"
    resp = _safe_request(url)
    if resp is None:
        return None
    data = resp.json()
    ref_ids: List[str] = data.get("referenced_works", [])
    cited_by_count = data.get("cited_by_count", 0) or 0
    return {
        "doi": doi,
        "openalex_id": data.get("id", ""),
        "referenced_works": ref_ids,
        "cited_by_count": cited_by_count,
        "publication_year": data.get("publication_year"),
        "title": data.get("title", ""),
        "authorships": data.get("authorships", []),
    }


def _resolve_openalex_ids(ref_ids: List[str]) -> List[Optional[Dict]]:
    """Batch resolve OpenAlex IDs to DOIs."""
    results: List[Optional[Dict]] = []
    batch_size = 50
    for i in range(0, len(ref_ids), batch_size):
        batch = ref_ids[i:i + batch_size]
        ids_param = "|".join(batch)
        url = "https://api.openalex.org/works"
        resp = _safe_request(url, params={"filter": f"openalex_id:{ids_param}", "per_page": batch_size})
        if resp is None:
            results.extend([None] * len(batch))
            continue
        data = resp.json()
        results_map: Dict[str, Dict] = {}
        for result in data.get("results", []):
            oid = result.get("id", "")
            results_map[oid] = {
                "doi": (result.get("doi") or "").replace("https://doi.org/", ""),
                "title": result.get("title", ""),
                "cited_by_count": result.get("cited_by_count", 0),
                "publication_year": result.get("publication_year"),
            }
        results.extend([results_map.get(oid) for oid in batch])
    return results


def fetch_reference_data(
    dois: List[str],
    cache: Dict[str, Any],
    max_refetch: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch reference data from OpenAlex for each DOI. Uses and updates cache."""
    to_fetch = [d for d in dois if d and d not in cache]
    if max_refetch is not None:
        to_fetch = to_fetch[:max_refetch]

    if to_fetch:
        log.info("Fetching OpenAlex data for %d DOIs...", len(to_fetch))
        with ThreadPoolExecutor(max_workers=5) as pool:
            fut_map = {pool.submit(_fetch_openlex_work, doi): doi for doi in to_fetch}
            for fut in tqdm(as_completed(fut_map), total=len(to_fetch), desc="OpenAlex"):
                doi = fut_map[fut]
                try:
                    result = fut.result()
                    if result:
                        cache[doi] = result
                except Exception as e:
                    log.debug("Fetch failed for %s: %s", doi, e)

        _save_cache(cache)

    log.info("Cache has %d / %d DOIs", len(cache), len(dois))
    return cache


# ---------------------------------------------------------------------------
# Build citation graph
# ---------------------------------------------------------------------------

def build_citation_graph(
    df: pd.DataFrame,
    cache: Dict[str, Any],
    model: Optional[SentenceTransformer] = None,
) -> nx.DiGraph:
    """Build directed citation graph from OpenAlex reference data.

    Falls back to semantic similarity edges when OpenAlex data is missing.
    """
    G = nx.DiGraph()
    doi_to_idx: Dict[str, int] = {}
    for idx, row in df.iterrows():
        doi = str(row.get("doi", "")).strip().lower()
        if doi and doi not in ("nan", "", "none"):
            doi_to_idx[doi] = idx
            title = str(row.get("title", ""))[:80]
            G.add_node(doi, title=title, idx=idx,
                       year=int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                       citation_count=int(row.get("citation_count", 0)))

    # Add edges from OpenAlex reference data
    edge_count = 0
    for doi in doi_to_idx:
        entry = cache.get(doi)
        if entry and entry.get("referenced_works"):
            ref_ids = entry["referenced_works"]
            resolved = _resolve_openalex_ids(ref_ids)
            for ref in resolved:
                if ref and ref.get("doi"):
                    ref_doi = ref["doi"].strip().lower()
                    if ref_doi in doi_to_idx:
                        G.add_edge(doi, ref_doi)
                        edge_count += 1

    log.info("Citation graph: %d nodes, %d directed edges (from OpenAlex)", G.number_of_nodes(), edge_count)

    # Fallback: semantic similarity for papers with missing reference data
    if model is not None and G.number_of_edges() < len(doi_to_idx):
        _add_semantic_edges(G, df, doi_to_idx, model)

    return G


def _add_semantic_edges(
    G: nx.DiGraph, df: pd.DataFrame, doi_to_idx: Dict[str, int], model: SentenceTransformer,
) -> None:
    """Add undirected edges based on abstract similarity for disconnected nodes."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return

    texts: List[str] = []
    node_dois: List[str] = []
    for doi in nodes:
        idx = doi_to_idx[doi]
        title = str(df.at[idx, "title"] if idx in df.index else "")
        abstract = str(df.at[idx, "abstract"] if idx in df.index else "")
        texts.append(f"{title}. {abstract}")
        node_dois.append(doi)

    log.info("Computing semantic similarity for %d papers...", len(texts))
    embs = model.encode(texts, show_progress_bar=True, batch_size=32)
    sim = util.cos_sim(embs, embs).numpy()

    added = 0
    for i in range(len(node_dois)):
        for j in range(i + 1, len(node_dois)):
            if sim[i][j] >= SEMANTIC_EDGE_THRESHOLD:
                G.add_edge(node_dois[i], node_dois[j], weight=float(sim[i][j]), semantic=True)
                added += 1
    log.info("Added %d semantic similarity edges", added)


# ---------------------------------------------------------------------------
# Network metrics
# ---------------------------------------------------------------------------

def compute_network_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Compute PageRank, betweenness, degree, and hub/authority scores."""
    metrics: Dict[str, Dict[str, float]] = {}

    if G.number_of_nodes() == 0:
        return metrics

    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    try:
        betweenness = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()))
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    try:
        hubs, authorities = nx.hits(G, max_iter=200, tol=1e-6, nstart=None)
    except Exception:
        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
            hubs = {n: pr.get(n, 0.0) * G.out_degree(n) for n in G.nodes()}
            authorities = {n: pr.get(n, 0.0) * G.in_degree(n) for n in G.nodes()}
        except Exception:
            hubs = {n: 0.0 for n in G.nodes()}
            authorities = {n: 0.0 for n in G.nodes()}

    for n in G.nodes():
        metrics[n] = {
            "pagerank": round(pagerank.get(n, 0.0), 6),
            "betweenness": round(betweenness.get(n, 0.0), 6),
            "in_degree": in_degree.get(n, 0),
            "out_degree": out_degree.get(n, 0),
            "hub_score": round(hubs.get(n, 0.0), 6),
            "authority_score": round(authorities.get(n, 0.0), 6),
        }

    return metrics


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def identify_foundational_papers(
    df: pd.DataFrame,
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    doi_to_idx: Dict[str, int],
    top_n: int = 10,
) -> List[Dict]:
    """Papers with high PageRank + early year + high citation count."""
    scores: List[Dict] = []
    for doi, m in metrics.items():
        idx = doi_to_idx.get(doi)
        if idx is None or idx not in df.index:
            continue
        row = df.loc[idx]
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        age = max(1, CURRENT_YEAR - year)
        # Composite score: PageRank * (citations / age) * recency bonus
        foundational_score = m["pagerank"] * (1 + np.log1p(cites)) * (1 + 1.0 / age)
        scores.append({
            "doi": doi,
            "title": str(row.get("title", ""))[:120],
            "authors": str(row.get("authors", ""))[:80],
            "year": year,
            "citation_count": cites,
            "pagerank": m["pagerank"],
            "foundational_score": round(foundational_score, 4),
        })
    scores.sort(key=lambda x: x["foundational_score"], reverse=True)
    return scores[:top_n]


def identify_influential_authors(
    df: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    top_n: int = 10,
) -> List[Dict]:
    """Aggregate influence per author: total citations + paper count + avg PageRank."""
    author_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"papers": 0, "total_citations": 0, "pagerank_sum": 0.0, "dois": []}
    )
    for doi, m in metrics.items():
        # Find the row in df with this DOI
        match = df[df["doi"].astype(str).str.strip().str.lower() == doi]
        if match.empty:
            continue
        row = match.iloc[0]
        authors_list = _parse_authors(str(row.get("authors", "")))
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        for author in authors_list:
            ad = author_data[author]
            ad["papers"] += 1
            ad["total_citations"] += cites
            ad["pagerank_sum"] += m["pagerank"]
            ad["dois"].append(doi)

    result: List[Dict] = []
    for author, ad in author_data.items():
        avg_pr = ad["pagerank_sum"] / max(ad["papers"], 1)
        influence_score = ad["total_citations"] * (1 + np.log1p(ad["papers"])) * (1 + avg_pr)
        result.append({
            "author": author,
            "paper_count": ad["papers"],
            "total_citations": ad["total_citations"],
            "avg_pagerank": round(avg_pr, 6),
            "influence_score": round(influence_score, 2),
        })
    result.sort(key=lambda x: x["influence_score"], reverse=True)
    return result[:top_n]


def detect_citation_clusters(G: nx.DiGraph) -> List[Dict]:
    """Detect citation communities using label propagation / connected components."""
    if G.number_of_nodes() == 0:
        return []

    # Use weakly connected components as clusters
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)

    clusters: List[Dict] = []
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        try:
            density = nx.density(subgraph)
        except Exception:
            density = 0.0
        clusters.append({
            "cluster_id": i,
            "size": len(comp),
            "density": round(density, 4),
            "nodes": list(comp)[:20],  # top 20 DOIs
        })
    return clusters


def detect_citation_bottlenecks(
    metrics: Dict[str, Dict[str, float]],
    top_n: int = 10,
) -> List[Dict]:
    """Papers with highest betweenness centrality."""
    bottleneck_list = [
        {"doi": doi, "betweenness": m["betweenness"]}
        for doi, m in metrics.items() if m["betweenness"] > 0
    ]
    bottleneck_list.sort(key=lambda x: x["betweenness"], reverse=True)
    return bottleneck_list[:top_n]


def detect_neglected_papers(
    df: pd.DataFrame,
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    doi_to_idx: Dict[str, int],
    top_n: int = 10,
) -> List[Dict]:
    """Papers with low citation count but high PageRank / authority (under-cited gems)."""
    scores: List[Dict] = []
    for doi, m in metrics.items():
        idx = doi_to_idx.get(doi)
        if idx is None or idx not in df.index:
            continue
        row = df.loc[idx]
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        if cites > 20:
            continue  # already well-cited
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        age = max(1, CURRENT_YEAR - year)
        neglect_score = m["pagerank"] * (1 + m["authority_score"]) * (1 + 1.0 / age) / max(1 + np.log1p(cites), 0.1)
        scores.append({
            "doi": doi,
            "title": str(row.get("title", ""))[:120],
            "authors": str(row.get("authors", ""))[:80],
            "year": year,
            "citation_count": cites,
            "pagerank": m["pagerank"],
            "authority_score": m["authority_score"],
            "neglect_score": round(neglect_score, 4),
        })
    scores.sort(key=lambda x: x["neglect_score"], reverse=True)
    return scores[:top_n]


def identify_core_references(
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    top_n: int = 10,
) -> List[Dict]:
    """Papers acting as hubs (high out-degree / hub score) — core reference lists."""
    hub_list = [
        {"doi": doi, "hub_score": m["hub_score"], "out_degree": m["out_degree"]}
        for doi, m in metrics.items() if m["out_degree"] > 0
    ]
    hub_list.sort(key=lambda x: x["hub_score"], reverse=True)
    return hub_list[:top_n]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _doi_title(doi: str, G: nx.DiGraph, df: pd.DataFrame, doi_to_idx: Dict[str, int]) -> str:
    """Get short title for a DOI."""
    if doi in G.nodes:
        raw = G.nodes[doi].get("title", "")
        if raw and str(raw).strip().lower() not in ("nan", "none", ""):
            return str(raw)[:80]
    idx = doi_to_idx.get(doi)
    if idx is not None and idx in df.index:
        raw = str(df.at[idx, "title"])
        if raw.strip().lower() not in ("nan", "none", ""):
            return raw[:80]
    return f"[Untitled — {doi[:30]}]"


def generate_report(
    df: pd.DataFrame,
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    foundational: List[Dict],
    influential_authors: List[Dict],
    clusters: List[Dict],
    bottlenecks: List[Dict],
    neglected: List[Dict],
    core_refs: List[Dict],
    doi_to_idx: Dict[str, int],
) -> str:
    """Generate citation_report.md."""
    lines: List[str] = []
    lines.append("# Citation Intelligence Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Corpus size:** {len(df)} papers")
    lines.append(f"- **Graph nodes:** {G.number_of_nodes()}")
    lines.append(f"- **Graph edges:** {G.number_of_edges()}")
    try:
        lines.append(f"- **Graph density:** {nx.density(G):.6f}")
    except Exception:
        pass
    lines.append("")

    # 1. Foundational Papers
    lines.append("## 1. Foundational Papers")
    lines.append("")
    lines.append("Papers that combine high PageRank, early publication, and strong citation impact.")
    lines.append("")
    if foundational:
        lines.append("| Rank | Title | Authors | Year | Citations | PageRank | Foundational Score |")
        lines.append("|------|-------|---------|------|-----------|----------|---------------------|")
        for rank, p in enumerate(foundational, 1):
            lines.append(f"| {rank} | {p['title'][:60]} | {p['authors'][:40]} | {p['year']} | {p['citation_count']} | {p['pagerank']:.4f} | {p['foundational_score']:.2f} |")
        lines.append("")
    else:
        lines.append("_Insufficient citation data to identify foundational papers._")
        lines.append("")

    # 2. Influential Authors
    lines.append("## 2. Most Influential Authors")
    lines.append("")
    lines.append("Aggregated influence based on total citations, paper count, and average PageRank.")
    lines.append("")
    if influential_authors:
        lines.append("| Rank | Author | Papers | Total Citations | Avg PageRank | Influence Score |")
        lines.append("|------|--------|--------|-----------------|--------------|-----------------|")
        for rank, a in enumerate(influential_authors, 1):
            lines.append(f"| {rank} | {a['author'][:40]} | {a['paper_count']} | {a['total_citations']} | {a['avg_pagerank']:.4f} | {a['influence_score']:.2f} |")
        lines.append("")
    else:
        lines.append("_Insufficient data to rank authors._")
        lines.append("")

    # 3. Citation Clusters
    lines.append("## 3. Citation Clusters / Communities")
    lines.append("")
    lines.append("Groups of papers densely connected by citation links.")
    lines.append("")
    if clusters:
        multi = [c for c in clusters if c["size"] > 1]
        single = [c for c in clusters if c["size"] == 1]
        for c in multi:
            lines.append(f"### Cluster {c['cluster_id'] + 1} ({c['size']} papers, density={c['density']:.4f})")
            lines.append("")
            for doi in c["nodes"][:10]:
                title = _doi_title(doi, G, df, doi_to_idx)
                lines.append(f"- {title}")
            if len(c["nodes"]) > 10:
                lines.append(f"- _... and {len(c['nodes']) - 10} more_")
            lines.append("")
        if single:
            lines.append(f"### Isolated Papers ({len(single)} papers)")
            lines.append("")
            for c in single[:10]:
                title = _doi_title(c["nodes"][0], G, df, doi_to_idx)
                lines.append(f"- {title}")
            if len(single) > 10:
                lines.append(f"- _... and {len(single) - 10} more_")
            lines.append("")
    else:
        lines.append("_No citation clusters detected._")
        lines.append("")

    # 4. Citation Bottlenecks
    lines.append("## 4. Citation Bottlenecks")
    lines.append("")
    lines.append("Papers with high betweenness centrality — they bridge disparate research threads.")
    lines.append("")
    if bottlenecks:
        for rank, b in enumerate(bottlenecks, 1):
            title = _doi_title(b["doi"], G, df, doi_to_idx)
            lines.append(f"{rank}. **{title}** — betweenness {b['betweenness']:.4f}")
        lines.append("")
    else:
        lines.append("_No bottleneck papers identified._")
        lines.append("")

    # 5. Neglected Papers (Hidden Gems)
    lines.append("## 5. Neglected Papers — Hidden Gems")
    lines.append("")
    lines.append("Under-cited papers with high structural importance (high PageRank but low citations).")
    lines.append("")
    if neglected:
        for rank, n in enumerate(neglected, 1):
            lines.append(f"{rank}. **{n['title'][:70]}** ({n['year']}) — {n['citation_count']} citations, "
                         f"PageRank {n['pagerank']:.4f}, neglect score {n['neglect_score']:.2f}")
        lines.append("")
    else:
        lines.append("_No neglected papers identified._")
        lines.append("")

    # 6. Core References
    lines.append("## 6. Core References (Hub Papers)")
    lines.append("")
    lines.append("Papers with high hub scores — they cite broadly and anchor the reference network.")
    lines.append("")
    if core_refs:
        for rank, c in enumerate(core_refs, 1):
            title = _doi_title(c["doi"], G, df, doi_to_idx)
            lines.append(f"{rank}. **{title}** — hub score {c['hub_score']:.4f}, out-degree {c['out_degree']}")
        lines.append("")
    else:
        lines.append("_No core reference papers identified._")
        lines.append("")

    # 7. Network Statistics Table
    lines.append("## 7. Network Statistics Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Papers in graph | {G.number_of_nodes()} |")
    lines.append(f"| Citation edges | {G.number_of_edges()} |")
    try:
        lines.append(f"| Graph density | {nx.density(G):.6f} |")
    except Exception:
        pass
    if metrics:
        pr_vals = [m["pagerank"] for m in metrics.values()]
        bt_vals = [m["betweenness"] for m in metrics.values()]
        lines.append(f"| Avg PageRank | {np.mean(pr_vals):.6f} |")
        lines.append(f"| Max PageRank | {max(pr_vals):.6f} |")
        lines.append(f"| Avg betweenness | {np.mean(bt_vals):.6f} |")
        lines.append(f"| Max betweenness | {max(bt_vals):.6f} |")
    lines.append("")

    # 8. Methodological Note
    lines.append("## 8. Methodological Note")
    lines.append("")
    lines.append("Citation data is sourced from OpenAlex. The citation graph includes only edges "
                 "where both the source and target papers are present in the corpus. "
                 "Semantic similarity edges (cosine ≥ 0.75 on MiniLM embeddings) are added "
                 "as a fallback where OpenAlex reference data is unavailable. "
                 "PageRank (alpha=0.85) and betweenness centrality are computed on the full directed graph.")
    lines.append("")
    if G.number_of_nodes() < len(df):
        lines.append(f"> **Note:** Only {G.number_of_nodes()} of {len(df)} papers have resolvable DOIs "
                     "and could be included in the citation graph.")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by citation_intelligence.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Citation network analysis and field intelligence."
    )
    parser.add_argument("--papers", type=str, default="search_results.csv",
                        help="Input CSV from search_papers.py")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv",
                        help="Consensus themes CSV (optional)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip OpenAlex API calls; use cached data only")
    parser.add_argument("--max-refetch", type=int, default=None,
                        help="Max DOIs to fetch from OpenAlex")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of results per category (default: 10)")
    args = parser.parse_args()

    papers_path = Path(args.papers)
    if not papers_path.exists():
        log.error("Input file not found: %s", papers_path)
        return

    df = pd.read_csv(args.papers)
    log.info("Loaded %d papers from %s", len(df), args.papers)

    # Load consensus themes (optional)
    consensus_path = Path(args.consensus)
    has_consensus = consensus_path.exists()
    if has_consensus:
        df_consensus = pd.read_csv(consensus_path)
        log.info("Loaded %d consensus records from %s", len(df_consensus), args.consensus)

    # Gather DOIs
    dois: List[str] = []
    for d in df["doi"]:
        d_str = str(d).strip().lower()
        if d_str and d_str not in ("nan", "none", ""):
            dois.append(d_str)

    # Load cache
    cache = _load_cache()

    # Fetch reference data
    if not args.skip_fetch and dois:
        cache = fetch_reference_data(dois, cache, max_refetch=args.max_refetch)

    # DOI -> index map
    doi_to_idx: Dict[str, int] = {}
    for idx, row in df.iterrows():
        d = str(row.get("doi", "")).strip().lower()
        if d and d not in ("nan", "none", ""):
            doi_to_idx[d] = idx

    # Load embedding model for semantic fallback
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build citation graph
    G = build_citation_graph(df, cache, model=model)

    # Compute metrics
    metrics = compute_network_metrics(G)
    log.info("Computed network metrics for %d nodes", len(metrics))

    # Run detections
    foundational = identify_foundational_papers(df, G, metrics, doi_to_idx, top_n=args.top_n)
    influential_authors = identify_influential_authors(df, metrics, top_n=args.top_n)
    clusters = detect_citation_clusters(G)
    bottlenecks = detect_citation_bottlenecks(metrics, top_n=args.top_n)
    neglected = detect_neglected_papers(df, G, metrics, doi_to_idx, top_n=args.top_n)
    core_refs = identify_core_references(G, metrics, top_n=args.top_n)

    # Save citation_network.csv (edge list)
    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source_doi": u,
            "target_doi": v,
            "source_title": _doi_title(u, G, df, doi_to_idx),
            "target_title": _doi_title(v, G, df, doi_to_idx),
            "weight": data.get("weight", 1.0),
            "edge_type": "semantic" if data.get("semantic") else "citation",
        })
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(out_dir / "citation_network.csv", index=False)
    log.info("Saved %d edges -> %s", len(edge_df), out_dir / "citation_network.csv")

    # Save citation_metrics.csv
    metrics_rows = []
    for doi, m in metrics.items():
        idx = doi_to_idx.get(doi)
        title = ""
        year = 0
        cites = 0
        if idx is not None and idx in df.index:
            title = str(df.at[idx, "title"])[:120]
            year = int(df.at[idx, "year"]) if pd.notna(df.at[idx, "year"]) else 0
            cites = int(df.at[idx, "citation_count"]) if pd.notna(df.at[idx, "citation_count"]) else 0
        metrics_rows.append({
            "doi": doi,
            "title": title,
            "year": year,
            "citation_count": cites,
            "pagerank": m["pagerank"],
            "betweenness": m["betweenness"],
            "in_degree": m["in_degree"],
            "out_degree": m["out_degree"],
            "hub_score": m["hub_score"],
            "authority_score": m["authority_score"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("pagerank", ascending=False)
    metrics_df.to_csv(out_dir / "citation_metrics.csv", index=False)
    log.info("Saved %d metrics -> %s", len(metrics_df), out_dir / "citation_metrics.csv")

    # Save foundational / influential / neglected lists as CSVs
    if foundational:
        pd.DataFrame(foundational).to_csv(out_dir / "foundational_papers.csv", index=False)
        log.info("Saved %d -> %s", len(foundational), out_dir / "foundational_papers.csv")
    if influential_authors:
        pd.DataFrame(influential_authors).to_csv(out_dir / "influential_authors.csv", index=False)
        log.info("Saved %d -> %s", len(influential_authors), out_dir / "influential_authors.csv")
    if bottlenecks:
        pd.DataFrame(bottlenecks).to_csv(out_dir / "citation_bottlenecks.csv", index=False)
        log.info("Saved %d -> %s", len(bottlenecks), out_dir / "citation_bottlenecks.csv")
    if neglected:
        pd.DataFrame(neglected).to_csv(out_dir / "hidden_gems.csv", index=False)
        log.info("Saved %d -> %s", len(neglected), out_dir / "hidden_gems.csv")

    # Generate report
    report_md = generate_report(
        df, G, metrics, foundational, influential_authors,
        clusters, bottlenecks, neglected, core_refs, doi_to_idx,
    )
    report_path = out_dir / "citation_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    log.info("Saved -> %s", report_path)

    # Print summary
    print()
    print("--- Citation Intelligence Complete ---")
    print(f"  Papers in graph:    {G.number_of_nodes()}")
    print(f"  Citation edges:     {G.number_of_edges()}")
    print(f"  Foundational:       {len(foundational)} papers")
    print(f"  Influential authors: {len(influential_authors)}")
    print(f"  Clusters:           {len(clusters)}")
    print(f"  Bottlenecks:        {len(bottlenecks)}")
    print(f"  Hidden gems:        {len(neglected)}")
    print(f"  Report:             {report_path}")
    print()


if __name__ == "__main__":
    main()
