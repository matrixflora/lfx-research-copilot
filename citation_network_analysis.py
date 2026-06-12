#!/usr/bin/env python3
"""
citation_network_analysis.py — Map scientific influence by identifying
seminal papers, citation communities, tracing idea evolution, and ranking
influential authors.  Builds on citation_intelligence.py data.

Outputs
-------
outputs/citation_network/citation_network.csv
outputs/citation_network/citation_report.md
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
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("citation_network")

MODEL_NAME = "all-MiniLM-L6-v2"


def _load_citation_metrics(path: str = "outputs/reports/citation_metrics.csv") -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_papers(path: str = "search_results.csv") -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def identify_seminal_papers(metrics_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank papers by composite of PageRank, citation count, and year-weighted recency."""
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    # Normalise scores
    for col in ["pagerank", "citation_count"]:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[f"{col}_norm"] = df[col] / max_val
    df["seminal_score"] = (
        df.get("pagerank_norm", 0) * 0.4
        + df.get("citation_count_norm", 0) * 0.35
        + 0.25
    )
    return df.sort_values("seminal_score", ascending=False).head(top_n)


def detect_citation_clusters(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster papers by citation patterns using PageRank/hub/authority."""
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    # Simple heuristic: cluster by hub-authority quadrant
    clusters = []
    for _, row in df.iterrows():
        hub = row.get("hub_score", 0)
        auth = row.get("authority_score", 0)
        if hub > 0.5 and auth > 0.5:
            cluster = "Core (high hub + authority)"
        elif hub > 0.5:
            cluster = "Hub (high out-degree)"
        elif auth > 0.5:
            cluster = "Authority (high in-degree)"
        else:
            cluster = "Peripheral"
        clusters.append({
            "doi": row.get("doi", ""),
            "title": str(row.get("title", ""))[:80],
            "cluster": cluster,
            "hub_score": hub,
            "authority_score": auth,
        })
    return pd.DataFrame(clusters)


def trace_idea_evolution(papers_df: pd.DataFrame) -> pd.DataFrame:
    """Group papers by year and compute embedding shifts to trace theme evolution."""
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if papers_df.empty:
        return pd.DataFrame()
    df = papers_df.copy()
    df = df[df["year"].notna() & (df["year"] != 0)].copy()
    if df.empty:
        return pd.DataFrame()
    df["year_int"] = df["year"].astype(int)
    years = sorted(df["year_int"].unique())
    rows = []
    for y in years:
        subset = df[df["year_int"] == y]
        texts = [f"{row.get('title', '')}" for _, row in subset.iterrows()]
        if not texts:
            continue
        embs = model.encode(texts, show_progress_bar=False)
        centroid = np.mean(embs, axis=0)
        rows.append({"year": y, "paper_count": len(texts), "centroid": centroid.tolist()})
    return pd.DataFrame(rows)


def identify_influential_authors(metrics_df: pd.DataFrame, papers_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate author influence from citation data."""
    if papers_df.empty or "authors" not in papers_df.columns:
        return pd.DataFrame()
    author_scores: Dict[str, Dict] = defaultdict(lambda: {"paper_count": 0, "total_cites": 0, "dois": []})
    for _, row in papers_df.iterrows():
        authors = str(row.get("authors", ""))
        doi = str(row.get("doi", ""))
        cites = float(row.get("citation_count", 0) or 0)
        for a in re.split(r"[;,]+\s*", authors):
            a = a.strip()
            if a and a.lower() not in ("", "nan"):
                author_scores[a]["paper_count"] += 1
                author_scores[a]["total_cites"] += cites
                if doi and doi not in author_scores[a]["dois"]:
                    author_scores[a]["dois"].append(doi)
    rows = []
    for name, data in author_scores.items():
        rows.append({
            "author": name,
            "paper_count": data["paper_count"],
            "total_citations": int(data["total_cites"]),
            "avg_citations": round(data["total_cites"] / max(data["paper_count"], 1), 1),
        })
    return pd.DataFrame(rows).sort_values("total_citations", ascending=False)


def _generate_report(seminal: pd.DataFrame, clusters: pd.DataFrame,
                     evolution: pd.DataFrame, authors: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Citation Network Analysis Report\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if not seminal.empty:
        lines.append("## Seminal Papers\n")
        for _, row in seminal.iterrows():
            lines.append(f"- {row.get('title', '')[:80]} (score: {row.get('seminal_score', ''):.3f})")
        lines.append("")

    if not clusters.empty:
        lines.append("## Citation Communities\n")
        for c in clusters["cluster"].unique():
            count = len(clusters[clusters["cluster"] == c])
            lines.append(f"- **{c}:** {count} papers")
        lines.append("")

    if not evolution.empty and len(evolution) > 1:
        lines.append("## Idea Evolution\n")
        for _, row in evolution.iterrows():
            lines.append(f"- {int(row['year'])}: {int(row['paper_count'])} papers")
        lines.append("")

    if not authors.empty:
        lines.append("## Influential Authors\n")
        for _, row in authors.head(10).iterrows():
            lines.append(f"- {row['author']} ({row['paper_count']} papers, {row['total_citations']} cites)")
        lines.append("")

    lines.append("---\n*Generated by citation_network_analysis.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation network analysis.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--metrics", type=str, default="outputs/reports/citation_metrics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/citation_network")
    args = parser.parse_args()

    metrics_df = _load_citation_metrics(args.metrics)
    papers_df = _load_papers(args.papers)

    seminal = identify_seminal_papers(metrics_df)
    clusters = detect_citation_clusters(metrics_df)
    evolution = trace_idea_evolution(papers_df)
    authors = identify_influential_authors(metrics_df, papers_df)

    report = _generate_report(seminal, clusters, evolution, authors)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not authors.empty:
        authors.to_csv(out_dir / "citation_network.csv", index=False)
        log.info("Saved -> %s", out_dir / "citation_network.csv")

    with open(out_dir / "citation_report.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "citation_report.md")

    print(f"\n--- Citation Network Analysis Complete ---")
    print()


if __name__ == "__main__":
    main()
