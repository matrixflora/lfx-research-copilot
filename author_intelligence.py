#!/usr/bin/env python3
"""
author_intelligence.py — Author-level analytics: influence, emerging talent,
collaboration networks, institutional leadership, and country trends.

Usage
-----
    python author_intelligence.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("author_intelligence")

CURRENT_YEAR = datetime.now().year


def _parse_authors(authors_str: str) -> List[str]:
    if not authors_str or str(authors_str).lower() in ("nan", "none", ""):
        return []
    parts = re.split(r"[;,]", str(authors_str))
    cleaned: List[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in ("", "nan", "none", "and"):
            cleaned.append(p)
    return cleaned


def identify_top_authors(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Rank authors by total citations, paper count, and h-index approximation."""
    author_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "citations": [], "years": [], "themes": set()})
    for _, row in df.iterrows():
        authors = _parse_authors(row.get("authors", ""))
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        theme = row.get("consensus_theme", "")
        for a in authors:
            ad = author_data[a]
            ad["papers"] += 1
            ad["citations"].append(cites)
            ad["years"].append(year)
            if theme:
                ad["themes"].add(theme)

    rows = []
    for author, ad in author_data.items():
        cites_sorted = sorted(ad["citations"], reverse=True)
        h_index = sum(1 for i, c in enumerate(cites_sorted, 1) if c >= i)
        rows.append({
            "author": author,
            "paper_count": ad["papers"],
            "total_citations": sum(ad["citations"]),
            "avg_citations": round(np.mean(ad["citations"]), 1) if ad["citations"] else 0.0,
            "h_index": h_index,
            "earliest_year": min(ad["years"]) if ad["years"] else 0,
            "latest_year": max(ad["years"]) if ad["years"] else 0,
            "themes": "; ".join(sorted(ad["themes"])),
        })
    result = pd.DataFrame(rows)
    result = result.sort_values("total_citations", ascending=False).head(top_n).reset_index(drop=True)
    return result


def identify_emerging_authors(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Authors with recent activity and rising trajectory."""
    author_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "citations": [], "years": [], "recent_cites": 0})
    cutoff = CURRENT_YEAR - 3
    for _, row in df.iterrows():
        authors = _parse_authors(row.get("authors", ""))
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        for a in authors:
            ad = author_data[a]
            ad["papers"] += 1
            ad["citations"].append(cites)
            ad["years"].append(year)
            if year >= cutoff:
                ad["recent_cites"] += cites

    rows = []
    for author, ad in author_data.items():
        if ad["recent_cites"] == 0:
            continue
        growth = ad["recent_cites"] / max(sum(ad["citations"]) - ad["recent_cites"], 1)
        rows.append({
            "author": author,
            "paper_count": ad["papers"],
            "total_citations": sum(ad["citations"]),
            "recent_citations": ad["recent_cites"],
            "growth_ratio": round(growth, 2),
            "latest_year": max(ad["years"]) if ad["years"] else 0,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("growth_ratio", ascending=False).head(top_n).reset_index(drop=True)
    return result


def identify_collaboration_networks(df: pd.DataFrame) -> pd.DataFrame:
    """Build co-author edge list from shared paper affiliations."""
    edges: List[Dict] = []
    for _, row in df.iterrows():
        authors = _parse_authors(row.get("authors", ""))
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                edges.append({
                    "author_a": authors[i],
                    "author_b": authors[j],
                    "paper_title": str(row.get("title", ""))[:80],
                    "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                })
    result = pd.DataFrame(edges)
    if not result.empty:
        collab_count = result.groupby(["author_a", "author_b"]).size().reset_index(name="co_author_count")
        result = result.drop_duplicates(subset=["author_a", "author_b"]).merge(collab_count, on=["author_a", "author_b"])
        result = result.sort_values("co_author_count", ascending=False).reset_index(drop=True)
    return result


def identify_institutional_leaders(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Extract institutional affiliations from author strings (heuristic)."""
    inst_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "citations": 0, "authors": set()})
    for _, row in df.iterrows():
        authors = _parse_authors(row.get("authors", ""))
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        # Heuristic: last part of author string before comma may be institution
        raw = str(row.get("authors", ""))
        parts = re.split(r"[;,]", raw)
        for p in parts:
            p = p.strip()
            if not p or p.lower() in ("nan", "none", ""):
                continue
            # Check if this looks like an institution (has common keywords)
            inst_kw = re.findall(r"(University|Institute|College|School|Centre|Center|Lab|Laboratory|Department|Academy|Foundation|Organization|Agency|Ministry|Hospital|Corporation|Inc\.|Ltd\.|LLC)", p, re.IGNORECASE)
            if inst_kw:
                inst_data[p]["papers"] += 1
                inst_data[p]["citations"] += cites
                for a in authors:
                    if a.lower() not in p.lower():
                        inst_data[p]["authors"].add(a)

    rows = []
    for inst, d in inst_data.items():
        rows.append({
            "institution": inst[:100],
            "paper_count": d["papers"],
            "total_citations": d["citations"],
            "unique_authors": len(d["authors"]),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["paper_count", "total_citations"], ascending=False).head(top_n).reset_index(drop=True)
    return result


def identify_country_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic country detection from author/institution strings."""
    country_map = {
        "USA": ["usa", "united states", "u.s.", "harvard", "stanford", "mit", "berkeley"],
        "UK": ["uk", "united kingdom", "england", "oxford", "cambridge", "london"],
        "Australia": ["australia", "sydney", "melbourne", "queensland"],
        "Canada": ["canada", "toronto", "vancouver", "mcgill", "ubc"],
        "Germany": ["germany", "berlin", "munich", "max planck", "fraunhofer"],
        "France": ["france", "paris", "sorbonne", "cnrs", "inria"],
        "India": ["india", "indian", "iit", "delhi", "mumbai", "bangalore"],
        "China": ["china", "beijing", "shanghai", "hong kong", "tsinghua"],
        "Brazil": ["brazil", "sao paulo", "rio de janeiro"],
        "South Africa": ["south africa", "cape town", "pretoria"],
        "Netherlands": ["netherlands", "amsterdam", "utrecht", "wur", "wageningen"],
        "Japan": ["japan", "tokyo", "kyoto", "osaka"],
        "Kenya": ["kenya", "nairobi"],
        "Nigeria": ["nigeria", "lagos", "ibadan"],
        "Hungary": ["hungary", "budapest"],
        "Indonesia": ["indonesia", "jakarta", "bandung"],
    }
    country_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "citations": 0})
    for _, row in df.iterrows():
        text = f"{row.get('authors', '')} {row.get('abstract', '')} {row.get('venue', '')}".lower()
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        matched = set()
        for country, keywords in country_map.items():
            if any(kw in text for kw in keywords):
                matched.add(country)
        for c in matched:
            country_data[c]["papers"] += 1
            country_data[c]["citations"] += cites

    rows = []
    for country, d in country_data.items():
        rows.append({"country": country, "paper_count": d["papers"], "total_citations": d["citations"]})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("paper_count", ascending=False).reset_index(drop=True)
    return result


def _generate_report(
    top_authors: pd.DataFrame,
    emerging: pd.DataFrame,
    collaboration: pd.DataFrame,
    institutions: pd.DataFrame,
    countries: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Author Landscape Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Most influential authors
    lines.append("## Most Influential Authors")
    lines.append("")
    if not top_authors.empty:
        lines.append(top_authors[["author", "paper_count", "total_citations", "h_index"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No data._\n")

    # Fastest growing authors
    lines.append("## Fastest Growing Authors (Recent Activity)")
    lines.append("")
    if not emerging.empty:
        lines.append(emerging[["author", "paper_count", "recent_citations", "growth_ratio"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No emerging authors detected._\n")

    # Collaboration clusters
    lines.append("## Collaboration Clusters")
    lines.append("")
    if not collaboration.empty:
        top_collab = collaboration.head(15)
        lines.append(top_collab[["author_a", "author_b", "co_author_count"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No collaboration data._\n")

    # Leading institutions
    lines.append("## Leading Institutions")
    lines.append("")
    if not institutions.empty:
        lines.append(institutions[["institution", "paper_count", "total_citations", "unique_authors"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No institutional data extracted._\n")

    # Leading countries
    lines.append("## Country Trends")
    lines.append("")
    if not countries.empty:
        lines.append(countries[["country", "paper_count", "total_citations"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No country data extracted._\n")

    lines.append("---")
    lines.append("*Generated by author_intelligence.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Author-level analytics.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv")
    args = parser.parse_args()

    papers_path = Path(args.papers)
    if not papers_path.exists():
        log.error("Papers file not found: %s", papers_path)
        return

    df = pd.read_csv(args.papers)
    # Merge consensus themes if available
    consensus_path = Path(args.consensus)
    if consensus_path.exists():
        df_c = pd.read_csv(consensus_path)
        merge_cols = [c for c in ["doi", "title"] if c in df_c.columns and c in df.columns]
        if merge_cols:
            suffix = "_consensus"
            df = df.merge(
                df_c[merge_cols + ["consensus_theme"]],
                on=merge_cols[0], how="left", suffixes=("", suffix)
            )
            if "consensus_theme" not in df.columns:
                fallback = f"consensus_theme{suffix}"
                if fallback in df.columns:
                    df["consensus_theme"] = df[fallback]

    log.info("Loaded %d papers", len(df))

    top_authors = identify_top_authors(df)
    emerging = identify_emerging_authors(df)
    collaboration = identify_collaboration_networks(df)
    institutions = identify_institutional_leaders(df)
    countries = identify_country_trends(df)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV outputs
    top_authors.to_csv(out_dir / "author_network.csv", index=False)
    log.info("Saved %d -> %s", len(top_authors), out_dir / "author_network.csv")

    if not emerging.empty:
        emerging.to_csv(out_dir / "emerging_authors.csv", index=False)
    if not collaboration.empty:
        collaboration.to_csv(out_dir / "collaboration_network.csv", index=False)
    if not institutions.empty:
        institutions.to_csv(out_dir / "leading_institutions.csv", index=False)
    if not countries.empty:
        countries.to_csv(out_dir / "country_trends.csv", index=False)

    # Generate report
    report = _generate_report(top_authors, emerging, collaboration, institutions, countries)
    report_path = out_dir / "author_landscape.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    print(f"\n--- Author Intelligence Complete ---")
    print(f"  Top authors:        {len(top_authors)}")
    print(f"  Emerging authors:   {len(emerging)}")
    print(f"  Collaboration edges: {len(collaboration)}")
    print(f"  Institutions:       {len(institutions)}")
    print(f"  Countries:          {len(countries)}")
    print()


if __name__ == "__main__":
    main()
