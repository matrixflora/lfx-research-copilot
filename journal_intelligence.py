#!/usr/bin/env python3
"""
journal_intelligence.py — Journal-level analytics: core venues, emerging
journals, theme–journal mapping, and publication opportunity analysis.

Usage
-----
    python journal_intelligence.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("journal_intelligence")

CURRENT_YEAR = datetime.now().year


def identify_core_journals(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    j_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "citations": [], "years": [], "themes": set()})
    for _, row in df.iterrows():
        venue = str(row.get("venue", "")).strip()
        if not venue or venue.lower() in ("nan", "", "none"):
            continue
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        theme = row.get("consensus_theme", "")
        j_data[venue]["papers"] += 1
        j_data[venue]["citations"].append(cites)
        j_data[venue]["years"].append(year)
        if theme:
            j_data[venue]["themes"].add(theme)
    rows = []
    for venue, d in j_data.items():
        rows.append({
            "journal": venue[:120],
            "paper_count": d["papers"],
            "total_citations": sum(d["citations"]),
            "avg_citations": round(np.mean(d["citations"]), 1) if d["citations"] else 0.0,
            "earliest_year": min(d["years"]) if d["years"] else 0,
            "latest_year": max(d["years"]) if d["years"] else 0,
            "themes_covered": len(d["themes"]),
            "themes": "; ".join(sorted(d["themes"])),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["paper_count", "total_citations"], ascending=False).head(top_n).reset_index(drop=True)
    return result


def identify_emerging_journals(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    cutoff = CURRENT_YEAR - 3
    j_data: Dict[str, Dict] = defaultdict(lambda: {"papers": 0, "recent": 0, "citations": 0, "recent_cites": 0})
    for _, row in df.iterrows():
        venue = str(row.get("venue", "")).strip()
        if not venue or venue.lower() in ("nan", "", "none"):
            continue
        cites = int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else 0
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        j_data[venue]["papers"] += 1
        j_data[venue]["citations"] += cites
        if year >= cutoff:
            j_data[venue]["recent"] += 1
            j_data[venue]["recent_cites"] += cites
    rows = []
    for venue, d in j_data.items():
        recent_ratio = d["recent"] / max(d["papers"], 1)
        rows.append({
            "journal": venue[:120],
            "paper_count": d["papers"],
            "recent_papers": d["recent"],
            "recent_ratio": round(recent_ratio, 2),
            "total_citations": d["citations"],
            "recent_citations": d["recent_cites"],
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result[result["recent_papers"] > 0]
    result = result.sort_values("recent_ratio", ascending=False).head(top_n).reset_index(drop=True)
    return result


def journal_theme_mapping(df: pd.DataFrame) -> pd.DataFrame:
    if "consensus_theme" not in df.columns:
        log.warning("consensus_theme column missing; skipping theme mapping.")
        return pd.DataFrame()
    valid = df[df["venue"].notna() & df["consensus_theme"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["venue"] = valid["venue"].astype(str).str.strip()
    valid["consensus_theme"] = valid["consensus_theme"].astype(str).str.strip()
    matrix = pd.crosstab(valid["venue"], valid["consensus_theme"], margins=True, margins_name="Total")
    return matrix


def publication_opportunity_analysis(core_journals: pd.DataFrame, theme_matrix: pd.DataFrame) -> pd.DataFrame:
    if theme_matrix.empty or "Total" not in theme_matrix.columns:
        return pd.DataFrame()
    rows = []
    for journal in theme_matrix.index:
        if journal == "Total":
            continue
        total = theme_matrix.loc[journal, "Total"] if "Total" in theme_matrix.columns else 0
        if total == 0:
            continue
        theme_cols = [c for c in theme_matrix.columns if c != "Total"]
        covered = sum(1 for c in theme_cols if theme_matrix.loc[journal, c] > 0)
        rows.append({
            "journal": journal[:120],
            "total_papers": total,
            "themes_covered": covered,
            "total_themes": len(theme_cols),
            "coverage_ratio": round(covered / max(len(theme_cols), 1), 2),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("coverage_ratio", ascending=False).reset_index(drop=True)
    return result


def _generate_report(core: pd.DataFrame, emerging: pd.DataFrame, theme_matrix: pd.DataFrame, opportunities: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Journal Landscape Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Most Relevant Journals")
    lines.append("")
    if not core.empty:
        lines.append(core[["journal", "paper_count", "total_citations", "avg_citations", "themes_covered"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No journal data._\n")
    lines.append("## Emerging Publication Venues")
    lines.append("")
    if not emerging.empty:
        lines.append(emerging[["journal", "recent_papers", "recent_ratio", "recent_citations"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No emerging journals detected._\n")
    lines.append("## Theme-Specific Journals")
    lines.append("")
    if not theme_matrix.empty:
        lines.append(theme_matrix.to_markdown())
        lines.append("")
    else:
        lines.append("_Theme mapping unavailable._\n")
    lines.append("## Recommended Submission Targets")
    lines.append("")
    if not opportunities.empty:
        lines.append("Journals with broadest theme coverage (potential for interdisciplinary work):\n")
        lines.append(opportunities[["journal", "total_papers", "themes_covered", "coverage_ratio"]].head(10).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_Opportunity analysis unavailable._\n")
    lines.append("---")
    lines.append("*Generated by journal_intelligence.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Journal-level analytics.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv")
    args = parser.parse_args()
    papers_path = Path(args.papers)
    if not papers_path.exists():
        log.error("Papers file not found: %s", papers_path)
        return
    df = pd.read_csv(args.papers)
    consensus_path = Path(args.consensus)
    if consensus_path.exists():
        df_c = pd.read_csv(consensus_path)
        merge_cols = [c for c in ["doi", "title"] if c in df_c.columns and c in df.columns]
        if merge_cols:
            df = df.merge(df_c[merge_cols + ["consensus_theme"]], on=merge_cols[0], how="left", suffixes=("", "_consensus"))
            if "consensus_theme" not in df.columns:
                df["consensus_theme"] = df.get("consensus_theme_consensus", "")
    log.info("Loaded %d papers", len(df))
    core = identify_core_journals(df)
    emerging = identify_emerging_journals(df)
    theme_matrix = journal_theme_mapping(df)
    opportunities = publication_opportunity_analysis(core, theme_matrix)
    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.to_csv(out_dir / "journal_rankings.csv", index=False)
    log.info("Saved %d -> %s", len(core), out_dir / "journal_rankings.csv")
    if not theme_matrix.empty:
        theme_matrix.to_csv(out_dir / "journal_theme_matrix.csv")
    if not opportunities.empty:
        opportunities.to_csv(out_dir / "submission_targets.csv", index=False)
    report = _generate_report(core, emerging, theme_matrix, opportunities)
    report_path = out_dir / "journal_landscape.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)
    print(f"\n--- Journal Intelligence Complete ---")
    print(f"  Core journals:     {len(core)}")
    print(f"  Emerging journals: {len(emerging)}")
    print()


if __name__ == "__main__":
    main()
