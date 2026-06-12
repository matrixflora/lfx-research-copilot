#!/usr/bin/env python3
"""
opportunity_ranking.py — Rank research opportunities by novelty, growth,
gap importance, and citation momentum.

Usage
-----
    python opportunity_ranking.py
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--gaps outputs/reports/research_gaps.md]
        [--papers search_results.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
log = logging.getLogger("opportunity_ranking")

CURRENT_YEAR = datetime.now().year


def compute_opportunity_scores(
    kb: Dict,
    gaps_text: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    themes = kb.get("themes", [])
    classifications = {}
    for cls_key in ("redundant_themes", "developing_themes", "trending_themes", "future_themes"):
        items = kb.get(cls_key, [])
        if isinstance(items, list):
            for item in items:
                t = item.get("theme", "")
                if t:
                    classifications[t] = cls_key.replace("_themes", "")

    rows = []
    for t in themes:
        label = t.get("theme", t.get("label", ""))
        paper_count = t.get("paper_count", 0)
        conf = t.get("confidence", 0.5)
        kw = ", ".join(t.get("keywords", []))

        # Novelty score — from evolution data or paper recency
        group = df[df["consensus_theme"] == label] if "consensus_theme" in df.columns else pd.DataFrame()
        if not group.empty:
            years = group["year"].dropna().astype(int)
            avg_year = years.mean() if len(years) > 0 else 2020
        else:
            avg_year = 2020
        novelty = min((avg_year - 2000) / 30.0, 1.0)
        novelty = max(0.0, novelty)

        # Growth — detect trend from year distribution
        if not group.empty and len(years) > 0:
            recent = (years >= CURRENT_YEAR - 3).sum()
            older = (years < CURRENT_YEAR - 3).sum()
            growth = recent / max(older, 1)
        else:
            growth = 1.0

        # Gap importance
        gap_importance = 0.5
        if gaps_text:
            if label.lower() in gaps_text.lower():
                gap_importance = 0.8
            if "sparse" in gaps_text.lower() and "only" in gaps_text.lower():
                gap_importance = 0.9

        # Citation momentum
        if not group.empty:
            cites = group["citation_count"].dropna().astype(float)
            recent_cites = cites[years >= CURRENT_YEAR - 3].sum() if len(years) > 0 else 0
            total_cites = cites.sum()
            citation_momentum = recent_cites / max(total_cites - recent_cites, 1)
        else:
            citation_momentum = 0.5

        opportunity = (
            novelty * 0.25
            + min(growth, 3.0) / 3.0 * 0.20
            + gap_importance * 0.30
            + min(citation_momentum, 3.0) / 3.0 * 0.25
        )

        if opportunity >= 0.6:
            level = "High Potential"
        elif opportunity >= 0.4:
            level = "Medium Potential"
        else:
            level = "Low Potential"

        rows.append({
            "theme": label,
            "paper_count": paper_count,
            "novelty_score": round(novelty, 3),
            "growth_score": round(growth, 3),
            "gap_importance": round(gap_importance, 3),
            "citation_momentum": round(citation_momentum, 3),
            "opportunity_score": round(opportunity, 3),
            "opportunity_level": level,
            "classification": classifications.get(label, "unknown"),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank research opportunities.")
    parser.add_argument("--knowledge-base", type=str, default="outputs/knowledge_base/knowledge_base.json")
    parser.add_argument("--gaps", type=str, default="outputs/reports/research_gaps.md")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    args = parser.parse_args()

    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        log.error("Knowledge base not found: %s", kb_path)
        return
    with open(kb_path) as f:
        kb = json.load(f)

    gaps_text = ""
    gaps_path = Path(args.gaps)
    if gaps_path.exists():
        gaps_text = gaps_path.read_text()

    papers_path = Path(args.papers)
    df = pd.DataFrame()
    if papers_path.exists():
        df = pd.read_csv(args.papers)
        consensus_path = Path("consensus_themes.csv")
        if consensus_path.exists():
            df_c = pd.read_csv(consensus_path)
            merge_cols = [c for c in ["doi", "title"] if c in df_c.columns and c in df.columns]
            if merge_cols:
                df = df.merge(df_c[merge_cols + ["consensus_theme"]], on=merge_cols[0], how="left", suffixes=("", "_consensus"))
                if "consensus_theme" not in df.columns:
                    df["consensus_theme"] = df.get("consensus_theme_consensus", "")

    scores = compute_opportunity_scores(kb, gaps_text, df)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "research_opportunities.csv", index=False)
    log.info("Saved %d -> %s", len(scores), out_dir / "research_opportunities.csv")

    print(f"\n--- Opportunity Ranking Complete ---")
    print(f"  Themes scored: {len(scores)}")
    for level in ["High Potential", "Medium Potential", "Low Potential"]:
        count = len(scores[scores["opportunity_level"] == level])
        if count:
            print(f"    {level}: {count}")
    print()


if __name__ == "__main__":
    main()
