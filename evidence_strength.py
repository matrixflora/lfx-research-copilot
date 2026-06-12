#!/usr/bin/env python3
"""
evidence_strength.py — Score thematic evidence strength across five
dimensions: study count, citation support, recency, methodological
diversity, and theme consistency.

Usage
-----
    python evidence_strength.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
        [--consensus-meta consensus_metadata.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("evidence_strength")

CURRENT_YEAR = datetime.now().year


def compute_evidence_scores(
    df: pd.DataFrame,
    consensus_meta: List[Dict],
) -> pd.DataFrame:
    """Score each theme on 5 dimensions and classify evidence strength."""
    if "consensus_theme" not in df.columns:
        # Build a placeholder theme for all papers
        df = df.copy()
        df["consensus_theme"] = "All Papers"
    themes = df["consensus_theme"].dropna().unique()
    theme_list = [t for t in themes if str(t).lower() not in ("nan", "", "none")]

    # Build confidence map from consensus_meta
    confidence_map: Dict[str, float] = {}
    for m in consensus_meta:
        label = m.get("label", "")
        confidence_map[label] = m.get("confidence", 0.5)

    rows = []
    for theme in theme_list:
        group = df[df["consensus_theme"] == theme]
        n = len(group)

        # 1. Study count score
        study_count_score = min(n / 10.0, 1.0)

        # 2. Citation support
        cites = group["citation_count"].dropna().astype(float)
        total_cites = cites.sum()
        citation_support = min(np.log1p(total_cites) / 6.0, 1.0)

        # 3. Recency
        years = group["year"].dropna().astype(int)
        if len(years) > 0:
            avg_year = years.mean()
            recency = min((avg_year - 2000) / 25.0, 1.0)
            recency = max(0.0, recency)
        else:
            recency = 0.5

        # 4. Methodological diversity (count unique sources/venues)
        venues = group["venue"].dropna().nunique() if "venue" in group else 0
        sources = group["source"].dropna().nunique() if "source" in group else 0
        meth_diversity = min((venues + sources) / 6.0, 1.0)

        # 5. Theme consistency (confidence from meta / consensus strength)
        conf = confidence_map.get(theme, 0.5)

        # Composite
        evidence_score = (
            study_count_score * 0.25
            + citation_support * 0.20
            + recency * 0.15
            + meth_diversity * 0.20
            + conf * 0.20
        )

        if evidence_score >= 0.7:
            classification = "Strong Evidence"
        elif evidence_score >= 0.5:
            classification = "Moderate Evidence"
        elif evidence_score >= 0.3:
            classification = "Weak Evidence"
        else:
            classification = "Very Weak Evidence"

        rows.append({
            "theme": theme,
            "paper_count": n,
            "total_citations": int(total_cites),
            "avg_year": round(avg_year, 1) if len(years) > 0 else 0,
            "study_count_score": round(study_count_score, 3),
            "citation_support_score": round(citation_support, 3),
            "recency_score": round(recency, 3),
            "method_diversity_score": round(meth_diversity, 3),
            "theme_consistency_score": round(conf, 3),
            "evidence_score": round(evidence_score, 3),
            "classification": classification,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("evidence_score", ascending=False).reset_index(drop=True)
    return result


def _generate_report(scores_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Evidence Strength Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Themes scored:** {len(scores_df)}")
    lines.append("")

    for cls in ["Strong Evidence", "Moderate Evidence", "Weak Evidence", "Very Weak Evidence"]:
        subset = scores_df[scores_df["classification"] == cls]
        if subset.empty:
            continue
        lines.append(f"## {cls}")
        lines.append("")
        lines.append(subset[["theme", "paper_count", "total_citations", "evidence_score"]].to_markdown(index=False))
        lines.append("")

    lines.append("## Score Components (All Themes)")
    lines.append("")
    display_cols = ["theme", "study_count_score", "citation_support_score", "recency_score",
                    "method_diversity_score", "theme_consistency_score", "evidence_score", "classification"]
    lines.append(scores_df[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Methodological Note")
    lines.append("")
    lines.append("Scores are composite indices (0–1) aggregated from five equally-weighted dimensions. ")
    lines.append("Classification thresholds: >=0.7 Strong, >=0.5 Moderate, >=0.3 Weak, <0.3 Very Weak.")
    lines.append("Small corpora (<50 papers) may produce preliminary classifications.")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by evidence_strength.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score thematic evidence strength.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv")
    parser.add_argument("--consensus-meta", type=str, default="consensus_metadata.json")
    args = parser.parse_args()

    papers_path = Path(args.papers)
    if not papers_path.exists():
        log.error("Papers file not found: %s", papers_path)
        return
    df = pd.read_csv(args.papers)

    # Load consensus themes
    consensus_path = Path(args.consensus)
    if consensus_path.exists():
        df_c = pd.read_csv(consensus_path)
        merge_cols = [c for c in ["doi", "title"] if c in df_c.columns and c in df.columns]
        if merge_cols:
            df = df.merge(df_c[merge_cols + ["consensus_theme"]], on=merge_cols[0], how="left", suffixes=("", "_consensus"))
            if "consensus_theme" not in df.columns:
                df["consensus_theme"] = df.get("consensus_theme_consensus", "")

    # Load consensus metadata
    meta_path = Path(args.consensus_meta)
    consensus_meta = []
    if meta_path.exists():
        with open(meta_path) as f:
            consensus_meta = json.load(f)

    log.info("Loaded %d papers, %d meta entries", len(df), len(consensus_meta))

    scores = compute_evidence_scores(df, consensus_meta)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "evidence_strength.csv", index=False)
    log.info("Saved %d -> %s", len(scores), out_dir / "evidence_strength.csv")

    report = _generate_report(scores)
    report_path = out_dir / "evidence_strength_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    print(f"\n--- Evidence Strength Scoring Complete ---")
    print(f"  Themes scored: {len(scores)}")
    for cls in ["Strong Evidence", "Moderate Evidence", "Weak Evidence", "Very Weak Evidence"]:
        count = len(scores[scores["classification"] == cls])
        if count:
            print(f"    {cls}: {count}")
    print()


if __name__ == "__main__":
    main()
