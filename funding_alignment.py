#!/usr/bin/env python3
"""
funding_alignment.py — Map research themes to funding priorities,
strategic directions, and identify high-alignment funding areas.

Usage
-----
    python funding_alignment.py
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--gaps outputs/reports/research_gaps.md]
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
log = logging.getLogger("funding_alignment")

# Strategic priority mapping
FUNDING_AREAS: List[Dict[str, Any]] = [
    {"name": "Digital Transformation", "keywords": ["digital", "transformation", "digitisation", "digitization", "industry 4.0", "smart"], "weight": 1.0},
    {"name": "Artificial Intelligence & ML", "keywords": ["artificial intelligence", "machine learning", "deep learning", "ai", "neural", "nlp"], "weight": 1.0},
    {"name": "Capacity Building & Training", "keywords": ["capacity building", "training", "education", "skill", "workforce", "human capital"], "weight": 0.9},
    {"name": "Agriculture & Food Security", "keywords": ["agriculture", "food", "farmer", "crop", "rural", "sustainable agriculture"], "weight": 0.9},
    {"name": "Health & Well-being", "keywords": ["health", "healthcare", "wellbeing", "well-being", "clinical", "patient", "mental health"], "weight": 0.9},
    {"name": "Climate Change & Sustainability", "keywords": ["climate", "sustainable", "sustainability", "green", "environment", "renewable"], "weight": 0.9},
    {"name": "Digital Inclusion & Equity", "keywords": ["digital divide", "inclusion", "equity", "inequality", "access", "indigenous", "marginalised", "marginalized"], "weight": 0.8},
    {"name": "Data Science & Analytics", "keywords": ["data science", "analytics", "big data", "data-driven", "data mining", "visualisation", "visualization"], "weight": 0.8},
    {"name": "Cybersecurity & Trust", "keywords": ["cybersecurity", "cyber security", "privacy", "trust", "security", "data protection"], "weight": 0.8},
    {"name": "Innovation & Entrepreneurship", "keywords": ["innovation", "entrepreneurship", "startup", "sme", "technology transfer", "commercialisation", "commercialization"], "weight": 0.7},
]


def match_strategic_priorities(themes: List[Dict]) -> pd.DataFrame:
    """Score funding areas against theme keywords."""
    rows = []
    for fa in FUNDING_AREAS:
        score = 0.0
        matched_themes: List[str] = []
        for t in themes:
            label = t.get("theme", t.get("label", "")).lower()
            kw_text = " ".join(t.get("keywords", [])).lower()
            combined = f"{label} {kw_text}"
            for kw in fa["keywords"]:
                if kw in combined:
                    score += fa["weight"]
                    matched_themes.append(t.get("theme", "")[:60])
                    break
        score = min(score / max(len(themes), 1) * 5, 1.0)
        rows.append({
            "funding_area": fa["name"],
            "alignment_score": round(score, 3),
            "matched_themes": len(set(matched_themes)),
            "themes": "; ".join(sorted(set(matched_themes)))[:200],
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("alignment_score", ascending=False).reset_index(drop=True)
    return result


def rank_alignment(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add alignment classification."""
    df = scores_df.copy()
    df["alignment_level"] = pd.cut(
        df["alignment_score"],
        bins=[-0.1, 0.3, 0.6, 1.0],
        labels=["Low Alignment", "Medium Alignment", "High Alignment"],
    )
    return df


def _generate_report(scores_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Funding Alignment Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("## Potential Funding Areas")
    lines.append("")
    if not scores_df.empty:
        lines.append(scores_df[["funding_area", "alignment_score", "matched_themes", "alignment_level"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No funding areas matched._\n")

    high = scores_df[scores_df["alignment_level"] == "High Alignment"]
    if not high.empty:
        lines.append("## High Alignment Themes")
        lines.append("")
        for _, row in high.iterrows():
            lines.append(f"- **{row['funding_area']}** (score: {row['alignment_score']:.2f})")
            if row["themes"]:
                lines.append(f"  - Matched themes: {row['themes'][:120]}")
        lines.append("")

    lines.append("## Strategic Research Directions")
    lines.append("")
    lines.append("Based on the alignment analysis, consider pursuing:")
    lines.append("")
    for _, row in scores_df.head(5).iterrows():
        lines.append(f"- **{row['funding_area']}** — alignment score {row['alignment_score']:.2f}")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by funding_alignment.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map themes to funding priorities.")
    parser.add_argument("--knowledge-base", type=str, default="outputs/knowledge_base/knowledge_base.json")
    parser.add_argument("--gaps", type=str, default="outputs/reports/research_gaps.md")
    args = parser.parse_args()

    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        log.error("Knowledge base not found: %s", kb_path)
        return
    with open(kb_path) as f:
        kb = json.load(f)

    themes = kb.get("themes", [])
    log.info("Loaded %d themes from knowledge base", len(themes))

    scores = match_strategic_priorities(themes)
    scores = rank_alignment(scores)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "funding_alignment.csv", index=False)
    log.info("Saved %d -> %s", len(scores), out_dir / "funding_alignment.csv")

    report = _generate_report(scores)
    report_path = out_dir / "funding_opportunities.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    print(f"\n--- Funding Alignment Complete ---")
    print(f"  Funding areas: {len(scores)}")
    for level in ["High Alignment", "Medium Alignment", "Low Alignment"]:
        count = len(scores[scores["alignment_level"] == level])
        if count:
            print(f"    {level}: {count}")
    print()


if __name__ == "__main__":
    main()
