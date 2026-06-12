#!/usr/bin/env python3
"""
meta_analysis_readiness.py — Assess whether the corpus contains comparable
studies suitable for meta-analysis based on common outcomes, interventions,
measurements, and evidence density.

Outputs
-------
outputs/reports/meta_analysis_candidates.csv
outputs/reports/meta_analysis_report.md
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
log = logging.getLogger("meta_analysis")

MODEL_NAME = "all-MiniLM-L6-v2"

OUTCOME_PATTERNS = [
    r"(?:outcome|endpoint|dependent variable|measured|assessed)",
    r"(?:effect|impact|influence|change|difference) (?:of|on|in)",
    r"(?:increase|decrease|improve|reduce|enhance|promote)",
]

INTERVENTION_PATTERNS = [
    r"(?:intervention|treatment|programme?|training|curriculum|course)",
    r"(?:policy|strategy|approach|method|technique|tool)",
]

MEASUREMENT_PATTERNS = [
    r"(?:scale|index|score|measure|instrument|survey|questionnaire)",
    r"(?:quantitative|qualitative|mixed.method|Likert|rating)",
    r"(?:pre.?test|post.?test|follow.up|baseline)",
]

INT_PAT = re.compile(r"n\s*=\s*(\d+)", re.IGNORECASE)


def assess_readiness(papers_path: str = "search_results.csv",
                     themes_path: str = "knowledge_base.json") -> pd.DataFrame:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    kb = json.load(open(themes_path)) if Path(themes_path).exists() else {}

    if df.empty:
        return df

    # Compute paper embeddings for pairwise similarity within themes
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    theme_papers: Dict[str, List[int]] = defaultdict(list)
    paper_list = []
    for idx, row in df.iterrows():
        title = str(row.get("title", ""))
        if title.lower() in ("nan", "", "none"):
            continue
        paper_list.append((idx, title, str(row.get("abstract", ""))))
        # Map to theme from KB
        for t in kb.get("themes", []):
            if title in t.get("papers", []):
                theme_papers[t["theme"]].append(idx)

    rows = []
    for theme, indices in theme_papers.items():
        if len(indices) < 2:
            continue
        group = df.iloc[indices]
        abstracts = [str(row.get("abstract", "")) for _, row in group.iterrows()]

        # Outcome overlap
        outcome_count = sum(1 for a in abstracts if any(re.search(p, a, re.IGNORECASE) for p in OUTCOME_PATTERNS))
        outcome_overlap = outcome_count / max(len(indices), 1)

        # Intervention overlap
        int_count = sum(1 for a in abstracts if any(re.search(p, a, re.IGNORECASE) for p in INTERVENTION_PATTERNS))
        int_overlap = int_count / max(len(indices), 1)

        # Measurement overlap
        meas_count = sum(1 for a in abstracts if any(re.search(p, a, re.IGNORECASE) for p in MEASUREMENT_PATTERNS))
        meas_overlap = meas_count / max(len(indices), 1)

        # Sample sizes
        sample_sizes = []
        for a in abstracts:
            m = INT_PAT.search(a)
            if m:
                sample_sizes.append(int(m.group(1)))
        has_sample_sizes = 1 if sample_sizes else 0
        avg_n = np.mean(sample_sizes) if sample_sizes else 0

        # Evidence density
        density = (outcome_overlap + int_overlap + meas_overlap) / 3.0

        # Feasibility
        feasibility = (
            0.30 * outcome_overlap
            + 0.25 * int_overlap
            + 0.20 * meas_overlap
            + 0.15 * has_sample_sizes
            + 0.10 * min(len(indices) / 10.0, 1.0)
        )

        if feasibility >= 0.6:
            level = "High Feasibility"
        elif feasibility >= 0.35:
            level = "Moderate Feasibility"
        else:
            level = "Low Feasibility"

        rows.append({
            "theme": theme,
            "paper_count": len(indices),
            "comparable_studies": len(indices),
            "common_outcomes": round(outcome_overlap, 3),
            "common_interventions": round(int_overlap, 3),
            "common_measurements": round(meas_overlap, 3),
            "has_sample_sizes": has_sample_sizes,
            "avg_sample_size": int(avg_n) if avg_n else 0,
            "evidence_density": round(density, 3),
            "feasibility_score": round(feasibility, 3),
            "feasibility_level": level,
        })

    return pd.DataFrame(rows)


def _generate_report(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Meta-Analysis Readiness Report\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if df.empty:
        lines.append("No meta-analysis candidates identified.\n")
        lines.append("---\n*Generated by meta_analysis_readiness.py*")
        return "\n".join(lines)

    for level in ["High Feasibility", "Moderate Feasibility", "Low Feasibility"]:
        subset = df[df["feasibility_level"] == level]
        if subset.empty:
            continue
        lines.append(f"## {level}\n")
        for _, row in subset.iterrows():
            lines.append(f"### {row['theme']}")
            lines.append(f"- Papers: {row['paper_count']}")
            lines.append(f"- Outcome overlap: {row['common_outcomes']:.0%}")
            lines.append(f"- Intervention overlap: {row['common_interventions']:.0%}")
            lines.append(f"- Measurement overlap: {row['common_measurements']:.0%}")
            lines.append(f"- Avg. sample size: {row['avg_sample_size']}")
            lines.append(f"- Feasibility score: {row['feasibility_score']:.2f}\n")

    lines.append("## Recommendations\n")
    high = df[df["feasibility_level"] == "High Feasibility"]
    if not high.empty:
        lines.append("### Suitable for Meta-Analysis")
        for _, row in high.iterrows():
            lines.append(f"- {row['theme']} ({row['paper_count']} papers)")
    mod = df[df["feasibility_level"] == "Moderate Feasibility"]
    if not mod.empty:
        lines.append("### Potentially Suitable with Additional Data")
        for _, row in mod.iterrows():
            lines.append(f"- {row['theme']} ({row['paper_count']} papers)")
    lines.append("")
    lines.append("---\n*Generated by meta_analysis_readiness.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess meta-analysis readiness.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--knowledge-base", type=str, default="knowledge_base.json")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    df = assess_readiness(args.papers, args.knowledge_base)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(out_dir / "meta_analysis_candidates.csv", index=False)
        log.info("Saved -> %s", out_dir / "meta_analysis_candidates.csv")

    report = _generate_report(df)
    with open(out_dir / "meta_analysis_report.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "meta_analysis_report.md")

    print(f"\n--- Meta-Analysis Readiness Complete ---")
    print(f"  Candidates: {len(df)}")
    for level in ["High Feasibility", "Moderate Feasibility", "Low Feasibility"]:
        c = len(df[df["feasibility_level"] == level])
        if c:
            print(f"    {level}: {c}")
    print()


if __name__ == "__main__":
    main()
