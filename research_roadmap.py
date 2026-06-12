#!/usr/bin/env python3
"""
research_roadmap.py — Generate a 1-year, 3-year, and 5-year research roadmap
for each theme, identifying knowledge gaps, required methods, key datasets,
milestones, and expected outcomes.

Usage
-----
    python research_roadmap.py
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--consensus consensus_themes.csv]
"""

from __future__ import annotations

import argparse
import json
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
log = logging.getLogger("research_roadmap")

CURRENT_YEAR = datetime.now().year


METHOD_SUGGESTIONS = {
    "sparse": "Qualitative exploration (interviews, case studies) followed by survey development",
    "developing": "Mixed-methods: longitudinal tracking + quasi-experimental design",
    "mature": "Systematic review + large-scale quantitative modelling (SEM, ML)",
}

DATASET_SUGGESTIONS = [
    "Primary data collection via survey instruments",
    "Publicly available administrative datasets",
    "Longitudinal panel data from existing repositories",
    "Qualitative interview transcripts and observational field notes",
    "Experimental data from controlled interventions",
    "Secondary analysis of existing multi-country survey data",
]


def generate_roadmap(themes: List[Dict], df_papers: pd.DataFrame) -> List[Dict]:
    """Generate a structured roadmap entry per theme."""
    entries: List[Dict] = []
    for t in themes:
        label = t.get("theme", t.get("label", ""))
        paper_count = t.get("paper_count", 0)
        confidence = t.get("confidence", 0.5)

        # Determine maturity
        if paper_count <= 2 or confidence < 0.5:
            maturity = "sparse"
        elif paper_count <= 5:
            maturity = "developing"
        else:
            maturity = "mature"

        # Knowledge gaps
        gaps = [
            f"Limited empirical evidence on mechanisms within {label}",
            f"Few longitudinal studies tracking change over time",
            f"Lack of validated measurement instruments",
        ]
        if maturity == "sparse":
            gaps.append("Fundamental descriptive and exploratory work needed")
        elif maturity == "developing":
            gaps.append("Need for comparative and cross-context studies")

        # Methods
        methods = METHOD_SUGGESTIONS.get(maturity, "Mixed-methods approach")
        methods_detail = [methods]
        if maturity == "sparse":
            methods_detail.append("Grounded theory for initial conceptual development")
        elif maturity == "developing":
            methods_detail.append("Structural equation modelling for theory testing")
        else:
            methods_detail.append("Meta-analysis for effect size estimation")

        # Datasets
        datasets = np.random.choice(DATASET_SUGGESTIONS, size=min(3, len(DATASET_SUGGESTIONS)), replace=False).tolist()

        # Milestones
        milestones = {
            "year_1": [
                f"Systematic review of {label} to refine research questions",
                "Develop and pilot measurement instruments",
                "Obtain ethical approval and begin data collection",
            ],
            "year_3": [
                f"Complete primary data collection for {label}",
                "Preliminary analysis and conference presentations",
                "Submit 2–3 journal articles on core findings",
            ],
            "year_5": [
                f"Longitudinal follow-up data collection",
                "Comparative analyses across contexts",
                f"Develop and disseminate evidence-based recommendations for {label}",
                "Establish research network / consortium",
            ],
        }

        # Expected outcomes
        outcomes = {
            "year_1": "Conceptual framework, validated instruments, pilot data",
            "year_3": "Core empirical findings, peer-reviewed publications, policy briefs",
            "year_5": "Longitudinal evidence, cross-context comparisons, practice guidelines",
        }

        entries.append({
            "theme": label,
            "maturity": maturity,
            "paper_count": paper_count,
            "gaps": gaps,
            "methods": methods_detail,
            "datasets": datasets,
            "milestones": milestones,
            "outcomes": outcomes,
        })
    return entries


def _generate_report(entries: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Research Roadmap")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Themes mapped:** {len(entries)}")
    lines.append("")

    for e in entries:
        lines.append(f"## {e['theme']}")
        lines.append(f"- **Maturity:** {e['maturity'].capitalize()} ({e['paper_count']} papers)")
        lines.append("")

        lines.append("### Knowledge Gaps")
        for g in e["gaps"]:
            lines.append(f"- {g}")
        lines.append("")

        lines.append("### Required Methods")
        for m in e["methods"]:
            lines.append(f"- {m}")
        lines.append("")

        lines.append("### Key Datasets / Data Sources")
        for d in e["datasets"]:
            lines.append(f"- {d}")
        lines.append("")

        lines.append("### 1-Year Roadmap")
        for m in e["milestones"]["year_1"]:
            lines.append(f"- [ ] {m}")
        lines.append(f"- **Expected outcome:** {e['outcomes']['year_1']}")
        lines.append("")

        lines.append("### 3-Year Roadmap")
        for m in e["milestones"]["year_3"]:
            lines.append(f"- [ ] {m}")
        lines.append(f"- **Expected outcome:** {e['outcomes']['year_3']}")
        lines.append("")

        lines.append("### 5-Year Roadmap")
        for m in e["milestones"]["year_5"]:
            lines.append(f"- [ ] {m}")
        lines.append(f"- **Expected outcome:** {e['outcomes']['year_5']}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by research_roadmap.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research roadmap.")
    parser.add_argument("--knowledge-base", type=str, default="outputs/knowledge_base/knowledge_base.json")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv")
    args = parser.parse_args()

    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        log.error("Knowledge base not found: %s", kb_path)
        return
    with open(kb_path) as f:
        kb = json.load(f)

    themes = kb.get("themes", [])
    if not themes:
        log.warning("No themes in knowledge base; trying consensus_themes.csv")
        consensus_path = Path(args.consensus)
        if consensus_path.exists():
            df_c = pd.read_csv(consensus_path)
            theme_labels = df_c["consensus_theme"].dropna().unique()
            themes = [{"theme": t, "paper_count": len(df_c[df_c["consensus_theme"] == t]), "keywords": []} for t in theme_labels if str(t).lower() not in ("nan", "", "none")]

    log.info("Loaded %d themes", len(themes))
    df_papers = pd.DataFrame()
    papers_path = Path("search_results.csv")
    if papers_path.exists():
        df_papers = pd.read_csv(papers_path)

    entries = generate_roadmap(themes, df_papers)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _generate_report(entries)
    report_path = out_dir / "research_roadmap.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    print(f"\n--- Research Roadmap Complete ---")
    print(f"  Themes mapped:  {len(entries)}")
    print()


if __name__ == "__main__":
    main()
