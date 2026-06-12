#!/usr/bin/env python3
"""
study_design_advisor.py — Recommend study designs, controls, sample sizes,
statistical tests, and validation strategies based on research gaps and
existing evidence.

Outputs
-------
outputs/reports/study_design_recommendations.md
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("study_design_advisor")

DESIGNS = {
    "exploratory": {
        "design": "Qualitative exploratory (interviews, focus groups, case studies)",
        "controls": "Purposive sampling, researcher reflexivity, member checking",
        "sample_size": "15–30 participants (saturation-based)",
        "statistical_tests": "Thematic analysis, grounded theory coding",
        "validation": "Triangulation, respondent validation, peer debriefing",
        "data_collection": "Semi-structured interviews, field observations, document analysis",
    },
    "correlational": {
        "design": "Cross-sectional correlational survey",
        "controls": "Demographic covariates, random sampling, common method bias checks",
        "sample_size": "200–400 (power analysis for medium effect, 80% power)",
        "statistical_tests": "Pearson/Spearman correlation, multiple regression, SEM",
        "validation": "Cross-validation, sensitivity analysis, multicollinearity assessment",
        "data_collection": "Online surveys, validated scales, Likert-type items",
    },
    "experimental": {
        "design": "Randomised controlled trial (RCT) or quasi-experimental",
        "controls": "Random assignment, pre-registration, blinding, attention checks",
        "sample_size": "100–300 per condition (power analysis for small-to-medium effect)",
        "statistical_tests": "t-test, ANOVA, ANCOVA, mixed-effects models",
        "validation": "Pre-registration, replication, intention-to-treat analysis",
        "data_collection": "Controlled tasks, standardised instruments, behavioural measures",
    },
    "longitudinal": {
        "design": "Longitudinal cohort or panel study",
        "controls": "Attrition analysis, time-invariant covariates, fixed effects",
        "sample_size": "200+ (accounting for attrition ~20%)",
        "statistical_tests": "Growth curve modelling, latent growth models, GEE",
        "validation": "Multiple imputation for missing data, sensitivity to attrition",
        "data_collection": "Repeated surveys, experience sampling, administrative records",
    },
    "mixed": {
        "design": "Sequential explanatory mixed methods (QUAN → qual)",
        "controls": "Integration validity, sampling frame alignment, meta-inferences",
        "sample_size": "QUAN: 200+; qual: 15–25",
        "statistical_tests": "Regression + thematic analysis, joint display analysis",
        "validation": "Integration checks, legitimation, inference quality assessment",
        "data_collection": "Surveys + in-depth interviews, quantitative + qualitative",
    },
}


def recommend_design(theme: str, paper_count: int, evidence_score: float) -> Dict:
    if paper_count <= 2 or evidence_score < 0.4:
        base = "exploratory"
    elif paper_count <= 5 or evidence_score < 0.6:
        base = "correlational"
    elif paper_count <= 10:
        base = "mixed"
    else:
        base = "experimental"

    rec = dict(DESIGNS.get(base, DESIGNS["exploratory"]))
    rec["theme"] = theme
    rec["paper_count"] = paper_count
    rec["evidence_score"] = evidence_score
    rec["recommended_design"] = base.capitalize()
    return rec


def _generate_report(recommendations: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Study Design Recommendations\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for rec in recommendations:
        lines.append(f"## {rec['theme']}")
        lines.append(f"- Current evidence: {rec['evidence_score']:.2f} ({rec['paper_count']} papers)")
        lines.append(f"- **Recommended design:** {rec['recommended_design']}\n")
        lines.append(f"### Design")
        lines.append(f"{rec['design']}\n")
        lines.append(f"### Controls")
        lines.append(f"{rec['controls']}\n")
        lines.append(f"### Sample Size")
        lines.append(f"{rec['sample_size']}\n")
        lines.append(f"### Statistical Tests")
        lines.append(f"{rec['statistical_tests']}\n")
        lines.append(f"### Validation Strategy")
        lines.append(f"{rec['validation']}\n")
        lines.append(f"### Data Collection")
        lines.append(f"{rec['data_collection']}\n")

    lines.append("## Quick Reference\n")
    lines.append("| Maturity | Recommended Design | Sample Size |\n")
    lines.append("|----------|-------------------|-------------|\n")
    for name, info in DESIGNS.items():
        lines.append(f"| {name.capitalize()} | {info['design'][:50]} | {info['sample_size']} |\n")

    lines.append("---\n*Generated by study_design_advisor.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend study designs.")
    parser.add_argument("--knowledge-base", type=str, default="knowledge_base.json")
    parser.add_argument("--evidence", type=str, default="outputs/reports/evidence_strength.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    kb = json.load(open(args.knowledge_base)) if Path(args.knowledge_base).exists() else {}
    evidence_df = pd.read_csv(args.evidence) if Path(args.evidence).exists() else pd.DataFrame()

    recommendations: List[Dict] = []
    for t in kb.get("themes", []):
        theme = t.get("theme", "")
        pc = t.get("paper_count", 0)
        ev_row = evidence_df[evidence_df["theme"] == theme]
        ev_score = float(ev_row.iloc[0]["evidence_score"]) if not ev_row.empty else 0.5
        recommendations.append(recommend_design(theme, pc, ev_score))

    report = _generate_report(recommendations)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "study_design_recommendations.md"
    with open(path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", path)

    print(f"\n--- Study Design Advisor Complete ---")
    print(f"  Recommendations: {len(recommendations)}")
    print()


if __name__ == "__main__":
    main()
