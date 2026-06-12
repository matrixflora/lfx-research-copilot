#!/usr/bin/env python3
"""
reviewer_simulator.py — Simulate peer review by evaluating manuscript quality,
identifying weak arguments, missing citations, unsupported claims, and
methodological concerns.

Outputs
-------
outputs/reports/reviewer_comments.md
outputs/reports/reviewer_scorecard.csv
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
log = logging.getLogger("reviewer_simulator")

WEAK_ARGUMENT_PATTERNS = [
    r"\b(clearly|obviously|undoubtedly|certainly|definitely)\b",
    r"\b(it is well known|as everyone knows|it goes without saying)\b",
    r"\b(prove|proves|proven|truly|absolutely)\b",
    r"\b(groundbreaking|revolutionary|unprecedented|game-changing)\b",
]

MISSING_CITATION_PATTERNS = [
    r"\b(recent studies|some research|prior work|previous work|past research) (has|have|suggests|shows|indicates)\b",
    r"\b(according to|as noted by|as argued by) (?!\[|\([\d{4}])",
    r"\b(evidence suggests|research indicates|studies show) (?!\[|\([\d{4}])",
]


def evaluate_manuscript(
    manuscript_dir: str = "outputs/manuscript",
    evidence_path: str = "outputs/evidence/evidence_matrix.csv",
) -> pd.DataFrame:
    rows: List[Dict] = []
    md_dir = Path(manuscript_dir)
    if not md_dir.exists():
        log.warning("Manuscript directory not found: %s", md_dir)
        return pd.DataFrame()

    evidence_df = pd.read_csv(evidence_path) if Path(evidence_path).exists() else pd.DataFrame()

    for md_file in sorted(md_dir.glob("*.md")):
        text = md_file.read_text()
        section = md_file.stem

        # Count weak arguments
        weak_count = 0
        weak_examples: List[str] = []
        for pat in WEAK_ARGUMENT_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                weak_count += 1
                weak_examples.append(m.group(0))

        # Count missing citations
        missing_cite_count = 0
        missing_examples: List[str] = []
        for pat in MISSING_CITATION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                missing_cite_count += 1
                missing_examples.append(m.group(0)[:80])

        # Count unsupported claims (statements > 80 chars without inline citation)
        unsupported = 0
        unsupported_examples: List[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) > 80 and not stripped.startswith(("#", "-", "|", ">", "[")):
                if not re.search(r"\([A-Z][a-z]+.*?\d{4}\)", stripped):
                    unsupported += 1
                    if len(unsupported_examples) < 3:
                        unsupported_examples.append(stripped[:100])

        # Methodological concerns heuristic
        method_concerns = 0
        if "limitation" in text.lower() or "limitations" not in text.lower():
            method_concerns += 1
        if "sample" not in text.lower() and section == "methods_draft":
            method_concerns += 2

        # Novelty assessment
        if "novel" not in text.lower() and "new" not in text.lower():
            novelty_concern = 1
        else:
            novelty_concern = 0

        # Impact assessment
        impact_score = 3
        if "implication" in text.lower():
            impact_score += 1
        if "significance" in text.lower():
            impact_score += 1
        if "limitation" in text.lower():
            impact_score -= 1

        total_issues = weak_count + missing_cite_count + unsupported + method_concerns + novelty_concern
        if total_issues <= 2:
            verdict = "Acceptable"
        elif total_issues <= 5:
            verdict = "Minor Revision"
        else:
            verdict = "Major Revision"

        rows.append({
            "section": section,
            "weak_arguments": weak_count,
            "weak_argument_examples": "; ".join(weak_examples[:3])[:200],
            "missing_citations": missing_cite_count,
            "missing_citation_examples": "; ".join(missing_examples[:3])[:200],
            "unsupported_claims": unsupported,
            "unsupported_examples": "; ".join(unsupported_examples)[:200],
            "methodological_concerns": method_concerns,
            "novelty_concern": novelty_concern,
            "impact_score": impact_score,
            "total_issues": total_issues,
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


def _generate_comments(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Reviewer Comments\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if df.empty:
        lines.append("No manuscript sections found for review.\n")
        lines.append("---\n*Generated by reviewer_simulator.py*")
        return "\n".join(lines)

    for _, row in df.iterrows():
        lines.append(f"## {row['section'].replace('_', ' ').title()}\n")
        lines.append(f"**Verdict:** {row['verdict']}\n")
        if row["weak_arguments"]:
            lines.append(f"- **Weak arguments ({row['weak_arguments']}):** {row['weak_argument_examples']}")
        if row["missing_citations"]:
            lines.append(f"- **Missing citations ({row['missing_citations']}):** {row['missing_citation_examples']}")
        if row["unsupported_claims"]:
            lines.append(f"- **Unsupported claims ({row['unsupported_claims']}):** {row['unsupported_examples']}")
        if row["methodological_concerns"]:
            lines.append(f"- **Methodological concerns:** {row['methodological_concerns']} issue(s)")
        if row["novelty_concern"]:
            lines.append("- **Novelty concern:** Limited novelty language detected")
        lines.append(f"- **Impact score:** {row['impact_score']}/5\n")

    lines.append("## Summary\n")
    for v in ["Major Revision", "Minor Revision", "Acceptable"]:
        c = len(df[df["verdict"] == v])
        if c:
            lines.append(f"- {v}: {c} section(s)")
    lines.append("")

    lines.append("---\n*Generated by reviewer_simulator.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate peer review.")
    parser.add_argument("--manuscript-dir", type=str, default="outputs/manuscript")
    parser.add_argument("--evidence", type=str, default="outputs/evidence/evidence_matrix.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    df = evaluate_manuscript(args.manuscript_dir, args.evidence)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(out_dir / "reviewer_scorecard.csv", index=False)
        log.info("Saved -> %s", out_dir / "reviewer_scorecard.csv")

    comments = _generate_comments(df)
    with open(out_dir / "reviewer_comments.md", "w") as f:
        f.write(comments)
    log.info("Saved -> %s", out_dir / "reviewer_comments.md")

    print(f"\n--- Reviewer Simulation Complete ---")
    print(f"  Sections evaluated: {len(df)}")
    print()


if __name__ == "__main__":
    main()
