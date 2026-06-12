#!/usr/bin/env python3
"""
reproducibility_auditor.py — Evaluate the reproducibility of studies in the
corpus based on data availability, code availability, sample size adequacy,
statistical reporting, and validation strategies.

Outputs
-------
outputs/reports/reproducibility_report.md
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
log = logging.getLogger("reproducibility_auditor")


DATA_AVAIL_PATTERNS = [
    r"(?:data (?:are|is|were) (?:available|accessible|publicly available|open access))",
    r"(?:data (?:has|have) been (?:deposited|uploaded|shared|made available))",
    r"(?:available at|available from|accessible at|repository|supplementary material)",
    r"(?:github\.com|zenodo|figshare|dryad|dataverse|osf\.io)",
]

CODE_AVAIL_PATTERNS = [
    r"(?:code (?:is|was|has been) (?:available|made available|shared|deposited))",
    r"(?:available at github|as open.?source|our code|scripts? (?:are|were) available)",
    r"(?:github\.com.*code|reproducib|replicat)",
]

SAMPLE_PATTERNS = [
    r"(?:n\s*=\s*\d+|n\s*=\s*\d+|sample\s+(?:size|of)\s+\d+|n\s*=\s*\d+ participants)",
    r"(?:recruited|enrolled|surveyed|included)\s+\d+",
]

STATS_PATTERNS = [
    r"(?:p\s*[<>=]\s*0\.\d+|p\s*=\s*0\.\d+)",
    r"(?:95%\s*CI|confidence interval|effect size|cohen|hedge|eta|r\s*=)",
    r"(?:mean|median|sd|standard deviation|sem|standard error)",
]

VALIDATION_PATTERNS = [
    r"(?:cross.?validation|train.?test|hold.?out|k.?fold|bootstrap)",
    r"(?:internal validation|external validation|validation cohort|replication)",
    r"(?:sensitivity analysis|robustness check|subgroup analysis)",
]

CONTROL_PATTERNS = [
    r"(?:control group|comparison group|control condition|baseline)",
    r"(?:randomised|randomized|RCT|random assignment)",
    r"(?:matched|propensity score|instrumental variable|quasi.experiment)",
]


def audit_papers(papers_path: str = "search_results.csv") -> pd.DataFrame:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        doi = str(row.get("doi", ""))
        abstract = str(row.get("abstract", ""))
        if not title or title.lower() in ("nan", ""):
            continue

        data_avail = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in DATA_AVAIL_PATTERNS) else 0
        code_avail = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in CODE_AVAIL_PATTERNS) else 0
        sample_size = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in SAMPLE_PATTERNS) else 0
        stats_report = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in STATS_PATTERNS) else 0
        validation = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in VALIDATION_PATTERNS) else 0
        controls = 1 if any(re.search(p, abstract, re.IGNORECASE) for p in CONTROL_PATTERNS) else 0

        score = (data_avail + code_avail + sample_size + stats_report + validation + controls) / 6.0

        if score >= 0.67:
            level = "High Reproducibility"
        elif score >= 0.33:
            level = "Moderate Reproducibility"
        else:
            level = "Low Reproducibility"

        risk = round((1.0 - score) * 100, 1)

        rows.append({
            "title": title[:120],
            "doi": doi,
            "data_available": data_avail,
            "code_available": code_avail,
            "sample_size_reported": sample_size,
            "statistical_reporting": stats_report,
            "validation_strategy": validation,
            "experimental_controls": controls,
            "reproducibility_score": round(score, 3),
            "risk_score": risk,
            "classification": level,
        })

    return pd.DataFrame(rows)


def _generate_report(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Reproducibility Audit\n")
    lines.append(f"- **Papers evaluated:** {len(df)}")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for level in ["High Reproducibility", "Moderate Reproducibility", "Low Reproducibility"]:
        subset = df[df["classification"] == level]
        if subset.empty:
            continue
        lines.append(f"## {level}\n")
        lines.append(f"{len(subset)} paper(s)\n")
        for _, row in subset.iterrows():
            lines.append(f"- {row['title']} (risk: {row['risk_score']}%)")
        lines.append("")

    # Dimension summary
    lines.append("## Dimension Coverage\n")
    dims = ["data_available", "code_available", "sample_size_reported",
            "statistical_reporting", "validation_strategy", "experimental_controls"]
    labels = ["Data Availability", "Code Availability", "Sample Size Adequacy",
              "Statistical Reporting", "Validation Strategy", "Experimental Controls"]
    for dim, label in zip(dims, labels):
        pct = df[dim].mean() * 100
        lines.append(f"- **{label}:** {pct:.0f}%")

    lines.append("")
    avg_score = df["reproducibility_score"].mean()
    lines.append(f"**Overall Reproducibility Score:** {avg_score:.2f}/1.0\n")
    lines.append("---\n*Generated by reproducibility_auditor.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reproducibility.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    df = audit_papers(args.papers)
    report = _generate_report(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(out_dir / "reproducibility_scores.csv", index=False)
    with open(out_dir / "reproducibility_report.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "reproducibility_report.md")

    print(f"\n--- Reproducibility Audit Complete ---")
    print(f"  Papers evaluated: {len(df)}")
    for level in ["High Reproducibility", "Moderate Reproducibility", "Low Reproducibility"]:
        c = len(df[df["classification"] == level])
        if c:
            print(f"    {level}: {c}")
    print()


if __name__ == "__main__":
    main()
