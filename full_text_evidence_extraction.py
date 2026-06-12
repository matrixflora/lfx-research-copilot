#!/usr/bin/env python3
"""
full_text_evidence_extraction.py — Extract structured evidence from paper
abstracts (full-text proxies) into standardised categories.

Outputs
-------
outputs/evidence/evidence_matrix.csv
outputs/evidence/evidence_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("evidence_extraction")

# Regex patterns for each section
OBJECTIVE_PATTERNS = [
    r"(?:objective|aim|goal|purpose|we aimed|we sought|this study (?:aims|aimed|seeks|sought))[\s:]*?(.+?)(?:\.\s[A-Z]|$)",
    r"(?:to investigate|to examine|to explore|to understand|to assess|to evaluate|to determine|to identify|to develop)[\s\w]+?(?:\.\s|$)",
]
METHODS_PATTERNS = [
    r"(?:methods?|methodology|approach|design|we (?:used|employed|conducted|performed|collected|analysed|analysed|applied))[\s:]*?(.+?)(?:\.\s[A-Z]|$)",
    r"(?:we recruited|we enrolled|we surveyed|we interviewed|we randomised|we randomized|we extracted)[\s\w]+?(?:\.\s|$)",
    r"(?:data (?:were|was) (?:collected|obtained|gathered|analysed))[\s\w]+?(?:\.\s|$)",
]
RESULTS_PATTERNS = [
    r"(?:results?|findings?|we found|we observed|we identified|we demonstrate)[" "\s:]*?(.+?)(?:\.\s[A-Z]|$)",
    r"(?:significant|significantly|increased|decreased|higher|lower|improved|reduced)[\s\w]+?(?:\.\s|$)",
    r"(?:p\s*[<>=]\s*0\.\d+|p\s*=\s*0\.\d+)",
]
LIMITATIONS_PATTERNS = [
    r"(?:limitation|limitations|limiting|we acknowledge|we note|caveat|caveats|however|but|although|despite)[\s:]*?(.+?)(?:\.\s[A-Z]|$)",
    r"(?:further research|further studies|additional work|future work)[\s\w]+?(?:\.\s|$)",
]
CONCLUSION_PATTERNS = [
    r"(?:conclusion|conclusions|conclude?|we conclude|summary|in summary|in conclusion|overall|taken together)[\s:]*?(.+?)(?:\.\s[A-Z]|$)",
    r"(?:our findings suggest|our results suggest|this study demonstrates|these findings indicate)[\s\w]+?(?:\.\s|$)",
]


def extract_by_patterns(text: str, patterns: List[str], default: str = "Not specified") -> str:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()[:300]
    return default


def extract_objective(text: str) -> str:
    return extract_by_patterns(text, OBJECTIVE_PATTERNS)


def extract_methods(text: str) -> str:
    return extract_by_patterns(text, METHODS_PATTERNS)


def extract_results(text: str) -> str:
    return extract_by_patterns(text, RESULTS_PATTERNS)


def extract_limitations(text: str) -> str:
    return extract_by_patterns(text, LIMITATIONS_PATTERNS, "None stated")


def extract_conclusions(text: str) -> str:
    return extract_by_patterns(text, CONCLUSION_PATTERNS)


def build_evidence_matrix(papers_path: str, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    if df.empty:
        log.warning("No papers found at %s", papers_path)
        return df

    rows = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        doi = str(row.get("doi", ""))
        abstract = str(row.get("abstract", ""))
        if not title or title.lower() in ("nan", "none", ""):
            continue
        rows.append({
            "Paper_ID": doi or title[:40],
            "Title": title[:200],
            "Objective": extract_objective(abstract),
            "Methods": extract_methods(abstract),
            "Results": extract_results(abstract),
            "Limitations": extract_limitations(abstract),
            "Conclusions": extract_conclusions(abstract),
            "DOI": doi,
        })

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "evidence_matrix.csv", index=False)
    log.info("Saved %d -> %s", len(result), output_dir / "evidence_matrix.csv")
    return result


def _generate_summary(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Evidence Summary\n")
    lines.append(f"- **Papers processed:** {len(df)}")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Coverage\n")
    for col in ["Objective", "Methods", "Results", "Limitations", "Conclusions"]:
        specified = (df[col] != "Not specified").sum() if col != "Limitations" else (df[col] != "None stated").sum()
        lines.append(f"- **{col}:** {specified}/{len(df)} ({specified/len(df)*100:.0f}%)")
    lines.append("")

    lines.append("## Sample Entries\n")
    for _, row in df.head(5).iterrows():
        lines.append(f"### {row['Title'][:80]}")
        for col in ["Objective", "Methods", "Results", "Limitations", "Conclusions"]:
            val = str(row[col])[:150]
            lines.append(f"- **{col}:** {val}")
        lines.append("")

    lines.append("---\n*Generated by full_text_evidence_extraction.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured evidence from papers.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/evidence")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df = build_evidence_matrix(args.papers, out_dir)

    summary = _generate_summary(df)
    with open(out_dir / "evidence_summary.md", "w") as f:
        f.write(summary)
    log.info("Saved -> %s", out_dir / "evidence_summary.md")

    print(f"\n--- Evidence Extraction Complete ---")
    print(f"  Papers processed: {len(df)}")
    print()


if __name__ == "__main__":
    main()
