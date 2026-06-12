#!/usr/bin/env python3
"""
methodology_mining.py — Extract and analyse research methods, study designs,
statistical tests, sample sizes, data collection approaches, computational
methods, and ML methods from paper titles and abstracts.

Usage
-----
    python methodology_mining.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("methodology_mining")

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

DESIGN_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Systematic Review", [r"\bsystematic review\b", r"\bmeta-analysis\b", r"\bmeta analysis\b", r"\bprisma\b"]),
    ("Literature Review", [r"\bliterature review\b", r"\bnarrative review\b", r"\bscoping review\b", r"\bmapping review\b"]),
    ("Randomised Controlled Trial", [r"\brct\b", r"\brandomized controlled\b", r"\brandomised controlled\b"]),
    ("Quasi-Experimental", [r"\bquasi-experimental\b", r"\bquasi experimental\b", r"\bpre-post\b", r"\binterrupted time series\b"]),
    ("Cross-Sectional", [r"\bcross-sectional\b", r"\bcross sectional\b", r"\bone-time survey\b"]),
    ("Longitudinal", [r"\blongitudinal\b", r"\bcohort\b", r"\bpanel data\b", r"\bfollow-up\b"]),
    ("Case Study", [r"\bcase study\b", r"\bcase report\b", r"\bcase analysis\b"]),
    ("Qualitative", [r"\bqualitative\b", r"\binterview\b", r"\bfocus group\b", r"\bgrounded theory\b", r"\bphenomenolog\b", r"\bethnograph\b"]),
    ("Mixed Methods", [r"\bmixed.method\b", r"\bmixed method\b"]),
    ("Survey", [r"\bsurvey\b", r"\bquestionnaire\b", r"\bself-report\b"]),
    ("Experimental", [r"\bexperiment\b", r"\bcontrolled trial\b", r"\btreatment group\b", r"\bintervention group\b"]),
    ("Conceptual / Theoretical", [r"\bframework\b", r"\bconceptual\b", r"\btheoretical\b", r"\bmodel\b", r"\btaxonomy\b", r"\btypology\b"]),
    ("Simulation", [r"\bsimulation\b", r"\bmodeling\b", r"\bmodelling\b", r"\bcomputational\b"]),
    ("Action Research", [r"\baction research\b", r"\bdesign science\b", r"\bparticipatory\b"]),
]

STAT_TEST_PATTERNS: List[Tuple[str, List[str]]] = [
    ("t-test", [r"\bt.test\b", r"\bstudent.t\b", r"\bt.test\b"]),
    ("ANOVA", [r"\banova\b", r"\bmanova\b", r"\bancova\b"]),
    ("Chi-square", [r"\bchi.square\b", r"\bchi square\b", r"\bchisq\b"]),
    ("Regression", [r"\bregression\b", r"\blogistic\b", r"\bols\b", r"\bglm\b"]),
    ("Correlation", [r"\bcorrelation\b", r"\bpearson\b", r"\bspearman\b"]),
    ("Factor Analysis", [r"\bfactor analysis\b", r"\bpca\b", r"\bprincipal component\b"]),
    ("Structural Equation", [r"\bsem\b", r"\bstructural equation\b", r"\bpath analysis\b"]),
    ("Non-parametric", [r"\bmann.whitney\b", r"\bwilcoxon\b", r"\bkruskal.wallis\b", r"\bfriedman\b"]),
    ("Bayesian", [r"\bbayesian\b", r"\bbayes\b", r"\bbayes factor\b"]),
    ("Machine Learning", [r"\bmachine learning\b", r"\bdeep learning\b", r"\bneural network\b", r"\brandom forest\b", r"\bsvm\b", r"\bgradient boosting\b"]),
]

DATA_COLLECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Primary Survey", [r"\bsurvey\b", r"\bquestionnaire\b", r"\bself-report\b"]),
    ("Interview", [r"\binterview\b", r"\bsemi-structured\b", r"\bunstructured interview\b"]),
    ("Focus Group", [r"\bfocus group\b"]),
    ("Observation", [r"\bobservation\b", r"\bfieldwork\b", r"\bethnograph\b"]),
    ("Secondary Data", [r"\bsecondary data\b", r"\badministrative data\b", r"\bpanel data\b"]),
    ("Experiment", [r"\blab experiment\b", r"\bfield experiment\b"]),
    ("Document Analysis", [r"\bdocument analysis\b", r"\bcontent analysis\b", r"\barchival\b"]),
    ("Sensor / IoT", [r"\bsensor\b", r"\bioT\b", r"\bwearable\b", r"\bremote sensing\b"]),
]

ML_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Random Forest", [r"\brandom forest\b", r"\brandom forest\b"]),
    ("Neural Network", [r"\bneural network\b", r"\bdeep learning\b", r"\bCNN\b", r"\bRNN\b", r"\btransformer\b", r"\bbert\b"]),
    ("Support Vector Machine", [r"\bsvm\b", r"\bsupport vector\b"]),
    ("Gradient Boosting", [r"\bgradient boosting\b", r"\bxgboost\b", r"\blightgbm\b", r"\bcatboost\b"]),
    ("Clustering", [r"\bclustering\b", r"\bk.means\b", r"\bhierarchical clustering\b", r"\bdbscan\b"]),
    ("Dimensionality Reduction", [r"\bdimensionality reduction\b", r"\bpca\b", r"\btsne\b", r"\bumap\b"]),
    ("Natural Language Processing", [r"\bnlp\b", r"\bnatural language\b", r"\btext mining\b", r"\btopic model\b", r"\bsentiment\b"]),
    ("Reinforcement Learning", [r"\breinforcement learning\b", r"\brl\b"]),
    ("Ensemble Methods", [r"\bensemble\b", r"\bbagging\b", r"\bboosting\b", r"\bstacking\b"]),
]

SAMPLE_SIZE_PATTERN = re.compile(r"\b[ns]\s*[=:]\s*(\d+)\b", re.IGNORECASE)


def extract_methods(texts: List[str]) -> Dict[str, Counter]:
    results: Dict[str, Counter] = {
        "study_designs": Counter(),
        "statistical_tests": Counter(),
        "data_collection": Counter(),
        "ml_methods": Counter(),
    }
    for text in texts:
        t = text.lower()
        for label, patterns in DESIGN_PATTERNS:
            if any(re.search(p, t) for p in patterns):
                results["study_designs"][label] += 1
        for label, patterns in STAT_TEST_PATTERNS:
            if any(re.search(p, t) for p in patterns):
                results["statistical_tests"][label] += 1
        for label, patterns in DATA_COLLECTION_PATTERNS:
            if any(re.search(p, t) for p in patterns):
                results["data_collection"][label] += 1
        for label, patterns in ML_PATTERNS:
            if any(re.search(p, t) for p in patterns):
                results["ml_methods"][label] += 1
    return results


def extract_sample_sizes(texts: List[str]) -> List[Dict]:
    samples = []
    for text in texts:
        matches = SAMPLE_SIZE_PATTERN.findall(text.lower())
        for m in matches:
            try:
                samples.append({"sample_size": int(m), "source": text[:80]})
            except ValueError:
                pass
    return samples


def analyze_methods_by_theme(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate study designs x consensus themes."""
    if "consensus_theme" not in df.columns:
        return pd.DataFrame()
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    themes = df["consensus_theme"].fillna("Unknown").tolist()
    rows = []
    for text, theme in zip(texts, themes):
        t = text.lower()
        designs = [label for label, patterns in DESIGN_PATTERNS if any(re.search(p, t) for p in patterns)]
        for d in designs:
            rows.append({"consensus_theme": theme, "study_design": d})
    if not rows:
        return pd.DataFrame()
    return pd.crosstab(
        pd.DataFrame(rows)["consensus_theme"],
        pd.DataFrame(rows)["study_design"],
        margins=True, margins_name="Total",
    )


def _generate_report(
    methods: Dict[str, Counter],
    sample_sizes: List[Dict],
    theme_design_matrix: pd.DataFrame,
    n_papers: int,
) -> str:
    lines: List[str] = []
    lines.append("# Methods Landscape Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Corpus:** {n_papers} papers")
    lines.append("")

    lines.append("## Most Common Study Designs")
    lines.append("")
    if methods["study_designs"]:
        for label, count in methods["study_designs"].most_common():
            lines.append(f"- **{label}:** {count} papers ({count/n_papers*100:.0f}%)")
        lines.append("")
    else:
        lines.append("_No study designs detected._\n")

    lines.append("## Statistical Tests Employed")
    lines.append("")
    if methods["statistical_tests"]:
        for label, count in methods["statistical_tests"].most_common():
            lines.append(f"- **{label}:** {count} occurrences")
        lines.append("")
    else:
        lines.append("_No statistical tests detected._\n")

    lines.append("## Data Collection Approaches")
    lines.append("")
    if methods["data_collection"]:
        for label, count in methods["data_collection"].most_common():
            lines.append(f"- **{label}:** {count} papers")
        lines.append("")
    else:
        lines.append("_No data collection approaches detected._\n")

    lines.append("## Computational / ML Methods")
    lines.append("")
    if methods["ml_methods"]:
        for label, count in methods["ml_methods"].most_common():
            lines.append(f"- **{label}:** {count} papers")
        lines.append("")
    else:
        lines.append("_No ML methods detected._\n")

    lines.append("## Sample Sizes")
    lines.append("")
    if sample_sizes:
        sizes = [s["sample_size"] for s in sample_sizes]
        lines.append(f"- **Range:** {min(sizes)} – {max(sizes)}")
        lines.append(f"- **Median:** {sorted(sizes)[len(sizes)//2]}")
        lines.append(f"- **Mean:** {sum(sizes)/len(sizes):.0f}")
        lines.append("")
    else:
        lines.append("_No sample size information extracted._\n")

    lines.append("## Study Designs by Theme")
    lines.append("")
    if not theme_design_matrix.empty:
        lines.append(theme_design_matrix.to_markdown())
        lines.append("")
    else:
        lines.append("_Theme cross-tabulation unavailable._\n")

    lines.append("## Emerging Methods")
    lines.append("")
    ml_total = sum(methods["ml_methods"].values()) if methods["ml_methods"] else 0
    if ml_total > 0:
        lines.append("Computational and ML methods are present in the corpus, indicating emerging methodological adoption.")
    else:
        lines.append("No computational/ML methods detected — potential area for methodological advancement.")
    lines.append("")

    lines.append("## Underused Methods / Methodological Gaps")
    lines.append("")
    all_designs = set(label for label, _ in DESIGN_PATTERNS)
    used_designs = set(methods["study_designs"].keys())
    unused = all_designs - used_designs
    if unused:
        for d in sorted(unused):
            lines.append(f"- **{d}** — not represented in the current corpus")
    else:
        lines.append("All common study designs are represented.")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by methodology_mining.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine research methods from paper metadata.")
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

    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    methods = extract_methods(texts)
    sample_sizes = extract_sample_sizes(texts)
    theme_design_matrix = analyze_methods_by_theme(df)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save methods database CSV
    rows = []
    for category, counter in methods.items():
        for label, count in counter.items():
            rows.append({"category": category, "method": label, "count": count, "percentage": round(count/len(df)*100, 1)})
    methods_df = pd.DataFrame(rows)
    methods_df.to_csv(out_dir / "methods_database.csv", index=False)
    log.info("Saved %d -> %s", len(methods_df), out_dir / "methods_database.csv")

    if not theme_design_matrix.empty:
        theme_design_matrix.to_csv(out_dir / "methods_by_theme.csv")

    report = _generate_report(methods, sample_sizes, theme_design_matrix, len(df))
    report_path = out_dir / "methods_summary.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    print(f"\n--- Methodology Mining Complete ---")
    print(f"  Study designs:     {len(methods['study_designs'])}")
    print(f"  Statistical tests: {len(methods['statistical_tests'])}")
    print(f"  ML methods:        {len(methods['ml_methods'])}")
    print()


if __name__ == "__main__":
    main()
