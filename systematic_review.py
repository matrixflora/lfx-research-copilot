#!/usr/bin/env python3
"""
systematic_review.py — Systematic and scoping review assistant.

Ingests search_results.csv and (optionally) consensus_themes.csv, then
applies inclusion/exclusion criteria, deduplicates, produces PRISMA flow
statistics, evidence tables, study characteristic extraction, an evidence
matrix, and risk-of-bias placeholders.

Usage
-----
    python systematic_review.py
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
        [--include-year-from YYYY]
        [--include-year-to YYYY]
        [--include-keywords kw1,kw2,...]
        [--exclude-keywords kw1,kw2,...]
        [--min-citations N]
        [--include-source source1,...]
        [--title-keywords kw1,kw2,...]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("systematic_review")

CURRENT_YEAR = datetime.now().year
SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreeningCriterion:
    field: str
    operator: str
    value: Any
    label: str

    def applies(self, row: pd.Series) -> bool:
        val = row.get(self.field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = "" if self.operator == "contains" else 0
        try:
            if self.operator == ">=":
                return float(val) >= float(self.value)
            elif self.operator == "<=":
                return float(val) <= float(self.value)
            elif self.operator == "contains":
                return str(self.value).lower() in str(val).lower()
            elif self.operator == "not_contains":
                return str(self.value).lower() not in str(val).lower()
            elif self.operator == "in":
                return str(val).lower() in {v.lower() for v in self.value}
        except (ValueError, TypeError):
            return False
        return False


# ---------------------------------------------------------------------------
# Inclusion / exclusion rule builders
# ---------------------------------------------------------------------------

def _build_inclusion_rules(args: argparse.Namespace) -> List[ScreeningCriterion]:
    rules: List[ScreeningCriterion] = []
    if args.include_year_from:
        rules.append(ScreeningCriterion("year", ">=", args.include_year_from,
                                         f"Year >= {args.include_year_from}"))
    if args.include_year_to:
        rules.append(ScreeningCriterion("year", "<=", args.include_year_to,
                                         f"Year <= {args.include_year_to}"))
    if args.include_keywords:
        for kw in args.include_keywords:
            rules.append(ScreeningCriterion(
                "abstract", "contains", kw.strip(),
                f"Abstract contains '{kw.strip()}'",
            ))
    if args.title_keywords:
        for kw in args.title_keywords:
            rules.append(ScreeningCriterion(
                "title", "contains", kw.strip(),
                f"Title contains '{kw.strip()}'",
            ))
    if args.min_citations is not None:
        rules.append(ScreeningCriterion(
            "citation_count", ">=", args.min_citations,
            f"Citations >= {args.min_citations}",
        ))
    if args.include_source:
        rules.append(ScreeningCriterion(
            "source", "in", args.include_source,
            f"Source in {args.include_source}",
        ))
    return rules


def _build_exclusion_rules(args: argparse.Namespace) -> List[ScreeningCriterion]:
    rules: List[ScreeningCriterion] = []
    if args.exclude_keywords:
        for kw in args.exclude_keywords:
            rules.append(ScreeningCriterion(
                "title", "not_contains", kw.strip(),
                f"Title excludes '{kw.strip()}'",
            ))
            rules.append(ScreeningCriterion(
                "abstract", "not_contains", kw.strip(),
                f"Abstract excludes '{kw.strip()}'",
            ))
    return rules


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates by DOI first, then by MiniLM title similarity."""
    n_before = len(df)
    # DOI-based dedup
    if "doi" in df.columns:
        df["_doi_clean"] = df["doi"].astype(str).str.strip().str.lower()
        df["_doi_clean"] = df["_doi_clean"].replace(["", "nan", "none"], np.nan)
        has_doi = df["_doi_clean"].notna()
        doi_groups = df.loc[has_doi].groupby("_doi_clean")
        keep_doi = doi_groups["citation_count"].idxmax() if "citation_count" in df.columns else doi_groups.head(1).index
        keep_doi = set(keep_doi)
        df["_keep"] = False
        df.loc[list(keep_doi), "_keep"] = True
        # Keep all rows without a DOI
        df.loc[~has_doi, "_keep"] = True
        df = df[df["_keep"]].drop(columns=["_doi_clean", "_keep"]).reset_index(drop=True)

    # Semantic dedup on remaining
    if len(df) > 1:
        titles = df["title"].fillna("").tolist()
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = model.encode(titles, show_progress_bar=False)
            sim_matrix = util.cos_sim(embs, embs).numpy()
            keep_idx: Set[int] = set(range(len(df)))
            for i in range(len(df)):
                if i not in keep_idx:
                    continue
                for j in range(i + 1, len(df)):
                    if j not in keep_idx:
                        continue
                    if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                        keep_idx.remove(j)
            df = df.iloc[sorted(keep_idx)].reset_index(drop=True)
        except Exception as e:
            log.warning("Semantic dedup failed (proceeding with DOI dedup only): %s", e)

    n_removed = n_before - len(df)
    log.info("Deduplication: %d before, %d after (%d removed)", n_before, len(df), n_removed)
    return df


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def _apply_screening(
    df: pd.DataFrame,
    inclusion_rules: List[ScreeningCriterion],
    exclusion_rules: List[ScreeningCriterion],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply screening rules. Returns (included, excluded, screening_log)."""
    log_records: List[Dict] = []
    included_rows: List[bool] = [True] * len(df)
    excluded_reasons: List[List[str]] = [[] for _ in range(len(df))]

    for idx, row in df.iterrows():
        # Inclusion rules – all must pass
        for rule in inclusion_rules:
            if not rule.applies(row):
                included_rows[idx] = False
                reason = f"Failed inclusion: {rule.label}"
                excluded_reasons[idx].append(reason)
                break

        # Exclusion rules – any one triggers exclusion (skip if already excluded)
        if included_rows[idx]:
            for rule in exclusion_rules:
                if rule.applies(row):
                    included_rows[idx] = False
                    reason = f"Excluded by: {rule.label}"
                    excluded_reasons[idx].append(reason)
                    break

        log_records.append({
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "year": row.get("year", ""),
            "decision": "included" if included_rows[idx] else "excluded",
            "reasons": "; ".join(excluded_reasons[idx]) if excluded_reasons[idx] else "",
        })

    included = df[included_rows].copy().reset_index(drop=True)
    excluded = df[[not r for r in included_rows]].copy().reset_index(drop=True)
    screening_log = pd.DataFrame(log_records)

    log.info("Screening complete: %d included, %d excluded", len(included), len(excluded))
    return included, excluded, screening_log


# ---------------------------------------------------------------------------
# Study characteristic extraction
# ---------------------------------------------------------------------------

_STUDY_TYPE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Systematic Review", [r"\bsystematic review\b", r"\bmeta-analysis\b", r"\bmeta analysis\b"]),
    ("Literature Review", [r"\bliterature review\b", r"\bnarrative review\b", r"\bscoping review\b"]),
    ("Survey", [r"\bsurvey\b", r"\bquestionnaire\b", r"\bresponse\b"]),
    ("Case Study", [r"\bcase study\b", r"\bcase report\b"]),
    ("Qualitative Study", [r"\bqualitative\b", r"\binterview\b", r"\bfocus group\b", r"\bthematic analysis\b"]),
    ("Quantitative Study", [r"\bquantitative\b", r"\bregression\b", r"\bstatistical\b", r"\bhypothesis\b"]),
    ("Mixed Methods", [r"\bmixed method\b", r"\bmixed-method\b"]),
    ("Experimental", [r"\brct\b", r"\brandomized\b", r"\brandomised\b", r"\bcontrolled trial\b", r"\bintervention\b"]),
    ("Conceptual / Theoretical", [r"\bframework\b", r"\bconceptual\b", r"\btheoretical\b", r"\bmodel\b"]),
    ("Policy / Commentary", [r"\bpolicy\b", r"\bcommentary\b", r"\bopinion\b", r"\bviewpoint\b", r"\bperspective\b"]),
]

_SETTING_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Higher Education", [r"\buniversity\b", r"\bhigher education\b", r"\btertiary\b", r"\bacademia\b"]),
    ("Healthcare", [r"\bhealth\b", r"\bclinical\b", r"\bmedical\b", r"\bhospital\b", r"\bpatient\b"]),
    ("Agriculture", [r"\bagricultur\b", r"\bfarmer\b", r"\bagri\b", r"\bcrop\b", r"\brural\b"]),
    ("Corporate / Industry", [r"\bcorporate\b", r"\bindustry\b", r"\bbusiness\b", r"\benterprise\b"]),
    ("Government / NGO", [r"\bgovernment\b", r"\bngo\b", r"\bnon.government\b", r"\bpublic sector\b"]),
    ("K-12 Education", [r"\bschool\b", r"\bprimary education\b", r"\bsecondary education\b"]),
    ("ICT / Digital", [r"\bdigital\b", r"\bict\b", r"\binformation technology\b", r"\bsoftware\b", r"\bplatform\b"]),
]


def _extract_study_type(text: str) -> str:
    text_lower = text.lower()
    for label, patterns in _STUDY_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower):
                return label
    return "Not specified"


def _extract_setting(text: str) -> str:
    text_lower = text.lower()
    for label, patterns in _SETTING_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower):
                return label
    return "Not specified"


def _extract_population(text: str) -> str:
    """Simple population extraction via known demographic keywords."""
    pop_keywords = [
        (r"\b(student|learner|trainee|undergraduate|graduate)\b", "Students / Trainees"),
        (r"\b(farmer|agricultural worker|smallholder)\b", "Farmers"),
        (r"\b(teacher|educator|trainer|instructor|faculty)\b", "Educators / Trainers"),
        (r"\b(patient|clinician|doctor|nurse|health worker)\b", "Healthcare Professionals"),
        (r"\b(engineer|developer|programmer|it professional)\b", "IT Professionals"),
        (r"\b(manager|executive|leader|director)\b", "Managers / Executives"),
        (r"\b(women|girl|female|gender)\b", "Women / Girls"),
        (r"\b(youth|adolescent|young)\b", "Youth"),
        (r"\b(community|rural|village)\b", "Community / Rural"),
        (r"\b(sme|entrepreneur|startup|small business)\b", "SMEs / Entrepreneurs"),
    ]
    text_lower = text.lower()
    found: List[str] = []
    for pat, label in pop_keywords:
        if re.search(pat, text_lower):
            found.append(label)
    return "; ".join(sorted(set(found))) if found else "Not specified"


def _extract_methodology(text: str) -> str:
    meth_keywords = [
        (r"\b(survey|questionnaire)\b", "Survey"),
        (r"\b(interview|focus group|ethnograph)\b", "Qualitative"),
        (r"\b(regression|correlation|statistical|quantitative)\b", "Quantitative"),
        (r"\b(mixed method|mixed-method)\b", "Mixed Methods"),
        (r"\b(case study|case analysis)\b", "Case Study"),
        (r"\b(systematic review|meta-analysis)\b", "Systematic Review"),
        (r"\b(rct|randomized|randomised|controlled trial)\b", "RCT / Controlled Trial"),
        (r"\b(action research|design science|design thinking)\b", "Action / Design Research"),
        (r"\b(literature review|scoping review)\b", "Literature Review"),
        (r"\b(simulation|modeling|forecast)\b", "Simulation / Modeling"),
        (r"\b(content analysis|thematic analysis)\b", "Content / Thematic Analysis"),
    ]
    text_lower = text.lower()
    found: List[str] = []
    for pat, label in meth_keywords:
        if re.search(pat, text_lower):
            found.append(label)
    return "; ".join(sorted(set(found))) if found else "Not specified"


def extract_study_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Add study_type, setting, population, methodology columns."""
    out = df.copy()
    texts = (out["title"].fillna("") + " " + out["abstract"].fillna("")).tolist()
    out["study_type"] = [_extract_study_type(t) for t in texts]
    out["setting"] = [_extract_setting(t) for t in texts]
    out["population"] = [_extract_population(t) for t in texts]
    out["methodology"] = [_extract_methodology(t) for t in texts]
    out["key_findings"] = ""  # placeholder for manual entry
    return out


# ---------------------------------------------------------------------------
# Risk of bias placeholders
# ---------------------------------------------------------------------------

_ROB_DOMAINS = [
    "selection_bias",
    "performance_bias",
    "detection_bias",
    "attrition_bias",
    "reporting_bias",
    "other_bias",
]


def _rob_assessment_template(n: int) -> pd.DataFrame:
    """Create Cochrane-style RoB assessment template."""
    rows = []
    for _ in range(n):
        rows.append({d: "" for d in _ROB_DOMAINS})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evidence matrix
# ---------------------------------------------------------------------------

def _build_evidence_matrix(
    df: pd.DataFrame,
    rows_field: str = "consensus_theme",
    cols_field: str = "study_type",
) -> pd.DataFrame:
    """Cross-tabulate two categorical fields."""
    if rows_field not in df.columns or cols_field not in df.columns:
        log.warning("Evidence matrix requires both '%s' and '%s' columns.", rows_field, cols_field)
        return pd.DataFrame()
    matrix = pd.crosstab(
        df[rows_field].fillna("Unknown"),
        df[cols_field].fillna("Not specified"),
        margins=True,
        margins_name="Total",
    )
    return matrix


# ---------------------------------------------------------------------------
# PRISMA flow
# ---------------------------------------------------------------------------

def _compute_prisma(
    n_identified: int,
    n_after_dedup: int,
    n_excluded_screening: int,
    n_excluded_fulltext: int,
    n_included: int,
) -> Dict[str, int]:
    return {
        "records_identified": n_identified,
        "records_after_deduplication": n_after_dedup,
        "records_screened": n_after_dedup,
        "records_excluded": n_excluded_screening,
        "full_text_assessed": n_after_dedup - n_excluded_screening,
        "full_text_excluded": n_excluded_fulltext,
        "studies_included_in_synthesis": n_included,
    }


def _prisma_markdown(flow: Dict[str, int]) -> str:
    lines: List[str] = []
    lines.append("## PRISMA 2020 Flow Diagram")
    lines.append("")
    lines.append("| Stage | Count |")
    lines.append("|-------|-------|")
    for stage, count in flow.items():
        label = stage.replace("_", " ").title()
        lines.append(f"| {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    flow: Dict[str, int],
    df_included: pd.DataFrame,
    df_excluded: pd.DataFrame,
    df_excluded_screening: pd.DataFrame,
    df_excluded_fulltext: pd.DataFrame,
    screening_log: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    rob: pd.DataFrame,
    inclusion_rules: List[ScreeningCriterion],
    exclusion_rules: List[ScreeningCriterion],
    consensus_available: bool,
    args: argparse.Namespace,
) -> str:
    """Generate systematic_review_report.md."""
    lines: List[str] = []
    lines.append("# Systematic Review Report")
    lines.append("")

    # Search strategy
    lines.append("## 1. Search Strategy")
    lines.append("")
    lines.append(f"- **Date of search:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"- **Databases searched:** Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, CORE")
    lines.append(f"- **Search query:** (from search_papers.py)")
    lines.append("")

    # Inclusion / exclusion criteria
    lines.append("## 2. Eligibility Criteria")
    lines.append("")
    lines.append("### Inclusion Criteria")
    if inclusion_rules:
        for r in inclusion_rules:
            lines.append(f"- {r.label}")
    else:
        lines.append("- No specific inclusion criteria applied")
    lines.append("")
    lines.append("### Exclusion Criteria")
    if exclusion_rules:
        for r in exclusion_rules:
            lines.append(f"- {r.label}")
    else:
        lines.append("- No specific exclusion criteria applied")
    lines.append("")

    # PRISMA flow
    lines.append(_prisma_markdown(flow))

    # Summary statistics
    lines.append("## 3. Screening Summary")
    lines.append("")
    lines.append(f"- **Records identified:** {flow['records_identified']}")
    lines.append(f"- **Duplicates removed:** {flow['records_identified'] - flow['records_after_deduplication']}")
    lines.append(f"- **Records screened:** {flow['records_screened']}")
    lines.append(f"- **Records excluded:** {flow['records_excluded']}")
    lines.append(f"- **Full-text assessed:** {flow['full_text_assessed']}")
    lines.append(f"- **Full-text excluded:** {flow['full_text_excluded']}")
    lines.append(f"- **Studies included:** {flow['studies_included_in_synthesis']}")
    lines.append("")

    # Included studies table
    lines.append("## 4. Included Studies")
    lines.append("")
    if len(df_included) > 0:
        display_cols = ["title", "authors", "year", "doi", "source", "citation_count"]
        if "study_type" in df_included.columns:
            display_cols.append("study_type")
        if "setting" in df_included.columns:
            display_cols.append("setting")
        if "consensus_theme" in df_included.columns:
            display_cols.append("consensus_theme")
        show = df_included[display_cols].copy()
        show["title"] = show["title"].str[:80]
        lines.append(show.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No studies met the inclusion criteria._")
        lines.append("")

    # Excluded studies table
    if len(df_excluded_screening) > 0 or len(df_excluded_fulltext) > 0:
        lines.append("## 5. Excluded Studies")
        lines.append("")
        df_excluded_all = pd.concat(
            [df_excluded_screening, df_excluded_fulltext], ignore_index=True
        )
        show_ex = df_excluded_all[["title", "year", "doi"]].copy()
        show_ex["title"] = show_ex["title"].str[:80]
        lines.append(show_ex.to_markdown(index=False))
        lines.append("")

    # Study characteristics
    if "study_type" in df_included.columns:
        lines.append("## 6. Study Characteristics")
        lines.append("")
        lines.append("### Study Types")
        lines.append("")
        st_counts = df_included["study_type"].fillna("Not specified").value_counts()
        for label, count in st_counts.items():
            lines.append(f"- **{label}:** {count}")
        lines.append("")

        lines.append("### Settings")
        lines.append("")
        set_counts = df_included["setting"].fillna("Not specified").value_counts()
        for label, count in set_counts.items():
            lines.append(f"- **{label}:** {count}")
        lines.append("")

        lines.append("### Populations")
        lines.append("")
        pop_counts = df_included["population"].fillna("Not specified").value_counts()
        for label, count in pop_counts.items():
            lines.append(f"- **{label}:** {count}")
        lines.append("")

    # Evidence matrix
    if not evidence_matrix.empty:
        lines.append("## 7. Evidence Matrix")
        lines.append("")
        lines.append("Cross-tabulation of **consensus theme** × **study type**.")
        lines.append("")
        lines.append(evidence_matrix.to_markdown())
        lines.append("")

    # Risk of bias
    lines.append("## 8. Risk of Bias Assessment")
    lines.append("")
    if len(rob) > 0:
        lines.append("Cochrane-style risk of bias domains (placeholders for manual entry):")
        lines.append("")
        lines.append("| Study | " + " | ".join(d.replace("_", " ").title() for d in _ROB_DOMAINS) + " |")
        lines.append("|-------|" + "|".join("---" for _ in _ROB_DOMAINS) + "|")
        for i in range(min(len(rob), len(df_included))):
            raw_title = df_included.iloc[i].get("title", f"Study {i+1}")
            title_short = str(raw_title)[:60] if not isinstance(raw_title, str) else raw_title[:60]
            vals = [str(rob.iloc[i].get(d, "")) for d in _ROB_DOMAINS]
            lines.append(f"| {title_short} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("_Note: Above assessments are placeholders. Each domain should be rated as Low / Unclear / High risk._")
        lines.append("")
    else:
        lines.append("_No studies available for risk of bias assessment._")
        lines.append("")

    # Evidence table
    lines.append("## 9. Evidence Table")
    lines.append("")
    if len(df_included) > 0:
        ev_cols = ["title", "authors", "year", "study_type", "setting", "population",
                    "methodology", "consensus_theme", "key_findings"]
        ev_cols = [c for c in ev_cols if c in df_included.columns]
        ev_table = df_included[ev_cols].copy()
        ev_table["title"] = ev_table["title"].str[:60] if "title" in ev_table else ev_table.iloc[:, 0]
        lines.append(ev_table.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_No studies to display._")
        lines.append("")

    lines.append("## 10. Research Gaps & Recommendations")
    lines.append("")
    gaps: List[str] = []
    if not evidence_matrix.empty and "Total" in evidence_matrix.index:
        for col in evidence_matrix.columns:
            if col == "Total":
                continue
            col_total = evidence_matrix.loc["Total", col] if "Total" in evidence_matrix.index else 0
            if col_total == 0:
                gaps.append(f"- No studies classified as **{col}** — potential methodological gap.")
    if not gaps:
        gaps.append("- Expand the search and screening to identify additional evidence gaps.")
    lines.extend(gaps)
    lines.append("")

    lines.append("---")
    lines.append("*Generated by systematic_review.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic / scoping review assistant."
    )
    parser.add_argument("--papers", type=str, default="search_results.csv",
                        help="Input CSV from search_papers.py")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv",
                        help="Consensus themes CSV from cluster_themes.py (optional)")
    parser.add_argument("--include-year-from", type=int, default=None,
                        help="Minimum publication year")
    parser.add_argument("--include-year-to", type=int, default=None,
                        help="Maximum publication year")
    parser.add_argument("--include-keywords", type=str, default=None,
                        help="Comma-separated keywords that must appear in abstract")
    parser.add_argument("--exclude-keywords", type=str, default=None,
                        help="Comma-separated keywords that cause exclusion")
    parser.add_argument("--min-citations", type=int, default=None,
                        help="Minimum citation count for inclusion")
    parser.add_argument("--include-source", type=str, default=None,
                        help="Comma-separated source names to restrict to")
    parser.add_argument("--title-keywords", type=str, default=None,
                        help="Comma-separated keywords that must appear in title")
    parser.add_argument("--output-dir", type=str, default="outputs/reports",
                        help="Output directory (default: outputs/reports)")
    args = parser.parse_args()

    # Check input
    papers_path = Path(args.papers)
    if not papers_path.exists():
        log.error("Input file not found: %s", papers_path)
        return

    # Parse multi-value args
    if args.include_keywords:
        args.include_keywords = [kw.strip() for kw in args.include_keywords.split(",") if kw.strip()]
    if args.exclude_keywords:
        args.exclude_keywords = [kw.strip() for kw in args.exclude_keywords.split(",") if kw.strip()]
    if args.include_source:
        args.include_source = [s.strip() for s in args.include_source.split(",") if s.strip()]
    if args.title_keywords:
        args.title_keywords = [kw.strip() for kw in args.title_keywords.split(",") if kw.strip()]

    # Load papers
    df_raw = pd.read_csv(args.papers)
    log.info("Loaded %d records from %s", len(df_raw), args.papers)

    # Load consensus themes (optional)
    consensus_path = Path(args.consensus)
    consensus_available = False
    if consensus_path.exists():
        df_consensus = pd.read_csv(consensus_path)
        consensus_available = True
        log.info("Loaded %d consensus records from %s", len(df_consensus), args.consensus)
        # Merge consensus theme into papers
        merge_cols = ["title", "doi"]
        merge_left = df_consensus[["title", "doi", "consensus_theme_id", "consensus_theme"]].copy()
        df_raw = df_raw.merge(merge_left, on="doi", how="left", suffixes=("", "_consensus"))
        if "consensus_theme" not in df_raw.columns:
            df_raw = df_raw.merge(merge_left, on="title", how="left", suffixes=("", "_consensus2"))
    else:
        log.info("Consensus themes not found; proceeding without theme labels.")

    # Deduplicate
    df_unique = _deduplicate(df_raw)
    n_identified = len(df_raw)
    n_after_dedup = len(df_unique)

    # Build rules
    inclusion_rules = _build_inclusion_rules(args)
    exclusion_rules = _build_exclusion_rules(args)

    # Apply screening
    df_included, df_excluded_screening, screening_log = _apply_screening(
        df_unique, inclusion_rules, exclusion_rules
    )

    # Full-text exclusion (none automated — placeholder for manual review)
    df_excluded_fulltext = pd.DataFrame()
    n_excluded_screening = len(df_excluded_screening)
    n_excluded_fulltext = 0
    n_included = len(df_included)

    # Extract study characteristics
    df_included = extract_study_characteristics(df_included)

    # PRISMA flow
    flow = _compute_prisma(n_identified, n_after_dedup, n_excluded_screening,
                           n_excluded_fulltext, n_included)

    # Evidence matrix
    evidence_matrix = pd.DataFrame()
    if consensus_available and "consensus_theme" in df_included.columns:
        evidence_matrix = _build_evidence_matrix(df_included, "consensus_theme", "study_type")

    # Risk of bias (placeholders)
    rob = _rob_assessment_template(len(df_included))

    # Build excluded studies for report
    df_excluded_all = df_excluded_screening.copy()
    if not df_excluded_fulltext.empty:
        df_excluded_all = pd.concat([df_excluded_all, df_excluded_fulltext], ignore_index=True)

    # Ensure output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV outputs
    df_included.to_csv(out_dir / "included_studies.csv", index=False)
    log.info("Saved %d -> %s", len(df_included), out_dir / "included_studies.csv")

    df_excluded_all.to_csv(out_dir / "excluded_studies.csv", index=False)
    log.info("Saved %d -> %s", len(df_excluded_all), out_dir / "excluded_studies.csv")

    screening_log.to_csv(out_dir / "screening_log.csv", index=False)
    log.info("Saved %d -> %s", len(screening_log), out_dir / "screening_log.csv")

    if not evidence_matrix.empty:
        evidence_matrix.to_csv(out_dir / "evidence_matrix.csv")
        log.info("Saved -> %s", out_dir / "evidence_matrix.csv")

    rob.to_csv(out_dir / "risk_of_bias.csv", index=False)
    log.info("Saved -> %s", out_dir / "risk_of_bias.csv")

    # Generate report
    report_md = _generate_report(
        flow,
        df_included,
        df_excluded_all,
        df_excluded_screening,
        df_excluded_fulltext,
        screening_log,
        evidence_matrix,
        rob,
        inclusion_rules,
        exclusion_rules,
        consensus_available,
        args,
    )
    report_path = out_dir / "systematic_review_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    log.info("Saved -> %s", report_path)

    # Print summary
    print()
    print("--- Systematic Review Complete ---")
    print(f"  Records identified:    {flow['records_identified']}")
    print(f"  After deduplication:   {flow['records_after_deduplication']}")
    print(f"  Excluded (screening):  {flow['records_excluded']}")
    print(f"  Excluded (full-text):  {flow['full_text_excluded']}")
    print(f"  Included:              {flow['studies_included_in_synthesis']}")
    print(f"  Report:                {report_path}")
    print(f"  Included studies:      {out_dir / 'included_studies.csv'}")
    print(f"  Excluded studies:      {out_dir / 'excluded_studies.csv'}")
    print(f"  Screening log:         {out_dir / 'screening_log.csv'}")
    if not evidence_matrix.empty:
        print(f"  Evidence matrix:       {out_dir / 'evidence_matrix.csv'}")
    print()


if __name__ == "__main__":
    main()
