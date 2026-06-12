#!/usr/bin/env python3
"""
hypothesis_generator.py — Generate structured research hypotheses from
consensus themes, research gaps, and theme evolution classifications.

Reads theme metadata, paper lists, gap descriptions, and evolution
classifications, then produces a hypothesis bank with testable research
questions, variables, and suggested methodologies.

Usage
-----
    python hypothesis_generator.py
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--gaps outputs/reports/research_gaps.md]
        [--papers search_results.csv]
        [--consensus consensus_themes.csv]
        [--consensus-meta consensus_metadata.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hypothesis_generator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    research_question: str
    hypothesis: str
    iv: str
    dv: str
    controls: List[str] = field(default_factory=list)
    methodology: str = ""
    expected_contribution: str = ""
    priority_score: float = 0.0
    theme: str = ""
    gap_type: str = ""


# ---------------------------------------------------------------------------
# Pattern templates for hypothesis generation
# ---------------------------------------------------------------------------

_STUDY_TYPES = [
    "cross-sectional survey",
    "quasi-experimental design",
    "longitudinal cohort study",
    "randomised controlled trial",
    "mixed-methods case study",
    "systematic literature review",
    "comparative case analysis",
    "action research",
]

_CONTROLS_POOL = [
    "participant age",
    "educational level",
    "geographic region",
    "organisational size",
    "years of experience",
    "socio-economic status",
    "prior domain knowledge",
    "technology literacy",
    "institutional type",
    "gender",
]

_CONTRIBUTIONS = [
    "Provides empirical evidence on underexplored relationships within the theme.",
    "Extends theoretical frameworks by integrating cross-domain concepts.",
    "Offers actionable insights for practitioners and policy-makers.",
    "Addresses a critical gap identified in the systematic review of the literature.",
    "Establishes a foundation for future longitudinal and experimental work.",
    "Contributes to theory-building by testing assumptions in a novel context.",
    "Supports evidence-based decision-making in programme design and resource allocation.",
    "Validates emerging constructs and measurement instruments for the field.",
]

_DIRECTION_ADVERBS = ["significantly", "positively", "negatively", "substantially"]
_DIRECTION_VERBS = ["enhances", "reduces", "improves", "moderates", "mediates", "predicts", "shapes", "influences", "determines", "is associated with"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _clean_kw(kw: str) -> str:
    """Normalise a keyword for use in templates."""
    kw = kw.strip().lower()
    kw = re.sub(r"[^a-z0-9\s]", "", kw)
    return kw


def _extract_gaps(gaps_path: str) -> List[Dict]:
    """Parse the research_gaps.md file into structured gap records."""
    p = Path(gaps_path)
    if not p.exists():
        log.warning("Gaps file not found: %s", gaps_path)
        return []

    text = p.read_text()
    gaps: List[Dict] = []

    # Parse "Sparse Themes" section
    sparse = re.findall(r"\*\*(.+?)\*\*\s*\((\d+) papers?\)", text)
    for theme, count in sparse:
        gaps.append({
            "theme": _clean_text(theme),
            "type": "sparse_theme",
            "description": f"Only {count} papers available",
            "paper_count": int(count),
        })

    # Parse "Potential Future Directions" section
    future_match = re.search(r"## Potential Future Directions\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if future_match:
        bullets = re.findall(r"- (.+)", future_match.group(1))
        for b in bullets:
            gaps.append({
                "theme": _clean_text(b),
                "type": "future_direction",
                "description": b,
                "paper_count": 0,
            })

    if not gaps:
        gaps.append({
            "theme": "General research gap",
            "type": "generic",
            "description": "No specific gaps identified in the current corpus",
            "paper_count": 0,
        })

    return gaps


def _load_evolution_classifications(kb: Dict) -> Dict[str, str]:
    """Return a dict mapping theme -> evolution classification."""
    mapping: Dict[str, str] = {}
    for cls_key, label in [
        ("redundant_themes", "redundant"),
        ("developing_themes", "developing"),
        ("trending_themes", "trending"),
        ("future_themes", "future"),
    ]:
        items = kb.get(cls_key, [])
        if isinstance(items, list):
            for item in items:
                theme = item.get("theme", "")
                if theme:
                    mapping[theme] = label
    return mapping


# ---------------------------------------------------------------------------
# Hypothesis template engine
# ---------------------------------------------------------------------------

def _pick(items: List[str], seed: int) -> str:
    """Deterministic pick using hash-like index."""
    return items[seed % len(items)]


def _generate_hypotheses_for_theme(
    theme: Dict,
    gap: Dict,
    evolution_class: str,
    themes_keywords: Dict[str, List[str]],
    seed: int,
) -> List[Hypothesis]:
    """Generate hypotheses for one theme, targeting its specific gap."""
    label = theme.get("theme", theme.get("label", ""))
    keywords = theme.get("keywords", [])
    # Use non-NaN keywords only
    keywords = [k for k in keywords if k.strip().lower() not in ("", "nan")]
    if not keywords:
        keywords = label.replace(",", " ").split()

    paper_count = theme.get("paper_count", 0)
    confidence = theme.get("confidence", 0.5)

    # Choose the two most meaningful keywords as IV/DV candidates
    kw_clean = [_clean_kw(k) for k in keywords
                if _clean_kw(k) and _clean_kw(k) not in ("nan", "none", "different")]
    kw_clean = list(dict.fromkeys(kw_clean))  # deduplicate preserving order

    # Build concept pairs
    pairs: List[Tuple[str, str]] = []
    for i in range(len(kw_clean)):
        for j in range(i + 1, len(kw_clean)):
            pairs.append((kw_clean[i], kw_clean[j]))

    if not pairs:
        pairs = [(kw_clean[0] if kw_clean else "concept_a",
                  "concept_b")]

    gap_type = gap.get("type", "generic")
    results: List[Hypothesis] = []

    for idx, (iv_raw, dv_raw) in enumerate(pairs[:2]):  # max 2 per theme
        local_seed = seed + idx + paper_count

        # --- IV / DV ---
        iv = iv_raw.title()
        dv = dv_raw.title()

        # --- Research question ---
        if gap_type == "sparse_theme":
            rq = f"How does {iv} influence {dv} in the context of {label}?"
        elif gap_type == "future_direction":
            rq = f"To what extent does {iv} predict {dv}, and what moderating factors shape this relationship?"
        else:
            rq = f"What is the relationship between {iv} and {dv} in {label}?"

        # --- Hypothesis ---
        adv = _pick(_DIRECTION_ADVERBS, local_seed)
        verb = _pick(_DIRECTION_VERBS, local_seed + 1)
        if confidence > 0.7:
            hyp = f"Increased {iv} {adv} {verb} {dv}."
        else:
            hyp = f"{iv} {adv} {verb} {dv}, controlling for contextual factors."

        # --- Controls ---
        n_controls = min(3, 2 + (local_seed % 3))
        controls = []
        for ci in range(n_controls):
            controls.append(_pick(_CONTROLS_POOL, local_seed + ci + 3))

        # --- Methodology ---
        if gap_type == "sparse_theme":
            method = f"Mixed-methods exploratory design: qualitative interviews to surface constructs, followed by a {_pick(_STUDY_TYPES, local_seed)} to test the hypothesis."
        elif gap_type == "future_direction":
            method = f"Longitudinal {_pick(_STUDY_TYPES, local_seed)} with multi-level modelling to capture temporal dynamics."
        else:
            method = f"{_pick(_STUDY_TYPES, local_seed).capitalize()} using validated instruments and structural equation modelling."

        # --- Expected contribution ---
        contrib = _pick(_CONTRIBUTIONS, local_seed + 4)

        # --- Priority score (0–1) ---
        priority = round(
            min(paper_count / 10.0, 1.0) * 0.25    # theme maturity
            + (0.8 if gap_type == "sparse_theme" else 0.5) * 0.30  # gap urgency
            + confidence * 0.25                     # theme confidence
            + (1.0 - min(len(keywords) / 10.0, 1.0)) * 0.20,  # novelty (fewer kw = more novel)
            3,
        )

        results.append(Hypothesis(
            research_question=rq,
            hypothesis=hyp,
            iv=iv,
            dv=dv,
            controls=controls,
            methodology=method,
            expected_contribution=contrib,
            priority_score=priority,
            theme=label,
            gap_type=gap_type,
        ))

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    all_hypotheses: List[Hypothesis],
    gaps: List[Dict],
    theme_classifications: Dict[str, str],
    corpus_size: int,
    theme_count: int,
) -> str:
    """Generate hypothesis_bank.md."""
    lines: List[str] = []
    lines.append("# Hypothesis Bank")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Corpus:** {corpus_size} papers, {theme_count} themes")
    lines.append(f"- **Gaps identified:** {len(gaps)}")
    lines.append(f"- **Hypotheses generated:** {len(all_hypotheses)}")
    lines.append("")

    if corpus_size < 50:
        lines.append("> **Note:** Based on a limited literature sample. "
                     "Hypotheses are exploratory and should be refined through "
                     "targeted domain review.")
        lines.append("")

    # Gap summary
    lines.append("## Gap Overview")
    lines.append("")
    for g in gaps:
        t = g.get("theme", "")
        gt = g.get("type", "generic").replace("_", " ").title()
        lines.append(f"- **{t}** — {gt}: {g.get('description', '')}")
    lines.append("")

    # Group hypotheses by gap type then theme
    sections: Dict[str, List[Hypothesis]] = defaultdict(list)
    for h in all_hypotheses:
        sections[h.gap_type].append(h)

    section_labels = {
        "sparse_theme": "Sparse Themes — Limited Evidence",
        "future_direction": "Future Research Directions",
        "generic": "General Research Gaps",
    }

    for gap_type, display_name in section_labels.items():
        items = sections.get(gap_type, [])
        if not items:
            continue
        lines.append(f"## {display_name}")
        lines.append("")

        for i, h in enumerate(items, 1):
            evo = theme_classifications.get(h.theme, "").title() if h.theme in theme_classifications else "Not classified"

            lines.append(f"### {i}. {h.theme}")
            lines.append("")
            lines.append(f"| Field | Detail |")
            lines.append(f"|-------|--------|")
            lines.append(f"| **Research Question** | {h.research_question} |")
            lines.append(f"| **Hypothesis** | {h.hypothesis} |")
            lines.append(f"| **Independent Variable** | {h.iv} |")
            lines.append(f"| **Dependent Variable** | {h.dv} |")
            lines.append(f"| **Control Variables** | {', '.join(h.controls)} |")
            lines.append(f"| **Suggested Methodology** | {h.methodology} |")
            lines.append(f"| **Expected Contribution** | {h.expected_contribution} |")
            lines.append(f"| **Priority Score** | {h.priority_score:.2f} (0–1) |")
            lines.append(f"| **Theme Classification** | {evo} |")
            lines.append("")

    # Cross-cutting research questions
    lines.append("## Cross-Cutting Research Questions")
    lines.append("")
    lines.append("Questions that bridge multiple themes and gaps:")
    lines.append("")
    cross = [
        "How do digital transformation initiatives interact with existing capacity-building frameworks across different sectors?",
        "What contextual factors determine the success or failure of technology adoption in resource-constrained settings?",
        "How can training programme design be optimised for diverse learner populations in the digital era?",
        "What metrics best capture the long-term impact of capacity-building interventions?",
        "How do policy environments shape the effectiveness of digital agriculture and food system innovations?",
    ]
    for c in cross:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by hypothesis_generator.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured research hypotheses from review outputs."
    )
    parser.add_argument("--knowledge-base", type=str,
                        default="outputs/knowledge_base/knowledge_base.json",
                        help="Path to knowledge_base.json")
    parser.add_argument("--gaps", type=str,
                        default="outputs/reports/research_gaps.md",
                        help="Path to research_gaps.md")
    parser.add_argument("--papers", type=str, default="search_results.csv",
                        help="Path to search_results.csv")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv",
                        help="Path to consensus_themes.csv")
    parser.add_argument("--consensus-meta", type=str,
                        default="consensus_metadata.json",
                        help="Path to consensus_metadata.json")
    parser.add_argument("--output-dir", type=str, default="outputs/reports",
                        help="Output directory")
    args = parser.parse_args()

    # --- Load knowledge base ---
    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        log.error("Knowledge base not found: %s", kb_path)
        return
    with open(kb_path) as f:
        kb = json.load(f)
    log.info("Loaded knowledge base (%d themes, %d papers)",
             kb.get("theme_count", 0), kb.get("corpus_size", 0))

    corpus_size = kb.get("corpus_size", 0)
    theme_count = kb.get("theme_count", 0)

    # --- Load themes from knowledge base ---
    themes = kb.get("themes", [])
    if not themes:
        log.warning("No themes in knowledge base; falling back to consensus_metadata.")
        meta_path = Path(args.consensus_meta)
        if meta_path.exists():
            with open(meta_path) as f:
                themes = json.load(f)
        else:
            log.error("No theme data available.")
            return

    # --- Load research gaps ---
    gaps = _extract_gaps(args.gaps)
    log.info("Loaded %d research gaps", len(gaps))

    # --- Load evolution classifications ---
    theme_classifications = _load_evolution_classifications(kb)
    log.info("Loaded %d theme classifications", len(theme_classifications))

    # --- Optionally read paper titles per theme for richer context ---
    consensus_path = Path(args.consensus)
    theme_papers: Dict[str, List[str]] = defaultdict(list)
    if consensus_path.exists():
        df_c = pd.read_csv(consensus_path)
        for _, row in df_c.iterrows():
            theme_label = row.get("consensus_theme", "")
            title = str(row.get("title", ""))
            if theme_label and title.lower() not in ("nan", ""):
                theme_papers[theme_label].append(title[:100])

    # --- Build keyword map from themes ---
    themes_keywords: Dict[str, List[str]] = {}
    for t in themes:
        label = t.get("theme", t.get("label", ""))
        kws = t.get("keywords", [])
        if isinstance(kws, list):
            themes_keywords[label] = kws

    # --- Generate hypotheses ---
    all_hypotheses: List[Hypothesis] = []
    seed_base = hash(datetime.now().strftime("%Y%m%d"))

    for gap in gaps:
        gap_theme = gap.get("theme", "").lower()

        # For future_direction and generic gaps, generate for ALL themes
        if gap.get("type") in ("future_direction", "generic"):
            matched_themes = themes
        else:
            # Find matching themes by name overlap
            matched_themes = []
            for t in themes:
                label = t.get("theme", t.get("label", "")).lower()
                if gap_theme and (gap_theme in label or label in gap_theme):
                    matched_themes.append(t)
            if not matched_themes:
                matched_themes = themes

        for t in matched_themes:
            label = t.get("theme", t.get("label", ""))
            evo = theme_classifications.get(label, "unknown")
            seed = seed_base + len(all_hypotheses)
            hyps = _generate_hypotheses_for_theme(t, gap, evo, themes_keywords, seed)
            all_hypotheses.extend(hyps)

    # Deduplicate near-identical hypotheses
    seen: set = set()
    unique: List[Hypothesis] = []
    for h in all_hypotheses:
        key = (h.research_question[:80], h.hypothesis[:80])
        if key not in seen:
            seen.add(key)
            unique.append(h)
    all_hypotheses = unique

    log.info("Generated %d unique hypotheses", len(all_hypotheses))

    # --- Output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _generate_report(
        all_hypotheses, gaps, theme_classifications,
        corpus_size, theme_count,
    )
    report_path = out_dir / "hypothesis_bank.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    # --- CSV export ---
    rows = []
    for h in all_hypotheses:
        rows.append({
            "theme": h.theme,
            "gap_type": h.gap_type,
            "research_question": h.research_question,
            "hypothesis": h.hypothesis,
            "iv": h.iv,
            "dv": h.dv,
            "controls": "; ".join(h.controls),
            "methodology": h.methodology,
            "expected_contribution": h.expected_contribution,
            "priority_score": h.priority_score,
        })
    csv_path = out_dir / "hypothesis_bank.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("Saved %d rows -> %s", len(rows), csv_path)

    # --- Summary ---
    gap_type_counts = defaultdict(int)
    for h in all_hypotheses:
        gap_type_counts[h.gap_type] += 1
    print()
    print("--- Hypothesis Generation Complete ---")
    print(f"  Gaps processed:         {len(gaps)}")
    print(f"  Hypotheses generated:  {len(all_hypotheses)}")
    for gt, count in sorted(gap_type_counts.items()):
        print(f"    {gt.replace('_', ' ').title()}: {count}")
    print(f"  Report:                {report_path}")
    print(f"  CSV:                   {csv_path}")
    print()


if __name__ == "__main__":
    main()
