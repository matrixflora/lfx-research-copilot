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
    rationale: str = ""
    measurable_variables: str = ""
    study_design: str = ""
    iv: str = ""
    dv: str = ""
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
_DIRECTION_VERBS = [
    "improves", "increases", "enhances", "strengthens", "boosts",
    "reduces", "decreases", "lowers", "diminishes",
    "accelerates", "facilitates", "promotes", "drives",
    "shapes", "moderates", "mediates", "predicts", "determines",
]

_POPULATION_CONTEXTS = [
    "among rural smallholder farmers",
    "in resource-constrained agricultural communities",
    "across low- and middle-income countries",
    "among extension service providers",
    "within peri-urban farming systems",
    "across diverse agro-ecological zones",
    "in government agricultural agencies",
    "among cooperative member farmers",
    "across supply chain actors",
    "in community-based development programmes",
]

_COMPARATIVE_CONTEXTS = [
    "compared with traditional extension approaches",
    "relative to conventional training methods",
    "compared with paper-based information channels",
    "relative to top-down technology transfer models",
    "compared with in-person workshops",
    "relative to standard practice guidelines",
    "compared with single-channel communication strategies",
    "relative to non-participatory approaches",
]

_CAPACITY_INTERVENTIONS = [
    "Digital capacity-building programmes",
    "Integrated training platforms",
    "Mobile-enabled advisory services",
    "Data-driven decision-support tools",
    "Peer-to-peer knowledge networks",
    "Multi-channel learning systems",
    "Participatory technology demonstrations",
    "Blended extension models",
]


def _generate_qwen_hypothesis(
    theme_label: str,
    keywords: List[str],
    gap_type: str,
    seed: int,
) -> Optional[str]:
    """Use Qwen 2.5 to generate a specific, well-formed hypothesis."""
    try:
        from src.agents.qwen_adapter import QwenAdapter
        qwen = QwenAdapter()
        kw_str = ", ".join(keywords[:6])
        gap_desc = {
            "sparse_theme": "limited existing evidence",
            "future_direction": "emerging research direction",
            "generic": "underexplored relationship",
        }.get(gap_type, "research gap")

        prompt = (
            f"Theme: {theme_label}\n"
            f"Keywords: {kw_str}\n"
            f"Gap type: {gap_desc}\n\n"
            "Generate one specific, testable research hypothesis (1 sentence, "
            "30-40 words). It must name a concrete intervention/program, "
            "specific measurable outcomes, relevant population context, and "
            "where appropriate a comparison condition. "
            "Use a strong causal verb. "
            "Return ONLY the hypothesis text, no labels or explanations."
        )
        result = qwen._call_model(prompt, max_new_tokens=60)
        if result:
            hyp = result.strip().strip('"').strip("'")
            words = hyp.split()
            if 12 <= len(words) <= 50 and hyp[0].isupper() and hyp.endswith("."):
                return hyp
    except Exception:
        pass
    return None


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
    """Generate specific, well-formed hypotheses for one theme."""
    label = theme.get("theme", theme.get("label", ""))
    keywords = theme.get("keywords", [])
    keywords = [k for k in keywords if k.strip().lower() not in ("", "nan")]
    if not keywords:
        keywords = label.replace(",", " ").split()

    paper_count = theme.get("paper_count", 0)
    confidence = theme.get("confidence", 0.5)
    gap_type = gap.get("type", "generic")

    kw_clean = [_clean_kw(k) for k in keywords
                if _clean_kw(k) and _clean_kw(k) not in ("nan", "none", "different")]
    kw_clean = list(dict.fromkeys(kw_clean))
    if not kw_clean:
        kw_clean = ["capacity", "outcome"]

    results: List[Hypothesis] = []
    max_hypotheses = 2

    for idx in range(max_hypotheses):
        local_seed = seed + idx + paper_count

        # --- Try Qwen first for a specific hypothesis ---
        qwen_hyp = _generate_qwen_hypothesis(label, kw_clean, gap_type, local_seed)
        if qwen_hyp:
            # Extract IV/DV from the Qwen hypothesis for structured fields
            words = qwen_hyp.split()
            iv = kw_clean[0].title() if kw_clean else "Intervention"
            dv = kw_clean[min(idx, len(kw_clean) - 1)].title() if kw_clean else "Outcome"

            n_controls = min(3, 2 + (local_seed % 3))
            controls = [
                _pick(_CONTROLS_POOL, local_seed + ci + 1)
                for ci in range(n_controls)
            ]
            rq = f"To what extent does {label.lower()} shape outcomes in this domain?"
            results.append(Hypothesis(
                research_question=rq,
                hypothesis=qwen_hyp,
                rationale=_generate_rationale(qwen_hyp, label, gap_type),
                measurable_variables=_generate_measurable_variables(iv, dv, qwen_hyp, label),
                study_design=_generate_study_design(qwen_hyp, iv, label),
                iv=iv,
                dv=dv,
                controls=controls,
                methodology=_pick(_STUDY_TYPES, local_seed).capitalize() + " with pre-post measurement and propensity score matching.",
                expected_contribution=_pick(_CONTRIBUTIONS, local_seed),
                priority_score=round(
                    min(paper_count / 10.0, 1.0) * 0.25
                    + (0.8 if gap_type == "sparse_theme" else 0.5) * 0.30
                    + confidence * 0.25
                    + (1.0 - min(len(kw_clean) / 10.0, 1.0)) * 0.20,
                    3,
                ),
                theme=label,
                gap_type=gap_type,
            ))
            continue

        # --- Fallback: template-based specific hypotheses ---
        intervention = _pick(_CAPACITY_INTERVENTIONS, local_seed)
        context = _pick(_POPULATION_CONTEXTS, local_seed + 1)

        # Use theme label as the outcome domain for richer phrasing
        theme_short = label.split(",")[0].strip()
        theme_words = [
            w for w in theme_short.split()
            if len(w) > 3 and w.lower() not in ("with", "from", "this", "that", "are")
        ]
        specific_outcome = " ".join(theme_words[:3]).lower()
        if not specific_outcome:
            specific_outcome = theme_short.lower()[:40]

        if idx == 0:
            comparison = _pick(_COMPARATIVE_CONTEXTS, local_seed + 2)
            verb = _pick(_DIRECTION_VERBS, local_seed + 3)
            hyp = (
                f"{intervention} {verb} {specific_outcome} "
                f"{context} {comparison}."
            ).replace("  ", " ")
            rq = (
                f"How do {intervention.lower()} affect {specific_outcome} "
                f"{context}?"
            ).replace("  ", " ")
        else:
            verb = _pick(_DIRECTION_VERBS, local_seed + 4)
            hyp = (
                f"Integrated approaches combining {intervention.lower()} "
                f"with {specific_outcome} {verb} "
                f"relevant outcomes "
                f"{context}."
            ).replace("  ", " ")
            rq = (
                f"How can integrated {specific_outcome} approaches "
                f"be optimised {context}?"
            ).replace("  ", " ")

        # --- Controls ---
        n_controls = min(3, 2 + (local_seed % 3))
        controls = [
            _pick(_CONTROLS_POOL, local_seed + ci + 5)
            for ci in range(n_controls)
        ]

        # --- Methodology ---
        if gap_type == "sparse_theme":
            method = (
                f"Mixed-methods exploratory design: qualitative interviews "
                f"to surface mechanisms, followed by a "
                f"{_pick(_STUDY_TYPES, local_seed)} "
                f"to test the hypothesis."
            )
        elif gap_type == "future_direction":
            method = (
                f"Longitudinal {_pick(_STUDY_TYPES, local_seed)} "
                f"with multi-level modelling to capture temporal dynamics "
                f"and heterogeneous treatment effects."
            )
        else:
            method = (
                f"{_pick(_STUDY_TYPES, local_seed).capitalize()} "
                f"using validated instruments and difference-in-differences "
                f"estimation."
            )

        # --- Expected contribution ---
        contrib = _pick(_CONTRIBUTIONS, local_seed + 6)

        # --- Priority score (0–1) ---
        priority = round(
            min(paper_count / 10.0, 1.0) * 0.25
            + (0.8 if gap_type == "sparse_theme" else 0.5) * 0.30
            + confidence * 0.25
            + (1.0 - min(len(kw_clean) / 10.0, 1.0)) * 0.20,
            3,
        )

        results.append(Hypothesis(
            research_question=rq,
            hypothesis=hyp,
            rationale=_generate_rationale(hyp, label, gap_type),
            measurable_variables=_generate_measurable_variables(intervention, specific_outcome.title(), hyp, label),
            study_design=_generate_study_design(hyp, intervention, label),
            iv=intervention,
            dv=specific_outcome.title(),
            controls=controls,
            methodology=method,
            expected_contribution=contrib,
            priority_score=priority,
            theme=label,
            gap_type=gap_type,
        ))

    return results


# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------

_RATIONALES: Dict[str, List[str]] = {
    "sparse_theme": [
        "Despite growing interest in this area, the empirical evidence base remains thin. Testing this hypothesis would provide much-needed rigorous evidence to support or refute current assumptions.",
        "Current literature offers conceptual frameworks but little causal evidence. This hypothesis targets a critical evidence gap where policy and practice decisions are being made without adequate empirical support.",
        "Preliminary studies suggest an effect but lack statistical power and controls. A focused investigation would clarify whether the observed patterns hold under more rigorous conditions.",
    ],
    "future_direction": [
        "Emerging trends point toward this relationship as a key driver of outcomes, yet no study has directly examined the causal pathway. This hypothesis would provide an early empirical test of a growing theoretical proposition.",
        "Several recent reviews identify this as a priority area for future research. Addressing it would contribute to shaping the next generation of interventions and studies in the field.",
        "Technological and methodological advances now make it feasible to test this relationship rigorously, where earlier work could only speculate. This hypothesis capitalises on those advances.",
    ],
    "generic": [
        "The relationship between these constructs is frequently asserted but rarely tested directly. This hypothesis would provide the first dedicated empirical examination.",
        "Existing studies approach this question indirectly or with limited scope. A purpose-designed investigation would resolve ambiguity in the current evidence base.",
        "Practitioners and policymakers need clearer guidance on this question. Testing this hypothesis would produce actionable evidence for programme design and resource allocation.",
    ],
}

_MEASURABLE_TEMPLATES: Dict[str, List[str]] = {
    "iv": [
        "Programme participation status (binary: enrolled / not enrolled)",
        "Frequency of intervention exposure (sessions attended per month)",
        "Intervention delivery modality (digital / in-person / blended)",
        "Duration of engagement with the programme (weeks or months)",
        "Dosage level (low / medium / high based on hours of contact)",
        "Access to intervention resources (yes / no; count of resources used)",
        "Implementation fidelity score (0-100 scale)",
        "Self-reported adoption rate (Likert scale 1-5)",
    ],
    "dv": [
        "Knowledge test score (standardised assessment, 0-100%)",
        "Technology adoption rate (proportion of target practices adopted)",
        "Behaviour change index (composite score of observed practices)",
        "Productivity or yield (units per hectare or per worker)",
        "Income or revenue change (percentage change from baseline)",
        "Skill competency score (observed assessment rubric, 0-100)",
        "Retention rate (proportion of knowledge or practice sustained at follow-up)",
        "Self-efficacy scale (validated instrument, e.g. Likert 1-5)",
        "Time to adoption (months from intervention start to first adoption)",
        "Cost-effectiveness ratio (cost per unit outcome achieved)",
    ],
}

_STUDY_DESIGNS: List[str] = [
    "Cluster-randomised controlled trial with treatment and control villages, baseline and endline surveys, and stratified random sampling by agro-ecological zone.",
    "Quasi-experimental difference-in-differences design comparing early and late adopters, with propensity score matching on observable characteristics.",
    "Stepped-wedge randomised rollout across administrative units, with repeated cross-sectional surveys at each step to estimate the intervention effect.",
    "Matched cohort study comparing programme participants with non-participants using coarsened exact matching on demographic and farm characteristics.",
    "Mixed-methods sequential explanatory design: quantitative pre-post survey with embedded qualitative interviews to explore mechanisms and contextual factors.",
    "Longitudinal panel study with three waves over 24 months, using fixed-effects models to estimate within-subject change over time.",
    "Regression discontinuity design exploiting a programme eligibility threshold, with robustness checks using alternative bandwidths and polynomial orders.",
    "Factorial randomised experiment testing two or more intervention components independently to identify which mechanisms drive outcomes.",
]


def _generate_qwen_rationale(hypothesis: str, theme: str, gap_type: str) -> Optional[str]:
    try:
        from src.agents.qwen_adapter import QwenAdapter
        qwen = QwenAdapter()
        prompt = (
            f"Theme: {theme}\nHypothesis: {hypothesis}\n\n"
            "Write a 2-3 sentence rationale for why this hypothesis is important "
            "to test. Mention the specific gap it addresses, why existing evidence "
            "is insufficient, and what answering it would contribute. "
            "Return ONLY the rationale text."
        )
        result = qwen._call_model(prompt, max_new_tokens=120)
        if result and len(result.split()) >= 15:
            return result.strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _generate_qwen_variables(hypothesis: str, iv_label: str, dv_label: str, theme: str) -> Optional[str]:
    try:
        from src.agents.qwen_adapter import QwenAdapter
        qwen = QwenAdapter()
        prompt = (
            f"Theme: {theme}\nHypothesis: {hypothesis}\n"
            f"Independent variable: {iv_label}\nDependent variable: {dv_label}\n\n"
            "List 2-3 specific, measurable indicators for each variable. "
            "Format:\nIV: indicator1, indicator2\nDV: indicator1, indicator2\n"
            "Return ONLY the formatted list."
        )
        result = qwen._call_model(prompt, max_new_tokens=100)
        if result and len(result.split()) >= 8:
            return result.strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _generate_qwen_study_design(hypothesis: str, iv_label: str, theme: str) -> Optional[str]:
    try:
        from src.agents.qwen_adapter import QwenAdapter
        qwen = QwenAdapter()
        prompt = (
            f"Theme: {theme}\nHypothesis: {hypothesis}\n"
            f"Independent variable: {iv_label}\n\n"
            "Recommend one specific study design to test this hypothesis. "
            "Describe the design type, comparison strategy, sampling approach, "
            "and key methodological features. 2-3 sentences. "
            "Return ONLY the design description."
        )
        result = qwen._call_model(prompt, max_new_tokens=120)
        if result and len(result.split()) >= 15:
            return result.strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _generate_rationale(hyp_text: str, theme: str, gap_type: str) -> str:
    qwen = _generate_qwen_rationale(hyp_text, theme, gap_type)
    if qwen:
        return qwen
    pool = _RATIONALES.get(gap_type, _RATIONALES["generic"])
    seed = hash(hyp_text) % len(pool)
    return pool[seed]


def _generate_measurable_variables(iv_label: str, dv_label: str, hyp_text: str, theme: str) -> str:
    qwen = _generate_qwen_variables(hyp_text, iv_label, dv_label, theme)
    if qwen:
        return qwen
    seed = hash(hyp_text)
    iv_pool = _MEASURABLE_TEMPLATES["iv"]
    dv_pool = _MEASURABLE_TEMPLATES["dv"]
    iv1 = iv_pool[seed % len(iv_pool)]
    iv2 = iv_pool[(seed + 1) % len(iv_pool)]
    dv1 = dv_pool[(seed + 2) % len(dv_pool)]
    dv2 = dv_pool[(seed + 3) % len(dv_pool)]
    return (
        f"Independent Variable ({iv_label}): {iv1}; {iv2}.\n"
        f"Dependent Variable ({dv_label}): {dv1}; {dv2}."
    )


def _generate_study_design(hyp_text: str, iv_label: str, theme: str) -> str:
    qwen = _generate_qwen_study_design(hyp_text, iv_label, theme)
    if qwen:
        return qwen
    seed = hash(hyp_text + theme) % len(_STUDY_DESIGNS)
    return _STUDY_DESIGNS[seed]


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
            lines.append(f"| **Rationale** | {h.rationale} |")
            lines.append(f"| **Independent Variable** | {h.iv} |")
            lines.append(f"| **Dependent Variable** | {h.dv} |")
            lines.append(f"| **Measurable Variables** | {h.measurable_variables.replace(chr(10), '<br>')} |")
            lines.append(f"| **Control Variables** | {', '.join(h.controls)} |")
            lines.append(f"| **Suggested Methodology** | {h.methodology} |")
            lines.append(f"| **Study Design Suggestion** | {h.study_design} |")
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
            "rationale": h.rationale,
            "iv": h.iv,
            "dv": h.dv,
            "measurable_variables": h.measurable_variables,
            "controls": "; ".join(h.controls),
            "methodology": h.methodology,
            "study_design": h.study_design,
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
