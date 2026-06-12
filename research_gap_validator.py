#!/usr/bin/env python3
"""
research_gap_validator.py — Validate research gaps by searching the corpus
for evidence that may already address claimed gaps.  Assigns confidence
scores and identifies truly underexplored areas.

Outputs
-------
outputs/reports/validated_research_gaps.md
outputs/reports/gap_confidence_scores.csv
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
log = logging.getLogger("gap_validator")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_evidence_scores(path: str = "outputs/reports/evidence_strength.csv") -> Dict[str, Dict]:
    df = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    scores = {}
    for _, row in df.iterrows():
        theme = str(row.get("theme", "")).strip()
        if theme:
            scores[theme] = dict(row)
    return scores


def load_consensus_metadata(path: str = "consensus_metadata.json") -> List[Dict]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def load_knowledge_base(path: str = "knowledge_base.json") -> Dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def extract_gap_statements(gaps_md: str, kb: Dict, evidence: Dict) -> List[Dict]:
    """Extract gap claims from research_gaps.md and knowledge base."""
    gaps: List[Dict] = []

    # Parse sparse themes from gaps_md
    for m in re.finditer(r'\*\*(.*?)\*\*\s*\((\d+) papers?\)', gaps_md):
        theme = m.group(1).strip()
        paper_count = int(m.group(2))
        gaps.append({
            "gap_statement": f"{theme} — only {paper_count} papers, limited evidence",
            "theme": theme,
            "paper_count": paper_count,
            "source": "research_gaps.md",
            "gap_type": "sparse_theme",
        })

    # Future directions from gaps_md
    for m in re.finditer(r'- Investigate \*\*(.*?)\*\*', gaps_md):
        theme = m.group(1).strip()
        gaps.append({
            "gap_statement": f"Investigate {theme} further with targeted literature searches",
            "theme": theme,
            "paper_count": 0,
            "source": "research_gaps.md",
            "gap_type": "future_direction",
        })

    # Thematic gaps from KB
    for t in kb.get("themes", []):
        theme = t.get("theme", "")
        pc = t.get("paper_count", 0)
        conf = t.get("confidence", 0.5)
        if pc <= 3 or conf < 0.65:
            gap_text = f"{theme} — {pc} papers, low confidence ({conf:.2f})"
            # Avoid duplicates
            if not any(g["theme"] == theme for g in gaps):
                gaps.append({
                    "gap_statement": gap_text,
                    "theme": theme,
                    "paper_count": pc,
                    "confidence": conf,
                    "source": "knowledge_base.json",
                    "gap_type": "low_confidence_theme" if conf < 0.65 else "sparse_theme",
                })

    return gaps


def validate_gaps(
    gaps: List[Dict],
    papers_df: pd.DataFrame,
    evidence: Dict[str, Dict],
    consensus_md: List[Dict],
    model: SentenceTransformer,
) -> pd.DataFrame:
    """Score each gap on how genuine it is.  HIGH score = real gap (little evidence).
    LOW score = false gap (substantial evidence already exists)."""
    rows = []
    paper_texts = []
    for _, r in papers_df.iterrows():
        t = str(r.get("title", ""))
        a = str(r.get("abstract", ""))
        paper_texts.append(f"{t} {a}")

    paper_embs = model.encode(paper_texts, show_progress_bar=False) if paper_texts else np.array([])
    paper_embs = paper_embs / np.linalg.norm(paper_embs, axis=1, keepdims=True) if paper_embs.ndim == 2 else paper_embs

    # Build theme->methods map from consensus metadata
    theme_methods: Dict[str, List[str]] = {}
    for cm in consensus_md:
        label = cm.get("label", "")
        methods = cm.get("methods", [])
        if label:
            theme_methods[label] = methods

    for g in gaps:
        theme = g["theme"]
        pc = g.get("paper_count", 0)
        gap_text = g["gap_statement"]

        # 1. Literature coverage — what fraction of papers address this?
        q_emb = model.encode([gap_text], show_progress_bar=False)[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        sims = np.dot(paper_embs, q_emb) if paper_embs.ndim == 2 else np.array([0.0])
        high_sim = float((sims > 0.35).sum())
        coverage = high_sim / max(len(sims), 1)  # fraction of corpus that addresses the gap
        # Inverted: coverage 0 = real gap, coverage 1 = false gap

        # 2. Evidence strength (inverted)
        ev = evidence.get(theme, {})
        ev_score = float(ev.get("evidence_score", 0.5))
        ev_inverted = 1.0 - ev_score  # 0 = strong evidence (false gap), 1 = weak (real gap)

        # 3. Paper count (inverted)
        pc_norm = min(pc / 10.0, 1.0)
        pc_inverted = 1.0 - pc_norm

        # 4. Citation poverty
        total_cites = float(ev.get("total_citations", 0))
        cite_poverty = 1.0 - min(total_cites / 500.0, 1.0)

        # 5. Method consensus (inverted)
        methods = theme_methods.get(theme, [])
        method_consensus = len(methods) / 3.0  # 0.33 to 1.0
        method_inverted = 1.0 - method_consensus

        # 6. Confidence (inverted)
        conf = float(g.get("confidence", ev.get("theme_consistency_score", 0.5)))
        conf_inverted = 1.0 - conf

        # Composite: HIGH = real gap
        gap_confidence = (
            coverage * 0.0          # inverted — 0 coverage = +gap
            + pc_inverted * 0.15
            + ev_inverted * 0.30
            + cite_poverty * 0.15
            + method_inverted * 0.15
            + conf_inverted * 0.10
            + (1.0 - coverage) * 0.15  # low coverage = real gap
        )

        if gap_confidence >= 0.65:
            verdict = "Confirmed Gap"
        elif gap_confidence >= 0.45:
            verdict = "Likely Gap"
        elif gap_confidence >= 0.25:
            verdict = "Uncertain"
        else:
            verdict = "Likely False Gap"

        supporting_papers = []
        if sims.ndim > 0:
            top_indices = np.argsort(sims)[-3:][::-1]
            for idx in top_indices:
                if idx < len(papers_df) and sims[idx] > 0.35:
                    row = papers_df.iloc[idx]
                    supporting_papers.append(str(row.get("title", ""))[:80])

        rows.append({
            "gap_statement": gap_text[:200],
            "theme": theme,
            "paper_count": pc,
            "evidence_score": round(ev_score, 3),
            "corpus_coverage": round(float(coverage), 3),
            "citation_poverty": round(cite_poverty, 3),
            "method_consensus_inverted": round(method_inverted, 3),
            "confidence_inverted": round(conf_inverted, 3),
            "gap_confidence_score": round(gap_confidence, 3),
            "verdict": verdict,
            "supporting_papers": "; ".join(supporting_papers)[:200],
        })

    return pd.DataFrame(rows)


def _generate_report(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Validated Research Gaps")
    lines.append(f"\n- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Gaps analysed:** {len(df)}")
    lines.append("")

    for verdict in ["Confirmed Gap", "Likely Gap", "Uncertain", "Likely False Gap"]:
        subset = df[df["verdict"] == verdict]
        if subset.empty:
            continue
        lines.append(f"## {verdict}s")
        lines.append("")
        for _, row in subset.iterrows():
            lines.append(f"### {row['theme']}")
            lines.append(f"- **Statement:** {row['gap_statement']}")
            lines.append(f"- **Gap Confidence:** {row['gap_confidence_score']:.2f}/1.0")
            lines.append(f"- **Evidence Score:** {row['evidence_score']}")
            lines.append(f"- **Corpus Coverage:** {row['corpus_coverage']:.1%}")
            if row["supporting_papers"]:
                lines.append(f"- **Top supporting papers:** {row['supporting_papers']}")
            lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    confirmed = df[df["verdict"] == "Confirmed Gap"]
    if not confirmed.empty:
        lines.append("### Pursue These Gaps")
        for _, row in confirmed.iterrows():
            lines.append(f"- **{row['theme']}** — score {row['gap_confidence_score']:.2f}")
    false_gaps = df[df["verdict"] == "Likely False Gap"]
    if not false_gaps.empty:
        lines.append("### Reconsider These Gaps (already addressed)")
        for _, row in false_gaps.iterrows():
            lines.append(f"- **{row['theme']}** — score {row['gap_confidence_score']:.2f}")
    lines.append("")
    lines.append("---\n*Generated by research_gap_validator.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate research gaps.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--knowledge-base", type=str, default="knowledge_base.json")
    parser.add_argument("--consensus-metadata", type=str, default="consensus_metadata.json")
    parser.add_argument("--evidence", type=str, default="outputs/reports/evidence_strength.csv")
    parser.add_argument("--gaps", type=str, default="outputs/reports/research_gaps.md")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    papers_df = pd.read_csv(args.papers) if Path(args.papers).exists() else pd.DataFrame()
    kb = load_knowledge_base(args.knowledge_base)
    evidence = load_evidence_scores(args.evidence)
    consensus_md = load_consensus_metadata(args.consensus_metadata)
    gaps_md = Path(args.gaps).read_text() if Path(args.gaps).exists() else ""

    gaps = extract_gap_statements(gaps_md, kb, evidence)
    if not gaps:
        log.warning("No gaps found to validate")
        return

    log.info("Validating %d gap claims ...", len(gaps))
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    df = validate_gaps(gaps, papers_df, evidence, consensus_md, model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gap_confidence_scores.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %d -> %s", len(df), csv_path)

    report = _generate_report(df)
    md_path = out_dir / "validated_research_gaps.md"
    with open(md_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", md_path)

    print(f"\n--- Research Gap Validation Complete ---")
    print(f"  Gaps analysed: {len(df)}")
    for v in ["Confirmed Gap", "Likely Gap", "Uncertain", "Likely False Gap"]:
        c = len(df[df["verdict"] == v])
        if c:
            print(f"    {v}: {c}")
    print()


if __name__ == "__main__":
    main()
