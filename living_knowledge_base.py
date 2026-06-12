#!/usr/bin/env python3
"""
living_knowledge_base.py — Incrementally update the knowledge base with new
themes, trends, citations, gaps, author data, journal data, methods, and
contradictions. Merges fresh analysis outputs into a single unified snapshot.

Usage
-----
    python living_knowledge_base.py
        [--knowledge-base outputs/knowledge_base/knowledge_base.json]
        [--themes outputs/themes/clustering_report.json]
        [--consensus consensus_themes.csv]
        [--evolution outputs/reports/theme_evolution.json]
        [--gaps outputs/reports/research_gaps.md]
        [--citations outputs/reports/citation_intelligence.json]
        [--hypotheses outputs/reports/hypothesis_bank.csv]
        [--authors outputs/reports/author_intelligence.csv]
        [--journals outputs/reports/journal_intelligence.csv]
        [--methods outputs/reports/methodology_profile.csv]
        [--contradictions outputs/reports/contradictory_findings.md]
        [--opportunities outputs/reports/research_opportunities.csv]
        [--funding outputs/reports/funding_alignment.csv]
        [--roadmap outputs/reports/research_roadmap.md]
        [--output outputs/knowledge_base/living_knowledge_base.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("living_knowledge_base")


def _try_json(path: Path) -> Dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not load %s: %s", path, e)
    return {}


def _try_md(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def _try_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.warning("Could not load %s: %s", path, e)
    return pd.DataFrame()


def merge_all(args: argparse.Namespace) -> Dict[str, Any]:
    kb = _try_json(Path(args.knowledge_base))
    themes = _try_json(Path(args.themes))
    evolution = _try_json(Path(args.evolution))
    gaps_md = _try_md(Path(args.gaps))
    contradictions_md = _try_md(Path(args.contradictions))
    roadmap_md = _try_md(Path(args.roadmap))

    citations = _try_json(Path(args.citations))
    hypotheses = _try_csv(Path(args.hypotheses))
    authors = _try_csv(Path(args.authors))
    journals = _try_csv(Path(args.journals))
    methods = _try_csv(Path(args.methods))
    opportunities = _try_csv(Path(args.opportunities))
    funding = _try_csv(Path(args.funding))

    consensus_path = Path(args.consensus)
    if consensus_path.exists():
        df_c = pd.read_csv(consensus_path)
    else:
        df_c = pd.DataFrame()

    snapshot: Dict[str, Any] = {
        "_meta": {
            "updated": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Living knowledge base — merged from all pipeline outputs",
        },
        "knowledge_base_summary": {
            "theme_count": len(kb.get("themes", [])),
            "paper_count": kb.get("total_papers", df_c["doi"].nunique() if "doi" in df_c else 0),
        },
        "themes": kb.get("themes", []),
        "theme_classifications": {
            k: v for k, v in kb.items() if k.endswith("_themes")
        },
        "clustering_report": themes,
        "theme_evolution": evolution,
        "citation_intelligence": {
            "paper_count": len(citations.get("papers", [])),
            "foundational": citations.get("foundational_papers", []),
            "influential_authors": citations.get("influential_authors", []),
            "bottlenecks": citations.get("bottlenecks", []),
            "neglected_papers": citations.get("neglected_papers", []),
            "clusters": citations.get("clusters", []),
            "graph_summary": citations.get("graph_summary", {}),
        },
        "hypothesis_bank": {
            "count": len(hypotheses),
            "hypotheses": hypotheses.to_dict(orient="records") if not hypotheses.empty else [],
        },
        "author_intelligence": {
            "count": len(authors),
            "data": authors.to_dict(orient="records") if not authors.empty else [],
        },
        "journal_intelligence": {
            "count": len(journals),
            "data": journals.to_dict(orient="records") if not journals.empty else [],
        },
        "methodology_profile": {
            "count": len(methods),
            "data": methods.to_dict(orient="records") if not methods.empty else [],
        },
        "research_gaps": gaps_md[:2000] if gaps_md else "",
        "contradictions": contradictions_md[:2000] if contradictions_md else "",
        "research_roadmap": roadmap_md[:2000] if roadmap_md else "",
        "opportunity_ranking": {
            "count": len(opportunities),
            "data": opportunities.to_dict(orient="records") if not opportunities.empty else [],
        },
        "funding_alignment": {
            "count": len(funding),
            "data": funding.to_dict(orient="records") if not funding.empty else [],
        },
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pipeline outputs into living knowledge base.")
    parser.add_argument("--knowledge-base", type=str, default="outputs/knowledge_base/knowledge_base.json")
    parser.add_argument("--themes", type=str, default="outputs/themes/clustering_report.json")
    parser.add_argument("--consensus", type=str, default="consensus_themes.csv")
    parser.add_argument("--evolution", type=str, default="outputs/reports/theme_evolution.json")
    parser.add_argument("--gaps", type=str, default="outputs/reports/research_gaps.md")
    parser.add_argument("--citations", type=str, default="outputs/reports/citation_intelligence.json")
    parser.add_argument("--hypotheses", type=str, default="outputs/reports/hypothesis_bank.csv")
    parser.add_argument("--authors", type=str, default="outputs/reports/author_intelligence.csv")
    parser.add_argument("--journals", type=str, default="outputs/reports/journal_intelligence.csv")
    parser.add_argument("--methods", type=str, default="outputs/reports/methodology_profile.csv")
    parser.add_argument("--contradictions", type=str, default="outputs/reports/contradictory_findings.md")
    parser.add_argument("--opportunities", type=str, default="outputs/reports/research_opportunities.csv")
    parser.add_argument("--funding", type=str, default="outputs/reports/funding_alignment.csv")
    parser.add_argument("--roadmap", type=str, default="outputs/reports/research_roadmap.md")
    parser.add_argument("--output", type=str, default="outputs/knowledge_base/living_knowledge_base.json")
    args = parser.parse_args()

    snapshot = merge_all(args)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    log.info("Saved living knowledge base -> %s", args.output)

    sections = [k for k in snapshot if k != "_meta"]
    print(f"\n--- Living Knowledge Base Complete ---")
    print(f"  Sections: {len(sections)}")
    print()


if __name__ == "__main__":
    main()
