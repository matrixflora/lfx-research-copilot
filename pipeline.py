#!/usr/bin/env python3
"""
pipeline.py — Research Copilot Advanced Analytics Pipeline.

Orchestrates all modules in the defined execution order. Each module is
invoked as a subprocess call using the system Python interpreter.

Usage
-----
    python pipeline.py [--skip paper_retrieval]
                       [--until hypothesis_generator]
                       [--quick]           # run priority modules only
                       [--life-science]    # auto-enable 7 life-science modules

Execution Order
---------------
paper_retrieval → pdf_manager → evidence_extraction → figure_table_extractor
→ evidence_synthesis → theme_discovery → theme_evolution_analysis
→ author_intelligence → journal_intelligence → methodology_mining
→ contradiction_detector → evidence_strength_scoring → research_gap_validation
→ study_design_advisor → statistical_consultant → bioinformatics_mode
→ meta_analysis_readiness → novelty_scoring → citation_network_analysis
→ hypothesis_generator → opportunity_ranking → funding_alignment
→ research_roadmap → systematic_review → citation_intelligence
→ knowledge_base_update → question_optimizer → dataset_discovery
→ grant_proposal_copilot → manuscript_copilot → reviewer_simulation
→ reproducibility_audit → protocol_generator → claim_support
→ claim_graph_generation → semantic_alert_system → explainability_engine
→ living_knowledge_base → project_manager → research_brief_generation
→ research_dashboard → research_memory_update
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asynchronously)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")

PIPELINE = [
    ("paper_retrieval",             "search_papers.py"),
    ("pdf_manager",                 "pdf_manager.py"),
    ("evidence_extraction",         "full_text_evidence_extraction.py"),
    ("figure_table_extraction",     "figure_table_extractor.py"),
    ("evidence_synthesis",          "evidence_synthesis.py"),
    ("theme_discovery",             "cluster_themes.py"),
    ("theme_evolution_analysis",    "theme_evolution.py"),
    ("systematic_review",           "systematic_review.py"),
    ("citation_intelligence",       "citation_intelligence.py"),
    ("author_intelligence",         "author_intelligence.py"),
    ("journal_intelligence",        "journal_intelligence.py"),
    ("methodology_mining",          "methodology_mining.py"),
    ("contradiction_detector",      "contradiction_detector.py"),
    ("evidence_strength_scoring",   "evidence_strength.py"),
    ("research_gap_validation",     "research_gap_validator.py"),
    ("study_design_advisor",        "study_design_advisor.py"),
    ("statistical_consultant",      "statistical_consultant.py"),
    ("bioinformatics_mode",         "bioinformatics_mode.py"),
    ("meta_analysis_readiness",     "meta_analysis_readiness.py"),
    ("novelty_scoring",             "research_novelty_scorer.py"),
    ("citation_network_analysis",   "citation_network_analysis.py"),
    ("hypothesis_generator",        "hypothesis_generator.py"),
    ("opportunity_ranking",         "opportunity_ranking.py"),
    ("funding_alignment",           "funding_alignment.py"),
    ("research_roadmap",            "research_roadmap.py"),
    ("question_optimizer",          "research_question_optimizer.py"),
    ("dataset_discovery",           "dataset_discovery.py"),
    ("knowledge_base_update",       "generate_reports.py"),
    ("grant_proposal_generation",   "grant_proposal_copilot.py"),
    ("manuscript_generation",       "manuscript_copilot.py"),
    ("reviewer_simulation",         "reviewer_simulator.py"),
    ("reproducibility_audit",       "reproducibility_auditor.py"),
    ("protocol_generation",         "protocol_generator.py"),
    ("claim_support",               "support_claims_with_references.py"),
    ("claim_graph_generation",      "scientific_claim_graph.py"),
    ("semantic_alerts",             "semantic_alert_system.py"),
    ("explainability",              "explainability_engine.py"),
    ("living_knowledge_base",       "living_knowledge_base.py"),
    ("project_manager",             "project_manager.py"),
    ("research_brief_generation",   "research_brief.py"),
    ("research_dashboard",          "research_dashboard.py"),
    ("research_memory_update",      "research_memory.py"),
]

PRIORITY_MODULES = {
    "evidence_extraction",
    "evidence_synthesis",
    "theme_discovery",
    "theme_evolution_analysis",
    "contradiction_detector",
    "methodology_mining",
    "citation_intelligence",
    "evidence_strength_scoring",
    "research_gap_validation",
    "hypothesis_generator",
    "opportunity_ranking",
    "novelty_scoring",
    "claim_support",
    "claim_graph_generation",
    "research_brief_generation",
    "research_dashboard",
    "research_memory_update",
}

LIFE_SCIENCE_PRIORITY = {
    "evidence_extraction",
    "research_gap_validation",
    "study_design_advisor",
    "statistical_consultant",
    "bioinformatics_mode",
    "meta_analysis_readiness",
    "novelty_scoring",
    "dataset_discovery",
    "protocol_generation",
}

PYTHON = sys.executable


def _run_module(name: str, script: str) -> float:
    """Run a module script, return elapsed seconds."""
    if script is None:
        log.info("[%s] No standalone script.", name)
        return 0.0

    script_path = Path(script)
    if not script_path.exists():
        alt = Path(__file__).parent / script
        if alt.exists():
            script_path = alt
        else:
            log.warning("[%s] Script not found: %s — skipping.", name, script)
            return 0.0

    log.info("=" * 60)
    log.info("[%s] Starting...", name)
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            check=False,
            capture_output=False,
            timeout=600,
        )
        if result.returncode != 0:
            log.warning("[%s] Exited with code %d", name, result.returncode)
    except FileNotFoundError:
        log.warning("[%s] Python interpreter not found.", name)
    except subprocess.TimeoutExpired:
        log.warning("[%s] Timed out.", name)
    elapsed = time.time() - t0
    log.info("[%s] Finished in %.1fs", name, elapsed)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Copilot Advanced Analytics Pipeline")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated module names to skip")
    parser.add_argument("--until", type=str, default=None,
                        help="Run until (and including) this module, then stop")
    parser.add_argument("--quick", action="store_true",
                        help="Run priority modules only")
    parser.add_argument("--life-science", action="store_true",
                        help="Auto-enable life science / bioinformatics modules")
    parser.add_argument("--topic", type=str, default=None,
                        help="Search topic (passed to search_papers.py)")
    args = parser.parse_args()

    skip_set: set = set()
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(",")}

    if args.life_science:
        active_set: set = PRIORITY_MODULES | LIFE_SCIENCE_PRIORITY
    elif args.quick:
        active_set = PRIORITY_MODULES
    else:
        active_set = {name for name, _ in PIPELINE}

    total_time = 0.0
    completed: list = []

    for name, script in PIPELINE:
        if name in skip_set:
            log.info("[%s] Skipped (--skip).", name)
            completed.append((name, "skipped"))
            continue
        if name not in active_set:
            log.info("[%s] Skipped (not in active set).", name)
            completed.append((name, "skipped"))
            continue

        elapsed = _run_module(name, script)
        total_time += elapsed
        completed.append((name, "ok" if elapsed > 0 else "noop"))

        if args.until and name == args.until:
            log.info("Stopping after --until=%s", name)
            break

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for name, status in completed:
        print(f"  {name:<35s} {status}")
    print(f"\nTotal time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
