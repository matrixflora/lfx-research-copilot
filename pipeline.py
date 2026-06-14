#!/usr/bin/env python3
"""
pipeline.py — LFX Research Copilot Agentic Pipeline.

Usage:
    python3 pipeline.py --agentic --query "<query>"
    python3 pipeline.py --agentic --query "<query>" --auto-loop
    python3 pipeline.py --agentic --query "<query>" --life-science
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.router_agent import RouterAgent
from src.agents.critic_agent import CriticAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.dashboard_agent import DashboardAgent
from src.agents.autonomous_loop import AutonomousResearchLoop


def ensure_directories() -> None:
    dirs = [
        "outputs/reports",
        "outputs/dashboard",
        "outputs/agent_logs",
        "outputs/themes",
        "outputs/knowledge_base",
        "outputs/projects",
        "projects",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def build_context(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "life_science": args.life_science,
        "quick": args.quick,
    }


def estimate_corpus_size() -> int:
    csv_path = Path("search_results.csv")
    if not csv_path.exists():
        return 0
    try:
        import pandas as pd
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def save_execution_log(
    query: str,
    confidence: float,
    modules: List[str],
    status: str,
    error: Optional[str] = None,
) -> None:
    log_entry = {
        "query": query,
        "confidence_score": confidence,
        "modules_executed": modules,
        "status": status,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    log_path = Path("outputs/agent_logs/execution_log.json")
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(log_entry)
    log_path.write_text(json.dumps(existing, indent=2))
    log.info("Execution log saved -> %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LFX Research Copilot — Agentic AI Research Pipeline"
    )
    parser.add_argument("--query", type=str, required=True,
                        help="Research query to investigate")
    parser.add_argument("--agentic", action="store_true",
                        help="Enable agentic AI research workflow")
    parser.add_argument("--auto-loop", action="store_true",
                        help="Enable autonomous iterative improvement loop")
    parser.add_argument("--life-science", action="store_true",
                        help="Enable life science / bioinformatics modules")
    parser.add_argument("--quick", action="store_true",
                        help="Run priority modules only")

    args = parser.parse_args()

    ensure_directories()
    query = args.query.strip()
    log.info("Query received: %s", query)

    modules_executed: List[str] = []
    final_confidence: float = 0.0
    status: str = "failed"
    error_message: Optional[str] = None

    try:
        log.info("Step 2: Initializing Memory Agent")
        memory_agent = MemoryAgent()

        log.info("Step 3: Loading previous memory if available")
        previous_memory = memory_agent.retrieve_memory()
        log.info("Previous memory loaded: %d searches, %d themes",
                 len(previous_memory.get("searches", [])),
                 len(previous_memory.get("themes", [])))

        log.info("Step 4: Initializing Orchestrator Agent")
        orchestrator = OrchestratorAgent()

        log.info("Step 5: Orchestrator invoking Router Agent")
        router = RouterAgent()
        corpus_size = estimate_corpus_size()
        log.info("Estimated corpus size: %d papers", corpus_size)

        log.info("Step 6: Router determining modules to execute")
        selected_modules = router.select_modules_for_corpus(corpus_size)

        if args.life_science:
            bio_modules = [
                "bioinformatics_mode", "dataset_discovery",
                "study_design_advisor", "statistical_consultant",
                "meta_analysis_readiness",
            ]
            for m in bio_modules:
                if m not in selected_modules:
                    selected_modules.append(m)

        if args.quick:
            quick_modules = {
                "search_papers", "cluster_themes", "evidence_synthesis",
                "contradiction_detector", "evidence_strength",
                "research_gap_validator", "hypothesis_generator",
                "opportunity_ranking", "research_brief", "research_dashboard",
            }
            selected_modules = [m for m in selected_modules if m in quick_modules]

        log.info("Modules selected: %s", selected_modules)
        modules_executed = selected_modules

        if args.auto_loop:
            log.info("Step 7: Running autonomous research loop")
            loop = AutonomousResearchLoop()
            loop_result = loop.run(
                user_query=query,
                corpus_size=corpus_size,
                context=build_context(args),
            )
            final_confidence = loop_result.get("best_confidence", 0.0)
            log.info("Auto-loop completed. Confidence: %.2f", final_confidence)
        else:
            log.info("Step 7: Executing selected modules via Orchestrator")
            orchestrator_result = orchestrator.run(
                user_query=query,
                corpus_size=corpus_size,
                context=build_context(args),
            )
            final_confidence = (
                orchestrator_result
                .get("critic_evaluation", {})
                .get("scores", {})
                .get("overall", 0.0)
            )

            log.info("Step 8: Reviewer Agent evaluating outputs")
            analysis = orchestrator.research_analysis or {}
            reviewer = ReviewerAgent()
            reviewer.review(analysis)

            log.info("Step 9: Critic Agent assigning confidence score")
            critic = CriticAgent()
            evaluation = critic.evaluate(analysis)
            final_confidence = evaluation.get("scores", {}).get("overall", 0.0)
            log.info("Confidence score: %.2f", final_confidence)

            log.info("Step 10: Checking confidence threshold")
            if final_confidence < 0.85 and args.auto_loop:
                log.info("Confidence %.2f < 0.85. Entering improvement loop.", final_confidence)
                max_iterations = 5
                for iteration in range(max_iterations):
                    log.info("Improvement iteration %d/%d", iteration + 1, max_iterations)
                    low_areas = evaluation.get("low_scoring_areas", [])
                    if not low_areas:
                        log.info("No low-scoring areas remaining.")
                        break
                    rerun_modules = []
                    for area in low_areas:
                        if area == "report_quality":
                            rerun_modules.extend(["research_brief", "research_dashboard"])
                        elif area == "citation_quality":
                            rerun_modules.extend(["citation_intelligence"])
                        elif area == "gap_quality":
                            rerun_modules.append("research_gap_validator")
                        elif area == "theme_quality":
                            rerun_modules.extend(["cluster_themes", "theme_evolution"])
                    log.info("Re-running modules: %s", rerun_modules)
                    for mod in rerun_modules:
                        orchestrator.researcher.execute_module(mod)
                    analysis = orchestrator.researcher.get_research_analysis()
                    evaluation = critic.evaluate(analysis)
                    final_confidence = evaluation.get("scores", {}).get("overall", 0.0)
                    log.info("Iteration %d confidence: %.2f", iteration + 1, final_confidence)
                    if final_confidence >= 0.85:
                        log.info("Target confidence reached.")
                        break

        log.info("Step 11: Updating project memory")
        memory_agent.add_search(query, estimate_corpus_size())
        kb_path = Path("knowledge_base.json")
        if kb_path.exists():
            try:
                kb = json.loads(kb_path.read_text())
                themes = kb.get("themes", [])
                if themes:
                    memory_agent.add_themes(themes)
            except Exception:
                pass

        log.info("Step 12: Generating outputs")
        try:
            brief_path = Path("outputs/reports/research_brief.md")
            existing_brief = Path("research_brief.md")
            if existing_brief.exists():
                brief_path.write_text(existing_brief.read_text())
                log.info("Research brief -> %s", brief_path)
            else:
                brief_path.write_text(f"# Research Brief\n\nQuery: {query}\n\nConfidence: {final_confidence:.2f}\n")
                log.info("Research brief (minimal) -> %s", brief_path)
        except Exception as e:
            log.warning("Could not generate research brief: %s", e)

        try:
            dashboard = DashboardAgent(memory=memory_agent)
            dashboard.generate()
            dash_src = Path("outputs/dashboard/dashboard.md")
            dash_dst = Path("outputs/dashboard/dashboard.md")
            dash_src.parent.mkdir(parents=True, exist_ok=True)
            if dash_src.exists():
                dash_dst.write_text(dash_src.read_text())
            log.info("Dashboard -> outputs/dashboard/dashboard.md")
        except Exception as e:
            log.warning("Could not generate dashboard: %s", e)

        try:
            exec_log_path = Path("outputs/agent_logs/execution_log.json")
            exec_log_path.parent.mkdir(parents=True, exist_ok=True)
            save_execution_log(query, final_confidence, modules_executed, "completed")
        except Exception as e:
            log.warning("Could not save execution log: %s", e)

        status = "completed"

    except Exception:
        error_message = traceback.format_exc()
        log.error("Pipeline failed:\n%s", error_message)
        status = "failed"
        save_execution_log(query, final_confidence, modules_executed, status, error_message)

    print()
    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Query:              {query}")
    print(f"  Status:             {status}")
    print(f"  Confidence Score:   {final_confidence:.2f}")
    print(f"  Modules Executed:   {len(modules_executed)}")
    for m in modules_executed:
        print(f"    - {m}")
    print(f"  Output Locations:")
    print(f"    Research Brief:    outputs/reports/research_brief.md")
    print(f"    Dashboard:         outputs/dashboard/dashboard.md")
    print(f"    Execution Log:     outputs/agent_logs/execution_log.json")
    print()
    log.info("Pipeline execution %s.", status)

    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
