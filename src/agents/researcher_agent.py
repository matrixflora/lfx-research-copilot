from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("researcher_agent")


class ResearcherAgent:
    """Executes literature investigation by coordinating existing modules."""

    MODULE_MAP = {
        "search_papers": "search_papers.py",
        "cluster_themes": "cluster_themes.py",
        "evidence_synthesis": "evidence_synthesis.py",
        "contradiction_detector": "contradiction_detector.py",
        "evidence_strength": "evidence_strength.py",
        "research_gap_validator": "research_gap_validator.py",
        "hypothesis_generator": "hypothesis_generator.py",
        "opportunity_ranking": "opportunity_ranking.py",
        "research_novelty_scorer": "research_novelty_scorer.py",
        "citation_intelligence": "citation_intelligence.py",
        "citation_network_analysis": "citation_network_analysis.py",
        "theme_evolution": "theme_evolution.py",
        "research_brief": "research_brief.py",
        "research_dashboard": "research_dashboard.py",
        "bioinformatics_mode": "bioinformatics_mode.py",
        "dataset_discovery": "dataset_discovery.py",
        "full_text_evidence_extraction": "full_text_evidence_extraction.py",
        "evidence_strength_scoring": "evidence_strength.py",
        "research_gap_validation": "research_gap_validator.py",
        "novelty_scoring": "research_novelty_scorer.py",
        "question_optimizer": "research_question_optimizer.py",
        "knowledge_base_update": "generate_reports.py",
        "grant_proposal_generation": "grant_proposal_copilot.py",
        "manuscript_generation": "manuscript_copilot.py",
        "reviewer_simulation": "reviewer_simulator.py",
        "reproducibility_audit": "reproducibility_auditor.py",
        "protocol_generation": "protocol_generator.py",
        "claim_support": "support_claims_with_references.py",
        "claim_graph_generation": "scientific_claim_graph.py",
        "semantic_alerts": "semantic_alert_system.py",
        "explainability": "explainability_engine.py",
        "living_knowledge_base": "living_knowledge_base.py",
        "project_manager": "project_manager.py",
        "research_memory_update": "research_memory.py",
        "systematic_review": "systematic_review.py",
        "author_intelligence": "author_intelligence.py",
        "journal_intelligence": "journal_intelligence.py",
        "methodology_mining": "methodology_mining.py",
        "study_design_advisor": "study_design_advisor.py",
        "statistical_consultant": "statistical_consultant.py",
        "meta_analysis_readiness": "meta_analysis_readiness.py",
        "funding_alignment": "funding_alignment.py",
        "research_roadmap": "research_roadmap.py",
        "figure_table_extraction": "figure_table_extractor.py",
        "pdf_manager": "pdf_manager.py",
        "trend_forecasting": "theme_evolution.py",
    }

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.execution_log: List[Dict] = []

    def execute_module(self, module_name: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        script = self.MODULE_MAP.get(module_name)
        if not script:
            log.warning("Unknown module: %s", module_name)
            return {"module": module_name, "status": "unknown", "error": "No script mapped"}

        script_path = Path(script)
        if not script_path.exists():
            script_path = Path(__file__).parent.parent.parent / script

        if not script_path.exists():
            log.warning("Script not found: %s", script)
            return {"module": module_name, "status": "not_found", "error": str(script_path)}

        log.info("Executing module: %s (%s)", module_name, script_path)
        t0 = datetime.now()
        try:
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)
            result = subprocess.run(
                cmd, check=False, capture_output=True, timeout=600
            )
            elapsed = (datetime.now() - t0).total_seconds()
            status = "success" if result.returncode == 0 else "failed"
            entry = {
                "module": module_name,
                "script": str(script_path),
                "status": status,
                "returncode": result.returncode,
                "elapsed_seconds": round(elapsed, 2),
                "stdout": result.stdout.decode("utf-8", errors="replace")[-1000:],
                "stderr": result.stderr.decode("utf-8", errors="replace")[-1000:],
            }
            self.execution_log.append(entry)
            self.results[module_name] = entry
            log.info("Module %s completed with status=%s in %.1fs", module_name, status, elapsed)
            return entry
        except subprocess.TimeoutExpired:
            entry = {"module": module_name, "status": "timeout", "error": "Exceeded 600s"}
            self.execution_log.append(entry)
            self.results[module_name] = entry
            return entry
        except Exception as e:
            entry = {"module": module_name, "status": "error", "error": str(e)}
            self.execution_log.append(entry)
            self.results[module_name] = entry
            return entry

    def execute_plan(self, steps: List[Dict]) -> Dict[str, Any]:
        for step in steps:
            module = step.get("module")
            if module:
                self.execute_module(module)
        return self.results

    def synthesize_results(self) -> Dict[str, Any]:
        return {
            "modules_executed": len(self.execution_log),
            "successful": sum(1 for e in self.execution_log if e.get("status") == "success"),
            "failed": sum(1 for e in self.execution_log if e.get("status") == "failed"),
            "total_time": round(sum(e.get("elapsed_seconds", 0) for e in self.execution_log), 2),
            "execution_log": self.execution_log,
            "analysis_ready": self._check_outputs_exist(),
        }

    def _check_outputs_exist(self) -> Dict[str, bool]:
        outputs = [
            "search_results.csv",
            "consensus_themes.csv",
            "theme_analysis_report.md",
            "knowledge_base.json",
            "research_gaps.md",
            "outputs/reports/research_brief.md",
            "outputs/dashboard/research_dashboard.md",
        ]
        return {p: Path(p).exists() for p in outputs}

    def get_research_analysis(self) -> Dict[str, Any]:
        analysis = {"summary": self.synthesize_results()}

        for fname in ["consensus_metadata.json", "search_results.json", "knowledge_base.json"]:
            p = Path(fname)
            if p.exists():
                try:
                    analysis[fname.replace(".json", "")] = json.loads(p.read_text())
                except Exception:
                    pass

        return analysis
