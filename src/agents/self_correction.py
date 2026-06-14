from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .critic_agent import CriticAgent
from .researcher_agent import ResearcherAgent
from .planner_agent import PlannerAgent
from .qwen_adapter import QwenAdapter

log = logging.getLogger("self_correction")


class SelfCorrection:
    """Autonomous self-correction loop that detects problems and re-runs modules."""

    MAX_RETRIES = 3

    def __init__(self, qwen_adapter: Optional[QwenAdapter] = None):
        self.critic = CriticAgent(qwen_adapter)
        self.researcher = ResearcherAgent()
        self.planner = PlannerAgent(qwen_adapter)
        self.qwen = qwen_adapter or QwenAdapter()
        self.log: List[Dict] = []

    def execute(self, research_analysis: Dict) -> Dict[str, Any]:
        log.info("Starting self-correction loop")
        current_analysis = research_analysis
        retry_count = 0

        while retry_count < self.MAX_RETRIES:
            evaluation = self.critic.evaluate(current_analysis)
            scores = evaluation.get("scores", {})
            overall = scores.get("overall", 0)

            entry = {
                "iteration": retry_count + 1,
                "scores": scores,
                "needs_repair": evaluation.get("needs_repair", False),
                "low_scoring_areas": evaluation.get("low_scoring_areas", []),
                "timestamp": datetime.now().isoformat(),
            }
            self.log.append(entry)
            log.info(
                "Iteration %d: overall=%.2f, needs_repair=%s",
                retry_count + 1, overall, entry["needs_repair"],
            )

            if not evaluation.get("needs_repair"):
                log.info("All quality thresholds met after %d iterations", retry_count + 1)
                break

            problems = self._detect_problems(current_analysis, evaluation)
            replan = self._replan(problems, current_analysis)
            self._rerun_modules(replan)
            current_analysis = self.researcher.get_research_analysis()
            retry_count += 1

        if retry_count >= self.MAX_RETRIES:
            log.warning("Max retries (%d) reached. Publishing best available result.", self.MAX_RETRIES)

        result = {
            "status": "corrected" if retry_count < self.MAX_RETRIES else "max_retries_reached",
            "iterations": retry_count + 1,
            "correction_log": self.log,
            "final_evaluation": self.critic.evaluate(current_analysis),
        }

        self._save_log(result)
        return result

    def evaluate(self, analysis: Dict) -> Dict:
        return self.critic.evaluate(analysis)

    def detect_problems(self, analysis: Dict, evaluation: Dict) -> List[str]:
        return self._detect_problems(analysis, evaluation)

    def replan(self, problems: List[str], analysis: Dict) -> List[Dict]:
        return self._replan(problems, analysis)

    def rerun_modules(self, plan: List[Dict]) -> None:
        self._rerun_modules(plan)

    def _detect_problems(self, analysis: Dict, evaluation: Dict) -> List[str]:
        problems = []
        low_areas = evaluation.get("low_scoring_areas", [])

        for area in low_areas:
            if area == "report_quality":
                problems.append("Research brief or dashboard missing or incomplete")
            elif area == "citation_quality":
                problems.append("Insufficient citations or paper coverage")
            elif area == "gap_quality":
                problems.append("Research gaps not properly identified")
            elif area == "theme_quality":
                problems.append("Theme clustering quality below threshold")

        return problems

    def _replan(self, problems: List[str], analysis: Dict) -> List[Dict]:
        modules_to_rerun = []
        for problem in problems:
            if "brief" in problem.lower() or "dashboard" in problem.lower():
                modules_to_rerun.append("research_brief")
                modules_to_rerun.append("research_dashboard")
            elif "citation" in problem.lower() or "paper" in problem.lower():
                modules_to_rerun.append("citation_intelligence")
                modules_to_rerun.append("citation_network_analysis")
            elif "gap" in problem.lower():
                modules_to_rerun.append("research_gap_validator")
                modules_to_rerun.append("hypothesis_generator")
            elif "theme" in problem.lower() or "cluster" in problem.lower():
                modules_to_rerun.append("cluster_themes")
                modules_to_rerun.append("theme_evolution")

        return [{"module": m, "task": f"Re-run {m}", "priority": 1} for m in dict.fromkeys(modules_to_rerun)]

    def _rerun_modules(self, plan: List[Dict]) -> None:
        for step in plan:
            module = step.get("module")
            if module:
                log.info("Self-correction re-running module: %s", module)
                self.researcher.execute_module(module)

    def _save_log(self, result: Dict) -> None:
        path = Path("self_correction_log.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Self-correction log -> %s", path)
