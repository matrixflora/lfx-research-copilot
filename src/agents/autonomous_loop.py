from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .orchestrator_agent import OrchestratorAgent
from .self_correction import SelfCorrection
from .critic_agent import CriticAgent
from .project_memory import ProjectMemory

log = logging.getLogger("autonomous_loop")


class AutonomousResearchLoop:
    """Continuously improves research until confidence target is met."""

    MAX_ITERATIONS = 5
    TARGET_CONFIDENCE = 0.85

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.self_correction = SelfCorrection()
        self.critic = CriticAgent()
        self.project_memory = ProjectMemory()
        self.iteration_log: List[Dict] = []

    def run(
        self,
        user_query: str,
        project_name: Optional[str] = None,
        corpus_size: int = 0,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        log.info("Starting autonomous research loop for: %s", user_query)

        project = self.project_memory.get_or_create(project_name or user_query[:50])
        project.update_research_question(user_query)

        iteration = 0
        best_result = None
        best_confidence = 0.0

        while iteration < self.MAX_ITERATIONS:
            log.info("=== Autonomous Loop Iteration %d/%d ===", iteration + 1, self.MAX_ITERATIONS)
            timestamp = datetime.now().isoformat()

            context = self._build_context(project)
            result = self.orchestrator.run(user_query, corpus_size, context)

            scores = result.get("critic_evaluation", {}).get("scores", {})
            confidence = scores.get("overall", 0.0)

            entry = {
                "iteration": iteration + 1,
                "confidence": confidence,
                "scores": scores,
                "timestamp": timestamp,
                "result_summary": self._summarize_result(result),
            }
            self.iteration_log.append(entry)
            log.info("Iteration %d confidence: %.2f (target: %.2f)", iteration + 1, confidence, self.TARGET_CONFIDENCE)

            if confidence > best_confidence:
                best_confidence = confidence
                best_result = result

            if confidence >= self.TARGET_CONFIDENCE:
                log.info("Target confidence reached at iteration %d", iteration + 1)
                break

            self._improve(iteration, scores)
            iteration += 1

        final = {
            "query": user_query,
            "project_name": project.name,
            "total_iterations": iteration + 1,
            "best_confidence": round(best_confidence, 2),
            "target_confidence": self.TARGET_CONFIDENCE,
            "target_met": best_confidence >= self.TARGET_CONFIDENCE,
            "iteration_log": self.iteration_log,
            "final_result": best_result,
            "timestamp": datetime.now().isoformat(),
        }

        project.update_status("completed" if final["target_met"] else "partial")
        project.save()
        self._save_final_report(final)

        return final

    def plan(self, user_query: str) -> Dict:
        return self.orchestrator.create_execution_plan(user_query)

    def execute(self, plan: Dict, corpus_size: int, query: str) -> Dict:
        return self.orchestrator.run(query, corpus_size)

    def evaluate(self, result: Dict) -> Dict:
        return self.critic.evaluate(result.get("research_analysis", {}))

    def improve(self, iteration: int, scores: Dict) -> None:
        self._improve(iteration, scores)

    def _improve(self, iteration: int, scores: Dict) -> None:
        low_areas = [
            name for name, score in scores.items()
            if name != "overall" and score < self.self_correction.critic.QUALITY_THRESHOLD
        ]
        if not low_areas:
            return

        log.info("Improvement needed in: %s", low_areas)
        analysis = self.orchestrator.research_analysis or {}
        problems = self.self_correction.detect_problems(analysis, {"low_scoring_areas": low_areas})
        replan = self.self_correction.replan(problems, analysis)
        self.self_correction.rerun_modules(replan)

    def _build_context(self, project: ProjectMemory) -> Dict:
        return {
            "project_name": project.name,
            "research_question": project.research_question,
            "papers_count": len(project.papers),
            "themes_count": len(project.themes),
            "gaps_count": len(project.gaps),
            "hypotheses_count": len(project.hypotheses),
        }

    def _summarize_result(self, result: Dict) -> Dict:
        analysis = result.get("research_analysis", {})
        summary = analysis.get("summary", {})
        return {
            "modules_executed": summary.get("modules_executed", 0),
            "successful": summary.get("successful", 0),
            "failed": summary.get("failed", 0),
            "total_time": summary.get("total_time", 0),
        }

    def _save_final_report(self, final: Dict) -> None:
        path = Path("autonomous_research_report.json")
        with open(path, "w") as f:
            json.dump(final, f, indent=2)
        log.info("Autonomous research report -> %s", path)

        md_path = Path("autonomous_research_report.md")
        lines = [
            "# Autonomous Research Report",
            "",
            f"**Query:** {final['query']}",
            f"**Project:** {final['project_name']}",
            f"**Total Iterations:** {final['total_iterations']}",
            f"**Best Confidence:** {final['best_confidence']:.2f}",
            f"**Target Confidence:** {final['target_confidence']}",
            f"**Target Met:** {'Yes' if final['target_met'] else 'No'}",
            "",
            "## Iteration Log",
            "",
        ]
        for entry in self.iteration_log:
            lines.append(f"### Iteration {entry['iteration']}")
            lines.append(f"- Confidence: {entry['confidence']:.2f}")
            lines.append(f"- Scores: {json.dumps(entry['scores'])}")
            lines.append("")
        md_path.write_text("\n".join(lines))
        log.info("Autonomous research report -> %s", md_path)
