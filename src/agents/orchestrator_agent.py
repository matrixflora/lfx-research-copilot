from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .planner_agent import PlannerAgent
from .router_agent import RouterAgent
from .memory_agent import MemoryAgent
from .researcher_agent import ResearcherAgent
from .reviewer_agent import ReviewerAgent
from .critic_agent import CriticAgent
from .dashboard_agent import DashboardAgent
from .agent_registry import AgentRegistry

log = logging.getLogger("orchestrator_agent")


class OrchestratorAgent:
    """Master controller for the agentic research workflow."""

    def __init__(self):
        self.planner = PlannerAgent()
        self.router = RouterAgent()
        self.memory = MemoryAgent()
        self.researcher = ResearcherAgent()
        self.reviewer = ReviewerAgent()
        self.critic = CriticAgent()
        self.dashboard = DashboardAgent(memory=self.memory)

        self.execution_plan: Optional[Dict] = None
        self.research_analysis: Optional[Dict] = None
        self.critic_evaluation: Optional[Dict] = None
        self.status: str = "idle"

    def run(self, user_query: str, corpus_size: int = 0, context: Optional[Dict] = None) -> Dict[str, Any]:
        log.info("Orchestrator starting for query: %s", user_query)
        self.status = "planning"

        plan = self.planner.create_execution_plan(user_query, context)
        self.execution_plan = plan
        log.info("Execution plan created with %d steps", len(plan.get("steps", [])))

        self.status = "routing"
        routed_steps = self.router.route(plan, corpus_size, user_query, context)
        log.info("Routing complete: %d steps selected", len(routed_steps))

        self.status = "researching"
        results = self.researcher.execute_plan(routed_steps)
        self.research_analysis = self.researcher.get_research_analysis()
        self.research_analysis["execution_plan"] = plan
        self.research_analysis["routed_steps"] = routed_steps
        self.research_analysis["query"] = user_query
        log.info("Research execution complete")

        self.status = "reviewing"
        self.reviewer.review(self.research_analysis)

        self.status = "evaluating"
        self.critic_evaluation = self.critic.evaluate(self.research_analysis)
        log.info("Critic evaluation: %s", self.critic_evaluation.get("scores", {}))

        if self.critic_evaluation.get("needs_repair"):
            self.status = "repairing"
            log.info("Quality threshold not met, triggering self-correction")
            self._trigger_repair()

        self.status = "updating_memory"
        try:
            self._update_memory()
        except Exception as e:
            log.warning("Memory update failed: %s", e)

        self.status = "dashboard"
        self.dashboard.generate()

        self.status = "completed"
        return self._build_final_output()

    def create_execution_plan(self, user_query: str) -> Dict:
        return self.planner.create_execution_plan(user_query)

    def assign_tasks(self, plan: Dict, corpus_size: int, query: str) -> List[Dict]:
        return self.router.route(plan, corpus_size, query)

    def monitor_execution(self) -> Dict:
        return self.researcher.synthesize_results()

    def aggregate_results(self) -> Dict:
        return self.research_analysis or {}

    def trigger_repair(self) -> None:
        self._trigger_repair()

    def _trigger_repair(self) -> None:
        low_areas = self.critic_evaluation.get("low_scoring_areas", [])
        retry_modules = self._map_low_scores_to_modules(low_areas)
        log.info("Re-running modules for low-scoring areas: %s", retry_modules)

        for module in retry_modules:
            self.researcher.execute_module(module)

        self.research_analysis = self.researcher.get_research_analysis()
        self.critic_evaluation = self.critic.evaluate(self.research_analysis)

        if self.critic_evaluation.get("needs_repair"):
            log.warning("Quality still below threshold after repair")

    def _map_low_scores_to_modules(self, low_areas: List[str]) -> List[str]:
        mapping = {
            "report_quality": ["research_brief", "research_dashboard"],
            "citation_quality": ["citation_intelligence", "citation_network_analysis"],
            "gap_quality": ["research_gap_validator"],
            "theme_quality": ["cluster_themes", "theme_evolution"],
        }
        modules = []
        for area in low_areas:
            modules.extend(mapping.get(area, []))
        return modules

    def _update_memory(self) -> None:
        analysis = self.research_analysis
        query = analysis.get("query", "")

        search_csv = Path("search_results.csv")
        if search_csv.exists():
            import pandas as pd
            df = pd.read_csv(search_csv)
            self.memory.add_search(query, len(df))

        kb = analysis.get("knowledge_base", {})
        themes = kb.get("themes", [])
        if themes:
            self.memory.add_themes(themes)

        gaps_path = Path("research_gaps.md")
        if gaps_path.exists():
            content = gaps_path.read_text()
            gap_lines = [l for l in content.split("\n") if l.startswith("###")]
            for g in gap_lines:
                self.memory.add_gaps([{"description": g.replace("###", "").strip()}])

        hyp_path = Path("outputs/reports/generated_hypotheses.json")
        if hyp_path.exists():
            try:
                hyps = json.loads(hyp_path.read_text())
                if isinstance(hyps, list):
                    self.memory.add_hypotheses([h.get("hypothesis", str(h)) for h in hyps])
            except Exception:
                pass

    def _build_final_output(self) -> Dict[str, Any]:
        return {
            "query": self.research_analysis.get("query", ""),
            "status": self.status,
            "execution_plan": self.execution_plan,
            "critic_evaluation": self.critic_evaluation,
            "research_analysis": self.research_analysis,
            "timestamp": datetime.now().isoformat(),
        }
