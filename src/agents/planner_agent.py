from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qwen_adapter import QwenAdapter

log = logging.getLogger("planner_agent")


class PlannerAgent:
    """Converts user requests into structured subgoals."""

    def __init__(self, qwen_adapter: Optional[QwenAdapter] = None):
        self.qwen = qwen_adapter or QwenAdapter()

    def create_execution_plan(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        log.info("Creating execution plan for: %s", user_query)

        plan = self.qwen.generate_plan(user_query, json.dumps(context) if context else None)

        if not plan:
            plan = self._fallback_plan(user_query)

        execution_plan = {
            "query": user_query,
            "created_at": datetime.now().isoformat(),
            "steps": plan,
            "total_steps": len(plan),
            "status": "planned",
        }

        self._save_plan(execution_plan)
        return execution_plan

    def _fallback_plan(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        steps = []
        bio_keywords = ["genom", "proteom", "bioinformat", "pathway", "sequenc", "gene"]

        is_bio = any(kw in query_lower for kw in bio_keywords)

        steps.append({"task": "Retrieve papers", "module": "search_papers", "priority": 1})
        steps.append({"task": "Extract evidence from full text", "module": "full_text_evidence_extraction", "priority": 2})
        steps.append({"task": "Synthesize evidence across papers", "module": "evidence_synthesis", "priority": 3})

        if is_bio:
            steps.append({"task": "Run bioinformatics mode", "module": "bioinformatics_mode", "priority": 4})
            steps.append({"task": "Discover relevant datasets", "module": "dataset_discovery", "priority": 5})

        steps.append({"task": "Discover themes via clustering", "module": "cluster_themes", "priority": 4 if not is_bio else 6})
        steps.append({"task": "Analyze theme evolution over time", "module": "theme_evolution", "priority": 5 if not is_bio else 7})
        steps.append({"task": "Detect contradictions across studies", "module": "contradiction_detector", "priority": 6 if not is_bio else 8})
        steps.append({"task": "Score evidence strength", "module": "evidence_strength", "priority": 7 if not is_bio else 9})
        steps.append({"task": "Validate research gaps", "module": "research_gap_validator", "priority": 8 if not is_bio else 10})
        steps.append({"task": "Generate hypotheses", "module": "hypothesis_generator", "priority": 9 if not is_bio else 11})
        steps.append({"task": "Rank research opportunities", "module": "opportunity_ranking", "priority": 10 if not is_bio else 12})
        steps.append({"task": "Score research novelty", "module": "research_novelty_scorer", "priority": 11 if not is_bio else 13})
        steps.append({"task": "Generate research dashboard", "module": "research_dashboard", "priority": 12 if not is_bio else 14})
        steps.append({"task": "Generate research brief", "module": "research_brief", "priority": 13 if not is_bio else 15})

        return steps

    def _save_plan(self, plan: Dict) -> None:
        path = Path("execution_plan.json")
        with open(path, "w") as f:
            json.dump(plan, f, indent=2)
        log.info("Saved execution plan -> %s", path)

    def revise_plan(self, plan: Dict, feedback: Dict) -> Dict:
        log.info("Revising plan based on feedback")
        completed = feedback.get("completed_steps", [])
        failed = feedback.get("failed_steps", [])
        low_quality = feedback.get("low_quality_steps", [])

        plan["steps"] = [
            s for s in plan["steps"]
            if s["task"] not in completed or s["task"] in failed or s["task"] in low_quality
        ]
        plan["status"] = "revised"
        plan["revised_at"] = datetime.now().isoformat()
        self._save_plan(plan)
        return plan
