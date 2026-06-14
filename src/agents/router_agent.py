from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("router_agent")


class RouterAgent:
    """Selects and sequences modules dynamically based on corpus and query."""

    def __init__(self):
        self.routing_log: List[Dict] = []

    def route(self, plan: Dict, corpus_size: int, query: str, context: Optional[Dict] = None) -> List[Dict]:
        log.info("Routing plan for corpus of %d papers", corpus_size)
        steps = plan.get("steps", [])
        query_lower = query.lower()

        is_bioinformatics = any(
            kw in query_lower
            for kw in ["genom", "proteom", "bioinformat", "pathway", "sequenc", "gene", "protein"]
        )
        is_large = corpus_size > 500
        is_small = corpus_size < 50

        routed = []
        for step in steps:
            module = step.get("module", "")
            task = step.get("task", "")

            if is_small and module == "citation_network_analysis":
                log.info("Skipping %s for small corpus", module)
                continue

            if is_large and module == "cluster_themes":
                routed.append({**step, "module": "citation_network_analysis", "note": "large corpus: using citation analysis instead of clustering"})
                continue

            if is_bioinformatics and module == "bioinformatics_mode":
                routed.append({**step, "bioinformatics": True, "note": "bioinformatics mode activated"})
                continue

            if not is_bioinformatics and module == "bioinformatics_mode":
                continue

            routed.append(step)

        self._log_routing(routed, corpus_size, is_bioinformatics)
        return routed

    def _log_routing(self, steps: List, corpus_size: int, is_bio: bool) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "corpus_size": corpus_size,
            "bioinformatics_mode": is_bio,
            "modules_selected": [s.get("module") for s in steps],
            "total_steps": len(steps),
        }
        self.routing_log.append(entry)
        path = Path("routing_log.json")
        with open(path, "w") as f:
            json.dump(self.routing_log, f, indent=2)
        log.info("Routing log -> %s", path)

    def select_modules_for_corpus(self, corpus_size: int) -> List[str]:
        if corpus_size < 50:
            return [
                "search_papers", "cluster_themes", "evidence_synthesis",
                "research_gap_validator", "hypothesis_generator",
                "opportunity_ranking", "research_brief",
            ]
        elif corpus_size > 500:
            return [
                "search_papers", "citation_intelligence", "trend_forecasting",
                "citation_network_analysis", "evidence_synthesis",
                "meta_analysis_readiness", "contradiction_detector",
                "evidence_strength", "research_gap_validator",
                "hypothesis_generator", "research_brief",
            ]
        else:
            return [
                "search_papers", "cluster_themes", "theme_evolution",
                "evidence_synthesis", "contradiction_detector",
                "evidence_strength", "research_gap_validator",
                "hypothesis_generator", "opportunity_ranking",
                "research_novelty_scorer", "research_brief",
            ]
