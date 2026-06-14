from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qwen_adapter import QwenAdapter

log = logging.getLogger("critic_agent")


class CriticAgent:
    """Self-evaluates research outputs and requests repairs when quality is low."""

    QUALITY_THRESHOLD = 0.6

    def __init__(self, qwen_adapter: Optional[QwenAdapter] = None):
        self.qwen = qwen_adapter or QwenAdapter()

    def evaluate(self, research_analysis: Dict) -> Dict[str, Any]:
        log.info("Critic evaluating research outputs")
        scores = {}

        scores["report_quality"] = self._score_report_quality(research_analysis)
        scores["citation_quality"] = self._score_citation_quality(research_analysis)
        scores["gap_quality"] = self._score_gap_quality(research_analysis)
        scores["theme_quality"] = self._score_theme_quality(research_analysis)

        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 2)

        needs_repair = any(v < self.QUALITY_THRESHOLD for v in scores.values())

        evaluation = {
            "scores": scores,
            "threshold": self.QUALITY_THRESHOLD,
            "needs_repair": needs_repair,
            "low_scoring_areas": [
                name for name, score in scores.items()
                if score < self.QUALITY_THRESHOLD
            ],
            "timestamp": datetime.now().isoformat(),
        }

        self._save_evaluation(evaluation)
        log.info("Critic evaluation: overall=%.2f, needs_repair=%s", overall, needs_repair)
        return evaluation

    def _score_report_quality(self, analysis: Dict) -> float:
        score = 0.7
        summary = analysis.get("summary", {})
        outputs = summary.get("analysis_ready", {})
        report_path = Path("outputs/reports/research_brief.md")

        if not report_path.exists():
            score -= 0.3
        if outputs.get("search_results.csv"):
            score += 0.1
        if outputs.get("consensus_themes.csv"):
            score += 0.1

        return round(min(max(score, 0.0), 1.0), 2)

    def _score_citation_quality(self, analysis: Dict) -> float:
        score = 0.5
        kb = analysis.get("knowledge_base", {})
        themes = kb.get("themes", [])
        if themes:
            total_papers = sum(len(t.get("papers", [])) for t in themes)
            score += min(total_papers / 50, 0.3)
            if total_papers > 0:
                score += 0.1

        return round(min(max(score, 0.0), 1.0), 2)

    def _score_gap_quality(self, analysis: Dict) -> float:
        score = 0.5
        gaps_path = Path("research_gaps.md")
        if gaps_path.exists():
            content = gaps_path.read_text()
            gap_count = content.count("###")
            score += min(gap_count / 10, 0.3)
            if gap_count > 0:
                score += 0.1

        return round(min(max(score, 0.0), 1.0), 2)

    def _score_theme_quality(self, analysis: Dict) -> float:
        score = 0.5
        consensus = analysis.get("consensus_metadata", [])
        if consensus:
            avg_confidence = sum(t.get("confidence", 0) for t in consensus) / len(consensus)
            score += avg_confidence * 0.3
            non_exploratory = sum(1 for t in consensus if not t.get("is_exploratory", False))
            score += (non_exploratory / len(consensus)) * 0.2

        return round(min(max(score, 0.0), 1.0), 2)

    def _save_evaluation(self, evaluation: Dict) -> None:
        path = Path("critic_evaluation.json")
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        if isinstance(existing, list):
            existing.append(evaluation)
        else:
            existing = [existing, evaluation]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        log.info("Critic evaluation saved -> %s", path)
