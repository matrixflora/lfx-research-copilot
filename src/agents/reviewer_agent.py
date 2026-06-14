from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("reviewer_agent")


class ReviewerAgent:
    """Acts as a peer reviewer to evaluate research outputs."""

    def __init__(self):
        self.feedback: List[Dict] = []

    def review(self, research_analysis: Dict) -> str:
        log.info("Starting peer review of research outputs")
        issues = []
        missing_evidence = self._check_missing_evidence(research_analysis)
        missing_citations = self._check_missing_citations(research_analysis)
        weak_claims = self._check_weak_claims(research_analysis)
        low_confidence = self._check_low_confidence(research_analysis)

        if missing_evidence:
            issues.append({"type": "missing_evidence", "details": missing_evidence})
        if missing_citations:
            issues.append({"type": "missing_citations", "details": missing_citations})
        if weak_claims:
            issues.append({"type": "weak_claims", "details": weak_claims})
        if low_confidence:
            issues.append({"type": "low_confidence", "details": low_confidence})

        report = self._generate_report(issues)
        self._save_feedback(report)
        return report

    def _check_missing_evidence(self, analysis: Dict) -> List[str]:
        issues = []
        summary = analysis.get("summary", {})
        outputs = summary.get("analysis_ready", {})

        if not outputs.get("search_results.csv"):
            issues.append("No search results found. Paper retrieval may be needed.")
        if not outputs.get("consensus_themes.csv"):
            issues.append("Theme discovery not performed.")
        if not outputs.get("knowledge_base.json"):
            issues.append("Knowledge base not generated.")

        return issues

    def _check_missing_citations(self, analysis: Dict) -> List[str]:
        issues = []
        kb = analysis.get("knowledge_base", {})
        themes = kb.get("themes", [])
        for theme in themes:
            papers = theme.get("papers", [])
            if len(papers) < 2:
                issues.append(f"Theme '{theme.get('theme', 'unknown')}' has only {len(papers)} paper(s)")
        return issues

    def _check_weak_claims(self, analysis: Dict) -> List[str]:
        issues = []
        consensus = analysis.get("consensus_metadata", [])
        for theme in consensus:
            confidence = theme.get("confidence", 0)
            if confidence < 0.5:
                issues.append(
                    f"Theme '{theme.get('label', 'unknown')}' has low confidence ({confidence:.2f})"
                )
            if theme.get("is_exploratory", False):
                issues.append(
                    f"Theme '{theme.get('label', 'unknown')}' is exploratory (single method)"
                )
        return issues

    def _check_low_confidence(self, analysis: Dict) -> List[str]:
        issues = []
        scores_path = Path("outputs/reports/evidence_strength.csv")
        if not scores_path.exists():
            issues.append("Evidence strength scoring not performed")
        return issues

    def _generate_report(self, issues: List[Dict]) -> str:
        lines = []
        lines.append("# Reviewer Feedback")
        lines.append("")
        lines.append(f"**Review Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append(f"**Issues Found:** {len(issues)}")
        lines.append("")

        if not issues:
            lines.append("No significant issues detected. Research outputs appear sound.")
        else:
            for issue in issues:
                lines.append(f"## {issue['type'].replace('_', ' ').title()}")
                lines.append("")
                for detail in issue["details"]:
                    lines.append(f"- {detail}")
                lines.append("")

        return "\n".join(lines)

    def _save_feedback(self, report: str) -> None:
        path = Path("reviewer_feedback.md")
        path.write_text(report)
        log.info("Reviewer feedback -> %s", path)
