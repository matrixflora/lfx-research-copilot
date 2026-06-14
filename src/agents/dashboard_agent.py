from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_agent import MemoryAgent

log = logging.getLogger("dashboard_agent")


class DashboardAgent:
    """Generates a user-facing dashboard of research state."""

    def __init__(self, memory: Optional[MemoryAgent] = None):
        self.memory = memory or MemoryAgent()

    def generate(self) -> str:
        log.info("Generating research dashboard")
        lines = []
        lines.append("# Research Dashboard")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        memory = self.memory.retrieve_memory()

        lines.append("## Active Projects")
        lines.append("")
        projects = memory.get("projects", [])
        if projects:
            for p in projects:
                lines.append(f"- {p.get('name', 'Unnamed')} ({p.get('status', 'unknown')})")
        else:
            lines.append("_No active projects._")
        lines.append("")

        lines.append("## Recent Queries")
        lines.append("")
        searches = memory.get("searches", [])
        if searches:
            for s in searches[-5:]:
                q = s.get("query", "")
                count = s.get("result_count", 0)
                ts = s.get("timestamp", "")[:10]
                lines.append(f"- `{q}` ({count} results, {ts})")
        else:
            lines.append("_No recent queries._")
        lines.append("")

        lines.append("## Emerging Themes")
        lines.append("")
        themes = memory.get("themes", [])
        if themes:
            for t in themes:
                name = t.get("name", t.get("theme", "Unknown"))
                count = t.get("paper_count", t.get("count", 0))
                lines.append(f"- **{name}** ({count} papers)")
        else:
            lines.append("_No themes identified yet._")
        lines.append("")

        lines.append("## Research Gaps")
        lines.append("")
        gaps = memory.get("research_gaps", [])
        if gaps:
            for g in gaps:
                desc = g.get("description", g.get("gap", "Unknown gap"))
                lines.append(f"- {desc}")
        else:
            lines.append("_No research gaps identified._")
        lines.append("")

        lines.append("## Hypotheses")
        lines.append("")
        hypotheses = memory.get("hypotheses", [])
        if hypotheses:
            for h in hypotheses[-5:]:
                hyp_text = h.get("hypothesis", "")
                lines.append(f"- {hyp_text}")
        else:
            lines.append("_No hypotheses generated._")
        lines.append("")

        lines.append("## Validation Status")
        lines.append("")
        critic_path = Path("critic_evaluation.json")
        if critic_path.exists():
            try:
                evals = json.loads(critic_path.read_text())
                if evals:
                    latest = evals[-1]
                    scores = latest.get("scores", {})
                    overall = scores.get("overall", 0)
                    lines.append(f"- **Overall Quality Score:** {overall:.2f}")
                    for area, score in scores.items():
                        if area != "overall":
                            lines.append(f"  - {area.replace('_', ' ').title()}: {score:.2f}")
                    if latest.get("needs_repair"):
                        lines.append("- **Status:** Repairs needed")
                    else:
                        lines.append("- **Status:** All quality thresholds met")
            except Exception:
                lines.append("_Unable to parse evaluation._")
        else:
            lines.append("_No evaluation performed yet._")
        lines.append("")

        dashboard = "\n".join(lines)
        path = Path("outputs/dashboard/dashboard.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dashboard)
        log.info("Dashboard -> %s", path)

        return dashboard
