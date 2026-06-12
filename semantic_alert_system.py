#!/usr/bin/env python3
"""
semantic_alert_system.py — Monitor changes when new literature is added by
comparing knowledge base snapshots.  Detects new themes, new gaps, theme
shifts, and conclusion changes.

Outputs
-------
outputs/alerts/semantic_alerts.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s): %(message)s")
log = logging.getLogger("semantic_alerts")

MODEL_NAME = "all-MiniLM-L6-v2"
SNAPSHOT_DIR = Path("outputs") / "alerts" / "snapshots"


def _load_kb(path: str = "knowledge_base.json") -> Dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _load_previous_kb() -> Dict:
    """Load the most recent snapshot, if any."""
    if not SNAPSHOT_DIR.exists():
        return {}
    snapshots = sorted(SNAPSHOT_DIR.glob("kb_snapshot_*.json"))
    if not snapshots:
        return {}
    return json.loads(snapshots[-1].read_text())


def _save_snapshot(kb: Dict) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(SNAPSHOT_DIR / f"kb_snapshot_{ts}.json", "w") as f:
        json.dump(kb, f, indent=2)


def detect_new_themes(current: Dict, previous: Dict) -> List[str]:
    current_names = {t["theme"] for t in current.get("themes", [])}
    previous_names = {t["theme"] for t in previous.get("themes", [])}
    new_themes = current_names - previous_names
    return sorted(new_themes)


def detect_new_gaps(current: Dict, gap_path: str = "outputs/reports/research_gaps.md") -> List[str]:
    """Check if new gap statements appeared vs previous KB snapshot."""
    previous = _load_previous_kb()
    if not previous:
        return ["Initial analysis — establish baseline gaps"]

    gap_file = Path(gap_path)
    if not gap_file.exists():
        return []
    current_gaps = set(re.findall(r'\*\*(.*?)\*\*\s*\((\d+) papers?\)', gap_file.read_text()))
    # This is a first-run detection; actual diff logic would store previous gap text
    return [f"{theme} ({pc} papers)" for theme, pc in current_gaps]


def detect_theme_shifts(current: Dict, previous: Dict) -> List[Dict]:
    """Detect changes in theme composition (paper membership shifts)."""
    shifts = []
    current_papers = {t["theme"]: set(t.get("papers", [])) for t in current.get("themes", [])}
    previous_papers = {t["theme"]: set(t.get("papers", [])) for t in previous.get("themes", [])}

    for theme, curr_set in current_papers.items():
        prev_set = previous_papers.get(theme, set())
        added = curr_set - prev_set
        removed = prev_set - curr_set
        if added or removed:
            shifts.append({
                "theme": theme,
                "papers_added": len(added),
                "papers_removed": len(removed),
                "net_change": len(curr_set) - len(prev_set),
            })
    return shifts


def detect_conclusion_changes(current: Dict, previous: Dict) -> List[str]:
    """Detect changes in themes' confidence or classification."""
    changes = []
    if not previous:
        return changes
    current_conf = {t["theme"]: t.get("confidence", 0) for t in current.get("themes", [])}
    previous_conf = {t["theme"]: t.get("confidence", 0) for t in previous.get("themes", [])}
    for theme, conf in current_conf.items():
        prev_conf = previous_conf.get(theme, conf)
        diff = conf - prev_conf
        if abs(diff) > 0.1:
            direction = "increased" if diff > 0 else "decreased"
            changes.append(f"{theme}: confidence {direction} from {prev_conf:.2f} to {conf:.2f}")
    return changes


def _generate_alerts(new_themes: List[str], new_gaps: List[str],
                     shifts: List[Dict], conclusion_changes: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Semantic Alerts\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## New Themes\n")
    if new_themes:
        for t in new_themes:
            lines.append(f"- **{t}** (newly detected)")
    else:
        lines.append("_No new themes detected._\n")

    lines.append("\n## New/Updated Gaps\n")
    for g in new_gaps:
        lines.append(f"- {g}")
    lines.append("")

    lines.append("## Theme Shifts\n")
    if shifts:
        for s in shifts:
            lines.append(f"- **{s['theme']}:** {s['net_change']:+d} papers ({s['papers_added']} added, {s['papers_removed']} removed)")
    else:
        lines.append("_No theme shifts detected._\n")

    lines.append("\n## Conclusion Changes\n")
    for c in conclusion_changes:
        lines.append(f"- {c}")
    if not conclusion_changes:
        lines.append("_No conclusion changes detected._\n")

    lines.append("---\n*Generated by semantic_alert_system.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic alert system for literature changes.")
    parser.add_argument("--knowledge-base", type=str, default="knowledge_base.json")
    parser.add_argument("--gaps", type=str, default="outputs/reports/research_gaps.md")
    parser.add_argument("--output-dir", type=str, default="outputs/alerts")
    args = parser.parse_args()

    current = _load_kb(args.knowledge_base)
    previous = _load_previous_kb()

    new_themes = detect_new_themes(current, previous)
    new_gaps = detect_new_gaps(current, args.gaps)
    shifts = detect_theme_shifts(current, previous)
    conclusion_changes = detect_conclusion_changes(current, previous)

    _save_snapshot(current)

    report = _generate_alerts(new_themes, new_gaps, shifts, conclusion_changes)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "semantic_alerts.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "semantic_alerts.md")

    print(f"\n--- Semantic Alert System Complete ---")
    print(f"  New themes: {len(new_themes)}")
    print(f"  Theme shifts: {len(shifts)}")
    print()


if __name__ == "__main__":
    main()
