from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("memory_agent")


class MemoryAgent:
    """Long-term project memory for cross-session continuity."""

    def __init__(self, memory_path: str = "memory.json"):
        self.memory_path = Path(memory_path)
        self.memory = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            with open(self.memory_path) as f:
                return json.load(f)
        return {
            "searches": [],
            "themes": [],
            "research_gaps": [],
            "hypotheses": [],
            "projects": [],
            "user_feedback": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def save(self) -> None:
        self.memory["updated_at"] = datetime.now().isoformat()
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f, indent=2)
        log.info("Memory saved -> %s", self.memory_path)

    def save_memory(self, data: Dict) -> None:
        self.memory.update(data)
        self.save()

    def retrieve_memory(self, key: Optional[str] = None) -> Any:
        if key:
            return self.memory.get(key)
        return self.memory

    def update_memory(self, key: str, value: Any) -> None:
        self.memory[key] = value
        self.save()

    def search_memory(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()

        for search in self.memory.get("searches", []):
            if query_lower in search.get("query", "").lower():
                results.append(search)

        for theme in self.memory.get("themes", []):
            if query_lower in theme.get("name", "").lower():
                results.append({"type": "theme", **theme})

        for gap in self.memory.get("research_gaps", []):
            if query_lower in gap.get("description", "").lower():
                results.append({"type": "gap", **gap})

        return results

    def add_search(self, query: str, result_count: int) -> None:
        self.memory["searches"].append({
            "query": query,
            "result_count": result_count,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def add_themes(self, themes: List[Dict]) -> None:
        for theme in themes:
            if theme not in self.memory["themes"]:
                self.memory["themes"].append(theme)
        self.save()

    def add_gaps(self, gaps: List[Dict]) -> None:
        for gap in gaps:
            if gap not in self.memory["research_gaps"]:
                self.memory["research_gaps"].append(gap)
        self.save()

    def add_hypotheses(self, hypotheses: List[str]) -> None:
        for h in hypotheses:
            entry = {"hypothesis": h, "timestamp": datetime.now().isoformat()}
            if entry not in self.memory["hypotheses"]:
                self.memory["hypotheses"].append(entry)
        self.save()

    def add_feedback(self, feedback: str) -> None:
        self.memory["user_feedback"].append({
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()
