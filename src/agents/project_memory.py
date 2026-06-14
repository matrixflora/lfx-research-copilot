from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("project_memory")


class ProjectMemory:
    """Persistent cross-session research project memory."""

    def __init__(self, storage_dir: str = "projects"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.name: str = ""
        self.research_question: str = ""
        self.papers: List[Dict] = []
        self.themes: List[Dict] = []
        self.gaps: List[Dict] = []
        self.hypotheses: List[Dict] = []
        self.reports: List[str] = []
        self.status: str = "active"
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = self.created_at

    def get_or_create(self, project_name: str) -> "ProjectMemory":
        path = self.storage_dir / f"{project_name.replace(' ', '_')}.json"
        if path.exists():
            return self.load(project_name)
        self.name = project_name
        self.save()
        log.info("Created new project: %s", project_name)
        return self

    def load(self, project_name: str) -> "ProjectMemory":
        path = self.storage_dir / f"{project_name.replace(' ', '_')}.json"
        if not path.exists():
            log.warning("Project not found: %s", project_name)
            return self.get_or_create(project_name)
        try:
            data = json.loads(path.read_text())
            self.name = data.get("name", project_name)
            self.research_question = data.get("research_question", "")
            self.papers = data.get("papers", [])
            self.themes = data.get("themes", [])
            self.gaps = data.get("gaps", [])
            self.hypotheses = data.get("hypotheses", [])
            self.reports = data.get("reports", [])
            self.status = data.get("status", "active")
            self.created_at = data.get("created_at", "")
            self.updated_at = data.get("updated_at", "")
            log.info("Loaded project: %s", project_name)
        except Exception as e:
            log.error("Failed to load project %s: %s", project_name, e)
        return self

    def save(self) -> None:
        path = self.storage_dir / f"{self.name.replace(' ', '_')}.json"
        data = {
            "name": self.name,
            "research_question": self.research_question,
            "papers": self.papers,
            "themes": self.themes,
            "gaps": self.gaps,
            "hypotheses": self.hypotheses,
            "reports": self.reports,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved project -> %s", path)

    def update_research_question(self, question: str) -> None:
        self.research_question = question
        self.save()

    def update_status(self, status: str) -> None:
        self.status = status
        self.save()

    def add_papers(self, papers: List[Dict]) -> None:
        existing_dois = {p.get("doi") for p in self.papers if p.get("doi")}
        for p in papers:
            if p.get("doi") not in existing_dois:
                self.papers.append(p)
        self.save()

    def add_themes(self, themes: List[Dict]) -> None:
        for t in themes:
            if t not in self.themes:
                self.themes.append(t)
        self.save()

    def add_gaps(self, gaps: List[Dict]) -> None:
        for g in gaps:
            if g not in self.gaps:
                self.gaps.append(g)
        self.save()

    def add_hypotheses(self, hypotheses: List[Dict]) -> None:
        for h in hypotheses:
            if h not in self.hypotheses:
                self.hypotheses.append(h)
        self.save()

    def add_report(self, report_path: str) -> None:
        if report_path not in self.reports:
            self.reports.append(report_path)
            self.save()

    def list_projects(self) -> List[str]:
        return [str(p.stem) for p in self.storage_dir.glob("*.json")]

    def delete(self) -> None:
        path = self.storage_dir / f"{self.name.replace(' ', '_')}.json"
        if path.exists():
            path.unlink()
            log.info("Deleted project: %s", self.name)
