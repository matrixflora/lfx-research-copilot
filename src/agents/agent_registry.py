from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

log = logging.getLogger("agent_registry")


class AgentRegistry:
    _agents: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
    ) -> None:
        cls._agents[name] = {
            "name": name,
            "module_path": module_path,
            "class_name": class_name,
            "description": description,
            "capabilities": capabilities or [],
        }
        log.info("Registered agent: %s (%s.%s)", name, module_path, class_name)

    @classmethod
    def discover(cls, base_path: Optional[str] = None) -> None:
        if base_path is None:
            base_path = str(Path(__file__).parent)
        p = Path(base_path)
        for fpath in p.glob("*_agent.py"):
            module_name = f"src.agents.{fpath.stem}"
            try:
                module = importlib.import_module(module_name)
                module_file = getattr(module, "__file__", "")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not (isinstance(attr, type) and attr_name.endswith("Agent")):
                        continue
                    attr_file = getattr(attr, "__module__", "")
                    if attr_file and attr_file != module_name:
                        continue
                    cls.register(
                        name=attr_name.replace("Agent", "").lower(),
                        module_path=module_name,
                        class_name=attr_name,
                        description=getattr(attr, "__doc__", "") or "",
                    )
            except Exception as e:
                log.warning("Failed to discover agent in %s: %s", fpath.name, e)

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        return cls._agents.get(name)

    @classmethod
    def load(cls, name: str) -> Optional[Any]:
        info = cls.get(name)
        if info is None:
            log.error("Agent '%s' not found in registry", name)
            return None
        try:
            module = importlib.import_module(info["module_path"])
            agent_class = getattr(module, info["class_name"])
            return agent_class()
        except Exception as e:
            log.error("Failed to load agent '%s': %s", name, e)
            return None

    @classmethod
    def list_agents(cls) -> List[Dict[str, Any]]:
        return list(cls._agents.values())

    @classmethod
    def find_by_capability(cls, capability: str) -> List[Dict[str, Any]]:
        return [
            info for info in cls._agents.values()
            if capability in info.get("capabilities", [])
        ]
