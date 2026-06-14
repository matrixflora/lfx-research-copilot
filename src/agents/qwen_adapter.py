from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("qwen_adapter")


class QwenAdapter:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto"
                )
                log.info("Loaded Qwen model: %s", self.model_name)
            except ImportError:
                log.warning(
                    "transformers not installed. Using fallback planning."
                )
                self._model = False
            except Exception as e:
                log.warning("Failed to load Qwen model: %s", e)
                self._model = False

    def _call_model(self, prompt: str, max_new_tokens: int = 512) -> str:
        self._load_model()
        if self._model is None or self._model is False:
            return ""
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            log.error("Qwen inference error: %s", e)
            return ""

    def generate_plan(self, user_query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt = (
            f"User query: {user_query}\n"
            f"Context: {context or 'No prior context.'}\n\n"
            "Break this research goal into concrete sub-tasks. "
            "Return as a JSON list of objects with keys 'task', 'module', 'priority'."
        )
        result = self._call_model(prompt)
        return self._parse_json_list(result)

    def evaluate_report(self, report_text: str) -> Dict[str, Any]:
        prompt = (
            f"Evaluate this research report for quality, completeness, "
            f"and rigor. Score each dimension 0.0-1.0. Return JSON.\n\n{report_text}"
        )
        result = self._call_model(prompt, max_new_tokens=256)
        return self._parse_json_dict(result)

    def identify_missing_steps(self, plan: List[Dict], results: Dict) -> List[str]:
        plan_str = json.dumps(plan, indent=2)
        results_str = json.dumps(results, indent=2)
        prompt = (
            f"Original plan:\n{plan_str}\n\n"
            f"Execution results:\n{results_str}\n\n"
            "What steps are missing or incomplete? Return a JSON list of strings."
        )
        result = self._call_model(prompt, max_new_tokens=256)
        return self._parse_json_list(result)

    def suggest_improvements(self, output_text: str) -> List[str]:
        prompt = (
            f"Suggest concrete improvements for this research output:\n\n{output_text}\n\n"
            "Return a JSON list of improvement suggestions."
        )
        result = self._call_model(prompt, max_new_tokens=256)
        return self._parse_json_list(result)

    def generate_research_questions(self, topic: str, n: int = 3) -> List[str]:
        prompt = (
            f"Generate {n} novel research questions about: {topic}\n\n"
            "Return a JSON list of strings."
        )
        result = self._call_model(prompt, max_new_tokens=256)
        return self._parse_json_list(result)

    def generate_hypotheses(self, research_question: str, n: int = 3) -> List[str]:
        prompt = (
            f"Generate {n} testable hypotheses for the question:\n"
            f"{research_question}\n\nReturn a JSON list of strings."
        )
        result = self._call_model(prompt, max_new_tokens=256)
        return self._parse_json_list(result)

    def _parse_json_list(self, text: str) -> list:
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    def _parse_json_dict(self, text: str) -> dict:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}
