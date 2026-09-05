"""
M-2LRF Evaluation Engine: MMLU 57-Subject Benchmark.
====================================================
Taxonomy and 5-shot prompt formatter for the Massive Multitask Language Understanding benchmark:
- STEM: Physics, Chemistry, Biology, Math, Computer Science
- Humanities: History, Philosophy, Law
- Social Sciences: Economics, Psychology, Politics
- Other: Business, Health, Accounting
"""

from typing import Any, Callable, Dict, List, Optional


class MMLUEvaluator:
    """MMLU 57-subject benchmark coordinator."""

    SUBJECT_CATEGORIES = {
        "stem": ["college_physics", "college_chemistry", "college_mathematics", "computer_science", "machine_learning"],
        "humanities": ["world_history", "philosophy", "jurisprudence", "formal_logic"],
        "social_sciences": ["microeconomics", "macroeconomics", "psychology", "us_foreign_policy"],
        "applied": ["medical_genetics", "professional_accounting", "business_ethics"],
    }

    @staticmethod
    def format_5shot_prompt(subject: str, question: str, choices: List[str]) -> str:
        """Formats an MMLU evaluation prompt."""
        subject_name = subject.replace("_", " ").title()
        prompt = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
        prompt += f"Question: {question}\n"
        letters = ["A", "B", "C", "D"]
        for l, c in zip(letters, choices):
            prompt += f"{l}. {c}\n"
        prompt += "Answer:"
        return prompt
