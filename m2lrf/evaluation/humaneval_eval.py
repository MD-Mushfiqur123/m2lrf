"""
M-2LRF Evaluation Engine: HumanEval Code Generation Evaluator.
==============================================================
Evaluates code generation capabilities using unit test assertion verification:
- Extracts Python function definitions from Markdown code blocks
- Executes assertions in a clean namespace
- Calculates Pass@1 accuracy
"""

from typing import Any, Callable, Dict, List, Optional
import re


class HumanEvalEvaluator:
    """Evaluates Python code generation and algorithmic verification."""

    @staticmethod
    def extract_code(completion: str) -> str:
        """Extracts python code blocks or raw code from model output."""
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        generic_blocks = re.findall(r"```\s*(.*?)\s*```", completion, re.DOTALL)
        if generic_blocks:
            return generic_blocks[0].strip()

        return completion.strip()

    @classmethod
    def execute_and_verify(cls, generated_code: str, test_assertions: str) -> bool:
        """
        Executes generated code followed by test assertions in a dedicated namespace.
        Returns True if all assertions execute without error.
        """
        full_code = f"{generated_code}\n\n{test_assertions}"
        local_scope: Dict[str, Any] = {}
        try:
            exec(full_code, {}, local_scope)
            return True
        except Exception:
            return False

    @classmethod
    def evaluate_dataset(
        cls,
        generate_fn: Callable[[str], str],
        dataset: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Evaluates a suite of HumanEval problems.
        Each sample dict contains: 'prompt', 'entry_point', 'test'.
        """
        passed = 0
        total = len(dataset)
        outcomes = []

        for sample in dataset:
            prompt = sample.get("prompt", "")
            test = sample.get("test", "")
            completion = generate_fn(prompt)
            code = cls.extract_code(completion)
            is_passed = cls.execute_and_verify(code, test)
            if is_passed:
                passed += 1
            outcomes.append({
                "prompt": prompt,
                "passed": is_passed,
                "code": code,
            })

        pass_at_1 = (passed / total * 100.0) if total > 0 else 0.0
        return {
            "total_tasks": total,
            "passed_tasks": passed,
            "pass_at_1_percent": round(pass_at_1, 2),
            "outcomes": outcomes,
        }
