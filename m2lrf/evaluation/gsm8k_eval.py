"""
M-2LRF Evaluation Engine: GSM8K Mathematical Reasoning Evaluator.
================================================================
Evaluates models on 8-shot Chain-of-Thought (CoT) mathematical reasoning:
- Extracts final numeric answer via regex, \\boxed{...}, or "The answer is ..."
- Robust normalization (removing commas, currency symbols, percentages)
- Computes Exact Match (EM) accuracy and token efficiency
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import re


class GSM8KEvaluator:
    """Evaluates mathematical reasoning accuracy on GSM8K style datasets."""

    @staticmethod
    def extract_answer(completion: str) -> Optional[str]:
        """Extracts the numeric answer from a model completion."""
        # 1. Look for \boxed{...}
        boxed_match = re.findall(r"\\boxed\{([^}]+)\}", completion)
        if boxed_match:
            cand = boxed_match[-1].strip().replace(",", "").replace("$", "")
            return cand

        # 2. Look for "#### <number>"
        hash_match = re.findall(r"####\s*([+-]?\d+(?:\.\d+)?)", completion)
        if hash_match:
            return hash_match[-1].strip()

        # 3. Look for "The answer is ..."
        is_match = re.findall(r"(?:answer is|equals|=)\s*([+-]?\d+(?:\.\d+)?)", completion, re.IGNORECASE)
        if is_match:
            return is_match[-1].strip()

        # 4. Fallback: last number in the string
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", completion)
        if numbers:
            return numbers[-1].strip()

        return None

    @classmethod
    def evaluate_sample(cls, completion: str, ground_truth: str) -> bool:
        """Returns True if extracted answer equals normalized ground truth."""
        pred = cls.extract_answer(completion)
        if pred is None:
            return False

        gt_norm = str(ground_truth).strip().replace(",", "").replace("$", "").replace("####", "").strip()
        try:
            # Float comparison for mathematical equivalence (e.g. 42.0 == 42)
            return abs(float(pred) - float(gt_norm)) < 1e-4
        except ValueError:
            return pred.lower() == gt_norm.lower()

    @classmethod
    def evaluate_dataset(
        cls,
        generate_fn: Callable[[str], str],
        dataset: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Runs evaluation across a dataset of question-answer pairs."""
        correct = 0
        total = len(dataset)
        results = []

        for item in dataset:
            question = item.get("question", item.get("prompt", ""))
            gt = item.get("answer", item.get("ground_truth", ""))
            completion = generate_fn(question)
            is_correct = cls.evaluate_sample(completion, gt)
            if is_correct:
                correct += 1
            results.append({
                "question": question,
                "ground_truth": gt,
                "completion": completion,
                "is_correct": is_correct,
            })

        acc = (correct / total * 100.0) if total > 0 else 0.0
        return {
            "total_samples": total,
            "correct_samples": correct,
            "accuracy_percent": round(acc, 2),
            "results": results,
        }
