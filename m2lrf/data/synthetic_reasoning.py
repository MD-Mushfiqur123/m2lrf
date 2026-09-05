"""
M-2LRF Data Engine: Synthetic Reasoning & Rule-Based Verifiers.
Inspired by Torchtune and DeepSeek-R1 RLVR (Reinforcement Learning with Verifiable Rewards).
Procedurally generates verifiable reasoning traces (Math, Olympiad, Algorithms, Logic)
with rule-based reward verifiers for GRPO training.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import random
import re


class MathProblem:
    """Represents a procedurally generated mathematical reasoning problem."""

    def __init__(self, prompt: str, solution_trace: str, ground_truth: str):
        self.prompt = prompt
        self.solution_trace = solution_trace
        self.ground_truth = str(ground_truth).strip()

    def to_dict(self) -> Dict[str, str]:
        return {
            "prompt": self.prompt,
            "solution_trace": self.solution_trace,
            "ground_truth": self.ground_truth,
        }


class MathRuleVerifier:
    """
    Rule-based verifier for mathematical solutions.
    Extracts answers from \\boxed{...} or the final line and compares with ground truth.
    Provides DeepSeek-R1 style reward scoring:
    - Correct answer: +1.0
    - Proper <think> formatting: +0.2
    - Incorrect answer: -0.5
    """

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        # 1. Look for \boxed{...}
        boxed_match = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match[-1].strip()

        # 2. Look for "The answer is ..."
        answer_match = re.findall(r"(?:answer is|equals|=)\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if answer_match:
            return answer_match[-1].strip()

        # 3. Last number in text
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if nums:
            return nums[-1].strip()

        return None

    @classmethod
    def verify(cls, completion: str, ground_truth: str) -> float:
        extracted = cls.extract_answer(completion)
        reward = 0.0

        # Reward proper thinking tags
        has_think = ("<think>" in completion and "</think>" in completion)
        if has_think:
            reward += 0.2

        if extracted is None:
            return reward - 0.5

        # Check equality (numerical or exact string)
        gt = ground_truth.strip()
        try:
            val_ext = float(extracted)
            val_gt = float(gt)
            if abs(val_ext - val_gt) < 1e-5:
                return reward + 1.0
        except ValueError:
            if extracted.lower() == gt.lower():
                return reward + 1.0

        return reward - 0.5


class SyntheticReasoningGenerator:
    """
    Procedural generator for high-density verifiable reasoning datasets.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_modular_arithmetic(self) -> MathProblem:
        a = self.rng.randint(11, 99)
        b = self.rng.randint(11, 99)
        m = self.rng.choice([5, 7, 9, 11, 13, 17])
        ans = (a * b) % m

        prompt = f"Calculate the remainder when {a} \\times {b} is divided by {m}."
        trace = (
            f"<think>\n"
            f"We need to find ({a} * {b}) mod {m}.\n"
            f"First, calculate the product: {a} * {b} = {a * b}.\n"
            f"Next, divide {a * b} by {m}:\n"
            f"{a * b} = {m} * { (a * b) // m } + {ans}.\n"
            f"Thus, the remainder is {ans}.\n"
            f"</think>\n"
            f"The remainder is \\boxed{{{ans}}}."
        )
        return MathProblem(prompt, trace, str(ans))

    def generate_linear_equation(self) -> MathProblem:
        x = self.rng.randint(-20, 20)
        a = self.rng.randint(2, 9)
        b = self.rng.randint(-50, 50)
        c = a * x + b

        prompt = f"Solve for x: {a}x + ({b}) = {c}."
        trace = (
            f"<think>\n"
            f"Given the linear equation {a}x + ({b}) = {c}.\n"
            f"Subtract {b} from both sides:\n"
            f"{a}x = {c} - ({b}) = {c - b}.\n"
            f"Divide by {a}:\n"
            f"x = {c - b} / {a} = {x}.\n"
            f"</think>\n"
            f"The solution is \\boxed{{{x}}}."
        )
        return MathProblem(prompt, trace, str(x))

    def generate_geometric_series(self) -> MathProblem:
        a = self.rng.randint(1, 5)
        r = 2
        n = self.rng.randint(4, 7)
        # S_n = a * (r^n - 1) / (r - 1)
        ans = a * (r**n - 1)

        prompt = f"Find the sum of the first {n} terms of the geometric progression with first term a = {a} and common ratio r = {r}."
        trace = (
            f"<think>\n"
            f"The sum of a geometric sequence is given by S_n = a * (r^n - 1) / (r - 1).\n"
            f"Here, a = {a}, r = {r}, and n = {n}.\n"
            f"Calculate r^n = {r}^{n} = {r**n}.\n"
            f"Then r^n - 1 = {r**n - 1}.\n"
            f"S_{n} = {a} * ({r**n - 1}) / ({r - 1}) = {ans}.\n"
            f"</think>\n"
            f"The sum is \\boxed{{{ans}}}."
        )
        return MathProblem(prompt, trace, str(ans))

    def generate_batch(self, num_samples: int = 100) -> List[MathProblem]:
        generators = [
            self.generate_modular_arithmetic,
            self.generate_linear_equation,
            self.generate_geometric_series,
        ]
        samples = []
        for _ in range(num_samples):
            gen_fn = self.rng.choice(generators)
            samples.append(gen_fn())
        return samples
