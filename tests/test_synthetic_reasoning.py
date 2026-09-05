"""
Unit tests for M-2LRF Synthetic Reasoning Generator and Rule-Based Verifiers.
"""

import unittest

from m2lrf.data.synthetic_reasoning import (
    MathProblem,
    MathRuleVerifier,
    SyntheticReasoningGenerator,
)


class TestSyntheticReasoning(unittest.TestCase):

    def setUp(self):
        self.generator = SyntheticReasoningGenerator(seed=12345)

    def test_modular_arithmetic_generator(self):
        problem = self.generator.generate_modular_arithmetic()
        self.assertIn("Calculate the remainder", problem.prompt)
        self.assertIn("<think>", problem.solution_trace)
        self.assertIn(f"\\boxed{{{problem.ground_truth}}}", problem.solution_trace)

        # Verify ground truth matches verifier
        reward = MathRuleVerifier.verify(problem.solution_trace, problem.ground_truth)
        self.assertAlmostEqual(reward, 1.2)  # 1.0 correct + 0.2 think tags

    def test_linear_equation_generator(self):
        problem = self.generator.generate_linear_equation()
        self.assertIn("Solve for x", problem.prompt)
        self.assertIn(f"\\boxed{{{problem.ground_truth}}}", problem.solution_trace)

        reward = MathRuleVerifier.verify(problem.solution_trace, problem.ground_truth)
        self.assertAlmostEqual(reward, 1.2)

    def test_geometric_series_generator(self):
        problem = self.generator.generate_geometric_series()
        self.assertIn("geometric progression", problem.prompt)
        reward = MathRuleVerifier.verify(problem.solution_trace, problem.ground_truth)
        self.assertAlmostEqual(reward, 1.2)

    def test_math_verifier_incorrect_response(self):
        wrong_completion = "<think>I think the answer is 999</think> The answer is \\boxed{999}."
        reward = MathRuleVerifier.verify(wrong_completion, ground_truth="42")
        # 0.2 think - 0.5 incorrect = -0.3
        self.assertAlmostEqual(reward, -0.3)

    def test_generate_batch(self):
        batch = self.generator.generate_batch(num_samples=25)
        self.assertEqual(len(batch), 25)
        for p in batch:
            d = p.to_dict()
            self.assertIn("prompt", d)
            self.assertIn("solution_trace", d)
            self.assertIn("ground_truth", d)


if __name__ == "__main__":
    unittest.main()
