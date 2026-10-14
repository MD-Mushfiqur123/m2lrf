import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from benchmarks.eval_downstream_tasks import compute_perplexity


class UniformCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=4)

    def forward(self, input_ids):
        logits = torch.zeros(
            input_ids.size(0), input_ids.size(1), 5, device=input_ids.device
        )
        return SimpleNamespace(logits=logits)


class TestWikiTextPerplexityEvaluation(unittest.TestCase):
    def test_scores_every_next_token_within_context_limit(self):
        model = UniformCausalModel()
        tokens = torch.tensor([[0, 1, 2, 3, 4, 0]])

        perplexity, scored_tokens = compute_perplexity(model, tokens, stride=2)

        self.assertAlmostEqual(perplexity, 5.0, places=5)
        self.assertEqual(scored_tokens, tokens.size(1) - 1)


if __name__ == "__main__":
    unittest.main()
