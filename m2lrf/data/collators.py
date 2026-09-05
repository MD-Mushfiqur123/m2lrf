"""
M-2LRF Data Collators (Axolotl-Inspired)
=========================================
Collators supporting completion-only loss masking and dynamic padding.
"""

from typing import List, Dict, Any, Optional
import torch


class CompletionOnlyDataCollator:
    """
    Masks user prompt tokens with ignore_index (-100) so gradients are calculated
    strictly on assistant / completion tokens.
    """
    def __init__(
        self,
        tokenizer: Any,
        response_template: str = "### Response:\n",
        ignore_index: int = -100
    ):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.ignore_index = ignore_index
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = list(input_ids)
            
            # Find response_template start
            found_idx = -1
            r_len = len(self.response_token_ids)
            for i in range(len(input_ids) - r_len + 1):
                if input_ids[i:i+r_len] == self.response_token_ids:
                    found_idx = i + r_len
                    break

            if found_idx != -1:
                # Mask prompt up to found_idx
                for k in range(found_idx):
                    labels[k] = self.ignore_index
            labels_list.append(labels)

        # Pad dynamically
        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(ids) for ids in input_ids_list)

        padded_inputs = []
        padded_labels = []
        attention_masks = []

        for ids, lbls in zip(input_ids_list, labels_list):
            p_len = max_len - len(ids)
            padded_inputs.append(ids + [pad_id] * p_len)
            padded_labels.append(lbls + [self.ignore_index] * p_len)
            attention_masks.append([1] * len(ids) + [0] * p_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
        }
