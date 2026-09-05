"""
M-2LRF Multiplexed Sequence Packing (Axolotl-Inspired)
======================================================
Packs multiple variable-length sequences into constant-length tensors.
Eliminates padding waste while generating block-diagonal attention masks to guarantee
zero cross-sample attention contamination.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch


class SequencePacker:
    """
    Packs tokenized samples up to max_seq_length.
    """
    def __init__(
        self,
        max_seq_length: int = 4096,
        pad_token_id: int = 0,
        ignore_index: int = -100
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def pack(
        self,
        tokenized_samples: List[Dict[str, List[int]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Packs a list of tokenized samples (containing 'input_ids' and 'labels').
        Returns list of packed batches with 2D block-diagonal attention masks.
        """
        packed_batches = []
        curr_input_ids = []
        curr_labels = []
        curr_position_ids = []
        curr_seq_boundaries = [0]

        for sample in tokenized_samples:
            input_ids = sample["input_ids"]
            labels = sample.get("labels", input_ids)

            sample_len = len(input_ids)
            if sample_len > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                labels = labels[:self.max_seq_length]
                sample_len = self.max_seq_length

            # If adding this sample exceeds max_seq_length, finalize current batch
            if len(curr_input_ids) + sample_len > self.max_seq_length:
                if len(curr_input_ids) > 0:
                    packed_batches.append(
                        self._build_packed_tensor(
                            curr_input_ids,
                            curr_labels,
                            curr_position_ids,
                            curr_seq_boundaries
                        )
                    )
                curr_input_ids = []
                curr_labels = []
                curr_position_ids = []
                curr_seq_boundaries = [0]

            # Append sample
            curr_input_ids.extend(input_ids)
            curr_labels.extend(labels)
            curr_position_ids.extend(list(range(sample_len)))
            curr_seq_boundaries.append(len(curr_input_ids))

        # Final batch
        if len(curr_input_ids) > 0:
            packed_batches.append(
                self._build_packed_tensor(
                    curr_input_ids,
                    curr_labels,
                    curr_position_ids,
                    curr_seq_boundaries
                )
            )

        return packed_batches

    def _build_packed_tensor(
        self,
        input_ids: List[int],
        labels: List[int],
        position_ids: List[int],
        boundaries: List[int]
    ) -> Dict[str, torch.Tensor]:
        actual_len = len(input_ids)
        pad_len = self.max_seq_length - actual_len

        # Pad to max_seq_length
        padded_input_ids = input_ids + [self.pad_token_id] * pad_len
        padded_labels = labels + [self.ignore_index] * pad_len
        padded_position_ids = position_ids + [0] * pad_len

        # Build 2D block-diagonal causal attention mask
        # Shape: [max_seq_length, max_seq_length]
        # mask[i, j] = 1 if i and j belong to the same sequence AND j <= i
        mask_2d = torch.zeros((self.max_seq_length, self.max_seq_length), dtype=torch.bool)
        
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            # Standard causal lower-triangular block for this sequence
            sub_block = torch.tril(torch.ones((end - start, end - start), dtype=torch.bool))
            mask_2d[start:end, start:end] = sub_block

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "position_ids": torch.tensor(padded_position_ids, dtype=torch.long),
            "attention_mask": mask_2d
        }
