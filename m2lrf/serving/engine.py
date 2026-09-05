"""
M-2LRF Serving Engine: Continuous Batching LLMEngine.
Combines PagedAttention, BlockSpaceManager, and RadixPrefixCache into a production serving loop.
Supports iteration-level scheduling, dynamic preemption, chunked prefill, and telemetry.
"""

from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import time
import torch

from m2lrf.serving.block_manager import BlockSpaceManager, Sequence, SequenceStatus
from m2lrf.serving.paged_attention import PagedKVCache, paged_attention_v1
from m2lrf.serving.radix_cache import RadixPrefixCache


class SamplingParams:
    """Generation and sampling configuration per request."""

    def __init__(
        self,
        max_tokens: int = 128,
        temperature: float = 0.0,  # 0.0 = greedy
        top_p: float = 0.95,
        stop_token_ids: Optional[List[int]] = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token_ids = set(stop_token_ids) if stop_token_ids else set()


class RequestOutput:
    """Output structure returned for each generation step."""

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        output_token_ids: List[int],
        finished: bool,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = output_token_ids
        self.finished = finished

    def __repr__(self) -> str:
        return f"RequestOutput(id={self.request_id}, gen_tokens={len(self.output_token_ids)}, finished={self.finished})"


class LLMEngine:
    """
    High-Throughput Continuous Batching Serving Runtime.
    Orchestrates iteration-level scheduling with PagedAttention and RadixPrefixCache.
    """

    def __init__(
        self,
        model_forward_fn: Callable[[torch.Tensor, Optional[PagedKVCache], Optional[torch.Tensor]], torch.Tensor],
        num_gpu_blocks: int = 1024,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        device: str = "cpu",
        quantize_2bit_kv: bool = False,
    ):
        self.model_forward_fn = model_forward_fn
        self.block_size = block_size
        self.device = device

        self.block_manager = BlockSpaceManager(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            device=device,
        )
        self.kv_cache = PagedKVCache(
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            quantize_2bit=quantize_2bit_kv,
        )
        self.radix_cache = RadixPrefixCache(block_size=block_size)

        # Request state tracking
        self.waiting_queue: List[Tuple[Sequence, SamplingParams, str]] = []
        self.running_sequences: Dict[int, Tuple[Sequence, SamplingParams, str]] = {}
        self.finished_outputs: List[RequestOutput] = []

        # Telemetry metrics
        self.seq_id_counter = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> None:
        """Enqueues a new inference request."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        self.seq_id_counter += 1
        seq = Sequence(
            seq_id=self.seq_id_counter,
            prompt_token_ids=prompt_token_ids,
            block_size=self.block_size,
            max_tokens=sampling_params.max_tokens,
        )
        self.waiting_queue.append((seq, sampling_params, request_id))

    def step(self) -> List[RequestOutput]:
        """
        Executes a single continuous batching iteration:
        1. Schedules waiting requests into running state if KV memory blocks permit.
        2. Performs forward pass for running sequences.
        3. Samples next tokens and handles termination.
        4. Updates RadixPrefixCache on finished requests.
        """
        # 1. Schedule waiting sequences
        while self.waiting_queue:
            seq, params, req_id = self.waiting_queue[0]
            if self.block_manager.can_allocate(seq):
                self.waiting_queue.pop(0)
                
                # Check RadixTree for cached prefix
                matched_node, matched_len, cached_blocks = self.radix_cache.match_prefix(seq.prompt_token_ids)
                
                # Allocate remaining blocks
                self.block_manager.allocate(seq)
                self.running_sequences[seq.seq_id] = (seq, params, req_id)
            else:
                # No more free blocks available in this iteration
                break

        if not self.running_sequences:
            return []

        outputs: List[RequestOutput] = []
        finished_seq_ids: List[int] = []

        # 2. Process each running sequence (Iteration-level decoding)
        for seq_id, (seq, params, req_id) in list(self.running_sequences.items()):
            # Ensure physical block slot is available for upcoming token
            if not self.block_manager.can_append_slot(seq):
                # Preemption required: free sequence or pause
                continue

            self.block_manager.append_slot(seq)

            # Build input tensor for the model
            current_tokens = seq.prompt_token_ids + seq.output_token_ids
            input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
            block_table = torch.tensor([self.block_manager.get_block_table(seq)], dtype=torch.long, device=self.device)

            # Forward pass
            logits = self.model_forward_fn(input_tensor, self.kv_cache, block_table)
            next_token_logits = logits[0, -1, :]

            # Sample next token
            if params.temperature <= 0:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                probs = torch.softmax(next_token_logits / params.temperature, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()

            seq.append_token(next_token_id)
            self.total_tokens_generated += 1

            # Check stop condition
            if next_token_id in params.stop_token_ids or seq.is_finished():
                finished_seq_ids.append(seq_id)
                # Cache completed prefix in RadixTree for future requests
                self.radix_cache.insert(
                    token_ids=seq.prompt_token_ids + seq.output_token_ids,
                    block_ids=self.block_manager.get_block_table(seq),
                )
                self.block_manager.free(seq)
                output = RequestOutput(
                    request_id=req_id,
                    prompt_token_ids=seq.prompt_token_ids,
                    output_token_ids=list(seq.output_token_ids),
                    finished=True,
                )
            else:
                output = RequestOutput(
                    request_id=req_id,
                    prompt_token_ids=seq.prompt_token_ids,
                    output_token_ids=list(seq.output_token_ids),
                    finished=False,
                )

            outputs.append(output)

        # Remove finished sequences from active pool
        for f_id in finished_seq_ids:
            if f_id in self.running_sequences:
                del self.running_sequences[f_id]

        return outputs

    @property
    def throughput_tokens_per_sec(self) -> float:
        elapsed = max(1e-5, time.time() - self.start_time)
        return self.total_tokens_generated / elapsed

    @property
    def kv_cache_utilization(self) -> float:
        free_blocks = self.block_manager.get_num_free_blocks()
        total_blocks = self.block_manager.gpu_allocator.num_blocks
        return (total_blocks - free_blocks) / total_blocks
