"""
M-2LRF Serving Engine: BlockSpaceManager and Physical/Logical Block Allocator.
Inspired by vLLM's virtual memory architecture for KV-cache management.
Treats GPU memory as paged memory blocks to eliminate internal and external fragmentation.
"""

from typing import Dict, List, Optional, Set, Tuple
import time


class PhysicalBlock:
    """Represents a physical allocation unit in GPU/CPU KV cache memory."""

    def __init__(self, block_id: int, block_size: int = 16, device: str = "cuda"):
        self.block_id = block_id
        self.block_size = block_size
        self.device = device
        self.ref_count = 0
        self.last_accessed: float = time.time()

    def touch(self) -> None:
        self.last_accessed = time.time()

    def __repr__(self) -> str:
        return f"PhysicalBlock(id={self.block_id}, size={self.block_size}, refs={self.ref_count})"


class BlockAllocator:
    """Manages the pool of free physical memory blocks."""

    def __init__(self, num_blocks: int, block_size: int = 16, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size, device=device)
            for i in range(num_blocks)
        ]
        self.all_blocks: Dict[int, PhysicalBlock] = {b.block_id: b for b in self.free_blocks}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise MemoryError("Out of KV-cache memory blocks! Free blocks exhausted.")
        block = self.free_blocks.pop(0)
        block.ref_count = 1
        block.touch()
        return block

    def free(self, block_id: int) -> None:
        if block_id not in self.all_blocks:
            raise ValueError(f"Invalid block_id {block_id}")
        block = self.all_blocks[block_id]
        if block.ref_count > 1:
            block.ref_count -= 1
        elif block.ref_count == 1:
            block.ref_count = 0
            self.free_blocks.append(block)
        else:
            raise RuntimeError(f"Double free detected on block_id {block_id}")


class SequenceStatus:
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED = "FINISHED"


class Sequence:
    """Represents an active generation request sequence."""

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        block_size: int = 16,
        max_tokens: int = 512,
    ):
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.status = SequenceStatus.WAITING

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_required_blocks(self) -> int:
        return (self.total_len + self.block_size - 1) // self.block_size

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


class BlockSpaceManager:
    """
    Manages logical-to-physical block tables for active sequences.
    Supports Copy-on-Write (CoW) for parallel beam search or multi-candidate sampling.
    """

    def __init__(self, num_gpu_blocks: int, block_size: int = 16, device: str = "cuda"):
        self.block_size = block_size
        self.device = device
        self.gpu_allocator = BlockAllocator(num_gpu_blocks, block_size, device=device)
        self.block_tables: Dict[int, List[int]] = {}  # seq_id -> list of physical block_ids

    def can_allocate(self, seq: Sequence) -> bool:
        """Checks if enough free GPU blocks exist to hold the sequence's prompt."""
        return self.gpu_allocator.num_free_blocks >= seq.num_required_blocks

    def allocate(self, seq: Sequence) -> None:
        """Allocates physical blocks for a new prompt sequence."""
        req_blocks = seq.num_required_blocks
        if self.gpu_allocator.num_free_blocks < req_blocks:
            raise MemoryError(f"Cannot allocate {req_blocks} blocks for sequence {seq.seq_id}")
        
        table: List[int] = []
        for _ in range(req_blocks):
            block = self.gpu_allocator.allocate()
            table.append(block.block_id)
        self.block_tables[seq.seq_id] = table
        seq.status = SequenceStatus.RUNNING

    def can_append_slot(self, seq: Sequence) -> bool:
        """Checks if a new slot can be allocated for the next token."""
        cur_blocks = len(self.block_tables.get(seq.seq_id, []))
        new_total_len = seq.total_len + 1
        new_req_blocks = (new_total_len + self.block_size - 1) // self.block_size
        if new_req_blocks > cur_blocks:
            return self.gpu_allocator.num_free_blocks > 0
        return True

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """
        Ensures a physical slot is available for the next token.
        If a new block is needed, allocates one.
        If the last block has ref_count > 1 (CoW scenario), copies the block to a new one.
        Returns (source_block_id, new_block_id) if CoW occurred, else None.
        """
        table = self.block_tables[seq.seq_id]
        cur_len = seq.total_len
        capacity = len(table) * self.block_size
        
        # Check if we exceeded the current allocated block capacity
        if cur_len > capacity:
            new_block = self.gpu_allocator.allocate()
            table.append(new_block.block_id)
            return None
        
        # Check Copy-on-Write: if the last block is shared with other sequences
        last_block_id = table[-1]
        last_block = self.gpu_allocator.all_blocks[last_block_id]
        if last_block.ref_count > 1:
            # Fork/CoW triggered: allocate a private physical block
            new_block = self.gpu_allocator.allocate()
            self.gpu_allocator.free(last_block_id)
            table[-1] = new_block.block_id
            return (last_block_id, new_block.block_id)

        last_block.touch()
        return None

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        """
        Forks a child sequence sharing the parent's block table via Copy-on-Write.
        Increments the ref_count on all physical blocks.
        """
        if parent_seq.seq_id not in self.block_tables:
            raise KeyError(f"Parent sequence {parent_seq.seq_id} has no allocated block table.")
        
        parent_table = self.block_tables[parent_seq.seq_id]
        child_table = list(parent_table)
        for block_id in child_table:
            block = self.gpu_allocator.all_blocks[block_id]
            block.ref_count += 1
            block.touch()
        
        self.block_tables[child_seq.seq_id] = child_table
        child_seq.status = SequenceStatus.RUNNING

    def free(self, seq: Sequence) -> None:
        """Frees all physical blocks mapped to the sequence."""
        if seq.seq_id in self.block_tables:
            table = self.block_tables.pop(seq.seq_id)
            for block_id in table:
                self.gpu_allocator.free(block_id)
        seq.status = SequenceStatus.FINISHED

    def get_block_table(self, seq: Sequence) -> List[int]:
        """Returns the physical block IDs allocated to the sequence."""
        return self.block_tables.get(seq.seq_id, [])

    def get_num_free_blocks(self) -> int:
        return self.gpu_allocator.num_free_blocks
