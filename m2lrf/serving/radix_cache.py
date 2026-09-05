"""
M-2LRF Serving Engine: SGLang-Inspired RadixTree Prefix Cache.
Enables automatic KV-cache reuse for multi-turn dialogues, system prompts, and RAG contexts.
Implements tree-structured prefix matching with LRU cache eviction.
"""

from typing import Dict, List, Optional, Set, Tuple
import time


class RadixTreeNode:
    """Represents a node in the RadixTree prefix cache."""

    def __init__(
        self,
        token_ids: Tuple[int, ...],
        block_ids: List[int],
        parent: Optional["RadixTreeNode"] = None,
    ):
        self.token_ids = token_ids  # Compressed path of token IDs
        self.block_ids = list(block_ids)  # Physical block IDs allocated to this token chunk
        self.parent = parent
        self.children: Dict[int, "RadixTreeNode"] = {}  # keyed by first token ID of child branch
        self.ref_count = 0  # Number of active sequences currently referencing this node
        self.last_accessed: float = time.time()

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def touch(self) -> None:
        self.last_accessed = time.time()
        if self.parent:
            self.parent.touch()

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"RadixTreeNode(tokens={len(self.token_ids)}, blocks={len(self.block_ids)}, refs={self.ref_count})"


class RadixPrefixCache:
    """
    Adaptive RadixTree Prefix Cache.
    Maintains cached KV-cache physical blocks organized as a compressed prefix tree.
    Allows new requests to match existing KV-cache state and skip redundant prefill computation.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = RadixTreeNode(token_ids=(), block_ids=[])
        self.total_queries = 0
        self.total_hits = 0
        self.total_matched_tokens = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[Optional[RadixTreeNode], int, List[int]]:
        """
        Finds the longest cached prefix for the given prompt tokens.
        Args:
            token_ids: list of input token IDs
        Returns:
            (last_matched_node, matched_token_count, accumulated_block_ids)
        """
        self.total_queries += 1
        curr_node = self.root
        matched_tokens = 0
        accumulated_blocks: List[int] = []
        tokens_tuple = tuple(token_ids)

        idx = 0
        while idx < len(tokens_tuple):
            first_token = tokens_tuple[idx]
            if first_token not in curr_node.children:
                break
            
            child = curr_node.children[first_token]
            edge_len = len(child.token_ids)
            remaining_query_len = len(tokens_tuple) - idx

            # Check if query matches child edge
            match_len = 0
            for i in range(min(edge_len, remaining_query_len)):
                if child.token_ids[i] == tokens_tuple[idx + i]:
                    match_len += 1
                else:
                    break

            if match_len == edge_len:
                # Full edge match, advance down the tree
                idx += edge_len
                matched_tokens += edge_len
                accumulated_blocks.extend(child.block_ids)
                curr_node = child
                curr_node.touch()
            else:
                # Partial match along edge: partial blocks cannot be safely sliced without re-indexing
                # So we stop at the parent node boundary
                break

        if matched_tokens > 0:
            self.total_hits += 1
            self.total_matched_tokens += matched_tokens

        return (curr_node if curr_node is not self.root else None, matched_tokens, accumulated_blocks)

    def insert(
        self,
        token_ids: List[int],
        block_ids: List[int],
    ) -> RadixTreeNode:
        """
        Inserts a completed sequence of tokens and their associated physical blocks into the RadixTree.
        Splits existing nodes if the path branches in the middle of a compressed node.
        """
        tokens_tuple = tuple(token_ids)
        curr_node = self.root
        idx = 0
        block_idx = 0

        while idx < len(tokens_tuple):
            first_token = tokens_tuple[idx]
            
            if first_token not in curr_node.children:
                # Create a new leaf child branch
                remaining_tokens = tokens_tuple[idx:]
                remaining_blocks = block_ids[block_idx:]
                new_child = RadixTreeNode(
                    token_ids=remaining_tokens,
                    block_ids=remaining_blocks,
                    parent=curr_node,
                )
                curr_node.children[first_token] = new_child
                new_child.touch()
                return new_child

            child = curr_node.children[first_token]
            edge_tokens = child.token_ids
            edge_len = len(edge_tokens)
            
            # Find common prefix length with this edge
            common_len = 0
            while (
                common_len < edge_len
                and (idx + common_len) < len(tokens_tuple)
                and edge_tokens[common_len] == tokens_tuple[idx + common_len]
            ):
                common_len += 1

            if common_len == edge_len:
                # Full match of edge, continue deeper
                idx += edge_len
                block_idx += len(child.block_ids)
                curr_node = child
                curr_node.touch()
            else:
                # Split edge: create an intermediate split node
                split_tokens = edge_tokens[:common_len]
                # Proportionate block splitting
                split_num_blocks = (common_len + self.block_size - 1) // self.block_size
                split_blocks = child.block_ids[:split_num_blocks]

                split_node = RadixTreeNode(
                    token_ids=split_tokens,
                    block_ids=split_blocks,
                    parent=curr_node,
                )
                curr_node.children[first_token] = split_node

                # Update existing child to become child of split_node
                remaining_child_tokens = edge_tokens[common_len:]
                child.token_ids = remaining_child_tokens
                child.block_ids = child.block_ids[split_num_blocks:]
                child.parent = split_node
                split_node.children[remaining_child_tokens[0]] = child

                idx += common_len
                block_idx += split_num_blocks

                # If there are still new tokens remaining, add a branch under split_node
                if idx < len(tokens_tuple):
                    new_branch_tokens = tokens_tuple[idx:]
                    new_branch_blocks = block_ids[block_idx:]
                    new_branch_child = RadixTreeNode(
                        token_ids=new_branch_tokens,
                        block_ids=new_branch_blocks,
                        parent=split_node,
                    )
                    split_node.children[new_branch_tokens[0]] = new_branch_child
                    new_branch_child.touch()
                    return new_branch_child
                else:
                    split_node.touch()
                    return split_node

        return curr_node

    def evict_lru(self, num_blocks_to_free: int) -> List[int]:
        """
        Evicts unreferenced leaf nodes in Least-Recently-Used (LRU) order
        until at least `num_blocks_to_free` physical blocks are released.
        Returns the list of freed block IDs.
        """
        freed_blocks: List[int] = []

        while len(freed_blocks) < num_blocks_to_free:
            # Find all unreferenced leaves
            unreferenced_leaves: List[RadixTreeNode] = []
            self._collect_unreferenced_leaves(self.root, unreferenced_leaves)

            if not unreferenced_leaves:
                # No more unreferenced nodes can be evicted
                break

            # Sort by last_accessed ascending (oldest first)
            unreferenced_leaves.sort(key=lambda n: n.last_accessed)
            oldest_leaf = unreferenced_leaves[0]

            # Evict leaf
            freed_blocks.extend(oldest_leaf.block_ids)
            parent = oldest_leaf.parent
            if parent and oldest_leaf.token_ids:
                first_token = oldest_leaf.token_ids[0]
                if first_token in parent.children:
                    del parent.children[first_token]

        return freed_blocks

    def _collect_unreferenced_leaves(
        self, node: RadixTreeNode, result: List[RadixTreeNode]
    ) -> None:
        if node is not self.root and node.is_leaf() and node.ref_count == 0:
            result.append(node)
        for child in node.children.values():
            self._collect_unreferenced_leaves(child, result)

    def inc_ref(self, node: RadixTreeNode) -> None:
        """Increments ref_count on node and all its ancestors."""
        curr = node
        while curr and curr is not self.root:
            curr.ref_count += 1
            curr = curr.parent

    def dec_ref(self, node: RadixTreeNode) -> None:
        """Decrements ref_count on node and all its ancestors."""
        curr = node
        while curr and curr is not self.root:
            if curr.ref_count > 0:
                curr.ref_count -= 1
            curr = curr.parent
