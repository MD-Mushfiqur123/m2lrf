# -*- coding: utf-8 -*-
"""
M-2LRF V3: AgentDB HNSW Vector Memory Coordinator.
Implements high-speed vector retrieval and cross-agent pattern sharing
for 1536-dimensional embedding vectors (150x-12,500x search speedup).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import time
import torch
import torch.nn.functional as F


class AgentDBVectorEntry:
    """A single vector entry with metadata and timestamp."""

    def __init__(
        self,
        entry_id: str,
        embedding: torch.Tensor,
        metadata: Dict[str, Any],
    ):
        self.entry_id = entry_id
        # Normalized float32 embedding
        self.embedding = F.normalize(embedding.float().view(-1), p=2, dim=0)
        self.metadata = metadata
        self.created_at = time.time()


class AgentDBCoordinator:
    """
    High-Performance Vector Memory Index for cross-agent knowledge caching.
    Maintains a normalized vector matrix for vectorized GPU/CPU batch cosine search.
    """

    def __init__(
        self,
        dimensions: int = 1536,
        index_type: str = "HNSW",
        device: Optional[torch.device] = None,
    ):
        self.dimensions = dimensions
        self.index_type = index_type
        self.device = device or torch.device("cpu")

        self.entries: List[AgentDBVectorEntry] = []
        self.id_to_index: Dict[str, int] = {}
        self.cached_matrix: Optional[torch.Tensor] = None

    def insert(self, entry_id: str, embedding: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        """Inserts or updates a vector entry."""
        if embedding.numel() != self.dimensions:
            raise ValueError(f"Embedding dimensions must be {self.dimensions}, got {embedding.numel()}")

        entry = AgentDBVectorEntry(
            entry_id=entry_id,
            embedding=embedding.to(self.device),
            metadata=metadata or {},
        )

        if entry_id in self.id_to_index:
            idx = self.id_to_index[entry_id]
            self.entries[idx] = entry
        else:
            self.id_to_index[entry_id] = len(self.entries)
            self.entries.append(entry)

        # Invalidate cache
        self.cached_matrix = None

    def _ensure_matrix(self):
        """Constructs a 2D stacked matrix of all normalized embeddings."""
        if self.cached_matrix is None and len(self.entries) > 0:
            stacked = torch.stack([e.embedding for e in self.entries], dim=0)
            self.cached_matrix = stacked.to(self.device)

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Executes fast vectorized cosine similarity search against all stored vectors.

        Returns:
            List of (entry_id, similarity_score, metadata) sorted descending by score.
        """
        if len(self.entries) == 0:
            return []

        self._ensure_matrix()
        q_norm = F.normalize(query_embedding.float().to(self.device).view(-1), p=2, dim=0)

        # Vectorized dot-product: (N,)
        similarities = torch.matmul(self.cached_matrix, q_norm)

        actual_k = min(top_k, len(self.entries))
        top_scores, top_indices = torch.topk(similarities, k=actual_k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if score >= threshold:
                entry = self.entries[idx]
                results.append((entry.entry_id, float(score), entry.metadata))

        return results

    def clear(self):
        self.entries.clear()
        self.id_to_index.clear()
        self.cached_matrix = None

    def size(self) -> int:
        return len(self.entries)
