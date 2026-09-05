# -*- coding: utf-8 -*-
r"""
M-2LRF Vector Codec: Vector Post-Training Quantization (VPTQ) & Codebook Engine.
Implementation of multi-dimensional vector quantization for sub-2-bit and 2-bit models.

Mathematical Principle:
Standard scalar quantization treats each weight element w_ij independently.
Vector Quantization (Shannon's Rate-Distortion Theory) proves that quantizing
vectors of dimension k > 1 strictly outperforms scalar quantization:
    lim_{k -> inf} D_k(R) = D_{Shannon}(R) < D_1(R)

Given weight matrix W in R^{m x n}, we reshape into vectors of dimension k:
    W_vec in R^{N x k}, where N = (m * n) / k

A codebook C in R^{K x k} contains K = 2^b centroid vectors.
Each vector v_i is mapped to its nearest codebook centroid index:
    idx_i = argmin_{j in {1, ..., K}} || v_i - C_j ||_2^2

Residual Vector Quantization (RVQ):
To achieve sub-2-bit compression with high fidelity, multi-stage RVQ is applied:
    Stage 1: v_i ≈ C^{(1)}_{idx1_i}
    Residual: r_i = v_i - C^{(1)}_{idx1_i}
    Stage 2: r_i ≈ C^{(2)}_{idx2_i}
Reconstruction:
    \hat{v}_i = C^{(1)}_{idx1_i} + C^{(2)}_{idx2_i}
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorCodebook(nn.Module):
    """
    Codebook container for k-dimensional vector quantization.
    """

    def __init__(
        self,
        vector_dim: int = 2,
        num_centroids: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.num_centroids = num_centroids
        self.bits_per_vector = int(math.log2(num_centroids))
        self.effective_bpp = self.bits_per_vector / vector_dim

        # Centroids matrix in R^{num_centroids x vector_dim}
        self.register_buffer(
            "centroids",
            torch.randn(num_centroids, vector_dim, dtype=dtype),
        )

    def initialize_from_data(self, vectors: torch.Tensor, num_iters: int = 10):
        """
        Initializes codebook using k-means clustering over sample vectors.
        """
        N, d = vectors.shape
        assert d == self.vector_dim, f"Dimension mismatch: expected {self.vector_dim}, got {d}"

        # Initialize with random subset
        indices = torch.randperm(N)[: self.num_centroids]
        self.centroids.copy_(vectors[indices].float())

        # K-means iterations
        for _ in range(num_iters):
            # Compute squared Euclidean distances: (N, K)
            # ||x - c||^2 = ||x||^2 - 2 x c^T + ||c||^2
            dists = torch.cdist(vectors.float(), self.centroids)  # (N, K)
            assignments = torch.argmin(dists, dim=-1)             # (N,)

            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            counts = torch.zeros(self.num_centroids, device=vectors.device)

            for k in range(self.num_centroids):
                mask = assignments == k
                if torch.any(mask):
                    new_centroids[k] = vectors[mask].float().mean(dim=0)
                    counts[k] = mask.sum()
                else:
                    # Re-seed empty cluster
                    rand_idx = torch.randint(0, N, (1,))
                    new_centroids[k] = vectors[rand_idx].float()

            self.centroids.copy_(new_centroids)

    def quantize(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps vectors to nearest centroid indices.

        Args:
            vectors: Tensor of shape (N, vector_dim)

        Returns:
            Tuple of (indices in {0, ..., K-1}, quantized_vectors)
        """
        dists = torch.cdist(vectors.float(), self.centroids)
        indices = torch.argmin(dists, dim=-1)
        quantized = self.centroids[indices].to(vectors.dtype)
        return indices, quantized

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Looks up centroids for given indices."""
        return self.centroids[indices]


class ResidualVectorQuantizer:
    """
    Multi-stage Residual Vector Quantizer (RVQ) for high-fidelity low-bit compression.
    """

    def __init__(
        self,
        vector_dim: int = 2,
        num_stages: int = 2,
        centroids_per_stage: int = 16,
    ):
        self.vector_dim = vector_dim
        self.num_stages = num_stages
        self.centroids_per_stage = centroids_per_stage
        self.codebooks = [
            VectorCodebook(vector_dim=vector_dim, num_centroids=centroids_per_stage)
            for _ in range(num_stages)
        ]
        # Total bits = num_stages * log2(centroids_per_stage) / vector_dim
        bits_per_stage = int(math.log2(centroids_per_stage))
        self.effective_bpp = (num_stages * bits_per_stage) / vector_dim

    def fit(self, weight_matrix: torch.Tensor, num_iters: int = 10):
        """
        Fits multi-stage RVQ codebooks on weight_matrix.
        """
        flat = weight_matrix.reshape(-1, self.vector_dim)
        residual = flat.clone()

        for stage_idx in range(self.num_stages):
            codebook = self.codebooks[stage_idx]
            codebook.centroids = codebook.centroids.to(weight_matrix.device)
            codebook.initialize_from_data(residual, num_iters=num_iters)
            _, quant = codebook.quantize(residual)
            residual = residual - quant

    def quantize(self, weight_matrix: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Quantizes weight matrix through all RVQ stages.

        Returns:
            Tuple of (list_of_stage_indices, reconstructed_matrix)
        """
        orig_shape = weight_matrix.shape
        flat = weight_matrix.reshape(-1, self.vector_dim)
        residual = flat.clone()
        all_indices = []
        reconstruction = torch.zeros_like(flat)

        for stage_idx in range(self.num_stages):
            codebook = self.codebooks[stage_idx]
            codebook.centroids = codebook.centroids.to(weight_matrix.device)
            indices, quant = codebook.quantize(residual)
            all_indices.append(indices)
            reconstruction.add_(quant)
            residual.sub_(quant)

        reconstructed_matrix = reconstruction.reshape(orig_shape)
        return all_indices, reconstructed_matrix

    def dequantize(
        self,
        all_indices: List[torch.Tensor],
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Reconstructs full matrix from stage indices.
        """
        reconstruction = None
        for stage_idx, indices in enumerate(all_indices):
            codebook = self.codebooks[stage_idx]
            q = codebook.dequantize(indices)
            if reconstruction is None:
                reconstruction = q.clone()
            else:
                reconstruction.add_(q)

        return reconstruction.reshape(target_shape)
