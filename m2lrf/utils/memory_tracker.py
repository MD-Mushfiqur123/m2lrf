"""
M-2LRF Hardware & Memory Profiler (Torchtune-Inspired)
=======================================================
Tracks physical GPU VRAM, activation peaks, step throughput, and memory snapshots.
"""

from typing import Dict, Any, Optional
import time
import torch


class MemoryTracker:
    """
    Context manager and live profiler tracking GPU VRAM allocation and step latency.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = 0.0
        self.end_time = 0.0
        self.initial_allocated = 0
        self.peak_allocated = 0
        self.peak_reserved = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            self.initial_allocated = torch.cuda.memory_allocated(self.device)
        self.start_time = time.time()

    def stop(self) -> Dict[str, Any]:
        self.end_time = time.time()
        elapsed_s = self.end_time - self.start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
            self.peak_reserved = torch.cuda.max_memory_reserved(self.device)
            net_allocated = self.peak_allocated - self.initial_allocated
        else:
            net_allocated = 0
            self.peak_allocated = 0
            self.peak_reserved = 0

        return {
            "elapsed_s": round(elapsed_s, 4),
            "initial_allocated_mb": round(self.initial_allocated / (1024**2), 2),
            "peak_allocated_mb": round(self.peak_allocated / (1024**2), 2),
            "peak_reserved_mb": round(self.peak_reserved / (1024**2), 2),
            "net_allocated_mb": round(net_allocated / (1024**2), 2)
        }

    def summary(self, tokens_processed: Optional[int] = None) -> str:
        metrics = self.stop()
        text = "=" * 60 + "\n"
        text += "⚡ M-2LRF HARDWARE & MEMORY PROFILE SUMMARY\n"
        text += "=" * 60 + "\n"
        text += f"  • Elapsed Execution Time : {metrics['elapsed_s']} s\n"
        text += f"  • Peak Allocated VRAM    : {metrics['peak_allocated_mb']} MB\n"
        text += f"  • Peak Reserved VRAM     : {metrics['peak_reserved_mb']} MB\n"
        text += f"  • Net Activation Memory  : {metrics['net_allocated_mb']} MB\n"
        if tokens_processed:
            tok_per_sec = tokens_processed / max(1e-5, metrics['elapsed_s'])
            text += f"  • Throughput             : {tok_per_sec:.2f} tokens/s\n"
        text += "=" * 60
        return text
