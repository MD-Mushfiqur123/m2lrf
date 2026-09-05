"""
M-2LRF Supervised Fine-Tuning (SFT) Trainer (Axolotl-Inspired)
===============================================================
High-performance SFT trainer integrated with Fast Cross Entropy Loss,
gradient checkpointing, and sequence packing.
"""

from typing import Optional, Dict, Any, List
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m2lrf.kernels.fast_cross_entropy import fast_cross_entropy_loss


class M2LRFSFTTrainer:
    """
    Production-grade SFT training loop for 2-bit M-2LRF models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: Optional[int] = None,
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        device: Optional[str] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.lr = learning_rate
        self.batch_size = batch_size
        self.grad_accum = gradient_accumulation_steps
        self.epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Collect trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def train(self) -> Dict[str, Any]:
        """Executes the SFT training loop."""
        self.model.train()
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )

        total_steps = self.max_steps or (len(train_loader) * self.epochs // self.grad_accum)
        warmup_steps = int(total_steps * self.warmup_ratio)

        step = 0
        running_loss = 0.0
        step_losses = []
        t0 = time.time()

        print("=" * 80)
        print(f"🔥 [M-2LRF SFT Trainer] Starting fine-tuning run ({total_steps} total optimizer steps)")
        print(f"[*] Trainable Parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"[*] Learning Rate        : {self.lr} (Warmup: {warmup_steps} steps)")
        print(f"[*] Batch Accumulation   : {self.batch_size} x {self.grad_accum}")
        print("=" * 80)

        self.optimizer.zero_grad()

        for epoch in range(self.epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                logits = outputs.logits

                # Fused Fast Cross Entropy
                loss = fast_cross_entropy_loss(logits, labels, ignore_index=-100)
                loss = loss / self.grad_accum
                loss.backward()

                running_loss += loss.item() * self.grad_accum

                if (step + 1) % self.grad_accum == 0:
                    # Learning rate warmup schedule
                    opt_step = (step + 1) // self.grad_accum
                    if opt_step < warmup_steps:
                        curr_lr = self.lr * (opt_step / max(1, warmup_steps))
                    else:
                        progress = (opt_step - warmup_steps) / max(1, total_steps - warmup_steps)
                        curr_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = curr_lr

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    avg_loss = running_loss / self.grad_accum
                    step_losses.append(avg_loss)
                    running_loss = 0.0

                    if opt_step % self.logging_steps == 0 or opt_step == 1:
                        elapsed = time.time() - t0
                        print(f"  [Step {opt_step:04d}/{total_steps:04d}] Loss: {avg_loss:.4f} | LR: {curr_lr:.2e} | Elapsed: {elapsed:.1f}s")

                    if self.max_steps and opt_step >= self.max_steps:
                        break

                step += 1
            if self.max_steps and (step // self.grad_accum) >= self.max_steps:
                break

        print(f"[✓] Training successfully completed in {time.time() - t0:.2f}s")
        return {
            "total_steps": step // self.grad_accum,
            "final_loss": step_losses[-1] if step_losses else 0.0,
            "step_losses": step_losses,
            "total_time_s": time.time() - t0
        }
