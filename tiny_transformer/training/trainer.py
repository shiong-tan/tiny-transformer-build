"""
Transformer Training Loop.

This module implements a complete trainer for transformer language models:
- Training loop with AdamW optimizer
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Evaluation and metrics logging

See Also:
    - theory.md Section 8: Training Loop
    - theory.md Section 7: Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
import time
import json

from tiny_transformer.training.utils import (
    compute_perplexity,
    clip_gradient_norm,
    AverageMeter,
    format_time
)


@dataclass
class TrainerConfig:
    """
    Configuration for Trainer.

    Args:
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        grad_clip: Maximum gradient norm (None = no clipping)
        log_interval: Log metrics every N steps
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps (None = no saving)
        max_steps: Maximum training steps (None = train for num_epochs)
        num_epochs: Number of training epochs (if max_steps is None)
        warmup_steps: Number of warmup steps for LR schedule
        device: Training device (cuda/mps/cpu)
        checkpoint_dir: Directory to save checkpoints

    Example:
        >>> config = TrainerConfig(
        ...     learning_rate=1e-3,
        ...     num_epochs=10,
        ...     warmup_steps=1000
        ... )
    """
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    grad_clip: Optional[float] = 1.0
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: Optional[int] = 1000
    max_steps: Optional[int] = None
    num_epochs: int = 10
    warmup_steps: int = 100
    device: str = 'cpu'
    checkpoint_dir: str = 'checkpoints'


class Trainer:
    """
    Complete training loop for transformer models.

    Args:
        model: TinyTransformerLM model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Trainer configuration
        scheduler: Optional custom LR scheduler

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> from tiny_transformer.training import Trainer, TrainerConfig
        >>>
        >>> model = TinyTransformerLM(vocab_size=1000, d_model=128, ...)
        >>> trainer = Trainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        scheduler: Optional[LRScheduler] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config if config else TrainerConfig()

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas
        )

        # Setup learning rate scheduler
        if scheduler is None and self.config.max_steps:
            # Create default warmup cosine scheduler
            from tiny_transformer.training.scheduler import WarmupCosineScheduler
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
                peak_lr=self.config.learning_rate,
                min_lr=self.config.learning_rate * 0.1
            )
        else:
            self.scheduler = scheduler

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Dictionary of final metrics

        Raises:
            RuntimeError: If training fails
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Config: {self.config}")
        print("=" * 70)
        print()

        start_time = time.time()

        try:
            # Validate data loader
            if len(self.train_loader) == 0:
                raise RuntimeError(
                    "train_loader is empty. Cannot start training. "
                    "Verify your dataset has samples and batch_size configuration."
                )

            # Determine total steps
            if self.config.max_steps:
                total_steps = self.config.max_steps
                num_epochs = (total_steps // len(self.train_loader)) + 1
            else:
                num_epochs = self.config.num_epochs
                total_steps = num_epochs * len(self.train_loader)

            # Training loop
            for epoch in range(num_epochs):
                self.epoch = epoch

                # Train one epoch
                train_metrics = self.train_one_epoch()

                # Check if we've reached max steps
                if self.config.max_steps and self.step >= self.config.max_steps:
                    print(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                    break

            # Final evaluation
            if self.val_loader:
                print("\nFinal Evaluation:")
                final_metrics = self.evaluate()
            else:
                final_metrics = train_metrics

            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print(f"Training Complete! Time: {format_time(elapsed)}")
            print("=" * 70)

            return final_metrics

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            return {}
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip is not None:
                grad_norm = clip_gradient_norm(self.model, self.config.grad_clip)
            else:
                from tiny_transformer.training.utils import get_gradient_norm
                grad_norm = get_gradient_norm(self.model)

            # Optimizer step
            self.optimizer.step()

            # Learning rate schedule step
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            loss_meter.update(loss.item())
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                ppl = compute_perplexity(loss.item())

                print(
                    f"Epoch {self.epoch} | Step {self.step} | "
                    f"Loss: {loss.item():.4f} | PPL: {ppl:.2f} | "
                    f"LR: {current_lr:.2e} | Grad: {grad_norm:.2f}"
                )

            # Evaluation
            if self.val_loader and self.step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt', is_best=True)

                self.model.train()  # Back to training mode

            # Checkpointing
            if self.config.save_interval and self.step % self.config.save_interval == 0:
                self.save_checkpoint(f'step_{self.step}.pt')

            # Check max steps
            if self.config.max_steps and self.step >= self.config.max_steps:
                break

        return {'loss': loss_meter.avg, 'perplexity': compute_perplexity(loss_meter.avg)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        loss_meter = AverageMeter()

        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Compute loss (ignore padding tokens if specified)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.config.pad_token_id if self.config.pad_token_id is not None else -100
            )

            loss_meter.update(loss.item())

        metrics = {
            'loss': loss_meter.avg,
            'perplexity': compute_perplexity(loss_meter.avg)
        }

        print(
            f"  Validation | Loss: {metrics['loss']:.4f} | "
            f"PPL: {metrics['perplexity']:.2f}"
        )

        return metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

        if is_best:
            print(f"  ✓ New best model! Val loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from step {self.step}, epoch {self.epoch}")


if __name__ == "__main__":
    print("=" * 70)
    print("Trainer - Demo")
    print("=" * 70)
    print()

    # This is a minimal example showing the API
    # For a complete example, see the notebook or exercises

    from tiny_transformer.model import TinyTransformerLM, get_model_config
    from tiny_transformer.training.dataset import TextDataset, create_data_loaders

    # Create tiny model
    config = get_model_config("tiny")
    model = TinyTransformerLM(vocab_size=100, **config)

    # Create dummy dataset
    train_tokens = list(range(1000))
    val_tokens = list(range(1000, 1200))

    train_loader, val_loader = create_data_loaders(
        train_tokens,
        val_tokens,
        seq_len=32,
        batch_size=8
    )

    # Create trainer
    trainer_config = TrainerConfig(
        learning_rate=1e-3,
        num_epochs=2,
        warmup_steps=10,
        log_interval=5,
        eval_interval=10,
        save_interval=None,  # Don't save for demo
        device='cpu'
    )

    trainer = Trainer(model, train_loader, val_loader, trainer_config)

    print("Trainer initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("To train: trainer.train()")
    print()
    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
