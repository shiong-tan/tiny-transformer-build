"""
Learning Rate Schedulers for Transformer Training.

This module implements learning rate schedules commonly used for transformers:
- Warmup + Cosine Decay (most common)
- Warmup + Linear Decay
- Warmup only

See Also:
    - theory.md Section 5: Learning Rate Schedules
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional


class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.

    This is the most common LR schedule for transformer training:
    1. Linear warmup: 0 → peak_lr over warmup_steps
    2. Cosine decay: peak_lr → min_lr over (total_steps - warmup_steps)

    Formula:
        if step < warmup_steps:
            lr = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate (after warmup)
        min_lr: Minimum learning rate (at end of training)
        last_epoch: Last epoch index (for resuming)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     peak_lr=1e-3,
        ...     min_lr=1e-5
        ... )
        >>> for epoch in range(num_epochs):
        ...     for batch in train_loader:
        ...         optimizer.step()
        ...         scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        # Validation
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be greater than warmup_steps ({warmup_steps})"
            )
        if peak_lr <= min_lr:
            raise ValueError(
                f"peak_lr ({peak_lr}) must be greater than min_lr ({min_lr})"
            )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate for current step.

        Returns:
            List of learning rates (one per parameter group)
        """
        # Current step (last_epoch is actually step count in PyTorch)
        step = self.last_epoch

        if step <= self.warmup_steps:
            # Linear warmup phase
            # lr = peak_lr * (step / warmup_steps)
            if self.warmup_steps == 0:
                lr = self.peak_lr
            else:
                lr = self.peak_lr * (step / self.warmup_steps)
        else:
            # Cosine decay phase
            # Compute progress through decay phase [0, 1]
            decay_steps = self.total_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_steps

            # Clamp progress to [0, 1] to handle steps beyond total_steps gracefully
            # This ensures lr stays at min_lr after training completes
            progress = min(progress, 1.0)

            # Cosine annealing formula
            # lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        # Return list of LRs (one per param group)
        return [lr for _ in self.optimizer.param_groups]


class WarmupLinearScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by linear decay.

    1. Linear warmup: 0 → peak_lr over warmup_steps
    2. Linear decay: peak_lr → min_lr over (total_steps - warmup_steps)

    Formula:
        if step < warmup_steps:
            lr = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = peak_lr - (peak_lr - min_lr) * progress

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate (after warmup)
        min_lr: Minimum learning rate (at end of training)
        last_epoch: Last epoch index (for resuming)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupLinearScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     peak_lr=1e-3,
        ...     min_lr=1e-5
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        # Validation
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be greater than warmup_steps ({warmup_steps})"
            )
        if peak_lr <= min_lr:
            raise ValueError(
                f"peak_lr ({peak_lr}) must be greater than min_lr ({min_lr})"
            )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate for current step.

        Returns:
            List of learning rates (one per parameter group)
        """
        step = self.last_epoch

        if step <= self.warmup_steps:
            # Linear warmup phase
            if self.warmup_steps == 0:
                lr = self.peak_lr
            else:
                lr = self.peak_lr * (step / self.warmup_steps)
        else:
            # Linear decay phase
            decay_steps = self.total_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_steps

            # Clamp progress to [0, 1] to handle steps beyond total_steps gracefully
            # This ensures lr stays at min_lr after training completes
            progress = min(progress, 1.0)

            # Linear interpolation from peak_lr to min_lr
            lr = self.peak_lr - (self.peak_lr - self.min_lr) * progress

        return [lr for _ in self.optimizer.param_groups]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0
) -> LRScheduler:
    """
    Factory function to create learning rate scheduler.

    Args:
        name: Scheduler name ("cosine" or "linear")
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate
        min_lr: Minimum learning rate

    Returns:
        LRScheduler instance

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_scheduler(
        ...     "cosine",
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     peak_lr=1e-3
        ... )
    """
    if name == "cosine":
        return WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, peak_lr, min_lr
        )
    elif name == "linear":
        return WarmupLinearScheduler(
            optimizer, warmup_steps, total_steps, peak_lr, min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}. Choose 'cosine' or 'linear'")


def plot_schedule(
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0
):
    """
    Plot learning rate schedule for visualization.

    Args:
        scheduler_name: "cosine" or "linear"
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate
        min_lr: Minimum learning rate

    Example:
        >>> plot_schedule("cosine", warmup_steps=1000, total_steps=10000, peak_lr=1e-3)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for plotting")
        return

    # Create dummy optimizer
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=peak_lr)

    # Create scheduler
    scheduler = get_scheduler(
        scheduler_name, optimizer, warmup_steps, total_steps, peak_lr, min_lr
    )

    # Collect learning rates
    lrs = []
    for step in range(total_steps + 1):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, linewidth=2)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='End of warmup')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(f'{scheduler_name.capitalize()} Schedule\n'
              f'(warmup={warmup_steps}, total={total_steps}, peak={peak_lr}, min={min_lr})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Learning Rate Schedulers - Demo")
    print("=" * 70)
    print()

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test cosine scheduler
    print("Testing Cosine Schedule:")
    print("-" * 70)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        peak_lr=1e-3,
        min_lr=1e-5
    )

    # Sample learning rates at key points
    sample_steps = [0, 50, 100, 250, 500, 750, 1000]
    for step in sample_steps:
        # Manually set step
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        print(f"Step {step:4d}: lr = {lr:.6f}")

    print()

    # Test linear scheduler
    print("Testing Linear Schedule:")
    print("-" * 70)
    scheduler = WarmupLinearScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        peak_lr=1e-3,
        min_lr=1e-5
    )

    for step in sample_steps:
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        print(f"Step {step:4d}: lr = {lr:.6f}")

    print()
    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
    print()
    print("To visualize schedules:")
    print("  plot_schedule('cosine', warmup_steps=1000, total_steps=10000, peak_lr=1e-3)")
