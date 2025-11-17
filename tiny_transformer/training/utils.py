"""
Training Utilities.

This module provides utility functions for training transformers:
- Perplexity computation
- Gradient statistics
- Memory estimation
- Training debugging helpers

See Also:
    - theory.md Section 3: Loss Functions
    - theory.md Section 6: Gradient Clipping
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss) is a common metric for language models.
    Lower perplexity = better model.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value

    Example:
        >>> loss = 2.3
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl:.2f}")
        Perplexity: 9.97
    """
    return math.exp(loss)


def get_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.

    Args:
        model: PyTorch model

    Returns:
        Total gradient norm (L2 norm)

    Example:
        >>> grad_norm = get_gradient_norm(model)
        >>> if grad_norm > 10.0:
        ...     print("Warning: Large gradients!")
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradient_norm(
    model: nn.Module,
    max_norm: float
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        Total norm before clipping

    Example:
        >>> # Clip gradients to prevent explosion
        >>> norm = clip_gradient_norm(model, max_norm=1.0)
        >>> print(f"Gradient norm: {norm:.2f}")
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    ).item()


def count_model_parameters(
    model: nn.Module,
    trainable_only: bool = False
) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters

    Example:
        >>> total = count_model_parameters(model)
        >>> trainable = count_model_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def estimate_model_memory(
    model: nn.Module,
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    Estimate model memory requirements in MB.

    Args:
        model: PyTorch model
        dtype: Data type (float32 or float16)

    Returns:
        Dictionary with memory estimates

    Example:
        >>> mem = estimate_model_memory(model)
        >>> print(f"Model: {mem['parameters']:.1f} MB")
        >>> print(f"Training: {mem['total_training']:.1f} MB")
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2
    num_params = count_model_parameters(model)

    param_memory = num_params * bytes_per_param / (1024 ** 2)  # MB
    grad_memory = param_memory  # Same size as parameters
    optimizer_memory = 2 * param_memory  # AdamW has 2× states

    return {
        'parameters': param_memory,
        'gradients': grad_memory,
        'optimizer': optimizer_memory,
        'total_training': param_memory + grad_memory + optimizer_memory,
        'inference_only': param_memory
    }


def log_gradient_stats(
    model: nn.Module,
    step: int,
    log_fn: Optional[callable] = None
) -> Dict[str, float]:
    """
    Log gradient statistics for debugging.

    Args:
        model: PyTorch model
        step: Current training step
        log_fn: Optional logging function (default: print)

    Returns:
        Dictionary of gradient statistics

    Example:
        >>> stats = log_gradient_stats(model, step=100)
        >>> # Prints gradient statistics
    """
    if log_fn is None:
        log_fn = print

    # Collect gradient norms per layer
    layer_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norms[name] = param.grad.norm().item()

    # Compute statistics
    if layer_norms:
        norms = list(layer_norms.values())
        stats = {
            'total_norm': sum(n**2 for n in norms) ** 0.5,
            'max_norm': max(norms),
            'min_norm': min(norms),
            'mean_norm': sum(norms) / len(norms),
        }

        log_fn(
            f"Step {step} | Grad norm: total={stats['total_norm']:.4f}, "
            f"max={stats['max_norm']:.4f}, mean={stats['mean_norm']:.4f}"
        )

        return stats
    else:
        return {}


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed

    Example:
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> print(format_time(3661))
        1h 1m 1s
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


class AverageMeter:
    """
    Tracks running average of a metric.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in data_loader:
        ...     loss = train_step(batch)
        ...     loss_meter.update(loss)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update with new value.

        Args:
            val: New value
            n: Weight/count for this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


if __name__ == "__main__":
    print("=" * 70)
    print("Training Utilities - Demo")
    print("=" * 70)
    print()

    # Test perplexity
    loss = 2.3
    ppl = compute_perplexity(loss)
    print(f"Loss: {loss:.2f} → Perplexity: {ppl:.2f}")
    print()

    # Test AverageMeter
    meter = AverageMeter()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        meter.update(v)
    print(f"Average of {values}: {meter.avg:.2f}")
    print()

    # Test time formatting
    test_times = [30, 90, 3661, 7322]
    for t in test_times:
        print(f"{t:5d}s → {format_time(t)}")
    print()

    # Test device detection
    device = get_device()
    print(f"Best available device: {device}")
    print()

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
