"""
Training package for TinyTransformer models.

This package provides:
- TextDataset: Language modeling dataset with sliding windows
- CharTokenizer: Simple character-level tokenizer
- WarmupCosineScheduler: LR schedule with warmup and cosine decay
- WarmupLinearScheduler: LR schedule with warmup and linear decay
- Trainer: Complete training loop with checkpointing
- TrainerConfig: Training configuration
- Training utilities: perplexity, gradient clipping, etc.
"""

from tiny_transformer.training.dataset import (
    TextDataset,
    CharTokenizer,
    create_data_loaders
)
from tiny_transformer.training.scheduler import (
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    get_scheduler
)
from tiny_transformer.training.trainer import (
    Trainer,
    TrainerConfig
)
from tiny_transformer.training.utils import (
    compute_perplexity,
    get_gradient_norm,
    clip_gradient_norm,
    count_model_parameters,
    estimate_model_memory,
    set_seed,
    get_device,
    AverageMeter
)

__all__ = [
    # Dataset
    'TextDataset',
    'CharTokenizer',
    'create_data_loaders',
    # Schedulers
    'WarmupCosineScheduler',
    'WarmupLinearScheduler',
    'get_scheduler',
    # Trainer
    'Trainer',
    'TrainerConfig',
    # Utilities
    'compute_perplexity',
    'get_gradient_norm',
    'clip_gradient_norm',
    'count_model_parameters',
    'estimate_model_memory',
    'set_seed',
    'get_device',
    'AverageMeter',
]
