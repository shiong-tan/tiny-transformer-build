"""
Tiny Transformer: Educational Implementation of Transformer Models

A production-grade educational repository for learning to build transformer
models from scratch with best engineering practices.

Main modules:
- attention: Core attention mechanism (Module 02)
- feedforward: Position-wise feed-forward networks (Module 03)
- transformer_block: Complete transformer block (Module 03)
- embeddings: Token and positional embeddings (Module 04)
- model: Complete TinyTransformerLM model (Module 05)
- training: Training loop, datasets, schedulers (Module 06)
- sampling: Text generation and sampling strategies (Module 07)
- config: Configuration management
- utils: Utilities for shape checking, visualization, etc.
"""

__version__ = "0.1.0"

# Core Components
from tiny_transformer.attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    Attention,
)

from tiny_transformer.multi_head import MultiHeadAttention

from tiny_transformer.feedforward import FeedForward

from tiny_transformer.transformer_block import TransformerBlock

# Embeddings
from tiny_transformer.embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    TransformerEmbedding,
)

# Complete Model
from tiny_transformer.model import (
    TinyTransformerLM,
    get_model_config,
)

# Training
from tiny_transformer.training import (
    TextDataset,
    CharTokenizer,
    create_data_loaders,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    get_scheduler,
    Trainer,
    TrainerConfig,
    compute_perplexity,
    get_gradient_norm,
    clip_gradient_norm,
    set_seed,
    get_device,
    AverageMeter,
)

# Sampling & Generation
from tiny_transformer.sampling import (
    greedy_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    combined_sample,
    TextGenerator,
    GeneratorConfig,
)

# Configuration (legacy)
from tiny_transformer.config import (
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    ExperimentConfig,
    get_tiny_config,
    get_small_config,
    get_medium_config,
)

__all__ = [
    # Attention (Module 02)
    "scaled_dot_product_attention",
    "create_causal_mask",
    "Attention",
    "MultiHeadAttention",
    # Feed-Forward (Module 03)
    "FeedForward",
    # Transformer Block (Module 03)
    "TransformerBlock",
    # Embeddings (Module 04)
    "TokenEmbedding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEmbedding",
    "TransformerEmbedding",
    # Complete Model (Module 05)
    "TinyTransformerLM",
    "get_model_config",
    # Training (Module 06)
    "TextDataset",
    "CharTokenizer",
    "create_data_loaders",
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "get_scheduler",
    "Trainer",
    "TrainerConfig",
    "compute_perplexity",
    "get_gradient_norm",
    "clip_gradient_norm",
    "set_seed",
    "get_device",
    "AverageMeter",
    # Sampling (Module 07)
    "greedy_sample",
    "temperature_sample",
    "top_k_sample",
    "top_p_sample",
    "combined_sample",
    "TextGenerator",
    "GeneratorConfig",
    # Legacy Configuration
    "ModelConfig",
    "TrainingConfig",
    "SamplingConfig",
    "ExperimentConfig",
    "get_tiny_config",
    "get_small_config",
    "get_medium_config",
]
