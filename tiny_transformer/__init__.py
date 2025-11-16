"""
Tiny Transformer: Educational Implementation of Transformer Models

A production-grade educational repository for learning to build transformer
models from scratch with best engineering practices.

Main modules:
- attention: Core attention mechanism
- config: Configuration management
- utils: Utilities for shape checking, visualization, etc.
"""

__version__ = "0.1.0"

from tiny_transformer.attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    Attention,
)

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
    # Attention
    "scaled_dot_product_attention",
    "create_causal_mask",
    "Attention",
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "SamplingConfig",
    "ExperimentConfig",
    "get_tiny_config",
    "get_small_config",
    "get_medium_config",
]
