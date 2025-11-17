"""
Sampling and text generation package.

This package provides:
- Sampling strategies: greedy, temperature, top-k, top-p
- TextGenerator: High-level generation interface
- GeneratorConfig: Generation configuration
"""

from tiny_transformer.sampling.strategies import (
    greedy_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    combined_sample
)
from tiny_transformer.sampling.generator import (
    TextGenerator,
    GeneratorConfig
)

__all__ = [
    # Sampling strategies
    'greedy_sample',
    'temperature_sample',
    'top_k_sample',
    'top_p_sample',
    'combined_sample',
    # Generator
    'TextGenerator',
    'GeneratorConfig',
]
