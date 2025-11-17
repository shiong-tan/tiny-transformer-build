"""
Embedding components for transformer models.

This package provides:
- TokenEmbedding: Convert token IDs to continuous vectors with scaling
- SinusoidalPositionalEncoding: Fixed positional encoding using sin/cos
- LearnedPositionalEmbedding: Trainable positional embeddings
- TransformerEmbedding: Complete embedding layer combining tokens + positions
"""

from tiny_transformer.embeddings.token_embedding import TokenEmbedding
from tiny_transformer.embeddings.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding
)
from tiny_transformer.embeddings.combined_embedding import TransformerEmbedding

__all__ = [
    'TokenEmbedding',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEmbedding',
    'TransformerEmbedding',
]
