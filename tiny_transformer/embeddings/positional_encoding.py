"""
Positional Encoding for Transformer Models.

This module implements both sinusoidal (fixed) and learned positional encodings.
Positional encodings provide position information to the transformer, which otherwise
treats sequences as unordered sets.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Uses fixed sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This allows the model to easily learn to attend by relative positions.

    Args:
        d_model: Embedding dimension (must be even)
        max_len: Maximum sequence length to pre-compute (default: 5000)
        dropout: Dropout probability (default: 0.0)

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> pe = SinusoidalPositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(32, 128, 512)  # (B, T, d_model)
        >>> x_with_pos = pe(x)  # (32, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0
    ):
        super().__init__()

        assert d_model % 2 == 0, \
            f"d_model must be even for sinusoidal encoding, got {d_model}"

        self.d_model = d_model
        self.max_len = max_len

        # Dropout (applied after adding positional encoding)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-compute positional encodings: (max_len, d_model)
        pe = self._create_sinusoidal_encoding(max_len, d_model)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    @staticmethod
    def _create_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding matrix.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding matrix of shape (max_len, d_model)
        """
        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1).float()

        # Create dimension indices: [0, 2, 4, ..., d_model-2]
        # These are the even indices for the sine function
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)

        # Even indices: sine
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd indices: cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added, same shape as input

        Shape flow:
            (B, T, d_model) input
            → Add positional encoding (broadcast) → (B, T, d_model)
            → Optional dropout → (B, T, d_model)
        """
        # Validate input
        assert x.dim() == 3, \
            f"Expected 3D input (B, T, d_model), got shape {x.shape}"
        assert x.size(-1) == self.d_model, \
            f"Input dimension {x.size(-1)} != d_model {self.d_model}"

        batch_size, seq_len, d_model = x.shape

        # Check sequence length
        assert seq_len <= self.max_len, \
            f"Sequence length {seq_len} > max_len {self.max_len}"

        # Add positional encoding
        # self.pe shape: (max_len, d_model)
        # We take [:seq_len] and broadcast across batch
        # Broadcasting: (seq_len, d_model) → (B, seq_len, d_model)
        x = x + self.pe[:seq_len]

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"d_model={self.d_model}, max_len={self.max_len}, "
                f"dropout={self.dropout.p if self.dropout else 0}")


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Instead of using fixed sinusoidal patterns, this learns a separate embedding
    for each position. Used in BERT, GPT-2, and many modern transformers.

    Pros:
        - Can learn task-specific position representations
        - Often performs better on specific tasks

    Cons:
        - Cannot extrapolate to longer sequences than seen during training
        - Requires more parameters

    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        dropout: Dropout probability (default: 0.0)

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> pe = LearnedPositionalEmbedding(max_len=1024, d_model=512)
        >>> x = torch.randn(32, 128, 512)  # (B, T, d_model)
        >>> x_with_pos = pe(x)  # (32, 128, 512)
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model

        # Learnable position embeddings: (max_len, d_model)
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional embeddings added

        Shape flow:
            (B, T, d_model) input
            → Create position indices (T,)
            → Embed positions (T, d_model)
            → Add to input (broadcast) → (B, T, d_model)
            → Optional dropout → (B, T, d_model)
        """
        # Validate input
        assert x.dim() == 3, \
            f"Expected 3D input (B, T, d_model), got shape {x.shape}"
        assert x.size(-1) == self.d_model, \
            f"Input dimension {x.size(-1)} != d_model {self.d_model}"

        batch_size, seq_len, d_model = x.shape

        # Check sequence length
        assert seq_len <= self.max_len, \
            f"Sequence length {seq_len} > max_len {self.max_len}. " \
            f"Learned embeddings cannot extrapolate to unseen positions."

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = torch.arange(seq_len, device=x.device)

        # Embed positions: (seq_len,) → (seq_len, d_model)
        position_embeddings = self.position_embeddings(positions)

        # Add to input (broadcasts across batch dimension)
        # (B, T, d_model) + (T, d_model) → (B, T, d_model)
        x = x + position_embeddings

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"max_len={self.max_len}, d_model={self.d_model}, "
                f"dropout={self.dropout.p if self.dropout else 0}")


def test_sinusoidal_encoding():
    """Test SinusoidalPositionalEncoding."""
    print("Testing SinusoidalPositionalEncoding...")

    d_model = 512
    max_len = 1000
    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # Test 1: Shape
    x = torch.randn(32, 128, d_model)
    output = pe(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape}")

    # Test 2: Positional encoding matrix shape
    assert pe.pe.shape == (max_len, d_model)
    print(f"✓ PE matrix shape: {pe.pe.shape}")

    # Test 3: Sine and cosine pattern
    # Even indices should be sine, odd should be cosine
    # Check that values are in [-1, 1]
    assert pe.pe.min() >= -1.1 and pe.pe.max() <= 1.1
    print(f"✓ PE values in valid range: [{pe.pe.min():.2f}, {pe.pe.max():.2f}]")

    # Test 4: Deterministic (no randomness)
    output1 = pe(x)
    output2 = pe(x)
    assert torch.equal(output1, output2)
    print(f"✓ Deterministic output")

    # Test 5: Different sequence lengths
    for seq_len in [10, 50, 256, 512]:
        x = torch.randn(16, seq_len, d_model)
        output = pe(x)
        assert output.shape == (16, seq_len, d_model)
    print(f"✓ Works with different sequence lengths")

    print("All SinusoidalPositionalEncoding tests passed! ✓\n")


def test_learned_embedding():
    """Test LearnedPositionalEmbedding."""
    print("Testing LearnedPositionalEmbedding...")

    d_model = 512
    max_len = 1024
    pe = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model, dropout=0.0)

    # Test 1: Shape
    x = torch.randn(32, 128, d_model)
    output = pe(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape}")

    # Test 2: Parameters are learnable
    assert pe.position_embeddings.weight.requires_grad
    print(f"✓ Parameters are learnable")

    # Test 3: Different positions get different embeddings
    pe.eval()
    x = torch.zeros(1, 10, d_model)
    output = pe(x)
    # Each position should be different (not all zeros)
    assert not torch.allclose(output[0, 0], output[0, 1])
    print(f"✓ Different positions have different embeddings")

    # Test 4: Exceeding max_len raises error
    x_long = torch.randn(16, max_len + 1, d_model)
    try:
        pe(x_long)
        assert False, "Should raise error for seq_len > max_len"
    except AssertionError as e:
        print(f"✓ Correctly raises error for seq_len > max_len")

    print("All LearnedPositionalEmbedding tests passed! ✓\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Positional Encoding - Test Suite")
    print("=" * 70)
    print()

    test_sinusoidal_encoding()
    test_learned_embedding()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
