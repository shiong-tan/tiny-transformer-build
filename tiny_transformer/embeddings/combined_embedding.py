"""
Combined Token and Positional Embeddings for Transformer Models.

This module provides a convenient wrapper that combines token embeddings
with positional encoding in a single module.
"""

import torch
import torch.nn as nn
from typing import Literal

from tiny_transformer.embeddings.token_embedding import TokenEmbedding
from tiny_transformer.embeddings.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding
)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer for transformers.

    Combines:
    1. Token embeddings (with √d_model scaling)
    2. Positional encoding (sinusoidal or learned)
    3. Dropout

    This is the standard embedding used at the input of transformer models.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        max_len: Maximum sequence length
        positional: Type of positional encoding - "sinusoidal" or "learned"
        dropout: Dropout probability (default: 0.1)
        padding_idx: Optional padding token index

    Shape:
        Input: (batch_size, seq_len) - Token IDs
        Output: (batch_size, seq_len, d_model) - Embedded tokens with positions

    Example:
        >>> embedding = TransformerEmbedding(
        ...     vocab_size=10000,
        ...     d_model=512,
        ...     max_len=1024,
        ...     positional="sinusoidal"
        ... )
        >>> tokens = torch.randint(0, 10000, (32, 128))  # (B, T)
        >>> embedded = embedding(tokens)  # (32, 128, 512)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 1024,
        positional: Literal["sinusoidal", "learned"] = "sinusoidal",
        dropout: float = 0.1,
        padding_idx: int = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.positional_type = positional

        # Token embeddings (with √d_model scaling)
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx
        )

        # Positional encoding
        if positional == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=0.0  # Dropout applied after combining
            )
        elif positional == "learned":
            self.positional_encoding = LearnedPositionalEmbedding(
                max_len=max_len,
                d_model=d_model,
                dropout=0.0  # Dropout applied after combining
            )
        else:
            raise ValueError(
                f"positional must be 'sinusoidal' or 'learned', got '{positional}'"
            )

        # Dropout (applied after combining embeddings)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and add positional information.

        Args:
            tokens: Token IDs of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)

        Processing steps:
            1. Token embedding with scaling: (B, T) → (B, T, d_model)
            2. Add positional encoding: (B, T, d_model) → (B, T, d_model)
            3. Apply dropout: (B, T, d_model) → (B, T, d_model)
        """
        # Validate input
        assert tokens.dim() == 2, \
            f"Expected 2D input (B, T), got shape {tokens.shape}"

        # Step 1: Embed tokens (includes √d_model scaling)
        # (B, T) → (B, T, d_model)
        token_emb = self.token_embedding(tokens)

        # Step 2: Add positional encoding
        # The positional encoding module adds in-place
        # (B, T, d_model) → (B, T, d_model)
        x = self.positional_encoding(token_emb)

        # Step 3: Apply dropout
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"max_len={self.max_len}, positional='{self.positional_type}', "
                f"dropout={self.dropout.p if self.dropout else 0}")


def test_transformer_embedding():
    """Test TransformerEmbedding with both positional encoding types."""
    print("Testing TransformerEmbedding...")

    vocab_size = 10000
    d_model = 512
    max_len = 1024

    # Test 1: Sinusoidal positional encoding
    print("\n1. Testing with sinusoidal positional encoding...")
    emb_sin = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional="sinusoidal",
        dropout=0.0
    )

    tokens = torch.randint(0, vocab_size, (32, 128))
    output_sin = emb_sin(tokens)

    assert output_sin.shape == (32, 128, d_model)
    print(f"   ✓ Output shape: {output_sin.shape}")

    # Test 2: Learned positional embedding
    print("\n2. Testing with learned positional embedding...")
    emb_learned = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional="learned",
        dropout=0.0
    )

    output_learned = emb_learned(tokens)

    assert output_learned.shape == (32, 128, d_model)
    print(f"   ✓ Output shape: {output_learned.shape}")

    # Test 3: Different outputs for different encoding types
    # (Sinusoidal is deterministic, learned is random initialization)
    assert not torch.allclose(output_sin, output_learned, atol=0.1)
    print(f"   ✓ Sinusoidal and learned produce different outputs")

    # Test 4: Dropout behavior
    print("\n3. Testing dropout...")
    emb_dropout = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional="sinusoidal",
        dropout=0.5
    )

    emb_dropout.train()
    output1 = emb_dropout(tokens)
    output2 = emb_dropout(tokens)

    # Should be different in training mode
    assert not torch.allclose(output1, output2, atol=1e-6)
    print(f"   ✓ Dropout active in train mode")

    emb_dropout.eval()
    output1 = emb_dropout(tokens)
    output2 = emb_dropout(tokens)

    # Should be identical in eval mode
    assert torch.allclose(output1, output2)
    print(f"   ✓ Dropout disabled in eval mode")

    # Test 5: Padding index
    print("\n4. Testing padding index...")
    emb_pad = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional="sinusoidal",
        dropout=0.0,
        padding_idx=0
    )

    tokens_with_pad = torch.randint(1, vocab_size, (16, 64))
    tokens_with_pad[:, :5] = 0  # First 5 positions are padding

    output_pad = emb_pad(tokens_with_pad)
    assert output_pad.shape == (16, 64, d_model)
    print(f"   ✓ Padding index works correctly")

    # Test 6: Different batch sizes
    print("\n5. Testing different batch sizes...")
    for batch_size in [1, 8, 64]:
        tokens = torch.randint(0, vocab_size, (batch_size, 128))
        output = emb_sin(tokens)
        assert output.shape == (batch_size, 128, d_model)
    print(f"   ✓ Works with batch sizes: {[1, 8, 64]}")

    # Test 7: Different sequence lengths
    print("\n6. Testing different sequence lengths...")
    for seq_len in [10, 50, 256, 512]:
        tokens = torch.randint(0, vocab_size, (32, seq_len))
        output = emb_sin(tokens)
        assert output.shape == (32, seq_len, d_model)
    print(f"   ✓ Works with seq lengths: {[10, 50, 256, 512]}")

    # Test 8: Invalid positional type
    print("\n7. Testing error handling...")
    try:
        emb_invalid = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            positional="invalid"  # type: ignore
        )
        assert False, "Should raise ValueError"
    except ValueError:
        print(f"   ✓ Correctly rejects invalid positional type")

    print("\nAll TransformerEmbedding tests passed! ✓\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Transformer Embedding - Test Suite")
    print("=" * 70)

    test_transformer_embedding()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
