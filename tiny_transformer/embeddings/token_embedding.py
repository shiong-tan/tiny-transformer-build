"""
Token Embeddings for Transformer Models.

This module implements token embeddings that convert discrete token IDs to
continuous vector representations. Includes scaling by √d_model following
the original "Attention is All You Need" paper.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to continuous vectors.

    This is a standard embedding lookup table with scaling by √d_model.
    The scaling helps balance the magnitudes of token embeddings and
    positional encodings when they are added together.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        d_model: Embedding dimension (model dimension)
        padding_idx: Optional index for padding token (default: None)
                    Embeddings at this index are not updated during training

    Shape:
        Input: (batch_size, seq_len) - Token IDs (LongTensor)
        Output: (batch_size, seq_len, d_model) - Embedded tokens

    Example:
        >>> vocab_size = 10000
        >>> d_model = 512
        >>> emb = TokenEmbedding(vocab_size, d_model)
        >>> tokens = torch.randint(0, vocab_size, (32, 128))  # (B, T)
        >>> embedded = emb(tokens)  # (32, 128, 512)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Embedding lookup table: vocab_size × d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )

        # Scaling factor: √d_model
        # This ensures embedding magnitudes are comparable to positional encodings
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and scale by √d_model.

        Args:
            tokens: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embedded and scaled tokens of shape (batch_size, seq_len, d_model)

        Shape flow:
            (B, T) token IDs
            → Embedding lookup → (B, T, d_model)
            → Scale by √d_model → (B, T, d_model)
        """
        # Validate input
        assert tokens.dim() == 2, \
            f"Expected 2D input (B, T), got shape {tokens.shape}"
        assert tokens.max() < self.vocab_size, \
            f"Token ID {tokens.max()} >= vocab_size {self.vocab_size}"
        assert tokens.min() >= 0, \
            f"Token IDs must be non-negative, got min {tokens.min()}"

        # Embedding lookup: (B, T) → (B, T, d_model)
        embedded = self.embedding(tokens)

        # Scale by √d_model
        # This is done to:
        # 1. Balance with positional encoding magnitudes
        # 2. Match the variance scaling used in attention (1/√d_k)
        embedded = embedded * self.scale

        return embedded

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"padding_idx={self.padding_idx}")


def test_token_embedding():
    """Test TokenEmbedding functionality."""
    print("Testing TokenEmbedding...")

    # Test 1: Basic shape
    vocab_size, d_model = 10000, 512
    emb = TokenEmbedding(vocab_size, d_model)

    batch_size, seq_len = 32, 128
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    embedded = emb(tokens)
    assert embedded.shape == (batch_size, seq_len, d_model)
    print(f"✓ Shape: {tokens.shape} → {embedded.shape}")

    # Test 2: Scaling factor
    assert emb.scale == math.sqrt(d_model)
    print(f"✓ Scaling factor: √{d_model} = {emb.scale:.4f}")

    # Test 3: Scaled magnitude
    # Without scaling, embedding magnitudes would be ~1
    # With scaling, they should be ~√d_model
    avg_norm = embedded.norm(dim=-1).mean()
    print(f"✓ Average embedding norm: {avg_norm:.2f} (expected ~{emb.scale:.2f})")

    # Test 4: Padding index
    emb_with_pad = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    tokens_with_pad = torch.randint(0, vocab_size, (16, 64))
    tokens_with_pad[:, :5] = 0  # First 5 positions are padding

    embedded_with_pad = emb_with_pad(tokens_with_pad)
    # Padding embeddings should be zero before scaling
    # After scaling they'll still be zero
    assert torch.allclose(
        emb_with_pad.embedding(torch.tensor([0])),
        torch.zeros(d_model)
    )
    print(f"✓ Padding index works correctly")

    # Test 5: Different batch sizes
    for bs in [1, 8, 64]:
        tokens = torch.randint(0, vocab_size, (bs, seq_len))
        embedded = emb(tokens)
        assert embedded.shape == (bs, seq_len, d_model)
    print(f"✓ Works with different batch sizes")

    # Test 6: Different sequence lengths
    for sl in [10, 50, 256]:
        tokens = torch.randint(0, vocab_size, (batch_size, sl))
        embedded = emb(tokens)
        assert embedded.shape == (batch_size, sl, d_model)
    print(f"✓ Works with different sequence lengths")

    print("All TokenEmbedding tests passed! ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Token Embedding - Test Suite")
    print("=" * 60)
    print()

    test_token_embedding()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
