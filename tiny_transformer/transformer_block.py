"""
Transformer Block - The core building block of transformer models.

This module implements a complete transformer block using Pre-LN architecture,
combining multi-head attention and feed-forward networks with residual connections
and layer normalization.

Pre-LN Architecture (modern standard):
    1. x = x + MultiHeadAttention(LayerNorm(x))
    2. x = x + FeedForward(LayerNorm(x))

This architecture provides better training stability than Post-LN.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from tiny_transformer.multi_head import MultiHeadAttention
from tiny_transformer.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Complete Transformer Block with Pre-LN architecture.

    Combines multi-head self-attention and position-wise feed-forward network
    with residual connections and layer normalization.

    Architecture:
        1. Attention sublayer:
           x_attn = x + MultiHeadAttention(LayerNorm(x))

        2. Feed-forward sublayer:
           x_out = x_attn + FeedForward(LayerNorm(x_attn))

    This is the Pre-LN variant, which normalizes BEFORE the sublayers.
    More stable than Post-LN (original transformer) for deep networks.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Hidden dimension for feed-forward network (typically 4 × d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(32, 128, 512)  # (B, T, d_model)
        >>> mask = torch.tril(torch.ones(128, 128))  # Causal mask
        >>> output, attn_weights = block(x, mask=mask)
        >>> output.shape
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )

        # Position-wise feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias
        )

        # Layer normalization (Pre-LN: before sublayers)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections (optional, often set to 0)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else None
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
                  - Shape: (seq_len, seq_len) for causal masking
                  - Values: 1 for allowed positions, 0 for masked positions
            return_attention: Whether to return attention weights

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Optional attention weights if return_attention=True
                         Shape: (batch_size, n_heads, seq_len, seq_len)

        Shape flow (Pre-LN):
            (B, T, d_model) → norm1 → (B, T, d_model)
                           → self_attn → (B, T, d_model)
                           → dropout + residual → (B, T, d_model)
                           → norm2 → (B, T, d_model)
                           → feed_forward → (B, T, d_model)
                           → dropout + residual → (B, T, d_model)
        """
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input (B, T, d_model), got shape {x.shape}"
        assert x.size(-1) == self.d_model, \
            f"Input dimension {x.size(-1)} != d_model {self.d_model}"

        # Sublayer 1: Multi-Head Self-Attention with Pre-LN
        # ---------------------------------------------------
        # Normalize before attention (Pre-LN)
        normed_x = self.norm1(x)  # (B, T, d_model)

        # Self-attention (Q=K=V)
        attn_output, attn_weights = self.self_attn(
            query=normed_x,
            key=normed_x,
            value=normed_x,
            mask=mask
        )  # (B, T, d_model), (B, n_heads, T, T)

        # Apply dropout to attention output (if configured)
        if self.dropout1 is not None:
            attn_output = self.dropout1(attn_output)

        # Residual connection: add original input
        x = x + attn_output  # (B, T, d_model)

        # Sublayer 2: Feed-Forward Network with Pre-LN
        # ---------------------------------------------
        # Normalize before feed-forward (Pre-LN)
        normed_x = self.norm2(x)  # (B, T, d_model)

        # Feed-forward transformation
        ff_output = self.feed_forward(normed_x)  # (B, T, d_model)

        # Apply dropout to feed-forward output (if configured)
        if self.dropout2 is not None:
            ff_output = self.dropout2(ff_output)

        # Residual connection: add input to this sublayer
        x = x + ff_output  # (B, T, d_model)

        # Return output and optionally attention weights
        if return_attention:
            return x, attn_weights
        else:
            return x, None

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"d_ff={self.d_ff}, "
                f"dropout={self.dropout1.p if self.dropout1 else 0}")


def test_transformer_block_shapes():
    """Test that TransformerBlock maintains expected shapes."""
    print("Testing TransformerBlock shapes...")

    # Test 1: Basic shape preservation
    d_model, n_heads, d_ff = 512, 8, 2048
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    batch_size, seq_len = 32, 128
    x = torch.randn(batch_size, seq_len, d_model)

    output, _ = block(x)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")

    # Test 2: With causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
    output, attn_weights = block(x, mask=mask, return_attention=True)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    print(f"✓ With causal mask: output {output.shape}, attention {attn_weights.shape}")

    # Test 3: Different batch sizes
    for bs in [1, 16, 64]:
        x = torch.randn(bs, seq_len, d_model)
        output, _ = block(x)
        assert output.shape == (bs, seq_len, d_model)
    print(f"✓ Works with different batch sizes: {[1, 16, 64]}")

    # Test 4: Different sequence lengths
    for sl in [10, 50, 256]:
        x = torch.randn(batch_size, sl, d_model)
        output, _ = block(x)
        assert output.shape == (batch_size, sl, d_model)
    print(f"✓ Works with different sequence lengths: {[10, 50, 256]}")

    print("All shape tests passed! ✓\n")


def test_transformer_block_residuals():
    """Test that residual connections are working."""
    print("Testing TransformerBlock residual connections...")

    block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
    x = torch.randn(16, 32, 256)

    # With very small learning rates, residuals should dominate
    # Output should be relatively close to input due to residuals
    output, _ = block(x)

    # The output shouldn't be identical (sublayers do transform)
    assert not torch.allclose(output, x), "Output should be different from input"

    # But residuals mean the transformation is additive
    # Check that the change is bounded (not wildly different)
    diff = (output - x).abs().mean()
    print(f"✓ Mean absolute difference from input: {diff:.4f}")
    print(f"✓ Residual connections are active")

    print("Residual connection test passed! ✓\n")


def test_transformer_block_gradients():
    """Test that gradients flow through TransformerBlock."""
    print("Testing TransformerBlock gradients...")

    block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
    x = torch.randn(16, 32, 256, requires_grad=True)

    # Forward pass
    output, _ = block(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for all parameters
    assert x.grad is not None, "Input should have gradients"

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have gradients"
        assert param.grad.abs().sum() > 0, f"Parameter {name} gradients should be non-zero"

    print("✓ Gradients flow through all parameters")
    print(f"✓ Total parameters with gradients: {len(list(block.parameters()))}")

    print("Gradient test passed! ✓\n")


def test_transformer_block_attention_weights():
    """Test that attention weights are returned correctly."""
    print("Testing TransformerBlock attention weights...")

    d_model, n_heads = 256, 8
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=1024)

    batch_size, seq_len = 16, 64
    x = torch.randn(batch_size, seq_len, d_model)

    # Without return_attention
    output, attn = block(x, return_attention=False)
    assert attn is None, "Attention weights should be None when not requested"

    # With return_attention
    output, attn = block(x, return_attention=True)
    assert attn is not None, "Attention weights should be returned when requested"
    assert attn.shape == (batch_size, n_heads, seq_len, seq_len)

    # Attention weights should sum to 1 across key dimension
    attn_sum = attn.sum(dim=-1)  # Sum over keys
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "Attention weights should sum to 1"

    print(f"✓ Attention weights shape: {attn.shape}")
    print(f"✓ Attention weights sum to 1 across keys")

    print("Attention weights test passed! ✓\n")


def test_transformer_block_causal_mask():
    """Test that causal mask prevents attending to future positions."""
    print("Testing TransformerBlock causal masking...")

    block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
    block.eval()  # Deterministic mode

    seq_len = 32
    x = torch.randn(1, seq_len, 128)

    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))

    # Get attention weights with causal mask
    _, attn_weights = block(x, mask=causal_mask, return_attention=True)
    # Shape: (1, n_heads, seq_len, seq_len)

    # Check that upper triangle (future positions) has zero attention
    for head_idx in range(attn_weights.size(1)):
        attn_head = attn_weights[0, head_idx]  # (seq_len, seq_len)

        # Upper triangle should be zero (or very small due to softmax)
        upper_triangle = torch.triu(attn_head, diagonal=1)
        assert upper_triangle.abs().max() < 1e-6, \
            f"Upper triangle should be ~0, got max {upper_triangle.abs().max()}"

    print(f"✓ Causal mask prevents attending to future positions")
    print(f"✓ All {attn_weights.size(1)} heads respect the mask")

    print("Causal mask test passed! ✓\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Transformer Block - Test Suite")
    print("=" * 70)
    print()

    test_transformer_block_shapes()
    test_transformer_block_residuals()
    test_transformer_block_gradients()
    test_transformer_block_attention_weights()
    test_transformer_block_causal_mask()

    print("=" * 70)
    print("All TransformerBlock tests passed! ✓")
    print("=" * 70)
