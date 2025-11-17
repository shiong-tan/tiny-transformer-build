"""
Position-wise Feed-Forward Network for Transformer Blocks.

This module implements the feed-forward network (FFN) used in transformer blocks.
The FFN applies two linear transformations with a GELU activation in between.
It's applied to each position separately and identically (hence "position-wise").

Architecture:
    Input (d_model) → Linear → GELU → Linear → Dropout → Output (d_model)

The hidden dimension d_ff is typically 4 × d_model, providing increased capacity
for feature transformation.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applies two linear transformations with a GELU activation:
        FFN(x) = Linear2(Dropout(GELU(Linear1(x))))

    This is applied identically to each position in the sequence.

    Args:
        d_model: Model dimension (input and output dimension)
        d_ff: Hidden dimension for the feed-forward network
              Typically 4 × d_model (e.g., 2048 for d_model=512)
        dropout: Dropout probability applied after the second linear layer
        bias: Whether to use bias in linear layers (default: True)

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(32, 128, 512)  # (B, T, d_model)
        >>> output = ff(x)  # (32, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer: d_model → d_ff
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)

        # GELU activation (modern choice, used in GPT-2/3, BERT)
        # More smooth than ReLU, better gradient flow
        self.activation = nn.GELU()

        # Second linear layer: d_ff → d_model
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Shape flow:
            (B, T, d_model) → Linear1 → (B, T, d_ff)
                           → GELU    → (B, T, d_ff)
                           → Linear2 → (B, T, d_model)
                           → Dropout → (B, T, d_model)
        """
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input (B, T, d_model), got shape {x.shape}"
        assert x.size(-1) == self.d_model, \
            f"Input dimension {x.size(-1)} != d_model {self.d_model}"

        # First transformation: expand to d_ff
        # (B, T, d_model) → (B, T, d_ff)
        hidden = self.fc1(x)

        # Apply GELU activation
        # (B, T, d_ff) → (B, T, d_ff)
        hidden = self.activation(hidden)

        # Second transformation: project back to d_model
        # (B, T, d_ff) → (B, T, d_model)
        output = self.fc2(hidden)

        # Apply dropout if configured
        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}, dropout={self.dropout.p if self.dropout else 0}"


def test_feedforward_shapes():
    """Test that FeedForward maintains expected shapes."""
    print("Testing FeedForward shapes...")

    # Test 1: Basic shape preservation
    d_model, d_ff = 512, 2048
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)

    batch_size, seq_len = 32, 128
    x = torch.randn(batch_size, seq_len, d_model)

    output = ff(x)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")

    # Test 2: Different batch sizes
    for bs in [1, 16, 64]:
        x = torch.randn(bs, seq_len, d_model)
        output = ff(x)
        assert output.shape == (bs, seq_len, d_model)
    print(f"✓ Works with different batch sizes: {[1, 16, 64]}")

    # Test 3: Different sequence lengths
    for sl in [10, 50, 256]:
        x = torch.randn(batch_size, sl, d_model)
        output = ff(x)
        assert output.shape == (batch_size, sl, d_model)
    print(f"✓ Works with different sequence lengths: {[10, 50, 256]}")

    # Test 4: Different d_model and d_ff combinations
    for dm, df in [(128, 512), (256, 1024), (768, 3072)]:
        ff = FeedForward(d_model=dm, d_ff=df)
        x = torch.randn(16, 64, dm)
        output = ff(x)
        assert output.shape == (16, 64, dm)
    print(f"✓ Works with different dimensions: [(128,512), (256,1024), (768,3072)]")

    print("All shape tests passed! ✓\n")


def test_feedforward_gradients():
    """Test that gradients flow through FeedForward."""
    print("Testing FeedForward gradients...")

    ff = FeedForward(d_model=256, d_ff=1024, dropout=0.0)  # No dropout for determinism
    x = torch.randn(16, 32, 256, requires_grad=True)

    # Forward pass
    output = ff(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert ff.fc1.weight.grad is not None, "fc1 weights should have gradients"
    assert ff.fc2.weight.grad is not None, "fc2 weights should have gradients"

    # Check that gradients are non-zero
    assert x.grad.abs().sum() > 0, "Input gradients should be non-zero"
    assert ff.fc1.weight.grad.abs().sum() > 0, "fc1 gradients should be non-zero"
    assert ff.fc2.weight.grad.abs().sum() > 0, "fc2 gradients should be non-zero"

    print("✓ Gradients flow through all parameters")
    print("All gradient tests passed! ✓\n")


def test_feedforward_expansion():
    """Test that hidden dimension expands to d_ff."""
    print("Testing FeedForward dimension expansion...")

    d_model, d_ff = 128, 512
    ff = FeedForward(d_model=d_model, d_ff=d_ff)

    # Check parameter shapes
    assert ff.fc1.weight.shape == (d_ff, d_model), \
        f"fc1 should expand: expected {(d_ff, d_model)}, got {ff.fc1.weight.shape}"
    assert ff.fc2.weight.shape == (d_model, d_ff), \
        f"fc2 should contract: expected {(d_model, d_ff)}, got {ff.fc2.weight.shape}"

    print(f"✓ fc1 expands: (d_model={d_model}) → (d_ff={d_ff})")
    print(f"✓ fc2 contracts: (d_ff={d_ff}) → (d_model={d_model})")
    print("Dimension expansion test passed! ✓\n")


def test_feedforward_dropout():
    """Test that dropout is applied correctly."""
    print("Testing FeedForward dropout...")

    # With dropout
    ff_with_dropout = FeedForward(d_model=128, d_ff=512, dropout=0.5)

    # Without dropout
    ff_without_dropout = FeedForward(d_model=128, d_ff=512, dropout=0.0)

    x = torch.randn(16, 32, 128)

    # Training mode: outputs should differ due to dropout
    ff_with_dropout.train()
    out1 = ff_with_dropout(x)
    out2 = ff_with_dropout(x)

    # Different due to random dropout
    assert not torch.allclose(out1, out2), "Outputs should differ in training mode with dropout"
    print("✓ Dropout is active in training mode")

    # Eval mode: outputs should be identical
    ff_with_dropout.eval()
    out1 = ff_with_dropout(x)
    out2 = ff_with_dropout(x)

    assert torch.allclose(out1, out2), "Outputs should be identical in eval mode"
    print("✓ Dropout is disabled in eval mode")

    print("Dropout test passed! ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("FeedForward Network - Test Suite")
    print("=" * 60)
    print()

    test_feedforward_shapes()
    test_feedforward_gradients()
    test_feedforward_expansion()
    test_feedforward_dropout()

    print("=" * 60)
    print("All FeedForward tests passed! ✓")
    print("=" * 60)
