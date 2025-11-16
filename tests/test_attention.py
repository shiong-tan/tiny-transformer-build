"""
Tests for Attention Module

Verifies correctness of:
- Scaled dot-product attention
- Causal masking
- Shape invariants
- Attention weight properties
"""

"""
Tests for Attention Module

This test suite demonstrates best practices for testing neural network components:

1. **Shape Testing**: Verify tensor dimensions are correct
2. **Mathematical Properties**: Test attention weights sum to 1, are non-negative
3. **Gradient Flow**: Ensure backpropagation works
4. **Numerical Stability**: Test with large dimensions and long sequences
5. **Edge Cases**: Single batch items, different d_k/d_v dimensions

Why test these properties?
- Shape mismatches are the #1 bug in transformer implementations
- Attention weights must be valid probability distributions
- Gradient flow ensures the model can learn
- Numerical stability prevents NaN/Inf during training

Run tests with: pytest tests/test_attention.py -v
Run with coverage: pytest tests/test_attention.py --cov=tiny_transformer --cov-report=html
"""

import pytest
import torch
import torch.nn as nn

from tiny_transformer.attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    Attention
)
from tiny_transformer.utils import check_shape


class TestScaledDotProductAttention:
    """Tests for the core attention function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_k = 2, 10, 64
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # Output should have same shape as V
        assert output.shape == V.shape
        
        # Attention weights should be (batch_size, seq_len, seq_len)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights form valid probability distributions."""
        batch_size, seq_len, d_k = 2, 5, 32
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        _, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # Each row should sum to 1 (probability distribution)
        row_sums = attn_weights.sum(dim=-1)
        
        # Check all sums are close to 1.0
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative (from softmax)."""
        batch_size, seq_len, d_k = 2, 5, 32
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        _, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # All weights should be >= 0
        assert (attn_weights >= 0).all()
    
    def test_different_dk_dv(self):
        """Test attention with different key and value dimensions."""
        batch_size, seq_len = 2, 8
        d_k = 32
        d_v = 64  # Different from d_k
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_v)  # Different dimension
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # Output should have d_v dimension
        assert output.shape == (batch_size, seq_len, d_v)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_single_item_batch(self):
        """Test with batch size of 1."""
        seq_len, d_k = 10, 64
        
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)
        V = torch.randn(1, seq_len, d_k)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (1, seq_len, d_k)
        assert attn_weights.shape == (1, seq_len, seq_len)
    
    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        batch_size, seq_len, d_k = 2, 5, 32
        
        Q = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        
        output, _ = scaled_dot_product_attention(Q, K, V)
        
        # Compute dummy loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        
        # Check gradients are not all zeros
        assert Q.grad.abs().sum() > 0
        assert K.grad.abs().sum() > 0
        assert V.grad.abs().sum() > 0


class TestCausalMask:
    """Tests for causal masking."""
    
    def test_mask_shape(self):
        """Test that mask has correct shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
    
    def test_mask_lower_triangular(self):
        """Test that mask is lower triangular."""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        # Lower triangle (including diagonal) should be 0
        # Upper triangle should be -inf
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert mask[i, j] == float('-inf')
    
    def test_causal_attention_zeros_future(self):
        """Test that causal attention zeros out future positions."""
        batch_size, seq_len, d_k = 2, 5, 32
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        mask = create_causal_mask(seq_len)
        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Check that upper triangle is zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Position i should not attend to position j (where j > i)
                assert torch.allclose(
                    attn_weights[:, i, j],
                    torch.zeros(batch_size),
                    atol=1e-6
                )
    
    def test_causal_attention_still_sums_to_one(self):
        """Test that causal attention weights still sum to 1."""
        batch_size, seq_len, d_k = 2, 8, 32
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        mask = create_causal_mask(seq_len)
        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Each row should still sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


class TestAttentionModule:
    """Tests for the Attention nn.Module."""
    
    def test_module_forward(self):
        """Test that module forward pass works."""
        batch_size, seq_len, d_k = 2, 10, 64
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        attn_module = Attention(dropout=0.1)
        output, attn_weights = attn_module(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_module_with_mask(self):
        """Test module with causal mask."""
        batch_size, seq_len, d_k = 2, 8, 32
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        mask = create_causal_mask(seq_len)
        
        attn_module = Attention(dropout=0.0)  # No dropout for deterministic test
        output, attn_weights = attn_module(Q, K, V, mask)
        
        # Check upper triangle is zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    attn_weights[:, i, j],
                    torch.zeros(batch_size),
                    atol=1e-6
                )
    
    def test_dropout_reduces_values(self):
        """Test that dropout actually modifies attention weights during training."""
        torch.manual_seed(42)
        batch_size, seq_len, d_k = 4, 10, 64
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        # Test multiple times to ensure we see dropout effect
        attn_module = Attention(dropout=0.5)
        attn_module.train()  # Ensure in training mode
        
        outputs = []
        for _ in range(5):
            output, _ = attn_module(Q, K, V)
            outputs.append(output)
        
        # With dropout, outputs should be different each time
        assert not torch.allclose(outputs[0], outputs[1])


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_dk_stability(self):
        """Test that large d_k values don't cause numerical issues."""
        batch_size, seq_len = 2, 8
        d_k = 512  # Large dimension
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # Check no NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(attn_weights).any()
        assert not torch.isinf(attn_weights).any()
    
    def test_very_long_sequence(self):
        """Test with longer sequences."""
        batch_size = 1
        seq_len = 512  # Long sequence
        d_k = 64
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
        
        # Verify attention weights still sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
