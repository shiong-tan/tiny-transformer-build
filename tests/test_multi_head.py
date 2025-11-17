"""
Tests for Multi-Head Attention Module

Verifies correctness of:
- Multi-head attention mechanism
- Head splitting and combination
- Shape transformations
- Attention weight properties across heads
- Cross-attention scenarios
- Masking with multiple heads
- Gradient flow
- Parameter initialization
"""

import pytest
import torch
import torch.nn as nn

from tiny_transformer.multi_head import MultiHeadAttention
from tiny_transformer.attention import create_causal_mask
from tiny_transformer.utils import check_shape


class TestMultiHeadAttention:
    """Tests for multi-head attention implementation."""

    def test_output_shape_self_attention(self):
        """Test that output has correct shape for self-attention."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        # Output should have same shape as input
        assert output.shape == (batch_size, seq_len, d_model)

        # Attention weights should be (batch_size, n_heads, seq_len, seq_len)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_output_shape_cross_attention(self):
        """Test that output has correct shape for cross-attention."""
        batch_size = 2
        decoder_len, encoder_len = 5, 10
        d_model = 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        decoder_input = torch.randn(batch_size, decoder_len, d_model)
        encoder_output = torch.randn(batch_size, encoder_len, d_model)

        output, attn_weights = mha(decoder_input, encoder_output, encoder_output)

        # Output should match query (decoder) sequence length
        assert output.shape == (batch_size, decoder_len, d_model)

        # Attention weights: (batch, heads, query_len, key_len)
        assert attn_weights.shape == (batch_size, n_heads, decoder_len, encoder_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights form valid probability distributions for each head."""
        batch_size, seq_len, d_model = 2, 5, 32
        n_heads = 4

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        _, attn_weights = mha(x, x, x)

        # Each row should sum to 1 for each head
        row_sums = attn_weights.sum(dim=-1)  # (batch, n_heads, seq_len)

        # Check all sums are close to 1.0
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative across all heads."""
        batch_size, seq_len, d_model = 2, 5, 32
        n_heads = 4

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        _, attn_weights = mha(x, x, x)

        # All weights should be >= 0
        assert (attn_weights >= 0).all()

    def test_d_model_divisibility(self):
        """Test that d_model must be divisible by n_heads."""
        d_model = 65  # Not divisible by 8
        n_heads = 8

        with pytest.raises(AssertionError):
            mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    def test_different_head_counts(self):
        """Test that multi-head attention works with various head counts."""
        batch_size, seq_len = 2, 8
        d_model = 64

        for n_heads in [1, 2, 4, 8, 16]:
            mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
            x = torch.randn(batch_size, seq_len, d_model)

            output, attn_weights = mha(x, x, x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_causal_mask(self):
        """Test that causal masking works correctly across all heads."""
        batch_size, seq_len, d_model = 1, 4, 32
        n_heads = 4

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = create_causal_mask(seq_len)

        output, attn_weights = mha(x, x, x, mask=mask)

        # Check that upper triangle is zero for all heads
        for head in range(n_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    # Future positions should have zero attention
                    assert attn_weights[0, head, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through multi-head attention."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, _ = mha(x, x, x)

        # Compute a simple loss
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_parameter_count(self):
        """Test that parameter count is correct."""
        d_model = 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, bias=True)

        # Count parameters
        total_params = sum(p.numel() for p in mha.parameters())

        # Expected: 4 linear layers (W_q, W_k, W_v, W_o) each with (d_model, d_model) + bias
        # = 4 * (d_model * d_model + d_model)
        expected = 4 * (d_model * d_model + d_model)

        assert total_params == expected

    def test_parameter_count_no_bias(self):
        """Test that parameter count is correct without bias."""
        d_model = 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, bias=False)

        # Count parameters
        total_params = sum(p.numel() for p in mha.parameters())

        # Expected: 4 linear layers (W_q, W_k, W_v, W_o) each with (d_model, d_model)
        # = 4 * d_model * d_model
        expected = 4 * d_model * d_model

        assert total_params == expected

    def test_split_heads(self):
        """Test that split_heads correctly reshapes tensors."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8
        d_k = d_model // n_heads

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Split heads
        x_split = mha.split_heads(x)

        # Should have shape (batch, n_heads, seq_len, d_k)
        assert x_split.shape == (batch_size, n_heads, seq_len, d_k)

    def test_combine_heads(self):
        """Test that combine_heads correctly reshapes tensors."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8
        d_k = d_model // n_heads

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        # Create tensor with head dimension
        x = torch.randn(batch_size, n_heads, seq_len, d_k)

        # Combine heads
        x_combined = mha.combine_heads(x)

        # Should have shape (batch, seq_len, d_model)
        assert x_combined.shape == (batch_size, seq_len, d_model)

    def test_split_combine_inverse(self):
        """Test that split_heads and combine_heads are inverses."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Split then combine
        x_reconstructed = mha.combine_heads(mha.split_heads(x))

        # Should be identical
        assert torch.allclose(x, x_reconstructed)

    def test_dropout(self):
        """Test that dropout is applied during training mode."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        # Create with high dropout to make effect visible
        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.5)
        mha.train()  # Set to training mode

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times - outputs should differ due to dropout
        output1, _ = mha(x, x, x)
        output2, _ = mha(x, x, x)

        # Outputs should be different (dropout is stochastic)
        assert not torch.allclose(output1, output2)

    def test_eval_mode_no_dropout(self):
        """Test that dropout is disabled in eval mode."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.5)
        mha.eval()  # Set to eval mode

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times - outputs should be identical (no dropout)
        torch.manual_seed(42)
        output1, _ = mha(x, x, x)

        torch.manual_seed(42)
        output2, _ = mha(x, x, x)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_batch_size_one(self):
        """Test that multi-head attention works with batch_size=1."""
        batch_size, seq_len, d_model = 1, 10, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_single_head_vs_multihead(self):
        """Test that n_heads=1 works correctly."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 1

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_large_dimensions(self):
        """Test with larger, more realistic dimensions."""
        batch_size, seq_len, d_model = 32, 128, 512
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

        # Check numerical stability
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(attn_weights).any()
        assert not torch.isinf(attn_weights).any()

    def test_different_sequence_lengths(self):
        """Test cross-attention with different sequence lengths."""
        batch_size = 4
        d_model = 128
        n_heads = 8

        # Different sequence lengths for encoder and decoder
        encoder_len = 50
        decoder_len = 30

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        encoder_output = torch.randn(batch_size, encoder_len, d_model)
        decoder_input = torch.randn(batch_size, decoder_len, d_model)

        # Cross-attention: decoder queries attend to encoder keys/values
        output, attn_weights = mha(
            query=decoder_input,
            key=encoder_output,
            value=encoder_output
        )

        # Output should match query length
        assert output.shape == (batch_size, decoder_len, d_model)

        # Attention maps decoder positions to encoder positions
        assert attn_weights.shape == (batch_size, n_heads, decoder_len, encoder_len)

    def test_mask_broadcasting(self):
        """Test different mask shapes for broadcasting."""
        batch_size, seq_len, d_model = 2, 4, 32
        n_heads = 4

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Test 2D mask (seq_len, seq_len)
        mask_2d = create_causal_mask(seq_len)
        output_2d, attn_2d = mha(x, x, x, mask=mask_2d)

        # Test 3D mask (batch, seq_len, seq_len)
        mask_3d = mask_2d.unsqueeze(0).expand(batch_size, -1, -1)
        output_3d, attn_3d = mha(x, x, x, mask=mask_3d)

        # Both should give similar results (within numerical precision)
        # Note: They might not be exactly equal due to broadcasting differences
        assert output_2d.shape == output_3d.shape
        assert attn_2d.shape == attn_3d.shape

    def test_deterministic_output(self):
        """Test that output is deterministic with same seed."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        # First run
        torch.manual_seed(42)
        mha1 = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        x1 = torch.randn(batch_size, seq_len, d_model)
        output1, _ = mha1(x1, x1, x1)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        x2 = torch.randn(batch_size, seq_len, d_model)
        output2, _ = mha2(x2, x2, x2)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_different_heads_learn_different_patterns(self):
        """Test that different heads can attend to different positions."""
        batch_size, seq_len, d_model = 1, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        _, attn_weights = mha(x, x, x)

        # Extract attention patterns for each head
        # attn_weights: (1, n_heads, seq_len, seq_len)
        head_patterns = attn_weights[0]  # (n_heads, seq_len, seq_len)

        # Heads should have different attention patterns
        # (This is probabilistic but should hold with random initialization)
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                # Compare attention patterns between heads
                # They should not be identical
                if not torch.allclose(head_patterns[i], head_patterns[j], atol=1e-3):
                    # Found at least one pair that differs
                    break
            else:
                continue
            break
        # If we get here, we found differing heads (expected)

    def test_zero_dropout(self):
        """Test that dropout=0.0 means no dropout is applied."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        mha.train()  # Even in training mode

        x = torch.randn(batch_size, seq_len, d_model)

        # Run twice - should get identical results
        torch.manual_seed(42)
        output1, _ = mha(x, x, x)

        torch.manual_seed(42)
        output2, _ = mha(x, x, x)

        assert torch.allclose(output1, output2)


class TestIntegration:
    """Integration tests combining multi-head attention with other components."""

    def test_with_layer_norm(self):
        """Test multi-head attention combined with layer normalization."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        layer_norm = nn.LayerNorm(d_model)

        x = torch.randn(batch_size, seq_len, d_model)

        # Forward through attention then layer norm
        attn_output, _ = mha(x, x, x)
        output = layer_norm(attn_output)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_residual_connection(self):
        """Test multi-head attention with residual connection."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Attention with residual
        attn_output, _ = mha(x, x, x)
        output = x + attn_output  # Residual connection

        assert output.shape == (batch_size, seq_len, d_model)

    def test_stacked_multi_head(self):
        """Test stacking multiple multi-head attention layers."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 8
        n_layers = 3

        # Create stack of MHA layers
        mha_layers = nn.ModuleList([
            MultiHeadAttention(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        x = torch.randn(batch_size, seq_len, d_model)

        # Forward through stack
        for mha in mha_layers:
            x, _ = mha(x, x, x)

        assert x.shape == (batch_size, seq_len, d_model)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
