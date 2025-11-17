"""
Comprehensive tests for the TransformerBlock module.

Tests cover:
- Shape preservation across different inputs
- Pre-LN architecture correctness
- Residual connections
- Gradient flow through deep stacks
- Causal masking
- Attention weight return
- Integration with MultiHeadAttention and FeedForward
- Edge cases and validation
"""

import pytest
import torch
import torch.nn as nn

from tiny_transformer.transformer_block import TransformerBlock


class TestTransformerBlockShapes:
    """Test that TransformerBlock maintains correct shapes."""

    def test_output_shape_basic(self):
        """Test that output has same shape as input."""
        d_model, n_heads, d_ff = 512, 8, 2048
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, d_model)

        output, _ = block(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_output_shape_different_batch_sizes(self):
        """Test with various batch sizes."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        seq_len = 64
        for batch_size in [1, 8, 16, 32, 64]:
            x = torch.randn(batch_size, seq_len, 256)
            output, _ = block(x)
            assert output.shape == (batch_size, seq_len, 256)

    def test_output_shape_different_seq_lengths(self):
        """Test with various sequence lengths."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        batch_size = 16
        for seq_len in [10, 32, 64, 128, 256, 512]:
            x = torch.randn(batch_size, seq_len, 256)
            output, _ = block(x)
            assert output.shape == (batch_size, seq_len, 256)

    def test_output_shape_different_configurations(self):
        """Test with various model configurations."""
        configs = [
            (128, 4, 512),
            (256, 8, 1024),
            (512, 8, 2048),
            (768, 12, 3072),
            (1024, 16, 4096)
        ]

        for d_model, n_heads, d_ff in configs:
            block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            x = torch.randn(16, 32, d_model)
            output, _ = block(x)
            assert output.shape == (16, 32, d_model)

    def test_attention_weights_shape(self):
        """Test that attention weights have correct shape when returned."""
        d_model, n_heads = 256, 8
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=1024)

        batch_size, seq_len = 16, 64
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = block(x, return_attention=True)

        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestTransformerBlockAttention:
    """Test attention mechanism in TransformerBlock."""

    def test_attention_weights_not_returned_by_default(self):
        """Test that attention weights are not returned by default."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(16, 64, 256)

        output, attn_weights = block(x, return_attention=False)

        assert attn_weights is None

    def test_attention_weights_returned_when_requested(self):
        """Test that attention weights are returned when requested."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(16, 64, 256)

        output, attn_weights = block(x, return_attention=True)

        assert attn_weights is not None

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across key dimension."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.0)
        block.eval()

        x = torch.randn(16, 64, 256)
        _, attn_weights = block(x, return_attention=True)

        # Sum over key dimension (last dimension)
        attn_sum = attn_weights.sum(dim=-1)

        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_self_attention_is_symmetric_without_mask(self):
        """Test that self-attention pattern is symmetric without mask."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
        block.eval()

        # Use fixed input for determinism
        torch.manual_seed(42)
        x = torch.randn(1, 16, 128)

        _, attn_weights = block(x, mask=None, return_attention=True)

        # Self-attention without mask should attend to all positions
        # Check that attention is non-zero for all positions
        assert (attn_weights > 0).all()


class TestTransformerBlockCausalMask:
    """Test causal masking in TransformerBlock."""

    def test_causal_mask_prevents_future_attention(self):
        """Test that causal mask prevents attending to future positions."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
        block.eval()

        seq_len = 32
        x = torch.randn(1, seq_len, 128)

        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        _, attn_weights = block(x, mask=causal_mask, return_attention=True)

        # Check that upper triangle (future positions) has near-zero attention
        for head_idx in range(attn_weights.size(1)):
            attn_head = attn_weights[0, head_idx]
            upper_triangle = torch.triu(attn_head, diagonal=1)
            assert upper_triangle.abs().max() < 1e-6

    def test_different_mask_shapes(self):
        """Test that different mask shapes work correctly."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512)

        batch_size, seq_len = 16, 32
        x = torch.randn(batch_size, seq_len, 128)

        # 2D mask: (seq_len, seq_len)
        mask_2d = torch.tril(torch.ones(seq_len, seq_len))
        output_2d, _ = block(x, mask=mask_2d)
        assert output_2d.shape == (batch_size, seq_len, 128)

        # 3D mask: (batch_size, seq_len, seq_len)
        mask_3d = torch.tril(torch.ones(batch_size, seq_len, seq_len))
        output_3d, _ = block(x, mask=mask_3d)
        assert output_3d.shape == (batch_size, seq_len, 128)

        # 4D mask: (batch_size, n_heads, seq_len, seq_len)
        mask_4d = torch.tril(torch.ones(batch_size, 4, seq_len, seq_len))
        output_4d, _ = block(x, mask=mask_4d)
        assert output_4d.shape == (batch_size, seq_len, 128)


class TestTransformerBlockResiduals:
    """Test residual connections in TransformerBlock."""

    def test_residual_connections_exist(self):
        """Test that residual connections are active."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256)

        output, _ = block(x)

        # Output should be different from input (sublayers transform)
        assert not torch.allclose(output, x, atol=1e-2)

        # But residuals mean the change is additive, not completely different
        diff = (output - x).abs().mean()
        # Difference should be bounded (not wildly different)
        assert diff < 10.0  # Reasonable bound for random weights

    def test_residual_gradient_flow(self):
        """Test that residuals enable gradient flow to input."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        # With residuals, input should have strong gradients
        assert x.grad is not None
        assert x.grad.abs().mean() > 0


class TestTransformerBlockPreLN:
    """Test Pre-LN architecture specifics."""

    def test_has_two_layer_norms(self):
        """Test that block has two LayerNorm layers."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)

    def test_layer_norm_parameters(self):
        """Test that LayerNorm has correct parameters."""
        d_model = 256
        block = TransformerBlock(d_model=d_model, n_heads=8, d_ff=1024)

        # Both layer norms should normalize over d_model
        assert block.norm1.normalized_shape == (d_model,)
        assert block.norm2.normalized_shape == (d_model,)

    def test_pre_ln_vs_post_ln_stability(self):
        """Test that Pre-LN provides stable outputs."""
        # Pre-LN should have bounded output norms
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        x = torch.randn(16, 32, 256)
        output, _ = block(x)

        # Output norm shouldn't explode
        output_norm = output.norm(dim=-1).mean()
        input_norm = x.norm(dim=-1).mean()

        # With Pre-LN, norms should be reasonably bounded
        assert output_norm < input_norm * 5  # Should not explode


class TestTransformerBlockGradients:
    """Test gradient flow through TransformerBlock."""

    def test_gradients_exist_for_all_parameters(self):
        """Test that all parameters receive gradients."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"

    def test_gradients_flow_to_input(self):
        """Test that gradients flow back to input."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_through_stacked_blocks(self):
        """Test gradient flow through multiple stacked blocks."""
        blocks = nn.ModuleList([
            TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
            for _ in range(4)
        ])

        x = torch.randn(8, 16, 128, requires_grad=True)

        # Forward through stack
        for block in blocks:
            x, _ = block(x)

        loss = x.sum()
        loss.backward()

        # Check that all blocks have gradients
        for i, block in enumerate(blocks):
            for name, param in block.named_parameters():
                assert param.grad is not None, \
                    f"Block {i}, parameter {name} has no gradient"


class TestTransformerBlockDropout:
    """Test dropout behavior in TransformerBlock."""

    def test_dropout_active_in_train_mode(self):
        """Test that dropout is active during training."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.5)
        block.train()

        x = torch.randn(16, 32, 128)

        output1, _ = block(x)
        output2, _ = block(x)

        # Should be different due to dropout
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_dropout_disabled_in_eval_mode(self):
        """Test that dropout is disabled during evaluation."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.5)
        block.eval()

        x = torch.randn(16, 32, 128)

        output1, _ = block(x)
        output2, _ = block(x)

        # Should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestTransformerBlockComponents:
    """Test that TransformerBlock correctly integrates components."""

    def test_has_multihead_attention(self):
        """Test that block has MultiHeadAttention component."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        assert hasattr(block, 'self_attn')
        from tiny_transformer.multi_head import MultiHeadAttention
        assert isinstance(block.self_attn, MultiHeadAttention)

    def test_has_feedforward(self):
        """Test that block has FeedForward component."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        assert hasattr(block, 'feed_forward')
        from tiny_transformer.feedforward import FeedForward
        assert isinstance(block.feed_forward, FeedForward)

    def test_component_parameters_match(self):
        """Test that component parameters match block config."""
        d_model, n_heads, d_ff = 256, 8, 1024
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        # Check attention
        assert block.self_attn.d_model == d_model
        assert block.self_attn.n_heads == n_heads

        # Check feedforward
        assert block.feed_forward.d_model == d_model
        assert block.feed_forward.d_ff == d_ff


class TestTransformerBlockValidation:
    """Test input validation and error handling."""

    def test_wrong_input_dimension_fails(self):
        """Test that wrong input dimension raises error."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(16, 32, 128)  # Wrong d_model

        with pytest.raises(AssertionError):
            block(x)

    def test_wrong_input_rank_fails(self):
        """Test that wrong input rank raises error."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)

        # 2D input
        x_2d = torch.randn(16, 256)
        with pytest.raises(AssertionError):
            block(x_2d)

        # 4D input
        x_4d = torch.randn(16, 32, 64, 256)
        with pytest.raises(AssertionError):
            block(x_4d)

    def test_d_model_not_divisible_by_n_heads_fails(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(AssertionError):
            TransformerBlock(d_model=256, n_heads=7, d_ff=1024)  # 256 % 7 != 0


class TestTransformerBlockEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_dimensions(self):
        """Test with minimal valid dimensions."""
        block = TransformerBlock(d_model=8, n_heads=2, d_ff=32)
        x = torch.randn(2, 4, 8)
        output, _ = block(x)
        assert output.shape == (2, 4, 8)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        block = TransformerBlock(d_model=1024, n_heads=16, d_ff=4096)
        x = torch.randn(2, 8, 1024)
        output, _ = block(x)
        assert output.shape == (2, 8, 1024)

    def test_single_batch(self):
        """Test with single sample."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(1, 64, 256)
        output, _ = block(x)
        assert output.shape == (1, 64, 256)

    def test_single_token(self):
        """Test with single token sequence."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(32, 1, 256)
        output, _ = block(x)
        assert output.shape == (32, 1, 256)


class TestTransformerBlockIntegration:
    """Integration tests for TransformerBlock."""

    def test_stacking_multiple_blocks(self):
        """Test stacking multiple transformer blocks."""
        num_layers = 6
        blocks = nn.ModuleList([
            TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.0)
            for _ in range(num_layers)
        ])

        x = torch.randn(16, 64, 256)

        # Forward through all blocks
        for block in blocks:
            x, _ = block(x)

        assert x.shape == (16, 64, 256)

    def test_batch_processing_independence(self):
        """Test that batch samples are processed independently."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
        block.eval()

        # Process batch
        batch_x = torch.randn(16, 32, 128)
        batch_output, _ = block(batch_x)

        # Process samples individually
        for i in range(16):
            single_x = batch_x[i:i+1]
            single_output, _ = block(single_x)
            assert torch.allclose(batch_output[i:i+1], single_output, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with various inputs."""
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)

        # Very small values
        x_small = torch.randn(16, 32, 128) * 1e-6
        output_small, _ = block(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()

        # Very large values
        x_large = torch.randn(16, 32, 128) * 1e3
        output_large, _ = block(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
