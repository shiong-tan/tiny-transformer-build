"""
Comprehensive tests for the FeedForward network module.

Tests cover:
- Shape preservation across different inputs
- Dimension expansion to d_ff
- Gradient flow
- Activation function behavior
- Dropout behavior in train vs eval modes
- Edge cases and validation
"""

import pytest
import torch
import torch.nn as nn

from tiny_transformer.feedforward import FeedForward


class TestFeedForwardShapes:
    """Test that FeedForward maintains correct shapes."""

    def test_output_shape_basic(self):
        """Test that output has same shape as input."""
        d_model, d_ff = 512, 2048
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_output_shape_different_batch_sizes(self):
        """Test with various batch sizes."""
        d_model, d_ff = 256, 1024
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        seq_len = 64
        for batch_size in [1, 8, 16, 32, 64]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ff(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_output_shape_different_seq_lengths(self):
        """Test with various sequence lengths."""
        d_model, d_ff = 256, 1024
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        batch_size = 16
        for seq_len in [10, 32, 64, 128, 256, 512]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ff(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_output_shape_different_dimensions(self):
        """Test with various d_model and d_ff combinations."""
        test_configs = [
            (128, 512),
            (256, 1024),
            (512, 2048),
            (768, 3072),
            (1024, 4096)
        ]

        for d_model, d_ff in test_configs:
            ff = FeedForward(d_model=d_model, d_ff=d_ff)
            x = torch.randn(16, 32, d_model)
            output = ff(x)
            assert output.shape == (16, 32, d_model)

    def test_shape_with_single_sequence(self):
        """Test with batch_size=1."""
        ff = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(1, 64, 256)
        output = ff(x)
        assert output.shape == (1, 64, 256)

    def test_shape_with_single_token(self):
        """Test with seq_len=1."""
        ff = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(32, 1, 256)
        output = ff(x)
        assert output.shape == (32, 1, 256)


class TestFeedForwardParameters:
    """Test FeedForward parameter shapes and initialization."""

    def test_parameter_shapes(self):
        """Test that linear layer weights have correct shapes."""
        d_model, d_ff = 256, 1024
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        # fc1: d_model → d_ff
        assert ff.fc1.weight.shape == (d_ff, d_model)
        assert ff.fc1.bias.shape == (d_ff,)

        # fc2: d_ff → d_model
        assert ff.fc2.weight.shape == (d_model, d_ff)
        assert ff.fc2.bias.shape == (d_model,)

    def test_no_bias_option(self):
        """Test that bias=False works."""
        ff = FeedForward(d_model=256, d_ff=1024, bias=False)

        assert ff.fc1.bias is None
        assert ff.fc2.bias is None

    def test_parameter_count(self):
        """Test that parameter count is correct."""
        d_model, d_ff = 256, 1024
        ff = FeedForward(d_model=d_model, d_ff=d_ff, bias=True)

        total_params = sum(p.numel() for p in ff.parameters())

        # fc1: (d_ff × d_model) weights + d_ff bias
        # fc2: (d_model × d_ff) weights + d_model bias
        expected = (d_ff * d_model + d_ff) + (d_model * d_ff + d_model)

        assert total_params == expected

    def test_dimension_expansion(self):
        """Test that hidden dimension expands to d_ff."""
        d_model, d_ff = 128, 512
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        # Check that fc1 expands
        assert ff.fc1.weight.shape[0] == d_ff
        assert ff.fc1.weight.shape[1] == d_model

        # Check that fc2 contracts
        assert ff.fc2.weight.shape[0] == d_model
        assert ff.fc2.weight.shape[1] == d_ff


class TestFeedForwardGradients:
    """Test that gradients flow correctly through FeedForward."""

    def test_gradients_exist(self):
        """Test that all parameters receive gradients."""
        ff = FeedForward(d_model=256, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None

        # Check parameter gradients
        assert ff.fc1.weight.grad is not None
        assert ff.fc1.bias.grad is not None
        assert ff.fc2.weight.grad is not None
        assert ff.fc2.bias.grad is not None

    def test_gradients_nonzero(self):
        """Test that gradients are non-zero."""
        ff = FeedForward(d_model=256, d_ff=1024, dropout=0.0)
        x = torch.randn(16, 32, 256, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad.abs().sum() > 0
        assert ff.fc1.weight.grad.abs().sum() > 0
        assert ff.fc2.weight.grad.abs().sum() > 0

    def test_gradient_flow_through_activation(self):
        """Test that gradients flow through GELU activation."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.0)
        x = torch.randn(8, 16, 128, requires_grad=True)

        output = ff(x)
        loss = output.mean()
        loss.backward()

        # GELU should propagate gradients (unlike ReLU which can zero them)
        assert x.grad is not None
        # Count non-zero gradient elements
        nonzero_grads = (x.grad.abs() > 1e-10).sum()
        assert nonzero_grads > 0


class TestFeedForwardActivation:
    """Test GELU activation behavior."""

    def test_uses_gelu_activation(self):
        """Test that FeedForward uses GELU activation."""
        ff = FeedForward(d_model=256, d_ff=1024)
        assert isinstance(ff.activation, nn.GELU)

    def test_gelu_output_characteristics(self):
        """Test that GELU produces expected output characteristics."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.0)
        ff.eval()

        # GELU is smooth and non-zero even for negative inputs (unlike ReLU)
        x = torch.randn(10, 20, 64)
        output = ff(x)

        # Output should have both positive and negative values
        assert (output > 0).any()
        # Note: GELU can produce negative outputs for very negative inputs


class TestFeedForwardDropout:
    """Test dropout behavior in train vs eval modes."""

    def test_dropout_active_in_train_mode(self):
        """Test that dropout is active during training."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.5)
        ff.train()

        x = torch.randn(16, 32, 128)

        # Multiple forward passes should give different results due to dropout
        output1 = ff(x)
        output2 = ff(x)

        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_dropout_disabled_in_eval_mode(self):
        """Test that dropout is disabled during evaluation."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.5)
        ff.eval()

        x = torch.randn(16, 32, 128)

        # Multiple forward passes should give identical results
        output1 = ff(x)
        output2 = ff(x)

        assert torch.allclose(output1, output2)

    def test_no_dropout_when_zero(self):
        """Test that dropout=0.0 means no dropout."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.0)

        assert ff.dropout is None

    def test_dropout_exists_when_nonzero(self):
        """Test that dropout layer exists when dropout > 0."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.1)

        assert ff.dropout is not None
        assert isinstance(ff.dropout, nn.Dropout)
        assert ff.dropout.p == 0.1


class TestFeedForwardValidation:
    """Test input validation and error handling."""

    def test_wrong_input_dimension_fails(self):
        """Test that wrong input dimension raises error."""
        ff = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(16, 32, 128)  # Wrong d_model (128 instead of 256)

        with pytest.raises(AssertionError):
            ff(x)

    def test_wrong_input_rank_fails(self):
        """Test that wrong input rank raises error."""
        ff = FeedForward(d_model=256, d_ff=1024)

        # 2D input (missing sequence dimension)
        x_2d = torch.randn(16, 256)
        with pytest.raises(AssertionError):
            ff(x_2d)

        # 4D input (extra dimension)
        x_4d = torch.randn(16, 32, 64, 256)
        with pytest.raises(AssertionError):
            ff(x_4d)


class TestFeedForwardEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_dimensions(self):
        """Test with minimal dimensions."""
        ff = FeedForward(d_model=8, d_ff=32)
        x = torch.randn(2, 4, 8)
        output = ff(x)
        assert output.shape == (2, 4, 8)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        ff = FeedForward(d_model=2048, d_ff=8192)
        x = torch.randn(2, 8, 2048)
        output = ff(x)
        assert output.shape == (2, 8, 2048)

    def test_non_standard_d_ff_ratio(self):
        """Test with d_ff not equal to 4 × d_model."""
        # d_ff can be any value, not just 4×d_model
        ff = FeedForward(d_model=256, d_ff=512)  # 2× instead of 4×
        x = torch.randn(16, 32, 256)
        output = ff(x)
        assert output.shape == (16, 32, 256)

    def test_deterministic_with_zero_dropout(self):
        """Test that forward pass is deterministic with dropout=0."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.0)
        ff.eval()

        x = torch.randn(16, 32, 128)

        output1 = ff(x)
        output2 = ff(x)

        assert torch.equal(output1, output2)


class TestFeedForwardIntegration:
    """Integration tests with other components."""

    def test_batch_processing(self):
        """Test that batches are processed independently."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.0)
        ff.eval()

        # Process batch
        batch_x = torch.randn(16, 32, 128)
        batch_output = ff(batch_x)

        # Process samples individually
        for i in range(16):
            single_x = batch_x[i:i+1]  # Keep batch dimension
            single_output = ff(single_x)
            assert torch.allclose(batch_output[i:i+1], single_output, atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with various input ranges."""
        ff = FeedForward(d_model=128, d_ff=512, dropout=0.0)

        # Very small values
        x_small = torch.randn(16, 32, 128) * 1e-6
        output_small = ff(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()

        # Very large values
        x_large = torch.randn(16, 32, 128) * 1e3
        output_large = ff(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
