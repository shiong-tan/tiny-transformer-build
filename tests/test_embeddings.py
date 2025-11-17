"""
Comprehensive tests for embedding modules.

Tests cover:
- TokenEmbedding with scaling
- SinusoidalPositionalEncoding
- LearnedPositionalEmbedding
- TransformerEmbedding (combined)
- Shape preservation and validation
- Padding index behavior
- Extrapolation capabilities
"""

import pytest
import torch
import torch.nn as nn
import math

from tiny_transformer.embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    TransformerEmbedding
)


class TestTokenEmbedding:
    """Test TokenEmbedding module."""

    def test_output_shape(self):
        """Test output has correct shape."""
        emb = TokenEmbedding(vocab_size=10000, d_model=512)
        tokens = torch.randint(0, 10000, (32, 128))
        output = emb(tokens)
        assert output.shape == (32, 128, 512)

    def test_scaling_factor(self):
        """Test that embeddings are scaled by √d_model."""
        d_model = 512
        emb = TokenEmbedding(vocab_size=10000, d_model=d_model)
        assert emb.scale == math.sqrt(d_model)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        emb = TokenEmbedding(vocab_size=5000, d_model=256)
        for batch_size in [1, 8, 32, 64]:
            tokens = torch.randint(0, 5000, (batch_size, 64))
            output = emb(tokens)
            assert output.shape == (batch_size, 64, 256)

    def test_different_seq_lengths(self):
        """Test with various sequence lengths."""
        emb = TokenEmbedding(vocab_size=5000, d_model=256)
        for seq_len in [10, 50, 128, 256]:
            tokens = torch.randint(0, 5000, (32, seq_len))
            output = emb(tokens)
            assert output.shape == (32, seq_len, 256)

    def test_padding_idx(self):
        """Test padding index functionality."""
        emb = TokenEmbedding(vocab_size=10000, d_model=512, padding_idx=0)

        # Padding embedding should be zero (before scaling)
        pad_emb = emb.embedding(torch.tensor([0]))
        assert torch.allclose(pad_emb, torch.zeros(512))

    def test_token_bounds_validation(self):
        """Test that token IDs must be in valid range."""
        emb = TokenEmbedding(vocab_size=1000, d_model=128)

        # Token ID >= vocab_size should fail
        tokens_invalid = torch.randint(0, 1001, (16, 32))
        with pytest.raises(AssertionError):
            emb(tokens_invalid)

        # Negative token IDs should fail
        tokens_negative = torch.randint(-10, 10, (16, 32))
        with pytest.raises(AssertionError):
            emb(tokens_negative)


class TestSinusoidalPositionalEncoding:
    """Test SinusoidalPositionalEncoding module."""

    def test_output_shape(self):
        """Test output has same shape as input."""
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=1000)
        x = torch.randn(32, 128, 512)
        output = pe(x)
        assert output.shape == x.shape

    def test_pe_matrix_shape(self):
        """Test pre-computed PE matrix has correct shape."""
        d_model, max_len = 512, 1000
        pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        assert pe.pe.shape == (max_len, d_model)

    def test_pe_values_range(self):
        """Test PE values are in valid range [-1, 1]."""
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=1000)
        assert pe.pe.min() >= -1.0
        assert pe.pe.max() <= 1.0

    def test_deterministic(self):
        """Test that output is deterministic (no randomness)."""
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=1000, dropout=0.0)
        x = torch.randn(32, 128, 512)
        output1 = pe(x)
        output2 = pe(x)
        assert torch.equal(output1, output2)

    def test_different_seq_lengths(self):
        """Test with various sequence lengths."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
        for seq_len in [10, 50, 128, 512]:
            x = torch.randn(16, seq_len, 256)
            output = pe(x)
            assert output.shape == (16, seq_len, 256)

    def test_exceeding_max_len_fails(self):
        """Test that exceeding max_len raises error."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(16, 101, 256)
        with pytest.raises(AssertionError):
            pe(x)

    def test_dropout_behavior(self):
        """Test dropout is applied correctly."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000, dropout=0.5)
        x = torch.randn(32, 64, 256)

        pe.train()
        output1 = pe(x)
        output2 = pe(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        pe.eval()
        output1 = pe(x)
        output2 = pe(x)
        assert torch.allclose(output1, output2)

    def test_even_d_model_required(self):
        """Test that d_model must be even."""
        with pytest.raises(AssertionError):
            SinusoidalPositionalEncoding(d_model=513, max_len=1000)


class TestLearnedPositionalEmbedding:
    """Test LearnedPositionalEmbedding module."""

    def test_output_shape(self):
        """Test output has same shape as input."""
        pe = LearnedPositionalEmbedding(max_len=1024, d_model=512)
        x = torch.randn(32, 128, 512)
        output = pe(x)
        assert output.shape == x.shape

    def test_parameters_learnable(self):
        """Test that parameters are learnable."""
        pe = LearnedPositionalEmbedding(max_len=1024, d_model=512)
        assert pe.position_embeddings.weight.requires_grad

    def test_different_positions_different_embeddings(self):
        """Test that different positions get different embeddings."""
        pe = LearnedPositionalEmbedding(max_len=1024, d_model=512, dropout=0.0)
        pe.eval()

        x = torch.zeros(1, 10, 512)
        output = pe(x)

        # Different positions should have different embeddings
        assert not torch.allclose(output[0, 0], output[0, 1])
        assert not torch.allclose(output[0, 0], output[0, 5])

    def test_exceeding_max_len_fails(self):
        """Test that exceeding max_len raises error."""
        pe = LearnedPositionalEmbedding(max_len=100, d_model=256)
        x = torch.randn(16, 101, 256)
        with pytest.raises(AssertionError):
            pe(x)

    def test_different_seq_lengths(self):
        """Test with various sequence lengths."""
        pe = LearnedPositionalEmbedding(max_len=1024, d_model=256)
        for seq_len in [10, 50, 128, 512]:
            x = torch.randn(16, seq_len, 256)
            output = pe(x)
            assert output.shape == (16, seq_len, 256)

    def test_dropout_behavior(self):
        """Test dropout is applied correctly."""
        pe = LearnedPositionalEmbedding(max_len=1024, d_model=256, dropout=0.5)
        x = torch.randn(32, 64, 256)

        pe.train()
        output1 = pe(x)
        output2 = pe(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        pe.eval()
        output1 = pe(x)
        output2 = pe(x)
        assert torch.allclose(output1, output2)


class TestTransformerEmbedding:
    """Test TransformerEmbedding (combined) module."""

    def test_sinusoidal_output_shape(self):
        """Test output shape with sinusoidal encoding."""
        emb = TransformerEmbedding(
            vocab_size=10000,
            d_model=512,
            max_len=1024,
            positional="sinusoidal",
            dropout=0.0
        )
        tokens = torch.randint(0, 10000, (32, 128))
        output = emb(tokens)
        assert output.shape == (32, 128, 512)

    def test_learned_output_shape(self):
        """Test output shape with learned encoding."""
        emb = TransformerEmbedding(
            vocab_size=10000,
            d_model=512,
            max_len=1024,
            positional="learned",
            dropout=0.0
        )
        tokens = torch.randint(0, 10000, (32, 128))
        output = emb(tokens)
        assert output.shape == (32, 128, 512)

    def test_different_positional_types_different_outputs(self):
        """Test that sinusoidal and learned produce different results."""
        tokens = torch.randint(0, 10000, (32, 128))

        emb_sin = TransformerEmbedding(
            vocab_size=10000, d_model=512, positional="sinusoidal", dropout=0.0
        )
        emb_learned = TransformerEmbedding(
            vocab_size=10000, d_model=512, positional="learned", dropout=0.0
        )

        output_sin = emb_sin(tokens)
        output_learned = emb_learned(tokens)

        assert not torch.allclose(output_sin, output_learned, atol=0.1)

    def test_dropout_behavior(self):
        """Test dropout is applied correctly."""
        emb = TransformerEmbedding(
            vocab_size=10000,
            d_model=512,
            positional="sinusoidal",
            dropout=0.5
        )
        tokens = torch.randint(0, 10000, (32, 64))

        emb.train()
        output1 = emb(tokens)
        output2 = emb(tokens)
        assert not torch.allclose(output1, output2, atol=1e-6)

        emb.eval()
        output1 = emb(tokens)
        output2 = emb(tokens)
        assert torch.allclose(output1, output2)

    def test_padding_idx(self):
        """Test padding index functionality."""
        emb = TransformerEmbedding(
            vocab_size=10000,
            d_model=512,
            positional="sinusoidal",
            dropout=0.0,
            padding_idx=0
        )

        tokens = torch.randint(1, 10000, (16, 64))
        tokens[:, :5] = 0  # Padding

        output = emb(tokens)
        assert output.shape == (16, 64, 512)

    def test_invalid_positional_type(self):
        """Test that invalid positional type raises error."""
        with pytest.raises(ValueError):
            TransformerEmbedding(
                vocab_size=10000,
                d_model=512,
                positional="invalid"  # type: ignore
            )

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        emb = TransformerEmbedding(vocab_size=5000, d_model=256)
        for batch_size in [1, 8, 32, 64]:
            tokens = torch.randint(0, 5000, (batch_size, 64))
            output = emb(tokens)
            assert output.shape == (batch_size, 64, 256)

    def test_different_seq_lengths(self):
        """Test with various sequence lengths."""
        emb = TransformerEmbedding(vocab_size=5000, d_model=256, max_len=1024)
        for seq_len in [10, 50, 128, 512]:
            tokens = torch.randint(0, 5000, (32, seq_len))
            output = emb(tokens)
            assert output.shape == (32, seq_len, 256)


class TestEmbeddingIntegration:
    """Integration tests for embeddings."""

    def test_token_plus_positional_magnitudes(self):
        """Test that token and positional embeddings have comparable magnitudes."""
        emb = TransformerEmbedding(
            vocab_size=10000,
            d_model=512,
            positional="sinusoidal",
            dropout=0.0
        )
        emb.eval()

        tokens = torch.randint(0, 10000, (32, 128))
        output = emb(tokens)

        # Check that output has reasonable magnitude
        output_norm = output.norm(dim=-1).mean()
        # Should be roughly √d_model due to scaling
        assert 10.0 < output_norm < 50.0

    def test_gradient_flow(self):
        """Test that gradients flow through embeddings."""
        emb = TransformerEmbedding(
            vocab_size=1000,
            d_model=128,
            positional="learned",
            dropout=0.0
        )

        tokens = torch.randint(0, 1000, (16, 32))
        output = emb(tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert emb.token_embedding.embedding.weight.grad is not None
        assert emb.positional_encoding.position_embeddings.weight.grad is not None

    def test_batch_processing_independence(self):
        """Test that batch samples are processed independently."""
        emb = TransformerEmbedding(
            vocab_size=5000,
            d_model=256,
            positional="sinusoidal",
            dropout=0.0
        )
        emb.eval()

        tokens = torch.randint(0, 5000, (16, 64))
        batch_output = emb(tokens)

        # Process individually
        for i in range(16):
            single_output = emb(tokens[i:i+1])
            assert torch.allclose(batch_output[i:i+1], single_output, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
