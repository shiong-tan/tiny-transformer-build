"""
Comprehensive tests for TinyTransformerLM.

Tests cover:
- Forward pass shapes across all configurations
- Weight tying mechanism and parameter reduction
- Parameter counting and breakdown accuracy
- Basic generation functionality
- Hidden states extraction
- Gradient flow through complete model
- Causal masking integration
- Model configurations (tiny/small/medium/large)
- Edge cases and validation
"""

import pytest
import torch
import torch.nn as nn

from tiny_transformer.model import TinyTransformerLM, get_model_config


class TestTinyTransformerLMShapes:
    """Test output shapes across different configurations."""

    def test_forward_pass_basic_shape(self):
        """Test basic forward pass produces correct output shape."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (16, 64))
        logits, _ = model(tokens)

        assert logits.shape == (16, 64, 1000)

    def test_different_batch_sizes(self):
        """Test model works with various batch sizes."""
        model = TinyTransformerLM(
            vocab_size=5000,
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024
        )

        for batch_size in [1, 4, 16, 32]:
            tokens = torch.randint(0, 5000, (batch_size, 128))
            logits, _ = model(tokens)
            assert logits.shape == (batch_size, 128, 5000)

    def test_different_sequence_lengths(self):
        """Test model works with various sequence lengths."""
        model = TinyTransformerLM(
            vocab_size=5000,
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            max_len=512
        )

        for seq_len in [10, 32, 128, 256]:
            tokens = torch.randint(0, 5000, (8, seq_len))
            logits, _ = model(tokens)
            assert logits.shape == (8, seq_len, 5000)

    def test_all_preset_configurations(self):
        """Test all preset model configurations."""
        vocab_size = 1000
        batch_size = 8
        seq_len = 64

        for size in ["tiny", "small", "medium", "large"]:
            config = get_model_config(size)
            model = TinyTransformerLM(vocab_size=vocab_size, **config)

            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits, _ = model(tokens)

            assert logits.shape == (batch_size, seq_len, vocab_size), \
                f"Failed for {size} config"


class TestTinyTransformerLMWeightTying:
    """Test weight tying mechanism."""

    def test_weight_tying_enabled(self):
        """Test that weight tying shares parameters correctly."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=True
        )

        # lm_head.weight should be the same object as embedding weight
        assert model.lm_head.weight is model.embedding.token_embedding.embedding.weight

    def test_weight_tying_disabled(self):
        """Test that weights are separate when tying is disabled."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=False
        )

        # Should be different weight objects
        assert model.lm_head.weight is not model.embedding.token_embedding.embedding.weight

    def test_weight_tying_reduces_parameters(self):
        """Test that weight tying reduces total parameter count."""
        # Model with weight tying
        model_tied = TinyTransformerLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            tie_weights=True
        )

        # Model without weight tying
        model_untied = TinyTransformerLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            tie_weights=False
        )

        params_tied = model_tied.count_parameters()
        params_untied = model_untied.count_parameters()

        # Tied should have exactly vocab_size * d_model fewer parameters
        expected_diff = 10000 * 512
        assert params_untied - params_tied == expected_diff

    def test_weight_tying_gradients_flow(self):
        """Test that gradients flow through tied weights."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=True
        )

        tokens = torch.randint(0, 1000, (8, 32))
        logits, _ = model(tokens)
        loss = logits.sum()
        loss.backward()

        # Both embedding and lm_head should have gradients (same weight)
        assert model.embedding.token_embedding.embedding.weight.grad is not None
        assert model.lm_head.weight.grad is not None

        # They should be the same gradient object
        assert model.embedding.token_embedding.embedding.weight.grad is \
               model.lm_head.weight.grad


class TestTinyTransformerLMParameterCounting:
    """Test parameter counting utilities."""

    def test_count_parameters(self):
        """Test total parameter counting."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=True
        )

        # Count manually
        manual_count = sum(p.numel() for p in model.parameters())
        method_count = model.count_parameters()

        assert manual_count == method_count

    def test_count_trainable_parameters(self):
        """Test counting only trainable parameters."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        # Freeze some parameters
        for param in model.blocks[0].parameters():
            param.requires_grad = False

        trainable_count = model.count_parameters(trainable_only=True)
        total_count = model.count_parameters(trainable_only=False)

        assert trainable_count < total_count

    def test_parameter_breakdown(self):
        """Test parameter breakdown by component."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=True
        )

        breakdown = model.get_parameter_breakdown()

        # Check that all components are present
        assert 'embedding' in breakdown
        assert 'transformer_blocks' in breakdown
        assert 'ln_f' in breakdown
        assert 'lm_head' in breakdown
        assert 'total' in breakdown

        # Check that breakdown sums to total (accounting for weight tying)
        assert breakdown['total'] == model.count_parameters()

    def test_parameter_breakdown_without_tying(self):
        """Test parameter breakdown when weight tying is disabled."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            tie_weights=False
        )

        breakdown = model.get_parameter_breakdown()

        # Without weight tying, components should sum directly to total
        component_sum = (breakdown['embedding'] +
                        breakdown['transformer_blocks'] +
                        breakdown['ln_f'] +
                        breakdown['lm_head'])

        assert breakdown['total'] == component_sum


class TestTinyTransformerLMGeneration:
    """Test basic generation functionality."""

    def test_generate_produces_correct_shape(self):
        """Test that generate produces correct output shape."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        start_tokens = torch.randint(0, 1000, (1, 10))
        max_new_tokens = 20

        generated = model.generate(start_tokens, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, 10 + max_new_tokens)

    def test_generate_deterministic_with_seed(self):
        """Test that generation is reproducible with same seed."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        start_tokens = torch.randint(0, 1000, (1, 10))

        # Generate twice with same seed
        torch.manual_seed(42)
        gen1 = model.generate(start_tokens, max_new_tokens=20, temperature=1.0)

        torch.manual_seed(42)
        gen2 = model.generate(start_tokens, max_new_tokens=20, temperature=1.0)

        assert torch.equal(gen1, gen2)

    def test_generate_batch(self):
        """Test generation with batch size > 1."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        batch_size = 4
        start_tokens = torch.randint(0, 1000, (batch_size, 10))
        max_new_tokens = 15

        generated = model.generate(start_tokens, max_new_tokens=max_new_tokens)

        assert generated.shape == (batch_size, 10 + max_new_tokens)

    def test_generate_respects_max_len(self):
        """Test that generation properly truncates context when exceeding max_len."""
        max_len = 64
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_len=max_len
        )
        model.eval()

        # Create deterministic input longer than max_len
        torch.manual_seed(42)
        start_tokens = torch.randint(0, 1000, (1, 100))

        # Generate with truncation
        torch.manual_seed(123)  # Different seed for generation
        generated = model.generate(start_tokens, max_new_tokens=5, temperature=1.0)

        # Verify shape
        assert generated.shape == (1, 105)

        # Verify that the model actually used truncated context
        # Generate from just the last max_len tokens directly
        truncated_start = start_tokens[:, -max_len:]
        torch.manual_seed(123)  # Same seed for deterministic comparison
        generated_from_truncated = model.generate(truncated_start, max_new_tokens=5, temperature=1.0)

        # The newly generated tokens should match
        assert torch.equal(generated[:, -5:], generated_from_truncated[:, -5:])

    def test_generate_with_temperature(self):
        """Test generation with different temperatures."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        start_tokens = torch.randint(0, 1000, (1, 10))

        # Should work with various temperatures
        for temp in [0.5, 1.0, 2.0]:
            generated = model.generate(
                start_tokens,
                max_new_tokens=10,
                temperature=temp
            )
            assert generated.shape == (1, 20)


class TestTinyTransformerLMHiddenStates:
    """Test hidden state extraction."""

    def test_return_hidden_states(self):
        """Test that hidden states are returned when requested."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 32))
        logits, hidden_states = model(tokens, return_hidden_states=True)

        assert hidden_states is not None
        assert isinstance(hidden_states, list)

    def test_hidden_states_count(self):
        """Test correct number of hidden states."""
        n_layers = 4
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=n_layers,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 32))
        _, hidden_states = model(tokens, return_hidden_states=True)

        # Should have: embedding output + n_layers block outputs
        assert len(hidden_states) == n_layers + 1

    def test_hidden_states_shapes(self):
        """Test that all hidden states have correct shapes."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=512
        )

        batch_size, seq_len = 8, 32
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        _, hidden_states = model(tokens, return_hidden_states=True)

        # All hidden states should have shape (batch_size, seq_len, d_model)
        for h in hidden_states:
            assert h.shape == (batch_size, seq_len, 128)

    def test_no_hidden_states_by_default(self):
        """Test that hidden states are not returned by default."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 32))
        logits, hidden_states = model(tokens, return_hidden_states=False)

        assert hidden_states is None


class TestTinyTransformerLMGradients:
    """Test gradient flow through the model."""

    def test_gradient_flow_all_parameters(self):
        """Test that gradients flow to all parameters."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 32))
        logits, _ = model(tokens)
        loss = logits.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_flow_with_loss(self):
        """Test gradient flow with realistic loss computation."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 32))
        targets = torch.randint(0, 1000, (8, 32))

        logits, _ = model(tokens)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 1000),
            targets.view(-1)
        )

        loss.backward()

        # All parameters should have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        # First backward pass
        tokens1 = torch.randint(0, 1000, (8, 32))
        logits1, _ = model(tokens1)
        loss1 = logits1.sum()
        loss1.backward()

        # Store gradients
        first_grad = model.embedding.token_embedding.embedding.weight.grad.clone()

        # Second backward pass (accumulates)
        tokens2 = torch.randint(0, 1000, (8, 32))
        logits2, _ = model(tokens2)
        loss2 = logits2.sum()
        loss2.backward()

        # Gradients should have accumulated
        accumulated_grad = model.embedding.token_embedding.embedding.weight.grad

        assert not torch.allclose(first_grad, accumulated_grad)


class TestTinyTransformerLMCausalMasking:
    """Test causal masking integration."""

    def test_default_causal_mask(self):
        """Test that default mask is causal (lower triangular)."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (2, 32))

        # Forward with default mask (should be causal)
        logits_default, _ = model(tokens)

        # Forward with explicit causal mask
        seq_len = 32
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        logits_explicit, _ = model(tokens, mask=causal_mask)

        # Should produce same results
        assert torch.allclose(logits_default, logits_explicit, atol=1e-6)

    def test_custom_mask(self):
        """Test that custom masks are respected."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (2, 32))

        # Create a fully visible mask (all ones)
        full_mask = torch.ones(32, 32)

        logits_full, _ = model(tokens, mask=full_mask)

        # Create causal mask
        causal_mask = torch.tril(torch.ones(32, 32))
        logits_causal, _ = model(tokens, mask=causal_mask)

        # Should produce different results
        assert not torch.allclose(logits_full, logits_causal, atol=1e-3)


class TestTinyTransformerLMConfigurations:
    """Test preset model configurations."""

    def test_all_configs_valid(self):
        """Test that all preset configs are valid."""
        for size in ["tiny", "small", "medium", "large"]:
            config = get_model_config(size)

            # Check required keys
            assert 'd_model' in config
            assert 'n_heads' in config
            assert 'n_layers' in config
            assert 'd_ff' in config
            assert 'max_len' in config
            assert 'dropout' in config

    def test_config_creates_valid_model(self):
        """Test that configs create functional models."""
        for size in ["tiny", "small", "medium", "large"]:
            config = get_model_config(size)
            model = TinyTransformerLM(vocab_size=1000, **config)

            # Test forward pass
            tokens = torch.randint(0, 1000, (4, 64))
            logits, _ = model(tokens)

            assert logits.shape == (4, 64, 1000)

    def test_invalid_config_raises_error(self):
        """Test that invalid config name raises error."""
        with pytest.raises(ValueError):
            get_model_config("invalid_size")


class TestTinyTransformerLMValidation:
    """Test input validation and error handling."""

    def test_invalid_input_shape(self):
        """Test that invalid input shapes are caught."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        # 1D input should fail
        tokens_1d = torch.randint(0, 1000, (32,))
        with pytest.raises(AssertionError):
            model(tokens_1d)

        # 3D input should fail
        tokens_3d = torch.randint(0, 1000, (8, 32, 128))
        with pytest.raises(AssertionError):
            model(tokens_3d)

    def test_positional_encoding_types(self):
        """Test both positional encoding types work."""
        for positional in ["sinusoidal", "learned"]:
            model = TinyTransformerLM(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                positional=positional
            )

            tokens = torch.randint(0, 1000, (8, 32))
            logits, _ = model(tokens)

            assert logits.shape == (8, 32, 1000)

    def test_padding_idx(self):
        """Test model with padding index."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            padding_idx=0
        )

        tokens = torch.randint(1, 1000, (8, 32))
        tokens[:, :5] = 0  # Add padding

        logits, _ = model(tokens)
        assert logits.shape == (8, 32, 1000)


class TestTinyTransformerLMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_input(self):
        """Test with single token input."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (8, 1))
        logits, _ = model(tokens)

        assert logits.shape == (8, 1, 1000)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )

        tokens = torch.randint(0, 1000, (1, 32))
        logits, _ = model(tokens)

        assert logits.shape == (1, 32, 1000)

    def test_max_length_input(self):
        """Test with input at max_len."""
        max_len = 512
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_len=max_len
        )

        tokens = torch.randint(0, 1000, (4, max_len))
        logits, _ = model(tokens)

        assert logits.shape == (4, max_len, 1000)

    def test_forward_exceeding_max_len_fails(self):
        """Test that forward pass with seq_len > max_len raises error."""
        max_len = 100
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_len=max_len
        )

        # Sequence longer than max_len
        tokens = torch.randint(0, 1000, (8, max_len + 1))

        with pytest.raises(AssertionError):
            model(tokens)

    def test_batch_processing_independence(self):
        """Test that batch samples are processed independently."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            dropout=0.0
        )
        model.eval()

        # Process batch
        batch_tokens = torch.randint(0, 1000, (16, 32))
        batch_logits, _ = model(batch_tokens)

        # Process samples individually
        for i in range(16):
            single_tokens = batch_tokens[i:i+1]
            single_logits, _ = model(single_tokens)
            assert torch.allclose(batch_logits[i:i+1], single_logits, atol=1e-5), \
                f"Sample {i} differs between batch and individual processing"


class TestTinyTransformerLMNumericalStability:
    """Test numerical stability with various inputs."""

    def test_no_nans_or_infs(self):
        """Test model produces valid outputs without NaN or Inf."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=6,  # Deeper stack
            d_ff=512,
            dropout=0.0
        )
        model.eval()

        tokens = torch.randint(0, 1000, (16, 32))
        logits, _ = model(tokens)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_deep_stack_stability(self):
        """Test that deep stacks don't cause numerical issues."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=24,  # Very deep
            d_ff=1024,
            dropout=0.0
        )
        model.eval()

        tokens = torch.randint(0, 1000, (8, 64))
        logits, _ = model(tokens)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        # Check that logits are in reasonable range
        assert logits.abs().max() < 1e3

    def test_gradient_norms_bounded(self):
        """Test that gradients don't explode."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=12,
            d_ff=512,
            dropout=0.0
        )

        tokens = torch.randint(0, 1000, (16, 32))
        targets = torch.randint(0, 1000, (16, 32))

        logits, _ = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 1000),
            targets.view(-1)
        )
        loss.backward()

        # Check gradient norms are bounded
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert not torch.isnan(grad_norm)
                assert not torch.isinf(grad_norm)
                assert grad_norm < 1e3, f"Gradient explosion in {name}"


class TestTinyTransformerLMDropout:
    """Test dropout behavior in the model."""

    def test_dropout_active_in_train_mode(self):
        """Test that dropout causes non-determinism in training."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.5
        )
        model.train()

        tokens = torch.randint(0, 1000, (16, 32))

        logits1, _ = model(tokens)
        logits2, _ = model(tokens)

        # Should be different due to dropout
        assert not torch.allclose(logits1, logits2, atol=1e-6)

    def test_dropout_disabled_in_eval_mode(self):
        """Test that dropout is disabled during evaluation."""
        model = TinyTransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.5
        )
        model.eval()

        tokens = torch.randint(0, 1000, (16, 32))

        logits1, _ = model(tokens)
        logits2, _ = model(tokens)

        # Should be identical in eval mode
        assert torch.allclose(logits1, logits2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
