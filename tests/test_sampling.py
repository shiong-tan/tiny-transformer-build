"""
Comprehensive Tests for Sampling Package (Module 07).

Tests cover:
- Sampling strategies: greedy, temperature, top-k, top-p, combined
- TextGenerator: autoregressive generation, EOS handling
- Edge cases: extreme temperatures, boundary conditions
- Determinism and randomness properties

Run with: pytest tests/test_sampling.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from tiny_transformer.sampling import (
    greedy_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    combined_sample,
    TextGenerator,
    GeneratorConfig,
)
from tiny_transformer.model import TinyTransformerLM, get_model_config


# ============================================================================
# Test Greedy Sampling
# ============================================================================

class TestGreedySampling:
    """Test greedy sampling strategy."""

    def test_basic_functionality(self):
        """Test that greedy sampling returns highest probability token."""
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.1, 0.9]])  # (2, 3)

        tokens = greedy_sample(logits)

        # Should select argmax for each batch
        assert tokens.shape == (2,)
        assert tokens[0] == 1  # argmax([1.0, 3.0, 2.0])
        assert tokens[1] == 2  # argmax([0.5, 0.1, 0.9])

    def test_determinism(self):
        """Test that greedy sampling is deterministic."""
        logits = torch.randn(4, 100)

        tokens1 = greedy_sample(logits)
        tokens2 = greedy_sample(logits)

        # Should always return same tokens
        assert torch.equal(tokens1, tokens2)

    def test_batch_processing(self):
        """Test greedy sampling with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            logits = torch.randn(batch_size, 50)
            tokens = greedy_sample(logits)

            assert tokens.shape == (batch_size,)
            assert tokens.dtype == torch.long

    def test_large_vocabulary(self):
        """Test with large vocabulary size."""
        vocab_size = 10000
        logits = torch.randn(8, vocab_size)

        tokens = greedy_sample(logits)

        assert tokens.shape == (8,)
        assert (tokens >= 0).all()
        assert (tokens < vocab_size).all()


# ============================================================================
# Test Temperature Sampling
# ============================================================================

class TestTemperatureSampling:
    """Test temperature sampling strategy."""

    def test_basic_functionality(self):
        """Test basic temperature sampling."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = temperature_sample(logits, temperature=1.0)

        assert tokens.shape == (4,)
        assert tokens.dtype == torch.long

    def test_temperature_one_equals_softmax_sampling(self):
        """Test that temperature=1.0 samples from softmax distribution."""
        torch.manual_seed(42)
        logits = torch.randn(100, 50)

        # Sample many times
        samples = []
        for _ in range(1000):
            tokens = temperature_sample(logits, temperature=1.0)
            samples.append(tokens)

        samples = torch.stack(samples)

        # Should have variety (not all the same)
        unique_samples = torch.unique(samples, dim=0)
        assert unique_samples.size(0) > 1

    def test_low_temperature_more_deterministic(self):
        """Test that low temperature makes sampling more deterministic."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Sample multiple times with low temperature
        low_temp_samples = []
        for _ in range(20):
            tokens = temperature_sample(logits, temperature=0.1)
            low_temp_samples.append(tokens.item())

        # Most samples should be the same (close to greedy)
        from collections import Counter
        counts = Counter(low_temp_samples)
        most_common_freq = counts.most_common(1)[0][1]

        assert most_common_freq >= 15  # At least 75% the same

    def test_high_temperature_more_random(self):
        """Test that high temperature makes sampling more random."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Sample multiple times with high temperature
        high_temp_samples = []
        for _ in range(50):
            tokens = temperature_sample(logits, temperature=2.0)
            high_temp_samples.append(tokens.item())

        # Should have more variety
        unique_count = len(set(high_temp_samples))
        assert unique_count >= 10  # At least 10 different tokens

    def test_invalid_temperature_raises_error(self):
        """Test that temperature <= 0 raises error."""
        logits = torch.randn(4, 100)

        with pytest.raises(ValueError, match="Temperature must be > 0"):
            temperature_sample(logits, temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be > 0"):
            temperature_sample(logits, temperature=-1.0)

    def test_extreme_temperature_values(self):
        """Test sampling with extreme temperature values."""
        logits = torch.randn(4, 50)

        # Very low temperature (near greedy)
        tokens_low = temperature_sample(logits, temperature=0.01)
        assert tokens_low.shape == (4,)

        # Very high temperature (near uniform)
        tokens_high = temperature_sample(logits, temperature=10.0)
        assert tokens_high.shape == (4,)


# ============================================================================
# Test Top-K Sampling
# ============================================================================

class TestTopKSampling:
    """Test top-k sampling strategy."""

    def test_basic_functionality(self):
        """Test basic top-k sampling."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = top_k_sample(logits, k=10)

        assert tokens.shape == (4,)
        assert tokens.dtype == torch.long

    def test_samples_from_top_k(self):
        """Test that samples come from top-k tokens."""
        torch.manual_seed(42)
        batch_size = 1
        vocab_size = 100
        k = 5

        logits = torch.randn(batch_size, vocab_size)

        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)
        top_k_set = set(top_k_indices[0].tolist())

        # Sample many times
        samples = set()
        for _ in range(100):
            tokens = top_k_sample(logits, k=k)
            samples.add(tokens.item())

        # All samples should be in top-k set
        assert samples.issubset(top_k_set)

    def test_k_equals_one_is_greedy(self):
        """Test that k=1 is equivalent to greedy sampling."""
        logits = torch.randn(4, 100)

        greedy_tokens = greedy_sample(logits)
        top_k_tokens = top_k_sample(logits, k=1)

        assert torch.equal(greedy_tokens, top_k_tokens)

    def test_k_equals_vocab_size_is_normal_sampling(self):
        """Test that k=vocab_size samples from full distribution."""
        torch.manual_seed(42)
        vocab_size = 50
        logits = torch.randn(1, vocab_size)

        # Sample many times
        samples = []
        for _ in range(200):
            tokens = top_k_sample(logits, k=vocab_size, temperature=1.0)
            samples.append(tokens.item())

        # Should have good variety
        unique_count = len(set(samples))
        assert unique_count >= 20  # At least 40% of vocab

    def test_invalid_k_raises_error(self):
        """Test that k <= 0 raises error."""
        logits = torch.randn(4, 100)

        with pytest.raises(ValueError, match="k must be > 0"):
            top_k_sample(logits, k=0)

        with pytest.raises(ValueError, match="k must be > 0"):
            top_k_sample(logits, k=-1)

    def test_temperature_with_top_k(self):
        """Test temperature parameter with top-k."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Low temperature with top-k
        tokens_low = top_k_sample(logits, k=10, temperature=0.5)
        assert tokens_low.shape == (1,)

        # High temperature with top-k
        tokens_high = top_k_sample(logits, k=10, temperature=2.0)
        assert tokens_high.shape == (1,)


# ============================================================================
# Test Top-P (Nucleus) Sampling
# ============================================================================

class TestTopPSampling:
    """Test top-p (nucleus) sampling strategy."""

    def test_basic_functionality(self):
        """Test basic top-p sampling."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = top_p_sample(logits, p=0.9)

        assert tokens.shape == (4,)
        assert tokens.dtype == torch.long

    def test_p_equals_one_samples_full_distribution(self):
        """Test that p=1.0 samples from full distribution."""
        torch.manual_seed(42)
        vocab_size = 50
        logits = torch.randn(1, vocab_size)

        # Sample many times
        samples = []
        for _ in range(300):
            tokens = top_p_sample(logits, p=1.0)
            samples.append(tokens.item())

        # Should have good variety
        unique_count = len(set(samples))
        assert unique_count >= 20  # At least 40% of vocab

    def test_low_p_more_deterministic(self):
        """Test that low p makes sampling more deterministic."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Sample with very low p
        samples = []
        for _ in range(50):
            tokens = top_p_sample(logits, p=0.1)
            samples.append(tokens.item())

        # Should have less variety
        unique_count = len(set(samples))
        assert unique_count <= 10  # Limited to small nucleus

    def test_invalid_p_raises_error(self):
        """Test that p outside (0, 1] raises error."""
        logits = torch.randn(4, 100)

        with pytest.raises(ValueError, match="p must be in \\(0, 1\\]"):
            top_p_sample(logits, p=0.0)

        with pytest.raises(ValueError, match="p must be in \\(0, 1\\]"):
            top_p_sample(logits, p=1.5)

        with pytest.raises(ValueError, match="p must be in \\(0, 1\\]"):
            top_p_sample(logits, p=-0.5)

    def test_temperature_with_top_p(self):
        """Test temperature parameter with top-p."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        tokens_low = top_p_sample(logits, p=0.9, temperature=0.5)
        tokens_high = top_p_sample(logits, p=0.9, temperature=2.0)

        assert tokens_low.shape == (1,)
        assert tokens_high.shape == (1,)

    def test_always_samples_at_least_one_token(self):
        """Test that at least one token is always sampled."""
        # Create peaked distribution (one very high logit)
        logits = torch.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]])

        # Even with very low p, should sample something
        tokens = top_p_sample(logits, p=0.01)

        assert tokens.shape == (1,)
        # Should sample the peaked token
        assert tokens.item() == 2


# ============================================================================
# Test Combined Sampling
# ============================================================================

class TestCombinedSampling:
    """Test combined sampling (temperature + top-k + top-p)."""

    def test_basic_functionality(self):
        """Test combined sampling with all parameters."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = combined_sample(
            logits,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        assert tokens.shape == (4,)
        assert tokens.dtype == torch.long

    def test_temperature_only(self):
        """Test combined sampling with only temperature."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = combined_sample(logits, temperature=0.8)

        assert tokens.shape == (4,)

    def test_top_k_only(self):
        """Test combined sampling with only top-k."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = combined_sample(logits, top_k=10)

        assert tokens.shape == (4,)

    def test_top_p_only(self):
        """Test combined sampling with only top-p."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = combined_sample(logits, top_p=0.9)

        assert tokens.shape == (4,)

    def test_top_k_and_top_p(self):
        """Test combined sampling with both top-k and top-p."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        tokens = combined_sample(logits, top_k=50, top_p=0.9)

        assert tokens.shape == (4,)

    def test_all_parameters(self):
        """Test combined sampling with all parameters."""
        torch.manual_seed(42)
        logits = torch.randn(8, 200)

        tokens = combined_sample(
            logits,
            temperature=0.7,
            top_k=100,
            top_p=0.95
        )

        assert tokens.shape == (8,)


# ============================================================================
# Test TextGenerator
# ============================================================================

class TestTextGenerator:
    """Test TextGenerator class."""

    def test_initialization(self):
        """Test basic generator initialization."""
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)

        gen_config = GeneratorConfig(max_new_tokens=10)
        generator = TextGenerator(model, gen_config, device='cpu')

        assert generator.model is not None
        assert generator.config.max_new_tokens == 10
        assert generator.device == torch.device('cpu')

    def test_greedy_generation(self):
        """Test greedy text generation."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=10,
            do_sample=False  # Greedy
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))
        generated = generator.generate(start_tokens)

        # Should generate start_len + max_new_tokens
        assert generated.shape == (1, 5 + 10)

    def test_sampling_generation(self):
        """Test sampling-based generation."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=10,
            temperature=0.8,
            do_sample=True
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))
        generated = generator.generate(start_tokens)

        assert generated.shape == (1, 15)

    def test_top_k_generation(self):
        """Test generation with top-k sampling."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            do_sample=True
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))
        generated = generator.generate(start_tokens)

        assert generated.shape == (1, 15)

    def test_top_p_generation(self):
        """Test generation with top-p sampling."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))
        generated = generator.generate(start_tokens)

        assert generated.shape == (1, 15)

    def test_batch_generation(self):
        """Test batch generation."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(max_new_tokens=10, do_sample=False)
        generator = TextGenerator(model, gen_config, device='cpu')

        batch_size = 4
        start_tokens = torch.randint(0, 100, (batch_size, 5))
        generated = generator.generate(start_tokens)

        assert generated.shape == (batch_size, 15)

    def test_eos_token_handling(self):
        """Test that generation stops at EOS token."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        eos_token_id = 50
        gen_config = GeneratorConfig(
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=99
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))
        generated = generator.generate(start_tokens)

        # Length may be less than max if EOS was generated
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 5 + 20

    def test_deterministic_greedy(self):
        """Test that greedy generation is deterministic."""
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(max_new_tokens=10, do_sample=False)
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))

        generated1 = generator.generate(start_tokens)
        generated2 = generator.generate(start_tokens)

        # Greedy should be deterministic
        assert torch.equal(generated1, generated2)

    def test_override_config_parameters(self):
        """Test overriding config parameters at generation time."""
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(max_new_tokens=10, temperature=1.0)
        generator = TextGenerator(model, gen_config, device='cpu')

        start_tokens = torch.randint(0, 100, (1, 5))

        # Override max_new_tokens
        generated = generator.generate(start_tokens, max_new_tokens=20)

        assert generated.shape == (1, 5 + 20)  # Used overridden value

    def test_generate_batch_different_lengths(self):
        """Test generate_batch with different length inputs."""
        torch.manual_seed(42)
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=99
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        # Different length sequences
        start_1 = torch.tensor([1, 2, 3])
        start_2 = torch.tensor([4, 5])

        generated_list = generator.generate_batch([start_1, start_2])

        assert len(generated_list) == 2
        assert generated_list[0].shape[0] == 3 + 5
        assert generated_list[1].shape[0] == 2 + 5

    def test_generate_batch_requires_pad_token(self):
        """Test that generate_batch requires pad_token_id."""
        config = get_model_config('tiny')
        model = TinyTransformerLM(vocab_size=100, **config)
        model.eval()

        gen_config = GeneratorConfig(
            max_new_tokens=5,
            pad_token_id=None  # No pad token
        )
        generator = TextGenerator(model, gen_config, device='cpu')

        start_1 = torch.tensor([1, 2, 3])
        start_2 = torch.tensor([4, 5])

        with pytest.raises(ValueError, match="pad_token_id must be set"):
            generator.generate_batch([start_1, start_2])


# ============================================================================
# Test GeneratorConfig
# ============================================================================

class TestGeneratorConfig:
    """Test GeneratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()

        assert config.max_new_tokens == 50
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.do_sample is True
        assert config.eos_token_id is None
        assert config.pad_token_id is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = GeneratorConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=False,
            eos_token_id=1,
            pad_token_id=0
        )

        assert config.max_new_tokens == 100
        assert config.temperature == 0.8
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.do_sample is False
        assert config.eos_token_id == 1
        assert config.pad_token_id == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Running Sampling Package Tests")
    print("=" * 70)
    print()
    print("Run with: pytest tests/test_sampling.py -v")
    print()
