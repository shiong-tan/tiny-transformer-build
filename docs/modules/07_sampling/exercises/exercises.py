"""
Module 07: Sampling & Generation - Exercises

This module contains hands-on exercises for learning text generation and
sampling strategies. You'll practice:

1. Implementing greedy sampling
2. Temperature-based sampling for controlling randomness
3. Top-k sampling strategies
4. Top-p (nucleus) sampling
5. Combined sampling approaches
6. Using TextGenerator for autoregressive generation
7. Analyzing sampling temperature effects
8. Building custom generators with EOS handling
9. Comparing different sampling strategies
10. Creating interactive text completion systems
11. Temperature scheduling during generation
12. Building creative story continuation systems

Each exercise includes:
- Clear docstrings with learning objectives
- Type hints and shape annotations
- TODO sections for implementation
- Test assertions for self-assessment
- Progressive difficulty (Easy → Medium → Hard → Very Hard)

Prerequisites:
- Completed Modules 00-06 (especially Module 06 on training)
- Understanding of probability distributions
- Familiarity with PyTorch sampling operations
- Basic knowledge of softmax and multinomial sampling

Reference implementations:
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/sampling/strategies.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/sampling/generator.py

Theory reference:
- /Users/shiongtan/projects/tiny-transformer-build/docs/modules/07_sampling/theory.md

Time estimate: 5-6 hours for all exercises

Good luck! Let's generate some text!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math


# ==============================================================================
# EXERCISE 1: Implement Greedy Sampling
# Difficulty: Easy
# ==============================================================================

def exercise_01_greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """
    EXERCISE 01: Implement greedy sampling (always pick highest probability).

    Learning Objectives:
    - Understand greedy decoding strategy
    - Learn when greedy sampling is appropriate
    - Practice argmax operation on logits

    Greedy sampling always selects the token with highest probability:
    - Most deterministic strategy
    - No randomness involved
    - Often produces repetitive text
    - Fast and simple

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size)

    Returns:
        Token IDs of shape (batch_size,)

    Shape:
        Input: (batch_size, vocab_size)
        Output: (batch_size,)

    Example:
        >>> logits = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.1, 0.8]])
        >>> tokens = exercise_01_greedy_sampling(logits)
        >>> tokens.tolist()
        [1, 2]  # Indices of maximum values

    Theory reference: theory.md, Section "Greedy Sampling"

    Self-Assessment Questions:
    1. Why is greedy sampling deterministic?
    2. What are the disadvantages of always picking the highest probability?
    3. When might you want to use greedy sampling?
    """
    # TODO: Implement greedy sampling

    # Step 1: Find index of maximum value along vocabulary dimension
    # Hint: Use torch.argmax() on the last dimension
    # tokens = logits.argmax(dim=-1)

    # Step 2: Return the token indices
    # return tokens

    pass

    # Uncomment to test:
    # assert tokens.shape == (logits.shape[0],), f"Expected shape ({logits.shape[0]},), got {tokens.shape}"
    # assert tokens.dtype == torch.long, "Token IDs should be long integers"


# ==============================================================================
# EXERCISE 2: Implement Temperature Sampling
# Difficulty: Easy
# ==============================================================================

def exercise_02_temperature_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    EXERCISE 02: Implement temperature sampling to control randomness.

    Learning Objectives:
    - Understand how temperature affects probability distributions
    - Learn to scale logits before sampling
    - Practice multinomial sampling

    Temperature controls randomness:
    - T < 1.0: More confident (peaked distribution)
    - T = 1.0: Normal distribution
    - T > 1.0: More random (flattened distribution)
    - T → 0: Approaches greedy sampling
    - T → ∞: Approaches uniform sampling

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size)
        temperature: Temperature value (must be > 0)

    Returns:
        Token IDs of shape (batch_size,)

    Shape:
        Input: (batch_size, vocab_size)
        Output: (batch_size,)

    Example:
        >>> torch.manual_seed(42)
        >>> logits = torch.randn(2, 100)
        >>> tokens = exercise_02_temperature_sampling(logits, temperature=0.8)
        >>> tokens.shape
        torch.Size([2])

    Theory reference: theory.md, Section "Temperature Sampling"

    Self-Assessment Questions:
    1. What happens to probabilities when temperature is very low?
    2. How does temperature affect text diversity?
    3. Why divide by temperature instead of multiply?
    """
    # TODO: Implement temperature sampling

    # Step 1: Validate temperature
    # if temperature <= 0:
    #     raise ValueError(f"Temperature must be > 0, got {temperature}")

    # Step 2: Scale logits by temperature
    # scaled_logits = logits / temperature

    # Step 3: Convert scaled logits to probabilities using softmax
    # probs = F.softmax(scaled_logits, dim=-1)

    # Step 4: Sample from probability distribution
    # Use torch.multinomial to sample one token per batch
    # tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Step 5: Return sampled tokens
    # return tokens

    pass

    # Uncomment to test:
    # assert tokens.shape == (logits.shape[0],), f"Expected shape ({logits.shape[0]},), got {tokens.shape}"
    # assert tokens.dtype == torch.long, "Token IDs should be long integers"


# ==============================================================================
# EXERCISE 3: Compare Greedy vs Temperature
# Difficulty: Easy
# ==============================================================================

def exercise_03_compare_sampling_strategies(
    logits: torch.Tensor,
    num_samples: int = 5
) -> Dict[str, List[int]]:
    """
    EXERCISE 03: Compare greedy and temperature sampling behaviors.

    Learning Objectives:
    - Observe differences between deterministic and stochastic sampling
    - Understand temperature's effect on diversity
    - Practice analyzing sampling outputs

    You'll generate multiple samples with:
    - Greedy (deterministic - should be identical)
    - Temperature 0.5 (low randomness)
    - Temperature 1.0 (normal randomness)
    - Temperature 2.0 (high randomness)

    Args:
        logits: Logits for a single example of shape (vocab_size,)
        num_samples: Number of samples to generate for each strategy

    Returns:
        Dictionary mapping strategy name to list of sampled tokens

    Example:
        >>> torch.manual_seed(42)
        >>> logits = torch.randn(100)
        >>> results = exercise_03_compare_sampling_strategies(logits, num_samples=5)
        >>> len(results['greedy'])  # Should all be identical
        5
        >>> len(set(results['temp_2.0']))  # Should have variety
        > 1

    Theory reference: theory.md, Section "Comparing Sampling Strategies"

    Self-Assessment Questions:
    1. Why are all greedy samples identical?
    2. Which temperature setting gives most diverse outputs?
    3. How would you choose temperature for creative writing vs. factual Q&A?
    """
    # TODO: Implement comparison

    # Ensure logits is 2D (batch_size=1, vocab_size)
    # if logits.dim() == 1:
    #     logits = logits.unsqueeze(0)

    # results = {}

    # Step 1: Greedy sampling (should be deterministic)
    # greedy_samples = []
    # for _ in range(num_samples):
    #     token = exercise_01_greedy_sampling(logits)
    #     greedy_samples.append(token.item())
    # results['greedy'] = greedy_samples

    # Step 2: Temperature sampling with different values
    # for temp in [0.5, 1.0, 2.0]:
    #     temp_samples = []
    #     for _ in range(num_samples):
    #         token = exercise_02_temperature_sampling(logits, temperature=temp)
    #         temp_samples.append(token.item())
    #     results[f'temp_{temp}'] = temp_samples

    # return results

    pass

    # Uncomment to test:
    # assert len(results) == 4, "Should have 4 strategies"
    # assert len(set(results['greedy'])) == 1, "Greedy should be deterministic"


# ==============================================================================
# EXERCISE 4: Visualize Temperature Effects
# Difficulty: Easy
# ==============================================================================

def exercise_04_visualize_temperature(
    logits: torch.Tensor,
    temperatures: List[float] = [0.5, 1.0, 2.0, 5.0]
) -> Dict[str, torch.Tensor]:
    """
    EXERCISE 04: Compute probability distributions for different temperatures.

    Learning Objectives:
    - Visualize how temperature reshapes distributions
    - Understand the relationship between temperature and entropy
    - Learn to compute probability distributions

    For each temperature, compute the probability distribution after
    temperature scaling. This helps visualize how temperature affects
    the sharpness/flatness of the distribution.

    Args:
        logits: Logits of shape (vocab_size,)
        temperatures: List of temperature values to test

    Returns:
        Dictionary mapping temperature to probability distribution

    Example:
        >>> logits = torch.tensor([2.0, 1.0, 0.5])
        >>> dists = exercise_04_visualize_temperature(logits)
        >>> dists[0.5].sum().item()  # Should sum to 1.0
        1.0

    Theory reference: theory.md, Section "Temperature Sampling"

    Self-Assessment Questions:
    1. How does the distribution change as temperature increases?
    2. What is the entropy of the distribution at different temperatures?
    3. At what temperature does the distribution become nearly uniform?
    """
    # TODO: Implement visualization

    # Ensure logits is 1D
    # if logits.dim() > 1:
    #     logits = logits.squeeze()

    # distributions = {}

    # For each temperature, compute the probability distribution
    # for temp in temperatures:
    #     # Step 1: Scale logits
    #     scaled_logits = logits / temp
    #
    #     # Step 2: Convert to probabilities
    #     probs = F.softmax(scaled_logits, dim=-1)
    #
    #     # Step 3: Store distribution
    #     distributions[temp] = probs

    # return distributions

    pass

    # Uncomment to test:
    # for temp, probs in distributions.items():
    #     assert torch.allclose(probs.sum(), torch.tensor(1.0)), f"Probs should sum to 1 at T={temp}"


# ==============================================================================
# EXERCISE 5: Implement Top-K Sampling
# Difficulty: Medium
# ==============================================================================

def exercise_05_top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    EXERCISE 05: Implement top-k sampling.

    Learning Objectives:
    - Understand top-k filtering strategy
    - Learn to filter probability distributions
    - Practice combining filtering with temperature

    Top-k sampling:
    - Select only the k tokens with highest probabilities
    - Set all other probabilities to zero
    - Sample from the filtered distribution
    - Prevents sampling unlikely tokens

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        k: Number of top tokens to consider
        temperature: Temperature for sampling

    Returns:
        Token IDs of shape (batch_size,)

    Shape:
        Input: (batch_size, vocab_size)
        Output: (batch_size,)

    Example:
        >>> torch.manual_seed(42)
        >>> logits = torch.randn(2, 100)
        >>> tokens = exercise_05_top_k_sampling(logits, k=10, temperature=1.0)
        >>> tokens.shape
        torch.Size([2])

    Theory reference: theory.md, Section "Top-K Sampling"

    Self-Assessment Questions:
    1. Why is top-k useful for preventing unlikely tokens?
    2. How do you choose the value of k?
    3. What happens when k=1? When k=vocab_size?
    """
    # TODO: Implement top-k sampling

    # Step 1: Validate k
    # if k <= 0:
    #     raise ValueError(f"k must be > 0, got {k}")

    # Step 2: Apply temperature scaling
    # if temperature != 1.0:
    #     logits = logits / temperature

    # Step 3: Find top-k values and indices
    # Use torch.topk to get the k largest logits and their indices
    # top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # Step 4: Create filtered logits (all -inf except top-k)
    # filtered_logits = torch.full_like(logits, float('-inf'))
    # filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Step 5: Convert to probabilities and sample
    # probs = F.softmax(filtered_logits, dim=-1)
    # tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # return tokens

    pass

    # Uncomment to test:
    # assert tokens.shape == (logits.shape[0],), f"Expected shape ({logits.shape[0]},), got {tokens.shape}"


# ==============================================================================
# EXERCISE 6: Implement Top-P (Nucleus) Sampling
# Difficulty: Medium
# ==============================================================================

def exercise_06_top_p_sampling(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    EXERCISE 06: Implement top-p (nucleus) sampling.

    Learning Objectives:
    - Understand nucleus sampling strategy
    - Learn dynamic vocabulary filtering
    - Practice cumulative probability computation

    Top-p (nucleus) sampling:
    - Select smallest set of tokens whose cumulative probability >= p
    - Adapts to distribution shape (unlike fixed k in top-k)
    - More tokens selected when distribution is flat
    - Fewer tokens selected when distribution is peaked

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        p: Cumulative probability threshold (0 < p <= 1)
        temperature: Temperature for sampling

    Returns:
        Token IDs of shape (batch_size,)

    Shape:
        Input: (batch_size, vocab_size)
        Output: (batch_size,)

    Example:
        >>> torch.manual_seed(42)
        >>> logits = torch.randn(2, 100)
        >>> tokens = exercise_06_top_p_sampling(logits, p=0.9, temperature=1.0)
        >>> tokens.shape
        torch.Size([2])

    Theory reference: theory.md, Section "Top-P (Nucleus) Sampling"

    Self-Assessment Questions:
    1. Why is top-p more adaptive than top-k?
    2. What happens when p=1.0? When p→0?
    3. How would you combine top-k and top-p?
    """
    # TODO: Implement top-p sampling

    # Step 1: Validate p
    # if not 0 < p <= 1:
    #     raise ValueError(f"p must be in (0, 1], got {p}")

    # Step 2: Apply temperature scaling
    # if temperature != 1.0:
    #     logits = logits / temperature

    # Step 3: Convert to probabilities
    # probs = F.softmax(logits, dim=-1)

    # Step 4: Sort probabilities in descending order
    # sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Step 5: Compute cumulative probabilities
    # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Step 6: Create mask for tokens to remove
    # Remove tokens where cumsum > p (but keep at least one token)
    # sorted_indices_to_remove = cumulative_probs > p
    # Shift right by 1 to keep the first token that exceeds p
    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # sorted_indices_to_remove[..., 0] = False

    # Step 7: Map mask back to original order
    # indices_to_remove = sorted_indices_to_remove.scatter(
    #     dim=-1,
    #     index=sorted_indices,
    #     src=sorted_indices_to_remove
    # )

    # Step 8: Set removed indices to -inf in logits
    # filtered_logits = logits.clone()
    # filtered_logits[indices_to_remove] = float('-inf')

    # Step 9: Sample from filtered distribution
    # filtered_probs = F.softmax(filtered_logits, dim=-1)
    # tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)

    # return tokens

    pass

    # Uncomment to test:
    # assert tokens.shape == (logits.shape[0],), f"Expected shape ({logits.shape[0]},), got {tokens.shape}"


# ==============================================================================
# EXERCISE 7: Use TextGenerator for Basic Generation
# Difficulty: Medium
# ==============================================================================

def exercise_07_use_text_generator(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    EXERCISE 07: Use TextGenerator to generate text from a model.

    Learning Objectives:
    - Learn to use the high-level TextGenerator interface
    - Understand autoregressive generation loop
    - Practice configuring generation parameters

    TextGenerator provides a clean interface for:
    - Autoregressive text generation
    - Multiple sampling strategies
    - Batch generation
    - EOS token handling

    Args:
        model: TinyTransformerLM or compatible model
        start_tokens: Starting sequence of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        temperature: Temperature for sampling

    Returns:
        Generated tokens of shape (batch_size, seq_len + max_new_tokens)

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        >>> start = torch.randint(0, 100, (2, 10))
        >>> generated = exercise_07_use_text_generator(model, start, max_new_tokens=20)
        >>> generated.shape
        torch.Size([2, 30])

    Theory reference: theory.md, Section "Autoregressive Generation"

    Self-Assessment Questions:
    1. What is autoregressive generation?
    2. Why do we generate one token at a time?
    3. How does the model use previously generated tokens?
    """
    # TODO: Implement text generation

    # Step 1: Import TextGenerator and GeneratorConfig
    # from tiny_transformer.sampling import TextGenerator, GeneratorConfig

    # Step 2: Create generator configuration
    # config = GeneratorConfig(
    #     max_new_tokens=max_new_tokens,
    #     temperature=temperature,
    #     do_sample=True  # Use sampling instead of greedy
    # )

    # Step 3: Create TextGenerator
    # generator = TextGenerator(model, config)

    # Step 4: Generate text
    # generated = generator.generate(start_tokens)

    # return generated

    pass

    # Uncomment to test:
    # expected_len = start_tokens.size(1) + max_new_tokens
    # assert generated.shape[1] == expected_len, f"Expected length {expected_len}, got {generated.shape[1]}"


# ==============================================================================
# EXERCISE 8: Implement Combined Sampling Strategy
# Difficulty: Hard
# ==============================================================================

def exercise_08_combined_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    EXERCISE 08: Implement combined sampling with temperature + top-k + top-p.

    Learning Objectives:
    - Combine multiple sampling strategies
    - Understand the order of operations
    - Practice building flexible sampling functions

    Combined sampling applies filters in sequence:
    1. Temperature scaling
    2. Top-k filtering (if enabled)
    3. Top-p filtering (if enabled)
    4. Sample from final distribution

    This gives fine-grained control over generation quality and diversity.

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        temperature: Temperature scaling
        top_k: Optional top-k filtering
        top_p: Optional nucleus sampling threshold

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> torch.manual_seed(42)
        >>> logits = torch.randn(2, 1000)
        >>> tokens = exercise_08_combined_sampling(
        ...     logits,
        ...     temperature=0.8,
        ...     top_k=50,
        ...     top_p=0.9
        ... )

    Theory reference: theory.md, Section "Combined Sampling"

    Self-Assessment Questions:
    1. Why apply temperature before top-k and top-p?
    2. What's the interaction between top-k and top-p?
    3. How would you choose values for all three parameters?
    """
    # TODO: Implement combined sampling

    # Step 1: Apply temperature scaling
    # if temperature != 1.0:
    #     logits = logits / temperature

    # Step 2: Apply top-k filtering (if specified)
    # if top_k is not None and top_k > 0:
    #     # Find top-k values
    #     top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    #
    #     # Create filtered logits
    #     filtered_logits = torch.full_like(logits, float('-inf'))
    #     filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
    #     logits = filtered_logits

    # Step 3: Apply top-p filtering (if specified)
    # if top_p is not None and 0 < top_p < 1:
    #     # Convert to probabilities
    #     probs = F.softmax(logits, dim=-1)
    #
    #     # Sort probabilities
    #     sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    #     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #
    #     # Create removal mask
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = False
    #
    #     # Map back to original order
    #     indices_to_remove = sorted_indices_to_remove.scatter(
    #         dim=-1,
    #         index=sorted_indices,
    #         src=sorted_indices_to_remove
    #     )
    #
    #     # Filter logits
    #     logits[indices_to_remove] = float('-inf')

    # Step 4: Sample from final distribution
    # probs = F.softmax(logits, dim=-1)
    # tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # return tokens

    pass


# ==============================================================================
# EXERCISE 9: Analyze Sampling Temperature Effects
# Difficulty: Medium
# ==============================================================================

def exercise_09_analyze_temperature_effects(
    logits: torch.Tensor,
    temperatures: List[float] = [0.1, 0.5, 1.0, 1.5, 2.0],
    num_samples: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    EXERCISE 09: Analyze how temperature affects sampling statistics.

    Learning Objectives:
    - Measure temperature's effect on diversity
    - Compute statistical metrics for sampling
    - Understand entropy and diversity metrics

    For each temperature, compute:
    - Unique token ratio (diversity)
    - Entropy of sampled distribution
    - Most frequent token percentage

    Args:
        logits: Logits of shape (vocab_size,)
        temperatures: List of temperatures to analyze
        num_samples: Number of samples per temperature

    Returns:
        Dictionary mapping temperature to statistics dict

    Example:
        >>> logits = torch.randn(100)
        >>> stats = exercise_09_analyze_temperature_effects(logits)
        >>> stats[0.1]['diversity'] < stats[2.0]['diversity']
        True  # Higher temp = more diversity

    Theory reference: theory.md, Section "Temperature Sampling"

    Self-Assessment Questions:
    1. How does temperature affect the unique token ratio?
    2. What is the relationship between temperature and entropy?
    3. At what temperature do you get maximum diversity?
    """
    # TODO: Implement temperature analysis

    # Ensure logits is 2D
    # if logits.dim() == 1:
    #     logits = logits.unsqueeze(0)

    # results = {}

    # for temp in temperatures:
    #     # Sample num_samples tokens
    #     samples = []
    #     for _ in range(num_samples):
    #         token = exercise_02_temperature_sampling(logits, temperature=temp)
    #         samples.append(token.item())
    #
    #     # Compute statistics
    #     unique_tokens = len(set(samples))
    #     diversity = unique_tokens / num_samples
    #
    #     # Compute empirical distribution
    #     token_counts = {}
    #     for token in samples:
    #         token_counts[token] = token_counts.get(token, 0) + 1
    #
    #     # Entropy of empirical distribution
    #     entropy = 0.0
    #     for count in token_counts.values():
    #         p = count / num_samples
    #         if p > 0:
    #             entropy -= p * math.log2(p)
    #
    #     # Most frequent token percentage
    #     max_count = max(token_counts.values())
    #     max_freq = max_count / num_samples
    #
    #     results[temp] = {
    #         'diversity': diversity,
    #         'entropy': entropy,
    #         'max_frequency': max_freq,
    #         'unique_tokens': unique_tokens
    #     }

    # return results

    pass


# ==============================================================================
# EXERCISE 10: Build Custom Generator with EOS Handling
# Difficulty: Hard
# ==============================================================================

class Exercise10_CustomGenerator:
    """
    EXERCISE 10: Build a custom text generator with EOS token handling.

    Learning Objectives:
    - Implement autoregressive generation loop
    - Handle end-of-sequence tokens
    - Support batched generation with early stopping
    - Practice state tracking during generation

    The generator should:
    - Generate tokens autoregressively
    - Stop when EOS token is generated
    - Support batch generation
    - Handle sequences of different lengths

    Args:
        model: PyTorch model (TinyTransformerLM)
        eos_token_id: Token ID for end-of-sequence
        pad_token_id: Token ID for padding
        device: Device for generation

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        >>> generator = Exercise10_CustomGenerator(model, eos_token_id=0, pad_token_id=1)
        >>> start = torch.randint(2, 100, (2, 5))
        >>> output = generator.generate(start, max_new_tokens=20)

    Theory reference: theory.md, Section "Autoregressive Generation"

    Self-Assessment Questions:
    1. Why do we need EOS tokens?
    2. How do you handle different sequence lengths in a batch?
    3. What happens if a sequence never generates EOS?
    """

    def __init__(
        self,
        model: nn.Module,
        eos_token_id: int,
        pad_token_id: int,
        device: str = "cpu"
    ):
        # TODO: Implement initialization

        # Step 1: Store model and configuration
        # self.model = model.to(device)
        # self.eos_token_id = eos_token_id
        # self.pad_token_id = pad_token_id
        # self.device = device
        # self.model.eval()

        pass

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate text with EOS handling.

        Args:
            start_tokens: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            top_p: Optional top-p filtering

        Returns:
            Generated tokens of shape (batch_size, variable_length)
        """
        # TODO: Implement generation with EOS handling

        # Step 1: Move to device and initialize
        # start_tokens = start_tokens.to(self.device)
        # batch_size, start_len = start_tokens.shape
        # generated = start_tokens.clone()

        # Step 2: Track which sequences have finished
        # finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Step 3: Autoregressive generation loop
        # for step in range(max_new_tokens):
        #     # Get logits for next token
        #     logits, _ = self.model(generated)
        #     next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        #
        #     # Sample next token using combined sampling
        #     next_token = exercise_08_combined_sampling(
        #         next_token_logits,
        #         temperature=temperature,
        #         top_k=top_k,
        #         top_p=top_p
        #     )
        #
        #     # For finished sequences, use pad token
        #     next_token = torch.where(
        #         finished,
        #         torch.full_like(next_token, self.pad_token_id),
        #         next_token
        #     )
        #
        #     # Append to generated sequence
        #     next_token = next_token.unsqueeze(-1)
        #     generated = torch.cat([generated, next_token], dim=1)
        #
        #     # Update finished status
        #     finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
        #
        #     # Early stopping if all sequences finished
        #     if finished.all():
        #         break

        # return generated

        pass


# ==============================================================================
# EXERCISE 11: Compare All Sampling Strategies
# Difficulty: Hard
# ==============================================================================

def exercise_11_compare_all_strategies(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 30
) -> Dict[str, torch.Tensor]:
    """
    EXERCISE 11: Generate text with all sampling strategies and compare.

    Learning Objectives:
    - Compare different sampling strategies side-by-side
    - Understand trade-offs between strategies
    - Practice evaluating generation quality

    Generate text using:
    - Greedy decoding
    - Temperature sampling (T=0.5, 1.0, 2.0)
    - Top-k sampling (k=10, 50)
    - Top-p sampling (p=0.9, 0.95)
    - Combined sampling (T=0.8, k=50, p=0.9)

    Args:
        model: TinyTransformerLM model
        start_tokens: Starting sequence of shape (1, seq_len)
        max_new_tokens: Number of tokens to generate

    Returns:
        Dictionary mapping strategy name to generated sequence

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        >>> start = torch.randint(0, 100, (1, 10))
        >>> results = exercise_11_compare_all_strategies(model, start)
        >>> len(results)
        9  # Number of strategies

    Theory reference: theory.md, Section "Comparing Sampling Strategies"

    Self-Assessment Questions:
    1. Which strategy produces the most diverse output?
    2. Which strategy is most likely to produce coherent text?
    3. How would you evaluate which strategy is "best"?
    """
    # TODO: Implement strategy comparison

    # from tiny_transformer.sampling import TextGenerator, GeneratorConfig

    # results = {}

    # Step 1: Greedy decoding
    # config = GeneratorConfig(max_new_tokens=max_new_tokens, do_sample=False)
    # generator = TextGenerator(model, config)
    # results['greedy'] = generator.generate(start_tokens)

    # Step 2: Temperature sampling
    # for temp in [0.5, 1.0, 2.0]:
    #     config = GeneratorConfig(
    #         max_new_tokens=max_new_tokens,
    #         temperature=temp,
    #         do_sample=True
    #     )
    #     generator = TextGenerator(model, config)
    #     results[f'temp_{temp}'] = generator.generate(start_tokens)

    # Step 3: Top-k sampling
    # for k in [10, 50]:
    #     config = GeneratorConfig(
    #         max_new_tokens=max_new_tokens,
    #         top_k=k,
    #         do_sample=True
    #     )
    #     generator = TextGenerator(model, config)
    #     results[f'top_k_{k}'] = generator.generate(start_tokens)

    # Step 4: Top-p sampling
    # for p in [0.9, 0.95]:
    #     config = GeneratorConfig(
    #         max_new_tokens=max_new_tokens,
    #         top_p=p,
    #         do_sample=True
    #     )
    #     generator = TextGenerator(model, config)
    #     results[f'top_p_{p}'] = generator.generate(start_tokens)

    # Step 5: Combined sampling
    # config = GeneratorConfig(
    #     max_new_tokens=max_new_tokens,
    #     temperature=0.8,
    #     top_k=50,
    #     top_p=0.9,
    #     do_sample=True
    # )
    # generator = TextGenerator(model, config)
    # results['combined'] = generator.generate(start_tokens)

    # return results

    pass


# ==============================================================================
# EXERCISE 12: Interactive Text Completion System
# Difficulty: Very Hard
# ==============================================================================

class Exercise12_TextCompletionSystem:
    """
    EXERCISE 12: Build an interactive text completion system.

    Learning Objectives:
    - Create a user-facing generation system
    - Implement multiple completion suggestions
    - Practice tokenization and detokenization
    - Build a complete end-to-end application

    The system should:
    - Take text prompts from users
    - Generate multiple completions with different strategies
    - Rank completions by likelihood
    - Support interactive parameter tuning

    Args:
        model: TinyTransformerLM model
        tokenizer: Tokenizer with encode/decode methods
        device: Device for generation

    Example:
        >>> system = Exercise12_TextCompletionSystem(model, tokenizer)
        >>> completions = system.complete("Once upon a time", num_completions=3)
        >>> for i, text in enumerate(completions):
        ...     print(f"{i+1}. {text}")

    Theory reference: theory.md, Section "Applications"

    Self-Assessment Questions:
    1. How do you ensure diverse completions?
    2. How would you rank completions by quality?
    3. What metrics would you use to evaluate completions?
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,  # Assumes encode() and decode() methods
        device: str = "cpu"
    ):
        # TODO: Implement initialization

        # Step 1: Store model and tokenizer
        # self.model = model.to(device)
        # self.tokenizer = tokenizer
        # self.device = device
        # self.model.eval()

        pass

    def complete(
        self,
        prompt: str,
        num_completions: int = 3,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generate multiple completions for a prompt.

        Args:
            prompt: Input text prompt
            num_completions: Number of completions to generate
            max_new_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering

        Returns:
            List of completed text strings
        """
        # TODO: Implement completion generation

        # Step 1: Tokenize prompt
        # prompt_tokens = self.tokenizer.encode(prompt)
        # prompt_tensor = torch.tensor(prompt_tokens).unsqueeze(0).to(self.device)

        # Step 2: Generate multiple completions
        # from tiny_transformer.sampling import TextGenerator, GeneratorConfig
        # config = GeneratorConfig(
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     top_k=top_k,
        #     top_p=top_p,
        #     do_sample=True
        # )
        # generator = TextGenerator(self.model, config, device=self.device)

        # completions = []
        # for _ in range(num_completions):
        #     # Generate with different random seeds for diversity
        #     generated = generator.generate(prompt_tensor)
        #
        #     # Decode to text
        #     generated_tokens = generated[0].cpu().tolist()
        #     completion_text = self.tokenizer.decode(generated_tokens)
        #     completions.append(completion_text)

        # return completions

        pass

    def rank_completions(
        self,
        prompt: str,
        completions: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank completions by their likelihood under the model.

        Args:
            prompt: Original prompt
            completions: List of completion texts

        Returns:
            List of (completion, log_probability) tuples, sorted by probability
        """
        # TODO: Implement completion ranking

        # ranked = []

        # for completion in completions:
        #     # Tokenize completion
        #     tokens = self.tokenizer.encode(completion)
        #     tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
        #
        #     # Compute log probability
        #     with torch.no_grad():
        #         logits, _ = self.model(tokens_tensor[:, :-1])
        #
        #         # Compute log probabilities
        #         log_probs = F.log_softmax(logits, dim=-1)
        #
        #         # Get log prob of actual next tokens
        #         target_tokens = tokens_tensor[:, 1:]
        #         token_log_probs = log_probs.gather(
        #             dim=-1,
        #             index=target_tokens.unsqueeze(-1)
        #         ).squeeze(-1)
        #
        #         # Average log probability
        #         avg_log_prob = token_log_probs.mean().item()
        #
        #     ranked.append((completion, avg_log_prob))

        # Sort by log probability (higher is better)
        # ranked.sort(key=lambda x: x[1], reverse=True)

        # return ranked

        pass


# ==============================================================================
# EXERCISE 13: Temperature Scheduling During Generation
# Difficulty: Very Hard
# ==============================================================================

def exercise_13_temperature_scheduling(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    start_temp: float = 2.0,
    end_temp: float = 0.5,
    schedule: str = "linear"
) -> torch.Tensor:
    """
    EXERCISE 13: Implement temperature scheduling during generation.

    Learning Objectives:
    - Implement dynamic temperature adjustment
    - Understand benefits of temperature scheduling
    - Practice custom generation loops

    Temperature scheduling:
    - Start with high temperature (creative/diverse)
    - Gradually decrease temperature (focused/deterministic)
    - Helps balance exploration vs exploitation
    - Useful for creative tasks

    Supported schedules:
    - Linear: T(t) = start_temp - (start_temp - end_temp) * t / max_steps
    - Exponential: T(t) = start_temp * (end_temp / start_temp) ^ (t / max_steps)
    - Cosine: T(t) = end_temp + (start_temp - end_temp) * 0.5 * (1 + cos(π * t / max_steps))

    Args:
        model: TinyTransformerLM model
        start_tokens: Starting sequence of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        start_temp: Initial temperature
        end_temp: Final temperature
        schedule: Schedule type ("linear", "exponential", "cosine")

    Returns:
        Generated tokens of shape (batch_size, seq_len + max_new_tokens)

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        >>> start = torch.randint(0, 100, (1, 10))
        >>> generated = exercise_13_temperature_scheduling(
        ...     model, start, max_new_tokens=50, schedule="cosine"
        ... )

    Theory reference: theory.md, Section "Advanced Generation Techniques"

    Self-Assessment Questions:
    1. Why start with high temperature and decrease?
    2. Which schedule is best for different tasks?
    3. How would you adaptively adjust temperature based on model confidence?
    """
    # TODO: Implement temperature scheduling

    # model.eval()
    # generated = start_tokens.clone()

    # with torch.no_grad():
    #     for step in range(max_new_tokens):
    #         # Step 1: Compute temperature for this step
    #         progress = step / max(max_new_tokens - 1, 1)
    #
    #         if schedule == "linear":
    #             temp = start_temp - (start_temp - end_temp) * progress
    #         elif schedule == "exponential":
    #             temp = start_temp * ((end_temp / start_temp) ** progress)
    #         elif schedule == "cosine":
    #             temp = end_temp + (start_temp - end_temp) * 0.5 * (
    #                 1 + math.cos(math.pi * progress)
    #             )
    #         else:
    #             raise ValueError(f"Unknown schedule: {schedule}")
    #
    #         # Step 2: Get logits
    #         logits, _ = model(generated)
    #         next_token_logits = logits[:, -1, :]
    #
    #         # Step 3: Sample with current temperature
    #         next_token = exercise_02_temperature_sampling(
    #             next_token_logits,
    #             temperature=temp
    #         )
    #
    #         # Step 4: Append token
    #         next_token = next_token.unsqueeze(-1)
    #         generated = torch.cat([generated, next_token], dim=1)

    # return generated

    pass


# ==============================================================================
# EXERCISE 14: Story Continuation System
# Difficulty: Very Hard
# ==============================================================================

class Exercise14_StoryContinuation:
    """
    EXERCISE 14: Build a creative story continuation system.

    Learning Objectives:
    - Apply sampling strategies to creative writing
    - Implement coherence and diversity balancing
    - Practice multi-paragraph generation
    - Build a complete creative AI application

    The system should:
    - Continue stories with appropriate style
    - Generate multiple sentences coherently
    - Balance creativity and coherence
    - Support different genres/styles

    Args:
        model: TinyTransformerLM model
        tokenizer: Tokenizer with encode/decode
        device: Device for generation

    Example:
        >>> system = Exercise14_StoryContinuation(model, tokenizer)
        >>> story_start = "The dragon flew over the misty mountains"
        >>> continuation = system.continue_story(story_start, num_sentences=3)
        >>> print(continuation)

    Theory reference: theory.md, Section "Applications"

    Self-Assessment Questions:
    1. How do you maintain coherence across multiple sentences?
    2. What sampling parameters work best for creative writing?
    3. How would you adapt the system for different genres?
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cpu"
    ):
        # TODO: Implement initialization

        # self.model = model.to(device)
        # self.tokenizer = tokenizer
        # self.device = device
        # self.model.eval()

        pass

    def continue_story(
        self,
        story_start: str,
        num_sentences: int = 3,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        sentence_end_token: str = "."
    ) -> str:
        """
        Continue a story for multiple sentences.

        Args:
            story_start: Beginning of the story
            num_sentences: Number of sentences to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            sentence_end_token: Character marking sentence end

        Returns:
            Continued story text
        """
        # TODO: Implement story continuation

        # Step 1: Tokenize story start
        # current_text = story_start
        # prompt_tokens = self.tokenizer.encode(current_text)
        # current_tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(self.device)

        # Step 2: Generate sentence by sentence
        # from tiny_transformer.sampling import TextGenerator, GeneratorConfig

        # sentences_generated = 0
        # max_tokens_per_sentence = 50

        # while sentences_generated < num_sentences:
        #     # Configure generator
        #     config = GeneratorConfig(
        #         max_new_tokens=max_tokens_per_sentence,
        #         temperature=temperature,
        #         top_k=top_k,
        #         top_p=top_p,
        #         do_sample=True
        #     )
        #     generator = TextGenerator(self.model, config, device=self.device)
        #
        #     # Generate tokens
        #     generated = generator.generate(current_tokens)
        #
        #     # Decode
        #     generated_tokens = generated[0].cpu().tolist()
        #     generated_text = self.tokenizer.decode(generated_tokens)
        #
        #     # Check if we generated a sentence end
        #     if sentence_end_token in generated_text[len(current_text):]:
        #         # Find the sentence end
        #         new_text = generated_text[len(current_text):]
        #         sentence_end_pos = new_text.find(sentence_end_token)
        #
        #         # Update current text up to sentence end
        #         current_text = generated_text[:len(current_text) + sentence_end_pos + 1]
        #         sentences_generated += 1
        #
        #         # Update tokens for next iteration
        #         if sentences_generated < num_sentences:
        #             current_tokens = torch.tensor(
        #                 self.tokenizer.encode(current_text)
        #             ).unsqueeze(0).to(self.device)
        #     else:
        #         # No sentence end found, add entire generation
        #         current_text = generated_text
        #         current_tokens = generated
        #
        #         # Safety: break if we've generated too many tokens
        #         if len(current_tokens[0]) > len(prompt_tokens) + num_sentences * max_tokens_per_sentence:
        #             break

        # # Return continuation only (remove original prompt)
        # continuation = current_text[len(story_start):].strip()
        # return continuation

        pass

    def generate_with_style(
        self,
        story_start: str,
        style: str = "fantasy",
        num_sentences: int = 3
    ) -> str:
        """
        Generate story continuation with specific style parameters.

        Args:
            story_start: Beginning of story
            style: Style preset ("fantasy", "scifi", "mystery", "romance")
            num_sentences: Number of sentences to generate

        Returns:
            Styled continuation
        """
        # TODO: Implement style-based generation

        # Define style-specific parameters
        # style_params = {
        #     'fantasy': {'temperature': 0.9, 'top_k': 50, 'top_p': 0.92},
        #     'scifi': {'temperature': 0.8, 'top_k': 60, 'top_p': 0.9},
        #     'mystery': {'temperature': 0.7, 'top_k': 40, 'top_p': 0.85},
        #     'romance': {'temperature': 0.85, 'top_k': 45, 'top_p': 0.88}
        # }

        # params = style_params.get(style, {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9})

        # return self.continue_story(
        #     story_start,
        #     num_sentences=num_sentences,
        #     **params
        # )

        pass


# ==============================================================================
# Testing and Validation
# ==============================================================================

def run_all_tests():
    """
    Run tests for all exercises.

    Uncomment each test as you complete the corresponding exercise.
    """
    print("=" * 70)
    print("Module 07: Sampling & Generation - Exercise Tests")
    print("=" * 70)

    # Test Exercise 1
    # print("\n[TEST 1] Greedy Sampling")
    # logits = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.1, 0.8]])
    # tokens = exercise_01_greedy_sampling(logits)
    # assert tokens.tolist() == [1, 2], f"Expected [1, 2], got {tokens.tolist()}"
    # print("✓ Exercise 1 passed!")

    # Test Exercise 2
    # print("\n[TEST 2] Temperature Sampling")
    # torch.manual_seed(42)
    # logits = torch.randn(3, 100)
    # tokens = exercise_02_temperature_sampling(logits, temperature=0.8)
    # assert tokens.shape == (3,), f"Expected shape (3,), got {tokens.shape}"
    # assert tokens.dtype == torch.long, "Expected long dtype"
    # print("✓ Exercise 2 passed!")

    # Test Exercise 3
    # print("\n[TEST 3] Compare Sampling Strategies")
    # torch.manual_seed(42)
    # logits = torch.randn(100)
    # results = exercise_03_compare_sampling_strategies(logits, num_samples=10)
    # assert len(results) == 4, f"Expected 4 strategies, got {len(results)}"
    # assert len(set(results['greedy'])) == 1, "Greedy should be deterministic"
    # print("✓ Exercise 3 passed!")

    # Test Exercise 4
    # print("\n[TEST 4] Visualize Temperature")
    # logits = torch.tensor([2.0, 1.0, 0.5])
    # dists = exercise_04_visualize_temperature(logits)
    # for temp, probs in dists.items():
    #     assert torch.allclose(probs.sum(), torch.tensor(1.0)), f"Probs don't sum to 1 at T={temp}"
    # print("✓ Exercise 4 passed!")

    # Test Exercise 5
    # print("\n[TEST 5] Top-K Sampling")
    # torch.manual_seed(42)
    # logits = torch.randn(2, 100)
    # tokens = exercise_05_top_k_sampling(logits, k=10)
    # assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    # print("✓ Exercise 5 passed!")

    # Test Exercise 6
    # print("\n[TEST 6] Top-P Sampling")
    # torch.manual_seed(42)
    # logits = torch.randn(2, 100)
    # tokens = exercise_06_top_p_sampling(logits, p=0.9)
    # assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    # print("✓ Exercise 6 passed!")

    # Test Exercise 7
    # print("\n[TEST 7] Use TextGenerator")
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # start = torch.randint(0, 100, (2, 10))
    # generated = exercise_07_use_text_generator(model, start, max_new_tokens=20)
    # assert generated.shape == (2, 30), f"Expected shape (2, 30), got {generated.shape}"
    # print("✓ Exercise 7 passed!")

    # Test Exercise 8
    # print("\n[TEST 8] Combined Sampling")
    # torch.manual_seed(42)
    # logits = torch.randn(2, 100)
    # tokens = exercise_08_combined_sampling(logits, temperature=0.8, top_k=50, top_p=0.9)
    # assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    # print("✓ Exercise 8 passed!")

    # Test Exercise 9
    # print("\n[TEST 9] Analyze Temperature Effects")
    # logits = torch.randn(100)
    # stats = exercise_09_analyze_temperature_effects(logits)
    # assert 0.1 in stats, "Should have stats for T=0.1"
    # assert stats[0.1]['diversity'] < stats[2.0]['diversity'], "Higher T should give more diversity"
    # print("✓ Exercise 9 passed!")

    # Test Exercise 10
    # print("\n[TEST 10] Custom Generator with EOS")
    # model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # generator = Exercise10_CustomGenerator(model, eos_token_id=0, pad_token_id=1)
    # start = torch.randint(2, 100, (2, 5))
    # output = generator.generate(start, max_new_tokens=20)
    # assert output.shape[0] == 2, "Should preserve batch size"
    # print("✓ Exercise 10 passed!")

    # Test Exercise 11
    # print("\n[TEST 11] Compare All Strategies")
    # model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # start = torch.randint(0, 100, (1, 10))
    # results = exercise_11_compare_all_strategies(model, start, max_new_tokens=20)
    # assert len(results) >= 5, f"Expected at least 5 strategies, got {len(results)}"
    # print("✓ Exercise 11 passed!")

    # Test Exercise 13
    # print("\n[TEST 13] Temperature Scheduling")
    # model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # start = torch.randint(0, 100, (1, 10))
    # generated = exercise_13_temperature_scheduling(model, start, max_new_tokens=30)
    # assert generated.shape == (1, 40), f"Expected shape (1, 40), got {generated.shape}"
    # print("✓ Exercise 13 passed!")

    print("\n" + "=" * 70)
    print("All tests passed! Excellent work!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Module 07: Sampling & Generation - Exercises")
    print("=" * 70)
    print("\nWelcome! These exercises teach you about text generation and sampling.")
    print("\nExercise Structure:")
    print("  01. Implement Greedy Sampling (Easy)")
    print("  02. Implement Temperature Sampling (Easy)")
    print("  03. Compare Greedy vs Temperature (Easy)")
    print("  04. Visualize Temperature Effects (Easy)")
    print("  05. Implement Top-K Sampling (Medium)")
    print("  06. Implement Top-P (Nucleus) Sampling (Medium)")
    print("  07. Use TextGenerator for Basic Generation (Medium)")
    print("  08. Implement Combined Sampling Strategy (Hard)")
    print("  09. Analyze Sampling Temperature Effects (Medium)")
    print("  10. Build Custom Generator with EOS Handling (Hard)")
    print("  11. Compare All Sampling Strategies (Hard)")
    print("  12. Interactive Text Completion System (Very Hard)")
    print("  13. Temperature Scheduling During Generation (Very Hard)")
    print("  14. Story Continuation System (Very Hard)")
    print("\nInstructions:")
    print("1. Work through exercises 1-14 in order")
    print("2. Read each docstring carefully")
    print("3. Implement the TODO sections")
    print("4. Uncomment test assertions to verify")
    print("5. Run this file to test your solutions")
    print("\nTips:")
    print("- Refer to theory.md for concepts")
    print("- Check tiny_transformer/sampling/*.py for reference")
    print("- Use print() to debug shapes and values")
    print("- Experiment with different temperature values")
    print("- Compare outputs from different sampling strategies")
    print("=" * 70)

    # Uncomment when ready to test:
    # run_all_tests()

    print("\nGood luck with text generation! 🚀")
