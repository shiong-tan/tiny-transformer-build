"""
Module 07: Sampling & Generation - Complete Solutions

This file contains comprehensive reference solutions for all 14 sampling exercises.
Each solution includes:
- Complete, working implementation
- Educational comments explaining WHY, not just WHAT
- Shape annotations and assertions
- Best practices from reference implementations
- Notes about common mistakes and alternative approaches

Study these solutions to understand:
1. Greedy vs. stochastic sampling strategies
2. Temperature effects on probability distributions
3. Top-k and top-p filtering techniques
4. Combined sampling approaches
5. Autoregressive text generation
6. EOS token handling
7. Sampling strategy comparisons
8. Temperature scheduling
9. Building complete generation systems

Author: Educational reference implementation
See: /Users/shiongtan/projects/tiny-transformer-build/docs/modules/07_sampling/theory.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math


# ==============================================================================
# SOLUTION 1: Greedy Sampling
# ==============================================================================

class Solution01_GreedySampling:
    """
    SOLUTION 01: Greedy sampling - always pick highest probability token.

    Key Concepts:
    1. Deterministic: Always produces same output for same input
    2. Argmax: Find index of maximum value along vocabulary dimension
    3. No randomness: No temperature, no sampling

    Why Greedy?
    - Fastest and simplest strategy
    - Good for tasks requiring consistency (translations, code)
    - Often produces repetitive text in creative tasks

    When to Use:
    - Question answering (want most likely answer)
    - Code generation (want correct syntax)
    - Short sequences (less repetition)

    Common Mistakes:
    - Using on wrong dimension (dim=0 instead of dim=-1)
    - Not checking output dtype (should be long for token IDs)
    - Expecting diversity (greedy is deterministic!)

    See: tiny_transformer/sampling/strategies.py
    """

    @staticmethod
    def solution():
        def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
            """
            Implement greedy sampling.

            Args:
                logits: Shape (batch_size, vocab_size)

            Returns:
                tokens: Shape (batch_size,)
            """
            # Find index of maximum value along vocabulary dimension
            # argmax returns the index (token ID) of the maximum logit
            # dim=-1 means "last dimension" which is vocab_size
            tokens = logits.argmax(dim=-1)

            # Validate shape and dtype
            assert tokens.shape == (logits.shape[0],), \
                f"Expected shape ({logits.shape[0]},), got {tokens.shape}"
            assert tokens.dtype == torch.long, \
                "Token IDs should be long integers"

            return tokens

        # Example usage
        print("Solution 01: Greedy Sampling")
        print("=" * 70)

        # Example: Always picks highest probability
        logits = torch.tensor([
            [1.0, 3.0, 2.0],  # Batch 0: token 1 has highest logit
            [0.5, 0.3, 0.8]   # Batch 1: token 2 has highest logit
        ])

        tokens = greedy_sampling(logits)
        print(f"Logits:\n{logits}")
        print(f"Selected tokens: {tokens.tolist()}")  # [1, 2]
        print()

        # Verify determinism: calling multiple times gives same result
        tokens2 = greedy_sampling(logits)
        assert torch.equal(tokens, tokens2), "Greedy should be deterministic!"
        print("Verified: Greedy sampling is deterministic")
        print("=" * 70)

        return greedy_sampling


# ==============================================================================
# SOLUTION 2: Temperature Sampling
# ==============================================================================

class Solution02_TemperatureSampling:
    """
    SOLUTION 02: Temperature sampling for controlling randomness.

    Key Concepts:
    1. Temperature scaling: logits / T reshapes distribution
    2. T < 1.0: Sharpens distribution (more confident)
    3. T > 1.0: Flattens distribution (more random)
    4. T → 0: Approaches greedy sampling
    5. T → ∞: Approaches uniform sampling

    Mathematical Effect:
    - Softmax(logits/T) emphasizes differences when T < 1
    - Example: logits=[1, 2, 3] with T=0.5 makes 3 much more likely
    - Same logits with T=2.0 makes probabilities more uniform

    Why Temperature?
    - Control creativity vs. coherence trade-off
    - Creative writing: T=0.8-1.2 (more diverse)
    - Factual tasks: T=0.5-0.8 (more focused)

    Common Mistakes:
    - Temperature ≤ 0 (invalid! causes division by zero)
    - Forgetting to apply softmax after scaling
    - Using sample() instead of multinomial()

    See: tiny_transformer/sampling/strategies.py
    """

    @staticmethod
    def solution():
        def temperature_sampling(
            logits: torch.Tensor,
            temperature: float = 1.0
        ) -> torch.Tensor:
            """
            Sample with temperature scaling.

            Args:
                logits: Shape (batch_size, vocab_size)
                temperature: Temperature > 0

            Returns:
                tokens: Shape (batch_size,)
            """
            # Step 1: Validate temperature
            if temperature <= 0:
                raise ValueError(f"Temperature must be > 0, got {temperature}")

            # Step 2: Scale logits by temperature
            # Lower T → larger logits → sharper distribution
            # Higher T → smaller logits → flatter distribution
            scaled_logits = logits / temperature

            # Step 3: Convert to probabilities
            # Softmax converts logits to valid probability distribution
            # Sum of probabilities = 1.0
            probs = F.softmax(scaled_logits, dim=-1)

            # Step 4: Sample from probability distribution
            # multinomial samples indices according to probabilities
            # num_samples=1 means sample one token per batch
            # squeeze(-1) removes the extra dimension from shape
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Validate output
            assert tokens.shape == (logits.shape[0],), \
                f"Expected shape ({logits.shape[0]},), got {tokens.shape}"
            assert tokens.dtype == torch.long, \
                "Token IDs should be long integers"

            return tokens

        # Example usage
        print("\nSolution 02: Temperature Sampling")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(2, 100)

        # Compare different temperatures
        for temp in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            tokens = temperature_sampling(logits, temperature=temp)
            print(f"Temperature {temp}: sampled tokens = {tokens.tolist()}")

        print("\nNote: Lower temperature → more confident (peaked distribution)")
        print("      Higher temperature → more random (flat distribution)")
        print("=" * 70)

        return temperature_sampling


# ==============================================================================
# SOLUTION 3: Compare Sampling Strategies
# ==============================================================================

class Solution03_CompareSamplingStrategies:
    """
    SOLUTION 03: Compare greedy and temperature sampling behaviors.

    Key Concepts:
    1. Greedy is deterministic (all samples identical)
    2. Temperature adds stochasticity (samples vary)
    3. Higher temperature → more diversity

    Observations:
    - Greedy: Same token every time
    - T=0.5: Low diversity, mostly high-probability tokens
    - T=1.0: Normal diversity
    - T=2.0: High diversity, includes low-probability tokens

    Use Cases:
    - Greedy: Factual Q&A, code generation
    - T=0.5-0.8: Summarization, translation
    - T=0.8-1.2: Creative writing, dialogue
    - T=1.5-2.0: Brainstorming, exploration

    Common Mistakes:
    - Not setting random seed (results not reproducible)
    - Expecting greedy to vary (it's deterministic!)
    - Using too high temperature (gibberish)

    See: theory.md, Section "Comparing Sampling Strategies"
    """

    @staticmethod
    def solution():
        def compare_sampling_strategies(
            logits: torch.Tensor,
            num_samples: int = 5
        ) -> Dict[str, List[int]]:
            """
            Compare different sampling strategies.

            Args:
                logits: Shape (vocab_size,) or (1, vocab_size)
                num_samples: Number of samples per strategy

            Returns:
                Dictionary mapping strategy name to list of samples
            """
            # Ensure logits is 2D for compatibility
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Import sampling functions from previous solutions
            greedy_fn = Solution01_GreedySampling.solution()
            temp_fn = Solution02_TemperatureSampling.solution()

            results = {}

            # Strategy 1: Greedy (deterministic)
            greedy_samples = []
            for _ in range(num_samples):
                token = greedy_fn(logits)
                greedy_samples.append(token.item())
            results['greedy'] = greedy_samples

            # Strategy 2-4: Temperature sampling with varying temperatures
            for temp in [0.5, 1.0, 2.0]:
                temp_samples = []
                for _ in range(num_samples):
                    token = temp_fn(logits, temperature=temp)
                    temp_samples.append(token.item())
                results[f'temp_{temp}'] = temp_samples

            return results

        # Example usage
        print("\nSolution 03: Compare Sampling Strategies")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(100)

        results = compare_sampling_strategies(logits, num_samples=10)

        for strategy, samples in results.items():
            unique = len(set(samples))
            print(f"{strategy:12} → {samples}")
            print(f"             Unique tokens: {unique}/10")
            print()

        # Validate greedy is deterministic
        assert len(set(results['greedy'])) == 1, \
            "Greedy should produce same token every time!"

        print("Observation: Higher temperature → more diversity")
        print("=" * 70)

        return compare_sampling_strategies


# ==============================================================================
# SOLUTION 4: Visualize Temperature Effects
# ==============================================================================

class Solution04_VisualizeTemperature:
    """
    SOLUTION 04: Visualize how temperature reshapes distributions.

    Key Concepts:
    1. Temperature changes distribution shape, not order
    2. Low T: Probability mass concentrates on top tokens
    3. High T: Probability mass spreads more evenly
    4. Entropy increases with temperature

    Mathematical Insight:
    - Entropy = -Σ p(x) log p(x)
    - Low T → low entropy (confident)
    - High T → high entropy (uncertain)

    Visualization Tips:
    - Plot probabilities as bar charts
    - Show cumulative distribution
    - Compare entropy across temperatures

    Common Mistakes:
    - Not normalizing (probs should sum to 1)
    - Comparing different logits (use same input!)
    - Forgetting temperature affects relative differences

    See: theory.md, Section "Temperature Sampling"
    """

    @staticmethod
    def solution():
        def visualize_temperature(
            logits: torch.Tensor,
            temperatures: List[float] = [0.5, 1.0, 2.0, 5.0]
        ) -> Dict[float, torch.Tensor]:
            """
            Compute probability distributions for different temperatures.

            Args:
                logits: Shape (vocab_size,)
                temperatures: List of temperatures to test

            Returns:
                Dictionary mapping temperature to probability distribution
            """
            # Ensure logits is 1D
            if logits.dim() > 1:
                logits = logits.squeeze()

            distributions = {}

            for temp in temperatures:
                # Scale logits by temperature
                scaled_logits = logits / temp

                # Convert to probabilities
                probs = F.softmax(scaled_logits, dim=-1)

                # Store distribution
                distributions[temp] = probs

                # Validate probability sum
                assert torch.allclose(probs.sum(), torch.tensor(1.0)), \
                    f"Probabilities should sum to 1 at T={temp}"

            return distributions

        # Example usage
        print("\nSolution 04: Visualize Temperature Effects")
        print("=" * 70)

        # Create peaked distribution (one dominant logit)
        logits = torch.tensor([1.0, 3.0, 0.5, 0.3, 0.1])

        dists = visualize_temperature(logits, temperatures=[0.5, 1.0, 2.0])

        print(f"Original logits: {logits.tolist()}\n")

        for temp, probs in dists.items():
            print(f"Temperature {temp}:")
            print(f"  Probabilities: {probs.tolist()}")
            print(f"  Max prob: {probs.max().item():.4f}")

            # Compute entropy: -Σ p log p
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            print(f"  Entropy: {entropy.item():.4f}")
            print()

        print("Observation: Lower temperature → lower entropy (more confident)")
        print("            Higher temperature → higher entropy (more uncertain)")
        print("=" * 70)

        return visualize_temperature


# ==============================================================================
# SOLUTION 5: Top-K Sampling
# ==============================================================================

class Solution05_TopKSampling:
    """
    SOLUTION 05: Top-k sampling - filter to k most likely tokens.

    Key Concepts:
    1. Keep only k highest probability tokens
    2. Set all others to -inf (probability = 0)
    3. Sample from filtered distribution
    4. Prevents sampling unlikely/nonsensical tokens

    Why Top-K?
    - Eliminates tail of distribution (rare tokens)
    - Maintains diversity among top tokens
    - Fixed vocabulary size regardless of distribution shape

    Choosing K:
    - k=1: Equivalent to greedy
    - k=10-50: Common for most tasks
    - k=100+: Very permissive (close to no filtering)
    - k=vocab_size: No filtering at all

    Common Mistakes:
    - k=0 (invalid, need at least one token)
    - k > vocab_size (no effect, but wastes computation)
    - Filtering after softmax (should filter logits!)
    - Not using scatter_ for batch filtering

    See: tiny_transformer/sampling/strategies.py
    """

    @staticmethod
    def solution():
        def top_k_sampling(
            logits: torch.Tensor,
            k: int,
            temperature: float = 1.0
        ) -> torch.Tensor:
            """
            Sample from top-k tokens.

            Args:
                logits: Shape (batch_size, vocab_size)
                k: Number of top tokens to consider
                temperature: Temperature for sampling

            Returns:
                tokens: Shape (batch_size,)
            """
            # Validate k
            if k <= 0:
                raise ValueError(f"k must be > 0, got {k}")

            # Apply temperature scaling first
            if temperature != 1.0:
                logits = logits / temperature

            # Find top-k values and their indices
            # topk returns (values, indices) sorted in descending order
            top_k_logits, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)

            # Create filtered logits initialized to -inf
            # -inf in logits → probability 0 after softmax
            filtered_logits = torch.full_like(logits, float('-inf'))

            # Scatter top-k values back to their positions
            # scatter_(dim, index, src) fills positions at index with values from src
            filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

            # Convert to probabilities and sample
            # Only top-k positions have non-zero probability
            probs = F.softmax(filtered_logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Validate output
            assert tokens.shape == (logits.shape[0],), \
                f"Expected shape ({logits.shape[0]},), got {tokens.shape}"

            return tokens

        # Example usage
        print("\nSolution 05: Top-K Sampling")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Compare different k values
        for k in [1, 5, 10, 50]:
            torch.manual_seed(42)
            tokens = top_k_sampling(logits, k=k, temperature=1.0)
            print(f"k={k:3d}: sampled token = {tokens.item()}")

        print("\nNote: k=1 is equivalent to greedy sampling")
        print("      Larger k allows more diversity")
        print("=" * 70)

        return top_k_sampling


# ==============================================================================
# SOLUTION 6: Top-P (Nucleus) Sampling
# ==============================================================================

class Solution06_TopPSampling:
    """
    SOLUTION 06: Top-p (nucleus) sampling - dynamic vocabulary filtering.

    Key Concepts:
    1. Select smallest set of tokens with cumulative probability ≥ p
    2. Adapts to distribution shape (unlike fixed k)
    3. Peaked distribution → fewer tokens selected
    4. Flat distribution → more tokens selected

    Why Top-P?
    - More adaptive than top-k
    - Automatically adjusts to model confidence
    - Prevents sampling very unlikely tokens
    - Commonly used in GPT models (p=0.9 or 0.95)

    Algorithm:
    1. Sort probabilities in descending order
    2. Compute cumulative sum
    3. Find cutoff where cumsum > p
    4. Remove tokens after cutoff
    5. Sample from remaining tokens

    Choosing P:
    - p=1.0: No filtering (sample from full distribution)
    - p=0.95: Very permissive (typical for creative tasks)
    - p=0.9: Balanced (common default)
    - p=0.8: Conservative (focused generation)
    - p→0: Approaches greedy

    Common Mistakes:
    - Not shifting mask (keeps first token exceeding p)
    - Removing tokens where cumsum ≥ p (should be >)
    - Forgetting to map mask back to original order
    - Not handling edge case where p is very small

    See: tiny_transformer/sampling/strategies.py
    """

    @staticmethod
    def solution():
        def top_p_sampling(
            logits: torch.Tensor,
            p: float,
            temperature: float = 1.0
        ) -> torch.Tensor:
            """
            Sample from nucleus (top-p).

            Args:
                logits: Shape (batch_size, vocab_size)
                p: Cumulative probability threshold (0 < p <= 1)
                temperature: Temperature for sampling

            Returns:
                tokens: Shape (batch_size,)
            """
            # Validate p
            if not 0 < p <= 1:
                raise ValueError(f"p must be in (0, 1], got {p}")

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for tokens to remove
            # Remove tokens where cumulative probability > p
            sorted_indices_to_remove = cumulative_probs > p

            # Shift the mask right by 1 to keep first token exceeding p
            # This ensures we always keep at least one token
            # Example: cumsum = [0.4, 0.7, 0.95], p=0.9
            #   Before shift: [F, F, T] → would keep first 2
            #   After shift:  [F, F, F] → keeps all 3 (first exceeding 0.9)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False  # Always keep top token

            # Map mask back to original order
            # scatter maps sorted positions back to original vocabulary positions
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )

            # Set removed indices to -inf in logits
            filtered_logits = logits.clone()
            filtered_logits[indices_to_remove] = float('-inf')

            # Sample from filtered distribution
            filtered_probs = F.softmax(filtered_logits, dim=-1)
            tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)

            # Validate output
            assert tokens.shape == (logits.shape[0],), \
                f"Expected shape ({logits.shape[0]},), got {tokens.shape}"

            return tokens

        # Example usage
        print("\nSolution 06: Top-P (Nucleus) Sampling")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # Compare different p values
        for p in [0.5, 0.9, 0.95, 1.0]:
            torch.manual_seed(42)
            tokens = top_p_sampling(logits, p=p, temperature=1.0)
            print(f"p={p:.2f}: sampled token = {tokens.item()}")

        print("\nNote: p=1.0 means no filtering")
        print("      Lower p → more focused sampling")
        print("=" * 70)

        return top_p_sampling


# ==============================================================================
# SOLUTION 7: Use TextGenerator
# ==============================================================================

class Solution07_UseTextGenerator:
    """
    SOLUTION 07: Use TextGenerator for autoregressive generation.

    Key Concepts:
    1. Autoregressive: Generate one token at a time
    2. Each new token depends on all previous tokens
    3. TextGenerator handles generation loop
    4. GeneratorConfig controls sampling parameters

    Why TextGenerator?
    - Abstracts away generation loop
    - Handles batching automatically
    - Supports all sampling strategies
    - Production-ready interface

    Generation Process:
    1. Start with prompt tokens
    2. Forward pass → get logits for next token
    3. Sample next token using configured strategy
    4. Append to sequence
    5. Repeat until max_new_tokens or EOS

    Common Mistakes:
    - Not setting do_sample=True for stochastic sampling
    - Expecting fast generation (it's sequential!)
    - Not handling EOS tokens
    - Forgetting to set model.eval()

    See: tiny_transformer/sampling/generator.py
    """

    @staticmethod
    def solution():
        def use_text_generator(
            model: nn.Module,
            start_tokens: torch.Tensor,
            max_new_tokens: int = 20,
            temperature: float = 1.0
        ) -> torch.Tensor:
            """
            Generate text using TextGenerator.

            Args:
                model: TinyTransformerLM or compatible
                start_tokens: Shape (batch_size, seq_len)
                max_new_tokens: Number of tokens to generate
                temperature: Sampling temperature

            Returns:
                generated: Shape (batch_size, seq_len + max_new_tokens)
            """
            # Import from reference implementation
            from tiny_transformer.sampling import TextGenerator, GeneratorConfig

            # Create generator configuration
            config = GeneratorConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,  # Use sampling (not greedy)
                top_k=None,      # No top-k filtering
                top_p=None       # No top-p filtering
            )

            # Create text generator
            generator = TextGenerator(model, config)

            # Generate text
            # This runs the autoregressive loop internally
            generated = generator.generate(start_tokens)

            # Validate output shape
            expected_len = start_tokens.size(1) + max_new_tokens
            assert generated.shape[1] == expected_len, \
                f"Expected length {expected_len}, got {generated.shape[1]}"

            return generated

        # Example usage
        print("\nSolution 07: Use TextGenerator")
        print("=" * 70)

        from tiny_transformer.model import TinyTransformerLM

        # Create small model
        model = TinyTransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256
        )
        model.eval()

        # Create start tokens
        start = torch.randint(0, 100, (2, 10))

        # Generate
        generated = use_text_generator(model, start, max_new_tokens=20, temperature=0.8)

        print(f"Start shape: {start.shape}")
        print(f"Generated shape: {generated.shape}")
        print(f"Start tokens: {start[0, :5].tolist()}...")
        print(f"Generated tokens: {generated[0, :15].tolist()}...")
        print("\nTextGenerator handles the autoregressive loop for you!")
        print("=" * 70)

        return use_text_generator


# ==============================================================================
# SOLUTION 8: Combined Sampling
# ==============================================================================

class Solution08_CombinedSampling:
    """
    SOLUTION 08: Combined sampling with temperature + top-k + top-p.

    Key Concepts:
    1. Apply filters sequentially for fine control
    2. Order matters: temperature → top-k → top-p
    3. Each filter further constrains the distribution

    Why Combine?
    - Temperature: Controls overall randomness
    - Top-k: Removes very unlikely tokens
    - Top-p: Adapts to distribution shape
    - Together: Precise control over generation quality

    Common Configurations:
    - Conservative: T=0.7, k=40, p=0.9
    - Balanced: T=0.8, k=50, p=0.95
    - Creative: T=1.0, k=100, p=0.98

    Order of Operations:
    1. Temperature scaling (affects all tokens)
    2. Top-k filtering (fixed cutoff)
    3. Top-p filtering (adaptive cutoff)
    4. Final sampling

    Why This Order?
    - Temperature first: Affects relative probabilities
    - Top-k second: Hard cutoff on number of tokens
    - Top-p last: Adaptive cutoff on probability mass

    Common Mistakes:
    - Wrong order (top-k after top-p doesn't make sense)
    - Both k and p too restrictive (too deterministic)
    - Not cloning logits (modifying in place)

    See: tiny_transformer/sampling/strategies.py
    """

    @staticmethod
    def solution():
        def combined_sampling(
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None
        ) -> torch.Tensor:
            """
            Combined sampling with multiple strategies.

            Args:
                logits: Shape (batch_size, vocab_size)
                temperature: Temperature scaling
                top_k: Optional top-k filtering
                top_p: Optional nucleus sampling

            Returns:
                tokens: Shape (batch_size,)
            """
            # Step 1: Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Step 2: Apply top-k filtering (if specified)
            if top_k is not None and top_k > 0:
                # Keep only top-k logits
                top_k_logits, top_k_indices = torch.topk(
                    logits,
                    k=min(top_k, logits.size(-1)),
                    dim=-1
                )

                # Create filtered logits
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
                logits = filtered_logits

            # Step 3: Apply top-p filtering (if specified)
            if top_p is not None and 0 < top_p < 1:
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sort probabilities
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create removal mask
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Map back to original order
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )

                # Filter logits
                logits[indices_to_remove] = float('-inf')

            # Step 4: Sample from final distribution
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            return tokens

        # Example usage
        print("\nSolution 08: Combined Sampling")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(2, 1000)

        # Test different combinations
        configs = [
            {"temperature": 0.8, "top_k": None, "top_p": None},
            {"temperature": 0.8, "top_k": 50, "top_p": None},
            {"temperature": 0.8, "top_k": None, "top_p": 0.9},
            {"temperature": 0.8, "top_k": 50, "top_p": 0.9},
        ]

        for config in configs:
            torch.manual_seed(42)
            tokens = combined_sampling(logits, **config)
            print(f"Config: {config}")
            print(f"  Tokens: {tokens.tolist()}\n")

        print("Combined sampling provides fine-grained control!")
        print("=" * 70)

        return combined_sampling


# ==============================================================================
# SOLUTION 9: Analyze Temperature Effects
# ==============================================================================

class Solution09_AnalyzeTemperature:
    """
    SOLUTION 09: Analyze temperature effects on diversity and entropy.

    Key Concepts:
    1. Diversity: Ratio of unique tokens to total samples
    2. Entropy: Measure of distribution uncertainty
    3. Max frequency: How often most common token appears

    Statistical Measures:
    - Diversity ∈ [0, 1]: 0 = all same, 1 = all unique
    - Entropy ≥ 0: Higher = more uncertain
    - Max frequency ∈ [0, 1]: How concentrated is distribution

    Expected Trends:
    - Lower temperature → lower diversity, lower entropy
    - Higher temperature → higher diversity, higher entropy
    - Very high T → approaches uniform (max entropy)

    Common Mistakes:
    - Too few samples (results unstable)
    - Not using same seed (results not comparable)
    - Confusing theoretical vs empirical distributions

    See: theory.md, Section "Temperature Sampling"
    """

    @staticmethod
    def solution():
        def analyze_temperature_effects(
            logits: torch.Tensor,
            temperatures: List[float] = [0.1, 0.5, 1.0, 1.5, 2.0],
            num_samples: int = 100
        ) -> Dict[str, Dict[str, float]]:
            """
            Analyze statistical effects of temperature.

            Args:
                logits: Shape (vocab_size,)
                temperatures: List of temperatures to test
                num_samples: Number of samples per temperature

            Returns:
                Dictionary mapping temperature to statistics
            """
            # Ensure logits is 2D
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Import sampling function
            temp_fn = Solution02_TemperatureSampling.solution()

            results = {}

            for temp in temperatures:
                # Sample many tokens
                samples = []
                for _ in range(num_samples):
                    token = temp_fn(logits, temperature=temp)
                    samples.append(token.item())

                # Compute diversity (unique tokens / total samples)
                unique_tokens = len(set(samples))
                diversity = unique_tokens / num_samples

                # Compute empirical distribution
                token_counts = {}
                for token in samples:
                    token_counts[token] = token_counts.get(token, 0) + 1

                # Compute entropy: H = -Σ p(x) log₂ p(x)
                # Higher entropy = more uncertain/diverse
                entropy = 0.0
                for count in token_counts.values():
                    p = count / num_samples
                    if p > 0:
                        entropy -= p * math.log2(p)

                # Find most frequent token
                max_count = max(token_counts.values())
                max_frequency = max_count / num_samples

                # Store statistics
                results[temp] = {
                    'diversity': diversity,
                    'entropy': entropy,
                    'max_frequency': max_frequency,
                    'unique_tokens': unique_tokens
                }

            return results

        # Example usage
        print("\nSolution 09: Analyze Temperature Effects")
        print("=" * 70)

        torch.manual_seed(42)
        logits = torch.randn(100)

        stats = analyze_temperature_effects(logits, num_samples=200)

        print(f"{'Temp':<8} {'Diversity':<12} {'Entropy':<12} {'Max Freq':<12}")
        print("-" * 50)
        for temp in sorted(stats.keys()):
            s = stats[temp]
            print(f"{temp:<8.1f} {s['diversity']:<12.3f} {s['entropy']:<12.3f} {s['max_frequency']:<12.3f}")

        print("\nObservation: Higher temperature → higher diversity & entropy")
        print("=" * 70)

        return analyze_temperature_effects


# ==============================================================================
# SOLUTION 10: Custom Generator with EOS Handling
# ==============================================================================

class Solution10_CustomGenerator:
    """
    SOLUTION 10: Custom generator with EOS token handling.

    Key Concepts:
    1. Autoregressive loop: Generate one token at a time
    2. EOS handling: Stop when end-of-sequence is generated
    3. Batching: Handle multiple sequences simultaneously
    4. Padding: Fill finished sequences with pad tokens

    Why EOS Tokens?
    - Natural stopping point (model decides when done)
    - Variable length outputs (not all sequences same length)
    - Better than fixed max_tokens (some need fewer)

    Batch Generation Challenges:
    - Different sequences finish at different times
    - Need to track which are finished
    - Must not generate more tokens for finished sequences
    - Solution: Use pad tokens and finished mask

    Common Mistakes:
    - Not tracking finished status per sequence
    - Continuing generation for finished sequences
    - Not using @torch.no_grad() (wastes memory)
    - Forgetting to set model.eval()

    See: tiny_transformer/sampling/generator.py
    """

    @staticmethod
    def solution():
        class CustomGenerator:
            """Custom text generator with EOS handling."""

            def __init__(
                self,
                model: nn.Module,
                eos_token_id: int,
                pad_token_id: int,
                device: str = "cpu"
            ):
                # Store model and configuration
                self.model = model.to(device)
                self.eos_token_id = eos_token_id
                self.pad_token_id = pad_token_id
                self.device = device

                # Set to evaluation mode
                self.model.eval()

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
                Generate with EOS handling.

                Args:
                    start_tokens: Shape (batch_size, seq_len)
                    max_new_tokens: Maximum tokens to generate
                    temperature: Sampling temperature
                    top_k: Optional top-k filtering
                    top_p: Optional top-p filtering

                Returns:
                    generated: Shape (batch_size, variable_length)
                """
                # Move to device
                start_tokens = start_tokens.to(self.device)
                batch_size, start_len = start_tokens.shape

                # Initialize with start tokens
                generated = start_tokens.clone()

                # Track which sequences have finished
                # finished[i] = True if sequence i generated EOS
                finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

                # Import combined sampling
                combined_fn = Solution08_CombinedSampling.solution()

                # Autoregressive generation loop
                for step in range(max_new_tokens):
                    # Forward pass to get logits for next token
                    logits, _ = self.model(generated)
                    next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

                    # Sample next token using combined sampling
                    next_token = combined_fn(
                        next_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )

                    # For finished sequences, use pad token instead
                    # This prevents generating more content after EOS
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, self.pad_token_id),
                        next_token
                    )

                    # Append to generated sequence
                    next_token = next_token.unsqueeze(-1)  # (batch_size, 1)
                    generated = torch.cat([generated, next_token], dim=1)

                    # Update finished status
                    # finished |= (next_token == EOS)
                    finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

                    # Early stopping: all sequences finished
                    if finished.all():
                        print(f"All sequences finished at step {step + 1}")
                        break

                return generated

        # Example usage
        print("\nSolution 10: Custom Generator with EOS")
        print("=" * 70)

        from tiny_transformer.model import TinyTransformerLM

        # Create model
        model = TinyTransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256
        )

        # Create generator (EOS=0, PAD=1)
        generator = CustomGenerator(model, eos_token_id=0, pad_token_id=1)

        # Generate (start with tokens ≥ 2 to avoid immediate EOS)
        start = torch.randint(2, 100, (2, 5))
        output = generator.generate(start, max_new_tokens=20, temperature=0.8)

        print(f"Start shape: {start.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Start tokens: {start[0].tolist()}")
        print(f"Output tokens: {output[0].tolist()}")
        print("\nGenerator stops when EOS is generated!")
        print("=" * 70)

        return CustomGenerator


# ==============================================================================
# SOLUTION 11: Compare All Strategies
# ==============================================================================

class Solution11_CompareAllStrategies:
    """
    SOLUTION 11: Compare all sampling strategies side-by-side.

    Key Concepts:
    1. Different strategies for different use cases
    2. Trade-offs between diversity and quality
    3. Configuration matters as much as strategy

    Strategy Comparison:
    - Greedy: Most deterministic, can be repetitive
    - Temperature: Simple randomness control
    - Top-k: Fixed vocabulary cutoff
    - Top-p: Adaptive vocabulary cutoff
    - Combined: Fine-grained control

    Evaluation Criteria:
    - Coherence: Does text make sense?
    - Diversity: How varied is output?
    - Quality: Grammar, factuality, etc.
    - Speed: How fast is generation?

    Common Mistakes:
    - Not using same prompt (unfair comparison)
    - Not setting same seed (randomness affects results)
    - Comparing single samples (need multiple runs)
    - Not considering task type (creative vs factual)

    See: theory.md, Section "Comparing Sampling Strategies"
    """

    @staticmethod
    def solution():
        def compare_all_strategies(
            model: nn.Module,
            start_tokens: torch.Tensor,
            max_new_tokens: int = 30
        ) -> Dict[str, torch.Tensor]:
            """
            Generate with all sampling strategies.

            Args:
                model: TinyTransformerLM
                start_tokens: Shape (1, seq_len)
                max_new_tokens: Tokens to generate

            Returns:
                Dictionary mapping strategy to generated sequence
            """
            from tiny_transformer.sampling import TextGenerator, GeneratorConfig

            results = {}

            # Strategy 1: Greedy
            config = GeneratorConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False  # Greedy mode
            )
            generator = TextGenerator(model, config)
            results['greedy'] = generator.generate(start_tokens)

            # Strategy 2-4: Temperature sampling
            for temp in [0.5, 1.0, 2.0]:
                config = GeneratorConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True
                )
                generator = TextGenerator(model, config)
                results[f'temp_{temp}'] = generator.generate(start_tokens)

            # Strategy 5-6: Top-k sampling
            for k in [10, 50]:
                config = GeneratorConfig(
                    max_new_tokens=max_new_tokens,
                    top_k=k,
                    do_sample=True
                )
                generator = TextGenerator(model, config)
                results[f'top_k_{k}'] = generator.generate(start_tokens)

            # Strategy 7-8: Top-p sampling
            for p in [0.9, 0.95]:
                config = GeneratorConfig(
                    max_new_tokens=max_new_tokens,
                    top_p=p,
                    do_sample=True
                )
                generator = TextGenerator(model, config)
                results[f'top_p_{p}'] = generator.generate(start_tokens)

            # Strategy 9: Combined sampling
            config = GeneratorConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
            generator = TextGenerator(model, config)
            results['combined'] = generator.generate(start_tokens)

            return results

        # Example usage
        print("\nSolution 11: Compare All Strategies")
        print("=" * 70)

        from tiny_transformer.model import TinyTransformerLM

        model = TinyTransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256
        )
        model.eval()

        start = torch.randint(0, 100, (1, 10))
        results = compare_all_strategies(model, start, max_new_tokens=20)

        print(f"Compared {len(results)} strategies:")
        for name in results.keys():
            print(f"  - {name}")

        print(f"\nPrompt length: {start.size(1)}")
        print(f"Generated length: {results['greedy'].size(1)}")
        print("=" * 70)

        return compare_all_strategies


# ==============================================================================
# SOLUTION 12: Interactive Text Completion
# ==============================================================================

class Solution12_TextCompletionSystem:
    """
    SOLUTION 12: Interactive text completion system.

    Key Concepts:
    1. End-to-end pipeline: text → tokens → generation → text
    2. Multiple completions for user choice
    3. Ranking by likelihood
    4. Production-ready interface

    System Components:
    - Tokenizer: text ↔ tokens
    - Model: tokens → logits
    - Generator: autoregressive sampling
    - Ranker: score completions by probability

    Ranking Strategy:
    - Compute log probability of each token
    - Average over sequence length
    - Higher average = better according to model

    Common Mistakes:
    - Not detokenizing properly (show tokens instead of text)
    - Ranking before generation (should rank results)
    - Not handling variable length sequences
    - Forgetting to normalize by length (favors shorter)

    See: Applications of text generation systems
    """

    @staticmethod
    def solution():
        class TextCompletionSystem:
            """Interactive text completion with ranking."""

            def __init__(
                self,
                model: nn.Module,
                tokenizer,  # Assumes encode/decode methods
                device: str = "cpu"
            ):
                self.model = model.to(device)
                self.tokenizer = tokenizer
                self.device = device
                self.model.eval()

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
                Generate multiple completions.

                Args:
                    prompt: Input text
                    num_completions: Number of completions
                    max_new_tokens: Max tokens per completion
                    temperature: Sampling temperature
                    top_k: Top-k filtering
                    top_p: Top-p filtering

                Returns:
                    List of completion texts
                """
                # Tokenize prompt
                prompt_tokens = self.tokenizer.encode(prompt)
                prompt_tensor = torch.tensor(prompt_tokens).unsqueeze(0).to(self.device)

                # Import generator
                from tiny_transformer.sampling import TextGenerator, GeneratorConfig

                # Configure generator
                config = GeneratorConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )
                generator = TextGenerator(self.model, config, device=self.device)

                # Generate multiple completions
                completions = []
                for i in range(num_completions):
                    # Generate (each call uses different random seed)
                    generated = generator.generate(prompt_tensor)

                    # Decode to text
                    generated_tokens = generated[0].cpu().tolist()
                    completion_text = self.tokenizer.decode(generated_tokens)
                    completions.append(completion_text)

                return completions

            def rank_completions(
                self,
                prompt: str,
                completions: List[str]
            ) -> List[Tuple[str, float]]:
                """
                Rank completions by model likelihood.

                Args:
                    prompt: Original prompt
                    completions: List of completion texts

                Returns:
                    List of (completion, score) sorted by score
                """
                ranked = []

                for completion in completions:
                    # Tokenize
                    tokens = self.tokenizer.encode(completion)
                    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)

                    # Compute log probability
                    with torch.no_grad():
                        # Forward pass
                        logits, _ = self.model(tokens_tensor[:, :-1])

                        # Compute log probabilities
                        log_probs = F.log_softmax(logits, dim=-1)

                        # Get log prob of actual next tokens
                        target_tokens = tokens_tensor[:, 1:]
                        token_log_probs = log_probs.gather(
                            dim=-1,
                            index=target_tokens.unsqueeze(-1)
                        ).squeeze(-1)

                        # Average log probability (normalize by length)
                        avg_log_prob = token_log_probs.mean().item()

                    ranked.append((completion, avg_log_prob))

                # Sort by log probability (higher is better)
                ranked.sort(key=lambda x: x[1], reverse=True)

                return ranked

        # Example usage
        print("\nSolution 12: Text Completion System")
        print("=" * 70)
        print("Interactive system for text completion with ranking")
        print("Components: tokenizer, generator, ranker")
        print("Use case: Autocomplete, writing assistance, code completion")
        print("=" * 70)

        return TextCompletionSystem


# ==============================================================================
# SOLUTION 13: Temperature Scheduling
# ==============================================================================

class Solution13_TemperatureScheduling:
    """
    SOLUTION 13: Dynamic temperature adjustment during generation.

    Key Concepts:
    1. Start high (creative/exploratory)
    2. End low (focused/deterministic)
    3. Smooth transition between extremes

    Why Temperature Scheduling?
    - Early tokens: Explore different directions
    - Later tokens: Commit to coherent completion
    - Balance creativity and coherence

    Schedule Types:
    - Linear: Constant rate of change
    - Exponential: Faster change initially
    - Cosine: Smooth, accelerating change

    Mathematical Formulas:
    Linear: T(t) = start + (end - start) * progress
    Exponential: T(t) = start * (end/start)^progress
    Cosine: T(t) = end + (start - end) * 0.5 * (1 + cos(π * progress))

    Common Mistakes:
    - Wrong schedule direction (increasing instead of decreasing)
    - Not clamping progress to [0, 1]
    - Division by zero in exponential (when start=0)
    - Forgetting to update temperature each step

    See: theory.md, Section "Advanced Generation Techniques"
    """

    @staticmethod
    def solution():
        def temperature_scheduling(
            model: nn.Module,
            start_tokens: torch.Tensor,
            max_new_tokens: int = 50,
            start_temp: float = 2.0,
            end_temp: float = 0.5,
            schedule: str = "linear"
        ) -> torch.Tensor:
            """
            Generate with temperature scheduling.

            Args:
                model: TinyTransformerLM
                start_tokens: Shape (batch_size, seq_len)
                max_new_tokens: Tokens to generate
                start_temp: Initial temperature
                end_temp: Final temperature
                schedule: "linear", "exponential", or "cosine"

            Returns:
                generated: Shape (batch_size, seq_len + max_new_tokens)
            """
            model.eval()
            generated = start_tokens.clone()

            # Import sampling function
            temp_fn = Solution02_TemperatureSampling.solution()

            with torch.no_grad():
                for step in range(max_new_tokens):
                    # Compute progress through generation [0, 1]
                    progress = step / max(max_new_tokens - 1, 1)

                    # Compute temperature for this step
                    if schedule == "linear":
                        # Linear interpolation
                        temp = start_temp - (start_temp - end_temp) * progress

                    elif schedule == "exponential":
                        # Exponential decay
                        temp = start_temp * ((end_temp / start_temp) ** progress)

                    elif schedule == "cosine":
                        # Cosine annealing (smooth curve)
                        temp = end_temp + (start_temp - end_temp) * 0.5 * (
                            1 + math.cos(math.pi * progress)
                        )

                    else:
                        raise ValueError(f"Unknown schedule: {schedule}")

                    # Get logits for next token
                    logits, _ = model(generated)
                    next_token_logits = logits[:, -1, :]

                    # Sample with current temperature
                    next_token = temp_fn(next_token_logits, temperature=temp)

                    # Append to sequence
                    next_token = next_token.unsqueeze(-1)
                    generated = torch.cat([generated, next_token], dim=1)

            return generated

        # Example usage
        print("\nSolution 13: Temperature Scheduling")
        print("=" * 70)

        from tiny_transformer.model import TinyTransformerLM

        model = TinyTransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256
        )

        start = torch.randint(0, 100, (1, 10))

        # Compare schedules
        for sched in ["linear", "exponential", "cosine"]:
            generated = temperature_scheduling(
                model, start,
                max_new_tokens=30,
                start_temp=2.0,
                end_temp=0.5,
                schedule=sched
            )
            print(f"{sched:12} schedule: generated {generated.size(1)} tokens")

        print("\nTemperature scheduling balances creativity and coherence!")
        print("=" * 70)

        return temperature_scheduling


# ==============================================================================
# SOLUTION 14: Story Continuation System
# ==============================================================================

class Solution14_StoryContinuation:
    """
    SOLUTION 14: Creative story continuation system.

    Key Concepts:
    1. Multi-sentence generation
    2. Sentence boundary detection
    3. Style-specific parameters
    4. Coherence across sentences

    Challenges:
    - Maintaining coherence over long sequences
    - Detecting sentence boundaries
    - Adapting style to genre
    - Preventing topic drift

    Genre Parameters:
    - Fantasy: High temp (creative), high p (diverse vocab)
    - Sci-fi: Medium temp (technical + creative)
    - Mystery: Lower temp (logical, coherent)
    - Romance: Medium-high temp (emotional variety)

    Common Mistakes:
    - Generating all at once (loses control)
    - Not checking for sentence ends
    - Same parameters for all genres
    - Not trimming prompt from output

    See: Applications of creative AI systems
    """

    @staticmethod
    def solution():
        class StoryContinuation:
            """Creative story continuation system."""

            def __init__(
                self,
                model: nn.Module,
                tokenizer,
                device: str = "cpu"
            ):
                self.model = model.to(device)
                self.tokenizer = tokenizer
                self.device = device
                self.model.eval()

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
                Continue story for multiple sentences.

                Args:
                    story_start: Beginning of story
                    num_sentences: Sentences to generate
                    temperature: Sampling temperature
                    top_k: Top-k filtering
                    top_p: Top-p filtering
                    sentence_end_token: Sentence delimiter

                Returns:
                    Continuation text (without original prompt)
                """
                # Import generator
                from tiny_transformer.sampling import TextGenerator, GeneratorConfig

                # Tokenize starting text
                current_text = story_start
                prompt_tokens = self.tokenizer.encode(current_text)
                current_tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(self.device)

                sentences_generated = 0
                max_tokens_per_sentence = 50

                # Generate sentence by sentence
                while sentences_generated < num_sentences:
                    # Configure generator
                    config = GeneratorConfig(
                        max_new_tokens=max_tokens_per_sentence,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True
                    )
                    generator = TextGenerator(self.model, config, device=self.device)

                    # Generate tokens
                    generated = generator.generate(current_tokens)

                    # Decode to text
                    generated_tokens = generated[0].cpu().tolist()
                    generated_text = self.tokenizer.decode(generated_tokens)

                    # Check if we generated a sentence end
                    new_text = generated_text[len(current_text):]
                    if sentence_end_token in new_text:
                        # Find sentence end position
                        sentence_end_pos = new_text.find(sentence_end_token)

                        # Update text up to sentence end
                        current_text = generated_text[:len(current_text) + sentence_end_pos + 1]
                        sentences_generated += 1

                        # Prepare for next sentence
                        if sentences_generated < num_sentences:
                            current_tokens = torch.tensor(
                                self.tokenizer.encode(current_text)
                            ).unsqueeze(0).to(self.device)
                    else:
                        # No sentence end, continue from current generation
                        current_text = generated_text
                        current_tokens = generated

                        # Safety: prevent infinite loop
                        if len(current_tokens[0]) > len(prompt_tokens) + num_sentences * max_tokens_per_sentence:
                            break

                # Return only the continuation (remove prompt)
                continuation = current_text[len(story_start):].strip()
                return continuation

            def generate_with_style(
                self,
                story_start: str,
                style: str = "fantasy",
                num_sentences: int = 3
            ) -> str:
                """
                Generate with style-specific parameters.

                Args:
                    story_start: Beginning of story
                    style: "fantasy", "scifi", "mystery", "romance"
                    num_sentences: Sentences to generate

                Returns:
                    Styled continuation
                """
                # Define style presets
                style_params = {
                    'fantasy': {
                        'temperature': 0.9,
                        'top_k': 50,
                        'top_p': 0.92
                    },
                    'scifi': {
                        'temperature': 0.8,
                        'top_k': 60,
                        'top_p': 0.9
                    },
                    'mystery': {
                        'temperature': 0.7,
                        'top_k': 40,
                        'top_p': 0.85
                    },
                    'romance': {
                        'temperature': 0.85,
                        'top_k': 45,
                        'top_p': 0.88
                    }
                }

                # Get parameters for style
                params = style_params.get(
                    style,
                    {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9}
                )

                # Generate with style parameters
                return self.continue_story(
                    story_start,
                    num_sentences=num_sentences,
                    **params
                )

        # Example usage
        print("\nSolution 14: Story Continuation System")
        print("=" * 70)
        print("Creative writing system with:")
        print("  - Multi-sentence generation")
        print("  - Genre-specific parameters")
        print("  - Sentence boundary detection")
        print("  - Coherence maintenance")
        print("\nUse cases: Creative writing, storytelling, content generation")
        print("=" * 70)

        return StoryContinuation


# ==============================================================================
# Testing and Validation
# ==============================================================================

def run_all_tests():
    """
    Run tests for all solutions.

    This validates that all implementations work correctly.
    """
    print("=" * 70)
    print("Module 07: Sampling & Generation - Solution Tests")
    print("=" * 70)

    # Test Solution 1
    print("\n[TEST 1] Greedy Sampling")
    greedy_fn = Solution01_GreedySampling.solution()
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.3, 0.8]])
    tokens = greedy_fn(logits)
    assert tokens.tolist() == [1, 2], f"Expected [1, 2], got {tokens.tolist()}"
    print("✓ Solution 1 passed!")

    # Test Solution 2
    print("\n[TEST 2] Temperature Sampling")
    temp_fn = Solution02_TemperatureSampling.solution()
    torch.manual_seed(42)
    logits = torch.randn(3, 100)
    tokens = temp_fn(logits, temperature=0.8)
    assert tokens.shape == (3,), f"Expected shape (3,), got {tokens.shape}"
    assert tokens.dtype == torch.long, "Expected long dtype"
    print("✓ Solution 2 passed!")

    # Test Solution 3
    print("\n[TEST 3] Compare Strategies")
    compare_fn = Solution03_CompareSamplingStrategies.solution()
    torch.manual_seed(42)
    logits = torch.randn(100)
    results = compare_fn(logits, num_samples=10)
    assert len(results) == 4, f"Expected 4 strategies, got {len(results)}"
    assert len(set(results['greedy'])) == 1, "Greedy should be deterministic"
    print("✓ Solution 3 passed!")

    # Test Solution 4
    print("\n[TEST 4] Visualize Temperature")
    viz_fn = Solution04_VisualizeTemperature.solution()
    logits = torch.tensor([2.0, 1.0, 0.5])
    dists = viz_fn(logits)
    for temp, probs in dists.items():
        assert torch.allclose(probs.sum(), torch.tensor(1.0)), \
            f"Probs don't sum to 1 at T={temp}"
    print("✓ Solution 4 passed!")

    # Test Solution 5
    print("\n[TEST 5] Top-K Sampling")
    topk_fn = Solution05_TopKSampling.solution()
    torch.manual_seed(42)
    logits = torch.randn(2, 100)
    tokens = topk_fn(logits, k=10)
    assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    print("✓ Solution 5 passed!")

    # Test Solution 6
    print("\n[TEST 6] Top-P Sampling")
    topp_fn = Solution06_TopPSampling.solution()
    torch.manual_seed(42)
    logits = torch.randn(2, 100)
    tokens = topp_fn(logits, p=0.9)
    assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    print("✓ Solution 6 passed!")

    # Test Solution 8
    print("\n[TEST 8] Combined Sampling")
    combined_fn = Solution08_CombinedSampling.solution()
    torch.manual_seed(42)
    logits = torch.randn(2, 100)
    tokens = combined_fn(logits, temperature=0.8, top_k=50, top_p=0.9)
    assert tokens.shape == (2,), f"Expected shape (2,), got {tokens.shape}"
    print("✓ Solution 8 passed!")

    print("\n" + "=" * 70)
    print("All solution tests passed!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Module 07: Sampling & Generation - Reference Solutions")
    print("=" * 70)
    print("\nThis file contains complete solutions for all 14 exercises.")
    print("\nSolution Structure:")
    print("  01. Greedy Sampling (Easy)")
    print("  02. Temperature Sampling (Easy)")
    print("  03. Compare Greedy vs Temperature (Easy)")
    print("  04. Visualize Temperature Effects (Easy)")
    print("  05. Top-K Sampling (Medium)")
    print("  06. Top-P (Nucleus) Sampling (Medium)")
    print("  07. Use TextGenerator (Medium)")
    print("  08. Combined Sampling Strategy (Hard)")
    print("  09. Analyze Temperature Effects (Medium)")
    print("  10. Custom Generator with EOS (Hard)")
    print("  11. Compare All Strategies (Hard)")
    print("  12. Text Completion System (Very Hard)")
    print("  13. Temperature Scheduling (Very Hard)")
    print("  14. Story Continuation System (Very Hard)")
    print("\nHow to Use:")
    print("1. Study each solution's implementation")
    print("2. Read the educational comments explaining WHY")
    print("3. Compare with your own implementations")
    print("4. Run tests to verify correctness")
    print("\nKey Takeaways:")
    print("- Greedy: Deterministic, fast, can be repetitive")
    print("- Temperature: Simple control over randomness")
    print("- Top-k: Fixed vocabulary filtering")
    print("- Top-p: Adaptive vocabulary filtering")
    print("- Combined: Fine-grained control over generation")
    print("- EOS handling: Proper sequence termination")
    print("- Scheduling: Dynamic parameter adjustment")
    print("=" * 70)

    # Run tests
    print("\n\nRunning solution tests...\n")
    run_all_tests()

    print("\n\nExcellent work completing Module 07!")
    print("You now understand sampling and text generation strategies.")
    print("\nNext steps:")
    print("- Experiment with different sampling parameters")
    print("- Try generating creative text with your trained models")
    print("- Implement beam search for better quality")
    print("- Study constrained generation techniques")
    print("=" * 70)
