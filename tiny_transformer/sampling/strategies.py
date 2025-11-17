"""
Sampling Strategies for Text Generation.

This module implements various sampling strategies:
- Greedy: Always select highest probability token
- Temperature: Control randomness with temperature scaling
- Top-K: Sample from K highest probability tokens
- Top-P (Nucleus): Sample from smallest set with cumulative probability >= p

See Also:
    - theory.md Sections 5-9: Sampling Strategies
"""

import torch
import torch.nn.functional as F
from typing import Optional


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling: select token with highest probability.

    Args:
        logits: Logits of shape (batch_size, vocab_size)

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> logits = torch.randn(4, 1000)
        >>> tokens = greedy_sample(logits)
        >>> tokens.shape
        torch.Size([4])
    """
    return logits.argmax(dim=-1)


def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Temperature sampling: scale logits and sample.

    Lower temperature → more deterministic
    Higher temperature → more random

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        temperature: Temperature value (> 0)

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> logits = torch.randn(4, 1000)
        >>> tokens = temperature_sample(logits, temperature=0.8)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    if temperature == 1.0:
        # No scaling needed
        probs = F.softmax(logits, dim=-1)
    else:
        # Scale logits by temperature
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

    # Sample from probability distribution
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sample(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-K sampling: sample from K highest probability tokens.

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        k: Number of top tokens to consider
        temperature: Temperature for sampling

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> logits = torch.randn(4, 1000)
        >>> tokens = top_k_sample(logits, k=50)
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Find top-k logits
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # Set all other logits to -inf (zero probability)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Sample from filtered distribution
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-P (nucleus) sampling: sample from smallest set with cumulative probability >= p.

    Dynamically adapts the number of tokens based on distribution shape.

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        p: Cumulative probability threshold (0 < p ≤ 1)
        temperature: Temperature for sampling

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> logits = torch.randn(4, 1000)
        >>> tokens = top_p_sample(logits, p=0.9)
    """
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

    # Find cutoff: first position where cumsum exceeds p
    # Shift right by 1 to keep at least one token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    # Set removed indices to -inf
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')

    # Sample from filtered distribution
    filtered_probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)


def combined_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Combined sampling: temperature + top-k + top-p.

    Apply all filtering methods in sequence:
    1. Temperature scaling
    2. Top-k filtering (if specified)
    3. Top-p filtering (if specified)
    4. Sample from final distribution

    Args:
        logits: Logits of shape (batch_size, vocab_size)
        temperature: Temperature for scaling
        top_k: Optional top-k filtering
        top_p: Optional nucleus sampling

    Returns:
        Token IDs of shape (batch_size,)

    Example:
        >>> logits = torch.randn(4, 1000)
        >>> tokens = combined_sample(
        ...     logits,
        ...     temperature=0.8,
        ...     top_k=50,
        ...     top_p=0.9
        ... )
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
        logits = filtered_logits

    # Apply top-p filtering
    if top_p is not None and 0 < top_p < 1:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )

        logits[indices_to_remove] = float('-inf')

    # Sample from final distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


if __name__ == "__main__":
    print("=" * 70)
    print("Sampling Strategies - Demo")
    print("=" * 70)
    print()

    # Create sample logits
    torch.manual_seed(42)
    batch_size = 2
    vocab_size = 100
    logits = torch.randn(batch_size, vocab_size)

    print(f"Logits shape: {logits.shape}")
    print()

    # Test greedy
    print("1. Greedy Sampling:")
    tokens = greedy_sample(logits)
    print(f"   Selected tokens: {tokens.tolist()}")
    print()

    # Test temperature
    print("2. Temperature Sampling:")
    for temp in [0.5, 1.0, 2.0]:
        tokens = temperature_sample(logits, temperature=temp)
        print(f"   T={temp:.1f}: {tokens.tolist()}")
    print()

    # Test top-k
    print("3. Top-K Sampling (k=10):")
    tokens = top_k_sample(logits, k=10)
    print(f"   Selected tokens: {tokens.tolist()}")
    print()

    # Test top-p
    print("4. Top-P Sampling (p=0.9):")
    tokens = top_p_sample(logits, p=0.9)
    print(f"   Selected tokens: {tokens.tolist()}")
    print()

    # Test combined
    print("5. Combined Sampling (T=0.8, k=50, p=0.9):")
    tokens = combined_sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    print(f"   Selected tokens: {tokens.tolist()}")
    print()

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
