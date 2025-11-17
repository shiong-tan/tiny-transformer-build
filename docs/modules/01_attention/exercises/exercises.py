"""
Attention Mechanism Practice Exercises

These exercises will help you understand the attention mechanism by implementing
each component from scratch. Work through them in order, as they build on each other.

Each exercise has:
- Clear docstring with task description
- Type hints for all parameters
- Expected input/output shapes
- Example usage
- TODO comments marking where to write code
- Test assertions (commented out - uncomment to verify your solution)

Tips:
- Read the docstrings carefully - they contain important hints
- Pay attention to tensor shapes at each step
- Use print() to debug tensor shapes if stuck
- The attention.py file in tiny_transformer/ is your reference
- Start simple, test often

Good luck! 🚀
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# EXERCISE 1: Basic Matrix Multiplication for Attention
# Difficulty: Easy
# ==============================================================================

def exercise_01_compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor
) -> torch.Tensor:
    """
    Compute raw attention scores using matrix multiplication.

    This is the first step of attention: measuring how much each query
    should attend to each key by computing their dot products.

    The formula is: scores = Q @ K^T

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)

    Returns:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
                scores[b, i, j] = how much query i attends to key j in batch b

    Example:
        >>> Q = torch.randn(2, 4, 8)  # batch=2, seq_len=4, d_k=8
        >>> K = torch.randn(2, 4, 8)
        >>> scores = exercise_01_compute_attention_scores(Q, K)
        >>> scores.shape
        torch.Size([2, 4, 4])

    Hint:
        Use @ for matrix multiplication and .transpose(-2, -1) to swap
        the last two dimensions of K.
    """
    # TODO: Implement this function
    # Step 1: Transpose key to shape (batch_size, d_k, seq_len)
    # Step 2: Multiply query by transposed key
    # Remember: (B, T, d_k) @ (B, d_k, T) -> (B, T, T)

    pass  # Remove this and add your implementation

    # Uncomment these assertions to test your implementation:
    # assert scores.dim() == 3, "Scores should be 3D"
    # assert scores.shape[0] == query.shape[0], "Batch dimension should match"
    # assert scores.shape[1] == query.shape[1], "Seq len should match query"
    # assert scores.shape[2] == key.shape[1], "Seq len should match key"


# ==============================================================================
# EXERCISE 2: Scaled Attention Scores
# Difficulty: Easy
# ==============================================================================

def exercise_02_scale_attention_scores(
    scores: torch.Tensor,
    d_k: int
) -> torch.Tensor:
    """
    Scale attention scores by sqrt(d_k) to prevent vanishing gradients.

    When d_k is large, dot products can grow very large in magnitude,
    causing the softmax to have very small gradients. Scaling prevents this.

    The formula is: scaled_scores = scores / sqrt(d_k)

    Args:
        scores: Raw attention scores of shape (batch_size, seq_len, seq_len)
        d_k: Dimension of the key/query vectors (used for scaling)

    Returns:
        scaled_scores: Scaled attention scores of same shape as input

    Example:
        >>> scores = torch.randn(2, 4, 4)
        >>> d_k = 64
        >>> scaled = exercise_02_scale_attention_scores(scores, d_k)
        >>> # Scaled values should be smaller in magnitude
        >>> assert scaled.abs().mean() < scores.abs().mean()

    Hint:
        Use math.sqrt() to compute the square root.
    """
    # TODO: Implement this function
    # Divide scores by sqrt(d_k)

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert scaled_scores.shape == scores.shape, "Shape should not change"
    # assert not torch.allclose(scaled_scores, scores), "Should be different from input"


# ==============================================================================
# EXERCISE 3: Create Causal Mask
# Difficulty: Medium
# ==============================================================================

def exercise_03_create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a causal (lower-triangular) mask for autoregressive models.

    In autoregressive generation, position i should only attend to positions
    0 through i (not future positions). We achieve this with a mask.

    The mask should be:
    - 0.0 for positions that CAN be attended to
    - -inf for positions that CANNOT be attended to

    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on (CPU or CUDA)

    Returns:
        mask: Causal mask of shape (seq_len, seq_len)

    Example:
        >>> mask = exercise_03_create_causal_mask(4)
        >>> print(mask)
        tensor([[  0., -inf, -inf, -inf],
                [  0.,   0., -inf, -inf],
                [  0.,   0.,   0., -inf],
                [  0.,   0.,   0.,   0.]])

    Visual interpretation:
        Row i shows what position i can attend to:
        Row 0: can only attend to position 0 (itself)
        Row 1: can attend to positions 0 and 1
        Row 2: can attend to positions 0, 1, and 2
        etc.

    Hints:
        1. Use torch.tril() to create a lower triangular matrix of ones
        2. Use .masked_fill() to replace 0s with -inf and 1s with 0.0
        3. Or use torch.triu() with offset=1 to get upper triangle
    """
    # TODO: Implement this function
    # Method 1: Using torch.tril
    #   - Create lower triangular matrix
    #   - Replace 0s with -inf, 1s with 0.0
    #
    # Method 2: Using torch.triu
    #   - Create upper triangular matrix (excluding diagonal)
    #   - Replace 1s with -inf, 0s with 0.0

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert mask.shape == (seq_len, seq_len), "Should be square matrix"
    # assert mask[0, 0] == 0, "Diagonal should be 0 (can attend to self)"
    # assert mask[0, 1] == float('-inf'), "Upper triangle should be -inf"
    # assert mask[1, 0] == 0, "Lower triangle should be 0"


# ==============================================================================
# EXERCISE 4: Apply Mask to Scores
# Difficulty: Easy
# ==============================================================================

def exercise_04_apply_mask(
    scores: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply a mask to attention scores.

    The mask contains -inf for positions to ignore and 0 for positions to keep.
    After adding the mask, masked positions will have -inf, which becomes 0
    after softmax.

    Args:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
        mask: Mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
              Contains 0 (keep) and -inf (mask out)

    Returns:
        masked_scores: Scores with mask applied, same shape as scores

    Example:
        >>> scores = torch.randn(2, 4, 4)
        >>> mask = exercise_03_create_causal_mask(4)
        >>> masked = exercise_04_apply_mask(scores, mask)
        >>> # Upper triangle should now be -inf
        >>> assert masked[0, 0, 1] == float('-inf')

    Hint:
        Simply add the mask to the scores. Broadcasting will handle
        different shapes automatically.
    """
    # TODO: Implement this function
    # Add mask to scores (broadcasting handles shape differences)

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert masked_scores.shape == scores.shape, "Shape should not change"
    # if mask.dim() == 2:
    #     # Check that broadcasting worked correctly
    #     for b in range(scores.shape[0]):
    #         assert torch.allclose(masked_scores[b, 0, 1:],
    #                              torch.full((scores.shape[2]-1,), float('-inf')))


# ==============================================================================
# EXERCISE 5: Compute Attention Weights with Softmax
# Difficulty: Easy
# ==============================================================================

def exercise_05_compute_attention_weights(
    scores: torch.Tensor
) -> torch.Tensor:
    """
    Convert attention scores to probabilities using softmax.

    Softmax normalizes the scores so each row sums to 1, giving us a
    probability distribution over which positions to attend to.

    Formula: attention_weights[i] = softmax(scores[i])

    Args:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
                Can contain -inf values from masking

    Returns:
        attention_weights: Probabilities of shape (batch_size, seq_len, seq_len)
                          Each row sums to 1.0

    Example:
        >>> scores = torch.randn(2, 4, 4)
        >>> weights = exercise_05_compute_attention_weights(scores)
        >>> # Each row should sum to 1
        >>> assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))

    Important:
        Apply softmax over the LAST dimension (dim=-1).
        This means each query position gets a probability distribution
        over all key positions.

    Hint:
        Use F.softmax() from torch.nn.functional
    """
    # TODO: Implement this function
    # Apply softmax over the last dimension

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert attention_weights.shape == scores.shape
    # assert torch.allclose(attention_weights.sum(dim=-1),
    #                      torch.ones(scores.shape[0], scores.shape[1]))
    # assert (attention_weights >= 0).all(), "Probabilities should be non-negative"
    # assert (attention_weights <= 1).all(), "Probabilities should be <= 1"


# ==============================================================================
# EXERCISE 6: Apply Attention to Values
# Difficulty: Easy
# ==============================================================================

def exercise_06_apply_attention_to_values(
    attention_weights: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Apply attention weights to values to get the final output.

    This is a weighted sum: each output position is a weighted combination
    of all value vectors, where weights come from attention.

    Formula: output = attention_weights @ V

    Args:
        attention_weights: Attention probabilities of shape (batch_size, seq_len, seq_len)
        value: Value tensor of shape (batch_size, seq_len, d_v)

    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)

    Example:
        >>> weights = torch.softmax(torch.randn(2, 4, 4), dim=-1)
        >>> V = torch.randn(2, 4, 8)
        >>> output = exercise_06_apply_attention_to_values(weights, V)
        >>> output.shape
        torch.Size([2, 4, 8])

    Hint:
        Matrix multiply: (B, T, T) @ (B, T, d_v) -> (B, T, d_v)
    """
    # TODO: Implement this function
    # Multiply attention_weights by value

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert output.shape == value.shape, "Output should have same shape as value"
    # assert output.dim() == 3, "Output should be 3D"


# ==============================================================================
# EXERCISE 7: Full Attention Implementation
# Difficulty: Medium
# ==============================================================================

def exercise_07_full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implement complete scaled dot-product attention.

    Combine all previous exercises into one complete attention function.

    Steps:
        1. Compute attention scores: Q @ K^T
        2. Scale by sqrt(d_k)
        3. Apply mask (if provided)
        4. Apply softmax to get attention weights
        5. Apply attention weights to values

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)
        value: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask of shape (seq_len, seq_len) with 0 and -inf

    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention probabilities of shape (batch_size, seq_len, seq_len)

    Example:
        >>> Q = torch.randn(2, 4, 8)
        >>> K = torch.randn(2, 4, 8)
        >>> V = torch.randn(2, 4, 8)
        >>> output, weights = exercise_07_full_attention(Q, K, V)
        >>> output.shape, weights.shape
        (torch.Size([2, 4, 8]), torch.Size([2, 4, 4]))

    Hint:
        You can reuse the functions from previous exercises!
    """
    # TODO: Implement this function
    # Use the pattern from tiny_transformer/attention.py

    # Step 1: Get d_k for scaling

    # Step 2: Compute scores

    # Step 3: Scale scores

    # Step 4: Apply mask if provided

    # Step 5: Compute attention weights with softmax

    # Step 6: Apply attention to values

    # Step 7: Return output and weights

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert output.shape == value.shape
    # assert attention_weights.shape == (query.shape[0], query.shape[1], key.shape[1])
    # assert torch.allclose(attention_weights.sum(dim=-1),
    #                      torch.ones(query.shape[0], query.shape[1]))


# ==============================================================================
# EXERCISE 8: Debug Broken Attention (Find and Fix the Bug)
# Difficulty: Medium
# ==============================================================================

def exercise_08_debug_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This attention implementation has bugs! Find and fix them.

    There are 3 bugs in this code. Your task:
    1. Read through the code carefully
    2. Identify what's wrong
    3. Fix the bugs
    4. Test that it produces correct output

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)
        value: Value tensor of shape (batch_size, seq_len, d_v)

    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention probabilities

    Hints:
        - Check the scaling factor
        - Check the transpose dimensions
        - Check the softmax dimension
        - What should the attention weights sum to?
    """
    # BUG 1: Wrong dimension for d_k
    d_k = query.size(0)  # Is this correct?

    # BUG 2: Wrong dimensions for transpose
    scores = query @ key.transpose(0, 1)  # Is this correct?

    # Scale scores
    scores = scores / math.sqrt(d_k)

    # BUG 3: Softmax over wrong dimension
    attention_weights = F.softmax(scores, dim=0)  # Is this correct?

    # Apply attention to values
    output = attention_weights @ value

    return output, attention_weights

    # Uncomment to test your fix:
    # Q = torch.randn(2, 4, 8)
    # K = torch.randn(2, 4, 8)
    # V = torch.randn(2, 4, 8)
    # output, weights = exercise_08_debug_attention(Q, K, V)
    # assert output.shape == V.shape, "Output shape should match value shape"
    # assert weights.shape == (2, 4, 4), "Weights should be (batch, seq, seq)"
    # assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4)), "Rows should sum to 1"


# ==============================================================================
# EXERCISE 9: Shape Manipulation Challenge
# Difficulty: Hard
# ==============================================================================

def exercise_09_reshape_for_multi_head(
    tensor: torch.Tensor,
    num_heads: int
) -> torch.Tensor:
    """
    Reshape a tensor for multi-head attention.

    Multi-head attention splits the d_model dimension into num_heads,
    each with dimension d_k = d_model // num_heads.

    This requires careful reshaping and transposing.

    Args:
        tensor: Input of shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        reshaped: Tensor of shape (batch_size, num_heads, seq_len, d_k)
                  where d_k = d_model // num_heads

    Example:
        >>> x = torch.randn(2, 4, 64)  # batch=2, seq=4, d_model=64
        >>> reshaped = exercise_09_reshape_for_multi_head(x, num_heads=8)
        >>> reshaped.shape
        torch.Size([2, 8, 4, 8])  # batch=2, heads=8, seq=4, d_k=8

    Steps:
        1. Get batch_size, seq_len, and d_model from input shape
        2. Calculate d_k = d_model // num_heads
        3. Reshape to (batch_size, seq_len, num_heads, d_k)
        4. Transpose to (batch_size, num_heads, seq_len, d_k)

    Hint:
        Use .view() for reshaping and .transpose() or .permute() for reordering
    """
    # TODO: Implement this function

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # batch_size, seq_len, d_model = tensor.shape
    # d_k = d_model // num_heads
    # assert reshaped.shape == (batch_size, num_heads, seq_len, d_k)
    # # Check that data is preserved (just reordered)
    # assert reshaped.reshape(batch_size, seq_len, d_model).allclose(tensor)


# ==============================================================================
# EXERCISE 10: Attention with Padding Mask
# Difficulty: Hard
# ==============================================================================

def exercise_10_create_padding_mask(
    sequence_lengths: torch.Tensor,
    max_len: int
) -> torch.Tensor:
    """
    Create a padding mask for variable-length sequences.

    When processing batches, sequences have different lengths and are padded.
    We need to prevent attention to padding positions.

    Args:
        sequence_lengths: Tensor of shape (batch_size,) with actual length of each sequence
        max_len: Maximum sequence length (length after padding)

    Returns:
        mask: Padding mask of shape (batch_size, 1, 1, max_len)
              0.0 for real tokens, -inf for padding tokens
              Shape is (B, 1, 1, T) to broadcast with attention scores (B, H, T, T)

    Example:
        >>> lengths = torch.tensor([3, 2, 4])  # 3 sequences of different lengths
        >>> mask = exercise_10_create_padding_mask(lengths, max_len=4)
        >>> # First sequence: length 3, so position 3 is padding
        >>> # Second sequence: length 2, so positions 2,3 are padding
        >>> # Third sequence: length 4, so no padding

    Steps:
        1. Create a range tensor [0, 1, 2, ..., max_len-1]
        2. Compare with sequence_lengths to find padding positions
        3. Convert to mask with 0 and -inf
        4. Reshape to (batch_size, 1, 1, max_len) for broadcasting

    Hints:
        - Use torch.arange() to create range
        - Use broadcasting to compare with lengths
        - Use .masked_fill() or boolean indexing
    """
    # TODO: Implement this function

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # batch_size = sequence_lengths.shape[0]
    # assert mask.shape == (batch_size, 1, 1, max_len)
    # # Check first sequence (length=3, max=4)
    # assert mask[0, 0, 0, 2] == 0.0, "Position 2 should be valid"
    # assert mask[0, 0, 0, 3] == float('-inf'), "Position 3 should be masked"


# ==============================================================================
# BONUS EXERCISE: Performance Optimization
# Difficulty: Advanced
# ==============================================================================

def bonus_optimized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_flash: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized attention implementation with memory-efficient options.

    This exercise explores performance optimizations:
    1. In-place operations where safe
    2. Fused operations
    3. Memory-efficient attention (if use_flash=True and available)

    Args:
        query, key, value: Standard attention inputs
        mask: Optional attention mask
        use_flash: Whether to use Flash Attention if available (PyTorch 2.0+)

    Returns:
        output: Attention output
        attention_weights: Attention probabilities (None if using Flash Attention)

    Challenge:
        Implement attention using torch.nn.functional.scaled_dot_product_attention
        when use_flash=True (if PyTorch 2.0+), otherwise use standard implementation.

    Hint:
        Check if F.scaled_dot_product_attention exists using hasattr()
    """
    # TODO: Implement this function
    # 1. Check PyTorch version and use_flash flag
    # 2. If available and requested, use F.scaled_dot_product_attention
    # 3. Otherwise, use your implementation from exercise 7

    pass  # Remove this and add your implementation


# ==============================================================================
# Testing and Validation
# ==============================================================================

def run_all_tests():
    """
    Run tests for all exercises.

    Uncomment each test as you complete the corresponding exercise.
    """
    print("=" * 70)
    print("Running Exercise Tests")
    print("=" * 70)

    # Test Exercise 1
    # print("\nExercise 1: Compute Attention Scores")
    # Q = torch.randn(2, 4, 8)
    # K = torch.randn(2, 4, 8)
    # scores = exercise_01_compute_attention_scores(Q, K)
    # assert scores.shape == (2, 4, 4), f"Expected (2,4,4), got {scores.shape}"
    # print("✓ Exercise 1 passed!")

    # Test Exercise 2
    # print("\nExercise 2: Scale Attention Scores")
    # scores = torch.randn(2, 4, 4) * 10  # Large values
    # scaled = exercise_02_scale_attention_scores(scores, 64)
    # assert scaled.shape == scores.shape
    # assert scaled.abs().mean() < scores.abs().mean()
    # print("✓ Exercise 2 passed!")

    # Test Exercise 3
    # print("\nExercise 3: Create Causal Mask")
    # mask = exercise_03_create_causal_mask(4)
    # assert mask.shape == (4, 4)
    # assert mask[0, 0] == 0 and mask[0, 1] == float('-inf')
    # assert mask[3, 0] == 0 and mask[3, 3] == 0
    # print("✓ Exercise 3 passed!")

    # Test Exercise 4
    # print("\nExercise 4: Apply Mask")
    # scores = torch.ones(2, 4, 4)
    # mask = exercise_03_create_causal_mask(4)
    # masked = exercise_04_apply_mask(scores, mask)
    # assert masked[0, 0, 1] == float('-inf')
    # print("✓ Exercise 4 passed!")

    # Test Exercise 5
    # print("\nExercise 5: Compute Attention Weights")
    # scores = torch.randn(2, 4, 4)
    # weights = exercise_05_compute_attention_weights(scores)
    # assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))
    # print("✓ Exercise 5 passed!")

    # Test Exercise 6
    # print("\nExercise 6: Apply Attention to Values")
    # weights = torch.softmax(torch.randn(2, 4, 4), dim=-1)
    # V = torch.randn(2, 4, 8)
    # output = exercise_06_apply_attention_to_values(weights, V)
    # assert output.shape == V.shape
    # print("✓ Exercise 6 passed!")

    # Test Exercise 7
    # print("\nExercise 7: Full Attention")
    # Q = torch.randn(2, 4, 8)
    # K = torch.randn(2, 4, 8)
    # V = torch.randn(2, 4, 8)
    # output, weights = exercise_07_full_attention(Q, K, V)
    # assert output.shape == V.shape
    # assert weights.shape == (2, 4, 4)
    # assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))
    # print("✓ Exercise 7 passed!")

    # Test Exercise 8
    # print("\nExercise 8: Debug Attention")
    # Q = torch.randn(2, 4, 8)
    # K = torch.randn(2, 4, 8)
    # V = torch.randn(2, 4, 8)
    # output, weights = exercise_08_debug_attention(Q, K, V)
    # assert output.shape == V.shape
    # assert weights.shape == (2, 4, 4)
    # assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    # print("✓ Exercise 8 passed!")

    # Test Exercise 9
    # print("\nExercise 9: Reshape for Multi-Head")
    # x = torch.randn(2, 4, 64)
    # reshaped = exercise_09_reshape_for_multi_head(x, num_heads=8)
    # assert reshaped.shape == (2, 8, 4, 8)
    # print("✓ Exercise 9 passed!")

    # Test Exercise 10
    # print("\nExercise 10: Padding Mask")
    # lengths = torch.tensor([3, 2, 4])
    # mask = exercise_10_create_padding_mask(lengths, max_len=4)
    # assert mask.shape == (3, 1, 1, 4)
    # assert mask[0, 0, 0, 2] == 0.0
    # assert mask[0, 0, 0, 3] == float('-inf')
    # assert mask[1, 0, 0, 1] == 0.0
    # assert mask[1, 0, 0, 2] == float('-inf')
    # print("✓ Exercise 10 passed!")

    print("\n" + "=" * 70)
    print("All tests passed! Great work! 🎉")
    print("=" * 70)


# ==============================================================================
# Solutions (for reference - try not to peek!)
# ==============================================================================

def solution_01(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Solution for Exercise 1"""
    return query @ key.transpose(-2, -1)


def solution_02(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    """Solution for Exercise 2"""
    return scores / math.sqrt(d_k)


def solution_03(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Solution for Exercise 3"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask


def solution_04(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Solution for Exercise 4"""
    return scores + mask


def solution_05(scores: torch.Tensor) -> torch.Tensor:
    """Solution for Exercise 5"""
    return F.softmax(scores, dim=-1)


def solution_06(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Solution for Exercise 6"""
    return attention_weights @ value


def solution_07(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solution for Exercise 7"""
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ value
    return output, attention_weights


def solution_08(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solution for Exercise 8 (fixed bugs)"""
    # FIX 1: d_k should be the last dimension, not batch size
    d_k = query.size(-1)

    # FIX 2: Transpose last two dimensions, not first two
    scores = query @ key.transpose(-2, -1)

    # Scale scores
    scores = scores / math.sqrt(d_k)

    # FIX 3: Softmax over last dimension (over keys)
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = attention_weights @ value

    return output, attention_weights


def solution_09(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Solution for Exercise 9"""
    batch_size, seq_len, d_model = tensor.shape
    d_k = d_model // num_heads

    # Reshape: (B, T, d_model) -> (B, T, num_heads, d_k)
    reshaped = tensor.view(batch_size, seq_len, num_heads, d_k)

    # Transpose: (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
    reshaped = reshaped.transpose(1, 2)

    return reshaped


def solution_10(sequence_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Solution for Exercise 10"""
    batch_size = sequence_lengths.shape[0]

    # Create range [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=sequence_lengths.device)

    # Expand for broadcasting: (1, max_len) vs (batch_size, 1)
    # Result: (batch_size, max_len) where True = padding position
    padding_mask = positions.unsqueeze(0) >= sequence_lengths.unsqueeze(1)

    # Convert to attention mask format
    mask = torch.zeros(batch_size, max_len, device=sequence_lengths.device)
    mask = mask.masked_fill(padding_mask, float('-inf'))

    # Reshape for broadcasting with attention scores (B, H, T, T)
    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

    return mask


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Attention Mechanism - Practice Exercises")
    print("=" * 70)
    print("\nWelcome! These exercises will help you master the attention mechanism.")
    print("\nInstructions:")
    print("1. Work through exercises 1-10 in order")
    print("2. Read each docstring carefully")
    print("3. Implement the TODO sections")
    print("4. Uncomment the assertions to test your code")
    print("5. Run this file to see if your solutions work")
    print("\nTips:")
    print("- Use print() to debug tensor shapes")
    print("- Refer to tiny_transformer/attention.py for guidance")
    print("- Solutions are at the bottom (try not to peek!)")
    print("=" * 70)

    # Uncomment this when you've completed some exercises:
    # run_all_tests()

    # Example: Test Exercise 1 individually
    print("\nExample: Testing Exercise 1")
    print("-" * 70)
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")

    # Uncomment when you implement exercise_01:
    # scores = exercise_01_compute_attention_scores(Q, K)
    # print(f"Scores shape: {scores.shape}")
    # print(f"Expected: torch.Size([2, 4, 4])")

    print("\nGood luck! 🚀")
