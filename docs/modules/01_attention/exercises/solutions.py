"""
Attention Mechanism Practice Exercises - COMPLETE SOLUTIONS

This file provides comprehensive, production-quality solutions to all exercises
from exercises.py. Each solution includes:

1. Complete implementation with all code filled in
2. Detailed explanatory comments (WHY, not just WHAT)
3. Shape annotations showing tensor dimensions at each step
4. Inline educational notes about key concepts and common mistakes
5. References to theory.md for deeper understanding
6. Alternative approaches where applicable

Study these solutions to:
- Understand best practices for implementing attention
- Learn defensive programming (shape checking, assertions)
- See how theory translates to code
- Recognize common pitfalls and how to avoid them

Reference implementation: /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/attention.py
Theory reference: /Users/shiongtan/projects/tiny-transformer-build/docs/modules/01_attention/theory.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# EXERCISE 1 SOLUTION: Basic Matrix Multiplication for Attention
# ==============================================================================

def solution_01_compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor
) -> torch.Tensor:
    """
    Compute raw attention scores using matrix multiplication.

    This is the first step of scaled dot-product attention: measuring similarity
    between queries and keys via their dot products.

    Theory reference: theory.md, Section "Mathematical Formulation", Step 1

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)

    Returns:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
                scores[b, i, j] = how much query i attends to key j in batch b

    Shape transformation breakdown:
        query: (B, T, d_k)
        key: (B, T, d_k)
        key.transpose(-2, -1): (B, d_k, T)
        query @ key.transpose(-2, -1): (B, T, d_k) @ (B, d_k, T) → (B, T, T)

    Key concepts:
        - Dot product measures similarity (high value = similar vectors)
        - We transpose K to align dimensions for matrix multiplication
        - Result is a T×T matrix of all pairwise query-key similarities
        - Using @ operator is more readable than torch.matmul()
        - Using -2, -1 for transpose ensures it works with any batch dimensions
    """
    # Step 1: Transpose key from (B, T, d_k) to (B, d_k, T)
    # Why transpose the last two dimensions?
    # - Query is (B, T, d_k): T queries, each of dimension d_k
    # - Key is (B, T, d_k): T keys, each of dimension d_k
    # - We want to compute dot product between each query and each key
    # - Matrix multiplication (B, T, d_k) @ (B, d_k, T) gives us (B, T, T)
    # - Element [i, j] is the dot product of query i with key j
    key_transposed = key.transpose(-2, -1)  # Shape: (B, d_k, T)

    # Step 2: Matrix multiply query by transposed key
    # This computes all pairwise dot products in one operation (efficient!)
    scores = query @ key_transposed  # Shape: (B, T, T)

    # Educational note: Why use @ instead of torch.matmul()?
    # - @ is the Python matrix multiplication operator (PEP 465)
    # - More concise and readable
    # - Equivalent to torch.matmul() but cleaner
    # - Preferred in modern PyTorch code

    # Alternative implementation (same result):
    # scores = torch.matmul(query, key.transpose(-2, -1))
    # scores = torch.bmm(query, key.transpose(-2, -1))  # For 3D tensors only

    # Common mistake: Using transpose(0, 1) instead of transpose(-2, -1)
    # WRONG: key.transpose(0, 1) would swap batch and sequence dimensions!
    # CORRECT: key.transpose(-2, -1) always swaps last two dimensions

    # Verification (optional, for debugging):
    # assert scores.dim() == 3, f"Expected 3D tensor, got {scores.dim()}D"
    # assert scores.shape[0] == query.shape[0], "Batch dimension mismatch"
    # assert scores.shape[1] == query.shape[1], "Query seq_len mismatch"
    # assert scores.shape[2] == key.shape[1], "Key seq_len mismatch"

    return scores


# ==============================================================================
# EXERCISE 2 SOLUTION: Scaled Attention Scores
# ==============================================================================

def solution_02_scale_attention_scores(
    scores: torch.Tensor,
    d_k: int
) -> torch.Tensor:
    """
    Scale attention scores by sqrt(d_k) to prevent vanishing gradients.

    This is a CRITICAL step that prevents softmax saturation when d_k is large.

    Theory reference: theory.md, Section "The Scaling Factor"

    Args:
        scores: Raw attention scores of shape (batch_size, seq_len, seq_len)
        d_k: Dimension of the key/query vectors (used for scaling)

    Returns:
        scaled_scores: Scaled attention scores of same shape as input

    Why we scale by sqrt(d_k):
        1. Without scaling, dot products have variance = d_k
        2. Large d_k → large dot product magnitudes
        3. Large values → softmax saturates (outputs near 0 or 1)
        4. Saturated softmax → vanishing gradients (∂softmax/∂x ≈ 0)
        5. Vanishing gradients → model can't learn effectively

    Mathematical justification:
        - If Q[i] and K[j] have components ~ N(0, 1) (standard normal)
        - Then Q[i] · K[j] = Σ(Q[i,k] × K[j,k]) has variance = d_k
        - Dividing by sqrt(d_k) normalizes variance back to 1
        - Var(Q·K / sqrt(d_k)) = Var(Q·K) / d_k = d_k / d_k = 1

    Example impact:
        d_k = 64:  scores range [-25, 25] → scaled range [-3, 3]
        d_k = 512: scores range [-70, 70] → scaled range [-3, 3]

    This keeps scores in a range where softmax behaves well!
    """
    # Compute the scaling factor
    # We use math.sqrt instead of torch.sqrt because:
    # - d_k is a scalar integer, not a tensor
    # - math.sqrt is slightly more efficient for scalars
    # - No need to create a tensor for a constant value
    scale = math.sqrt(d_k)

    # Divide scores by the scaling factor
    # This is an element-wise operation (broadcasts the scalar)
    scaled_scores = scores / scale

    # Alternative implementations (all equivalent):
    # scaled_scores = scores / math.sqrt(d_k)  # More concise, inline
    # scaled_scores = scores * (1.0 / math.sqrt(d_k))  # Multiplication form
    # scaled_scores = scores / d_k**0.5  # Using exponentiation

    # Common mistake: Using d_k instead of sqrt(d_k)
    # WRONG: scores / d_k  (under-scales, still has issues)
    # CORRECT: scores / math.sqrt(d_k)

    # Educational note: Why not scale by something else?
    # - Scaling by d_k: Under-scales, still problematic
    # - Scaling by 1/d_k: Over-scales, makes scores too small
    # - Scaling by sqrt(d_k): Just right! (Goldilocks principle)

    # Verification (optional):
    # Original scores might have std ≈ sqrt(d_k)
    # Scaled scores should have std ≈ 1
    # print(f"Original std: {scores.std():.2f}, Scaled std: {scaled_scores.std():.2f}")

    return scaled_scores


# ==============================================================================
# EXERCISE 3 SOLUTION: Create Causal Mask
# ==============================================================================

def solution_03_create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a causal (lower-triangular) mask for autoregressive models.

    In autoregressive generation (like GPT), position i should only attend to
    positions 0 through i (past and present), NOT future positions (i+1, i+2, ...).

    Theory reference: theory.md, Section "Masking in Attention"

    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on (CPU or CUDA)

    Returns:
        mask: Causal mask of shape (seq_len, seq_len)
              - 0.0 for positions that CAN be attended to
              - -inf for positions that CANNOT be attended to

    Visual example (seq_len=4):
        Position:  0    1    2    3
              0 [  0,  -∞,  -∞,  -∞ ]  ← Position 0 can only see itself
              1 [  0,   0,  -∞,  -∞ ]  ← Position 1 can see 0, 1
              2 [  0,   0,   0,  -∞ ]  ← Position 2 can see 0, 1, 2
              3 [  0,   0,   0,   0 ]  ← Position 3 can see all (0, 1, 2, 3)

    Why -inf and not 0 for masking?
        - We ADD the mask to scores: scores + mask
        - masked_score = original_score + (-∞) = -∞
        - softmax(-∞) = exp(-∞) / (sum...) = 0 / (sum...) = 0
        - Result: Masked positions get 0 attention weight

    Why 0.0 for allowed positions?
        - allowed_score = original_score + 0 = original_score (unchanged)
        - Positions we want to attend to remain unaffected
    """
    # Method 1: Using torch.tril (Lower Triangular)
    # =============================================

    # Step 1: Create a lower triangular matrix of ones
    # torch.tril creates:
    #   [[1, 0, 0, 0],
    #    [1, 1, 0, 0],
    #    [1, 1, 1, 0],
    #    [1, 1, 1, 1]]
    # Where 1 = can attend, 0 = cannot attend
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # Shape: (T, T)

    # Step 2: Convert 0 → -inf (positions to block)
    # masked_fill creates a new tensor with specified values replaced
    # Condition: mask == 0 (upper triangle)
    # Replace with: float('-inf')
    mask = mask.masked_fill(mask == 0, float('-inf'))

    # Step 3: Convert 1 → 0.0 (positions to keep)
    # Now 1s are in the lower triangle
    # We want them to be 0.0 so they don't affect scores when added
    mask = mask.masked_fill(mask == 1, 0.0)

    # Result:
    #   [[0.0, -∞, -∞, -∞],
    #    [0.0, 0.0, -∞, -∞],
    #    [0.0, 0.0, 0.0, -∞],
    #    [0.0, 0.0, 0.0, 0.0]]

    return mask


def solution_03_create_causal_mask_alternative(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Alternative implementation using torch.triu (Upper Triangular).

    Same result, different approach. Some find this more intuitive.
    """
    # Method 2: Using torch.triu (Upper Triangular)
    # =============================================

    # Step 1: Create an upper triangular matrix (excluding diagonal)
    # torch.triu with diagonal=1 creates:
    #   [[0, 1, 1, 1],
    #    [0, 0, 1, 1],
    #    [0, 0, 0, 1],
    #    [0, 0, 0, 0]]
    # Where 1 = future positions (should block)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)

    # Step 2: Convert to -inf and 0.0
    # Where mask == 1 (upper triangle), set to -inf
    # Where mask == 0 (lower triangle + diagonal), set to 0.0
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, 0.0)

    return mask


def solution_03_create_causal_mask_compact(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Most compact implementation (one-liner).

    Uses boolean indexing instead of masked_fill.
    """
    # Create lower triangular matrix and convert to float
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Use boolean indexing: invert (1-mask) and multiply by -inf
    # Where mask=1: (1-1) * -inf = 0
    # Where mask=0: (1-0) * -inf = -inf
    mask = (1 - mask) * float('-inf')

    return mask


# Common mistakes and how to detect them:
#
# MISTAKE 1: Using 1 and 0 instead of 0 and -inf
# mask = torch.tril(torch.ones(seq_len, seq_len))  # Wrong!
# This doesn't prevent attention, it just adds 1 to lower triangle scores
#
# MISTAKE 2: Using torch.triu without diagonal=1
# mask = torch.triu(torch.ones(seq_len, seq_len))  # Blocks diagonal too!
# Position i wouldn't be able to attend to itself
#
# MISTAKE 3: Forgetting to specify device
# If inputs are on CUDA but mask is on CPU, you'll get a device mismatch error
# Always pass device parameter or use .to(device)
#
# How to verify your mask is correct:
# 1. Print it and visually inspect (should be lower triangular with -inf)
# 2. Check mask[0, 0] == 0 (can attend to self)
# 3. Check mask[0, 1] == float('-inf') (can't attend to future)
# 4. Check mask[seq_len-1, 0] == 0 (last position can attend to first)


# ==============================================================================
# EXERCISE 4 SOLUTION: Apply Mask to Scores
# ==============================================================================

def solution_04_apply_mask(
    scores: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply a mask to attention scores.

    This is deceptively simple but crucial for causal attention and padding.

    Theory reference: theory.md, Section "Masking in Attention"

    Args:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
        mask: Mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
              Contains 0 (keep) and -inf (mask out)

    Returns:
        masked_scores: Scores with mask applied, same shape as scores

    Why addition instead of multiplication?
        Consider what we want:
        - Masked position: score → -∞ (so softmax gives 0)
        - Kept position: score → unchanged

        Addition achieves this:
        - masked_score = score + (-∞) = -∞ ✓
        - kept_score = score + 0 = score ✓

        Multiplication would NOT work:
        - masked_score = score × 0 = 0 (not -∞, softmax wouldn't give 0)
        - kept_score = score × 1 = score ✓

    Broadcasting behavior:
        PyTorch automatically broadcasts mask to match scores:
        - scores: (B, T, T)
        - mask: (T, T) → broadcasts to (B, T, T)
        - mask: (B, T, T) → exact match, no broadcasting

        This allows one mask for all batch elements (efficient!)
    """
    # Simply add the mask to scores
    # PyTorch handles broadcasting automatically
    masked_scores = scores + mask

    # That's it! The simplicity is beautiful.
    # All the complexity is in creating the right mask.

    # What's happening under the hood (broadcasting):
    # If scores is (B, T, T) and mask is (T, T):
    # 1. PyTorch expands mask to (1, T, T)
    # 2. Then broadcasts across batch dimension to (B, T, T)
    # 3. Performs element-wise addition
    #
    # Example:
    #   scores[0, 2, 3] = 5.2
    #   mask[2, 3] = -inf
    #   masked_scores[0, 2, 3] = 5.2 + (-inf) = -inf

    # Alternative (explicit broadcasting):
    # masked_scores = scores + mask.unsqueeze(0)  # Explicitly add batch dim
    # This is unnecessary since PyTorch does it automatically

    # Common mistake: Using multiplication
    # WRONG: masked_scores = scores * mask_of_ones_and_zeros
    # This sets masked positions to 0, not -inf
    # After softmax, 0 doesn't become 0 probability!
    # softmax([0, 1, 2]) = [0.09, 0.24, 0.67], not [0, 0.27, 0.73]

    # Educational note: In-place operations
    # scores += mask  # In-place, modifies scores
    # scores + mask   # Creates new tensor, preserves scores
    # Use in-place only if you're sure you don't need original scores
    # (e.g., for backward pass or debugging)

    # Verification (optional):
    # Check that masked positions are -inf
    # if mask.dim() == 2:
    #     for i in range(scores.shape[0]):
    #         assert torch.isinf(masked_scores[i, 0, 1]), "Masking failed"

    return masked_scores


# ==============================================================================
# EXERCISE 5 SOLUTION: Compute Attention Weights with Softmax
# ==============================================================================

def solution_05_compute_attention_weights(
    scores: torch.Tensor
) -> torch.Tensor:
    """
    Convert attention scores to probabilities using softmax.

    Softmax normalizes each row to sum to 1, giving us a probability distribution.

    Theory reference: theory.md, Section "Mathematical Formulation", Step 4

    Args:
        scores: Attention scores of shape (batch_size, seq_len, seq_len)
                Can contain -inf values from masking

    Returns:
        attention_weights: Probabilities of shape (batch_size, seq_len, seq_len)
                          Each row sums to 1.0

    Softmax formula:
        softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)

    Key properties:
        1. Output is always in [0, 1]
        2. Σⱼ softmax(x_j) = 1 (probability distribution)
        3. softmax is differentiable (enables learning)
        4. softmax(-∞) = 0 (this is why we use -inf for masking!)

    Why softmax over the LAST dimension (dim=-1)?
        scores[i, j] = relevance of position i to position j
        We want: for each query position i, create a distribution over all keys j
        softmax(scores[i, :]) gives us P(attend to key j | query i)

        If we used dim=-2 instead:
        - We'd normalize over queries instead of keys
        - Doesn't make sense: "probability of different queries given a key"
    """
    # Apply softmax over the last dimension (key dimension)
    # F.softmax is numerically stable (uses log-sum-exp trick internally)
    attention_weights = F.softmax(scores, dim=-1)

    # Shape: (B, T, T), same as input
    # For each batch b and query position i:
    #   attention_weights[b, i, :] is a probability distribution over keys
    #   sum(attention_weights[b, i, :]) = 1.0

    # What happens to -inf values from masking?
    # exp(-∞) = 0
    # So masked positions contribute 0 to both numerator and denominator
    # Result: They get probability 0 (exactly what we want!)

    # Example:
    #   scores = [2.0, 5.0, -∞, 1.0]
    #   exp(scores) = [e², e⁵, 0, e¹]
    #   sum = e² + e⁵ + 0 + e¹
    #   softmax = [e²/sum, e⁵/sum, 0/sum, e¹/sum]
    #             ≈ [0.05, 0.93, 0.00, 0.02]  (sums to 1.0)

    # Common mistake: Using softmax on wrong dimension
    # WRONG: F.softmax(scores, dim=-2)  # Normalizes over queries!
    # This would give: column sums = 1 instead of row sums = 1
    #
    # How to verify:
    # row_sums = attention_weights.sum(dim=-1)  # Should be all 1.0
    # col_sums = attention_weights.sum(dim=-2)  # Won't be 1.0 (expected)

    # Educational note: Numerical stability
    # PyTorch's F.softmax uses the log-sum-exp trick:
    #   softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    # This prevents overflow when x has large values
    # Example:
    #   Naive: exp(1000) = overflow!
    #   Stable: exp(1000 - 1000) = exp(0) = 1 ✓

    # Alternative (manual implementation, for understanding):
    # scores_shifted = scores - scores.max(dim=-1, keepdim=True)[0]
    # exp_scores = torch.exp(scores_shifted)
    # attention_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    # But use F.softmax in practice (faster and more reliable)

    return attention_weights


# ==============================================================================
# EXERCISE 6 SOLUTION: Apply Attention to Values
# ==============================================================================

def solution_06_apply_attention_to_values(
    attention_weights: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Apply attention weights to values to get the final output.

    This is the final step: weighted aggregation of value vectors.

    Theory reference: theory.md, Section "Mathematical Formulation", Step 5

    Args:
        attention_weights: Attention probabilities of shape (batch_size, seq_len, seq_len)
        value: Value tensor of shape (batch_size, seq_len, d_v)

    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)

    What's happening:
        For each query position i, we compute:
        output[i] = Σⱼ attention_weights[i,j] × value[j]

        This is a weighted average of all value vectors,
        where weights come from the attention distribution.

    Intuition:
        - High attention weight on position j → output strongly influenced by value[j]
        - Low attention weight on position j → value[j] barely affects output
        - Masked positions (weight=0) → don't contribute at all

    Example (simplified, d_v=2):
        attention_weights[2] = [0.1, 0.3, 0.6, 0.0]
        values = [[1, 2],    # Position 0
                  [3, 4],    # Position 1
                  [5, 6],    # Position 2
                  [7, 8]]   # Position 3 (masked)

        output[2] = 0.1×[1,2] + 0.3×[3,4] + 0.6×[5,6] + 0.0×[7,8]
                  = [0.1, 0.2] + [0.9, 1.2] + [3.0, 3.6] + [0, 0]
                  = [4.0, 5.0]

        The output is dominated by position 2 (weight=0.6)
    """
    # Matrix multiply attention weights by values
    # Shape: (B, T, T) @ (B, T, d_v) → (B, T, d_v)
    output = attention_weights @ value

    # Breaking down the matrix multiplication:
    # For batch b, query position i:
    #   output[b, i, :] = Σⱼ attention_weights[b, i, j] × value[b, j, :]
    #
    # This computes a weighted sum of all value vectors for each query position

    # Why does this shape work?
    # attention_weights: (B, T_q, T_k) - T_q queries, T_k keys
    # value: (B, T_k, d_v) - T_k value vectors, each d_v dimensional
    # Result: (B, T_q, d_v) - T_q output vectors, each d_v dimensional
    #
    # The middle dimension (T_k) is summed over (weighted aggregation)

    # Educational note: This is a batched matrix multiplication
    # PyTorch handles batch dimension automatically
    # Equivalent to:
    # output = torch.zeros(B, T_q, d_v)
    # for b in range(B):
    #     output[b] = attention_weights[b] @ value[b]
    # But vectorized version is much faster!

    # Common mistake: Swapping the order
    # WRONG: value @ attention_weights
    # Shape: (B, T, d_v) @ (B, T, T) → (B, d_v, T)
    # This gives wrong shape and wrong semantics!

    # Verification (optional):
    # assert output.shape == value.shape, "Output should match value shape"
    # assert output.shape == (B, T_q, d_v)

    # Alternative (explicit loop, for understanding):
    # output = torch.zeros_like(value)  # (B, T, d_v)
    # for i in range(T):
    #     output[:, i, :] = (attention_weights[:, i, :].unsqueeze(-1) * value).sum(dim=1)
    # But use matrix multiplication in practice (much faster!)

    return output


# ==============================================================================
# EXERCISE 7 SOLUTION: Full Attention Implementation
# ==============================================================================

def solution_07_full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete scaled dot-product attention implementation.

    Combines all previous exercises into one cohesive function.
    This is the canonical attention mechanism from "Attention Is All You Need".

    Theory reference: theory.md, Section "Scaled Dot-Product Attention"

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)
        value: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask of shape (seq_len, seq_len) with 0 and -inf

    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention probabilities of shape (batch_size, seq_len, seq_len)

    Full pipeline:
        1. Compute similarity: QK^T
        2. Scale: / sqrt(d_k)
        3. Mask (optional): + mask
        4. Normalize: softmax
        5. Aggregate: × V

    This is production-quality code matching tiny_transformer/attention.py
    """
    # Step 1: Get d_k for scaling
    # We get this from the last dimension of query (or key, they're the same)
    # query.size(-1) is more robust than query.shape[-1] for edge cases
    d_k = query.size(-1)  # Scalar: dimension of keys/queries

    # Step 2: Compute attention scores (similarity matrix)
    # Formula: scores = Q @ K^T
    # Shape: (B, T, d_k) @ (B, d_k, T) → (B, T, T)
    scores = query @ key.transpose(-2, -1)

    # At this point, scores[b, i, j] = dot product of query i with key j
    # High value = high similarity = should attend more
    # But values can be very large if d_k is large...

    # Step 3: Scale scores by sqrt(d_k)
    # Why? See theory.md section "The Scaling Factor"
    # Without this, large d_k causes softmax saturation and vanishing gradients
    scores = scores / math.sqrt(d_k)

    # Now scores have normalized variance (~1.0) regardless of d_k
    # This keeps them in a range where softmax behaves well

    # Step 4: Apply mask if provided
    # Mask contains 0.0 (keep) and -inf (block)
    # Adding -inf makes those positions have probability 0 after softmax
    if mask is not None:
        scores = scores + mask  # Broadcasting handles shape alignment

    # Masked positions now have -inf, which will become 0 after softmax

    # Step 5: Compute attention weights with softmax
    # Converts scores to probabilities: each row sums to 1.0
    # softmax(-inf) = 0, so masked positions get 0 weight
    attention_weights = F.softmax(scores, dim=-1)  # Shape: (B, T, T)

    # Step 6: Apply attention weights to values (weighted aggregation)
    # This creates a weighted mixture of value vectors
    # Shape: (B, T, T) @ (B, T, d_v) → (B, T, d_v)
    output = attention_weights @ value

    # Step 7: Return both output and weights
    # Weights are useful for visualization and debugging
    return output, attention_weights

    # Educational notes:
    #
    # 1. This function is stateless and purely functional
    #    - No learnable parameters (those are in the projections Q, K, V)
    #    - Easy to test and reason about
    #
    # 2. All operations are differentiable
    #    - Gradients flow back through all steps
    #    - PyTorch autograd handles everything automatically
    #
    # 3. Memory complexity: O(B × T²)
    #    - The attention weight matrix (B, T, T) is the bottleneck
    #    - For T=1024, B=32: ~128MB just for attention weights
    #
    # 4. Time complexity: O(B × T² × d_k)
    #    - Matrix multiplications dominate
    #    - Quadratic in sequence length (scaling challenge!)
    #
    # 5. Numerical stability
    #    - Scaling prevents overflow in scores
    #    - F.softmax uses log-sum-exp trick internally
    #    - -inf is handled correctly by softmax
    #
    # 6. This is used in every transformer layer
    #    - GPT, BERT, T5, etc. all use this exact mechanism
    #    - Multi-head attention runs this multiple times in parallel


# ==============================================================================
# EXERCISE 8 SOLUTION: Debug Broken Attention
# ==============================================================================

def solution_08_debug_attention_fixed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FIXED version with all bugs corrected.

    Original bugs:
        1. d_k = query.size(0) → Should be query.size(-1)
        2. key.transpose(0, 1) → Should be key.transpose(-2, -1)
        3. F.softmax(..., dim=0) → Should be F.softmax(..., dim=-1)

    Let's explain each bug in detail:
    """
    # BUG 1 FIXED: Get d_k from the correct dimension
    # ================================================
    # WRONG: d_k = query.size(0)
    # - query.size(0) is the BATCH SIZE, not the key dimension!
    # - If query shape is (32, 128, 64), this would give d_k=32
    # - But we want d_k=64 (the feature dimension)
    #
    # CORRECT: d_k = query.size(-1)
    # - query.size(-1) gets the LAST dimension (feature dimension)
    # - This is the actual dimension of keys/queries
    # - Using -1 is safer than hardcoding the index
    #
    # Why this bug causes problems:
    # - Scaling by wrong value (sqrt(32) instead of sqrt(64))
    # - Scores won't be properly normalized
    # - Model won't learn effectively
    d_k = query.size(-1)  # FIX 1: Last dimension, not first!

    # BUG 2 FIXED: Transpose the correct dimensions
    # ==============================================
    # WRONG: scores = query @ key.transpose(0, 1)
    # - transpose(0, 1) swaps BATCH and SEQUENCE dimensions
    # - If key is (B, T, d_k), this gives (T, B, d_k)
    # - Then (B, T, d_k) @ (T, B, d_k) fails or gives wrong result!
    #
    # CORRECT: scores = query @ key.transpose(-2, -1)
    # - transpose(-2, -1) swaps the LAST TWO dimensions
    # - If key is (B, T, d_k), this gives (B, d_k, T)
    # - Then (B, T, d_k) @ (B, d_k, T) → (B, T, T) ✓
    #
    # Why this bug causes problems:
    # - Shape mismatch errors (might crash)
    # - If it doesn't crash, gives completely wrong attention scores
    # - Each position would attend to wrong positions
    #
    # How to detect:
    # - Print shapes: print(f"scores shape: {scores.shape}")
    # - Should be (B, T, T), not some other shape
    scores = query @ key.transpose(-2, -1)  # FIX 2: Last two dims!

    # Scale scores (this was correct in original)
    scores = scores / math.sqrt(d_k)

    # BUG 3 FIXED: Softmax over the correct dimension
    # ================================================
    # WRONG: attention_weights = F.softmax(scores, dim=0)
    # - dim=0 normalizes over the BATCH dimension
    # - This means: sum over all batch elements = 1
    # - Doesn't make sense: we want each row to sum to 1, not each column!
    #
    # CORRECT: attention_weights = F.softmax(scores, dim=-1)
    # - dim=-1 normalizes over the LAST dimension (keys)
    # - This means: for each query, probabilities over keys sum to 1
    # - This is what we want: attention distribution for each position
    #
    # Why this bug causes problems:
    # - Attention weights don't form valid probability distributions
    # - Row sums won't be 1.0 (they'll be random values)
    # - Model can't interpret these as "how much to attend"
    # - Training will fail or be extremely unstable
    #
    # How to detect:
    # - Check row sums: attention_weights.sum(dim=-1)
    # - Should be all 1.0
    # - If not, softmax is on wrong dimension
    attention_weights = F.softmax(scores, dim=-1)  # FIX 3: Last dim!

    # Apply attention to values (this was correct in original)
    output = attention_weights @ value

    return output, attention_weights


def demonstrate_bugs():
    """
    Demonstrate what each bug causes and how to detect it.

    This is educational code showing debugging techniques.
    """
    print("=" * 70)
    print("DEBUGGING ATTENTION: Finding and Fixing Bugs")
    print("=" * 70)

    # Create test inputs
    B, T, d_k = 2, 4, 8
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    print(f"\nInput shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")

    # BUG 1 DEMONSTRATION: Wrong d_k
    print("\n" + "-" * 70)
    print("BUG 1: Using batch size instead of feature dimension")
    print("-" * 70)
    wrong_d_k = Q.size(0)  # Bug: batch size (2)
    correct_d_k = Q.size(-1)  # Correct: feature dim (8)
    print(f"Wrong d_k: {wrong_d_k} (batch size)")
    print(f"Correct d_k: {correct_d_k} (feature dimension)")
    print(f"Wrong scaling factor: {math.sqrt(wrong_d_k):.3f}")
    print(f"Correct scaling factor: {math.sqrt(correct_d_k):.3f}")
    print("Impact: Scores are scaled incorrectly → unstable training")

    # BUG 2 DEMONSTRATION: Wrong transpose
    print("\n" + "-" * 70)
    print("BUG 2: Transposing wrong dimensions")
    print("-" * 70)
    try:
        wrong_transpose = K.transpose(0, 1)  # Bug: swaps batch and seq
        print(f"K shape: {K.shape}")
        print(f"Wrong transpose: {wrong_transpose.shape} (swapped batch and seq!)")
        print("This breaks matrix multiplication or gives wrong result")
    except Exception as e:
        print(f"Error: {e}")

    correct_transpose = K.transpose(-2, -1)  # Correct
    print(f"Correct transpose: {correct_transpose.shape}")
    scores_correct = Q @ correct_transpose
    print(f"Correct scores shape: {scores_correct.shape} (B, T, T) ✓")

    # BUG 3 DEMONSTRATION: Wrong softmax dimension
    print("\n" + "-" * 70)
    print("BUG 3: Softmax on wrong dimension")
    print("-" * 70)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    wrong_softmax = F.softmax(scores, dim=0)  # Bug: over batch
    correct_softmax = F.softmax(scores, dim=-1)  # Correct: over keys

    print("Wrong softmax (dim=0):")
    print(f"  Shape: {wrong_softmax.shape}")
    print(f"  Row sums: {wrong_softmax[0].sum(dim=-1)}")  # Not 1.0!
    print(f"  Column sums (over batch): {wrong_softmax[:, 0, 0].sum()}")  # 1.0 (wrong!)

    print("\nCorrect softmax (dim=-1):")
    print(f"  Shape: {correct_softmax.shape}")
    print(f"  Row sums: {correct_softmax[0].sum(dim=-1)}")  # All 1.0 ✓
    print(f"  Column sums (over batch): {correct_softmax[:, 0, 0].sum()}")  # Not 1.0 (expected)

    print("\n" + "=" * 70)
    print("DEBUGGING TIPS:")
    print("=" * 70)
    print("1. Always print shapes at each step")
    print("2. Check that attention weights sum to 1.0 along the right dimension")
    print("3. Use -1, -2 for dimensions instead of hardcoded indices")
    print("4. Verify d_k is the feature dimension, not batch size")
    print("5. Visualize attention weights to spot patterns")


# ==============================================================================
# EXERCISE 9 SOLUTION: Shape Manipulation for Multi-Head Attention
# ==============================================================================

def solution_09_reshape_for_multi_head(
    tensor: torch.Tensor,
    num_heads: int
) -> torch.Tensor:
    """
    Reshape a tensor for multi-head attention.

    Multi-head attention splits d_model into num_heads parallel attention operations.
    This requires careful reshaping and dimension reordering.

    Theory context: Multi-head attention (covered in next module)
    Each head operates on d_k = d_model // num_heads dimensions

    Args:
        tensor: Input of shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        reshaped: Tensor of shape (batch_size, num_heads, seq_len, d_k)
                  where d_k = d_model // num_heads

    Visual example (B=2, T=4, d_model=64, num_heads=8):
        Input:  (2, 4, 64)
        Step 1: (2, 4, 8, 8)   ← Reshape: split d_model into (num_heads, d_k)
        Step 2: (2, 8, 4, 8)   ← Transpose: move num_heads before seq_len
        Output: (2, 8, 4, 8)   ← Ready for parallel attention!

    Why this shape?
        - Batch dimension first: efficient batching
        - num_heads second: separate attention operations
        - seq_len third: sequence positions
        - d_k last: feature dimension for each head

    This allows computing attention separately for each head in parallel.
    """
    # Extract dimensions from input
    batch_size, seq_len, d_model = tensor.shape

    # Calculate dimension per head
    # Each head gets d_model // num_heads features
    # Example: d_model=512, num_heads=8 → d_k=64
    d_k = d_model // num_heads

    # Defensive check: ensure d_model is divisible by num_heads
    # Without this, we might silently lose information
    assert d_model % num_heads == 0, \
        f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

    # Step 1: Reshape from (B, T, d_model) to (B, T, num_heads, d_k)
    # This splits the feature dimension into num_heads groups
    #
    # Visual example (B=1, T=2, d_model=4, num_heads=2):
    #   Before: [[1, 2, 3, 4],      (T=2, d_model=4)
    #            [5, 6, 7, 8]]
    #
    #   After:  [[[1, 2], [3, 4]],  (T=2, num_heads=2, d_k=2)
    #            [[5, 6], [7, 8]]]
    #
    # The view() operation doesn't move data, just changes how we interpret it
    reshaped = tensor.view(batch_size, seq_len, num_heads, d_k)

    # Step 2: Transpose to (B, num_heads, T, d_k)
    # We want num_heads dimension before seq_len
    # This groups all positions for each head together
    #
    # Why? So we can process each head independently:
    # - Head 0 operates on [:, 0, :, :] (all sequences, all positions)
    # - Head 1 operates on [:, 1, :, :] (all sequences, all positions)
    # - etc.
    #
    # transpose(1, 2) swaps dimensions 1 and 2:
    # - Dimension 0: batch_size (unchanged)
    # - Dimension 1: seq_len → num_heads
    # - Dimension 2: num_heads → seq_len
    # - Dimension 3: d_k (unchanged)
    reshaped = reshaped.transpose(1, 2)

    # Final shape: (B, num_heads, T, d_k)
    # Now each head sees (B, T, d_k) which is perfect for attention!

    # Alternative implementation using permute:
    # reshaped = tensor.view(batch_size, seq_len, num_heads, d_k)
    # reshaped = reshaped.permute(0, 2, 1, 3)
    # permute is more flexible (can reorder any dimensions)
    # transpose only swaps two dimensions (simpler, clearer for this case)

    # Educational note: view() vs reshape()
    # - view() requires contiguous memory, fails otherwise
    # - reshape() works always (copies if needed)
    # Use view() when you know tensor is contiguous (more explicit)
    # Use reshape() when you're unsure (safer, slightly slower)

    # Verification (optional but recommended in practice):
    # assert reshaped.shape == (batch_size, num_heads, seq_len, d_k), \
    #     f"Expected {(batch_size, num_heads, seq_len, d_k)}, got {reshaped.shape}"

    # Check that we didn't lose or duplicate data
    # Reshape should preserve all elements, just reorder them
    # reshaped.reshape(batch_size, seq_len, d_model) should equal original
    # (element-wise, though order might differ due to transpose)

    return reshaped


def solution_09_inverse_reshape(
    tensor: torch.Tensor
) -> torch.Tensor:
    """
    Inverse operation: reshape back from multi-head to single tensor.

    This is used after multi-head attention to combine all heads.

    Args:
        tensor: Input of shape (batch_size, num_heads, seq_len, d_k)

    Returns:
        reshaped: Tensor of shape (batch_size, seq_len, d_model)
                  where d_model = num_heads × d_k
    """
    batch_size, num_heads, seq_len, d_k = tensor.shape
    d_model = num_heads * d_k

    # Step 1: Transpose (B, num_heads, T, d_k) → (B, T, num_heads, d_k)
    # Move num_heads after seq_len
    tensor = tensor.transpose(1, 2)

    # Step 2: Reshape (B, T, num_heads, d_k) → (B, T, d_model)
    # Merge num_heads and d_k back into single dimension
    tensor = tensor.contiguous().view(batch_size, seq_len, d_model)

    # Note: .contiguous() is needed because transpose creates a non-contiguous tensor
    # view() requires contiguous memory, so we must call .contiguous() first
    # Alternative: use reshape() which handles this automatically
    # tensor = tensor.reshape(batch_size, seq_len, d_model)

    return tensor


# ==============================================================================
# EXERCISE 10 SOLUTION: Padding Mask
# ==============================================================================

def solution_10_create_padding_mask(
    sequence_lengths: torch.Tensor,
    max_len: int
) -> torch.Tensor:
    """
    Create a padding mask for variable-length sequences.

    In batched processing, sequences have different lengths and are padded.
    We need to prevent attention to padding positions.

    Theory reference: theory.md, Section "Masking in Attention" (Padding subsection)

    Args:
        sequence_lengths: Tensor of shape (batch_size,) with actual length of each sequence
        max_len: Maximum sequence length (length after padding)

    Returns:
        mask: Padding mask of shape (batch_size, 1, 1, max_len)
              0.0 for real tokens, -inf for padding tokens
              Shape is (B, 1, 1, T) to broadcast with attention scores (B, H, T, T)

    Example:
        sequence_lengths = [3, 2, 4]  # 3 sequences in batch
        max_len = 4

        Sequence 0: [tok, tok, tok, PAD]  → mask: [0, 0, 0, -∞]
        Sequence 1: [tok, tok, PAD, PAD]  → mask: [0, 0, -∞, -∞]
        Sequence 2: [tok, tok, tok, tok]  → mask: [0, 0, 0, 0]

    Broadcasting behavior:
        Mask shape: (B, 1, 1, T)
        Scores shape: (B, H, T, T)  (H = num_heads)
        When added: mask broadcasts to (B, H, T, T)
        Each head uses the same padding mask (makes sense!)
    """
    # Get batch size from sequence_lengths
    batch_size = sequence_lengths.shape[0]

    # Step 1: Create a range tensor [0, 1, 2, ..., max_len-1]
    # This represents all possible positions in the sequence
    # Shape: (max_len,)
    positions = torch.arange(max_len, device=sequence_lengths.device)
    # We use sequence_lengths.device to ensure mask is on same device (CPU/CUDA)

    # Step 2: Compare positions with sequence lengths to identify padding
    # We want: position >= length → is padding
    #
    # Broadcasting comparison:
    #   positions:        (max_len,)     → (1, max_len)
    #   sequence_lengths: (batch_size,)  → (batch_size, 1)
    #   Result:           (batch_size, max_len)
    #
    # Example:
    #   lengths = [3, 2, 4], max_len = 4
    #   positions = [0, 1, 2, 3]
    #
    #   positions.unsqueeze(0) = [[0, 1, 2, 3]]  (1, 4)
    #   lengths.unsqueeze(1) = [[3],             (3, 1)
    #                           [2],
    #                           [4]]
    #
    #   Comparison (positions >= lengths):
    #     [[0>=3, 1>=3, 2>=3, 3>=3],  → [F, F, F, T]
    #      [0>=2, 1>=2, 2>=2, 3>=2],  → [F, F, T, T]
    #      [0>=4, 1>=4, 2>=4, 3>=4]]  → [F, F, F, F]
    #
    # True = is padding, False = is real token
    padding_mask = positions.unsqueeze(0) >= sequence_lengths.unsqueeze(1)
    # Shape: (batch_size, max_len)

    # Step 3: Convert boolean mask to attention mask format
    # Create tensor of zeros (will be 0.0 for real tokens)
    mask = torch.zeros(batch_size, max_len, device=sequence_lengths.device)

    # Set padding positions to -inf
    # masked_fill: where padding_mask is True, set value to -inf
    mask = mask.masked_fill(padding_mask, float('-inf'))
    # Shape: (batch_size, max_len)
    # Real tokens: 0.0
    # Padding tokens: -inf

    # Step 4: Reshape for broadcasting with attention scores
    # Attention scores have shape (B, H, T, T) where H = num_heads
    # We want mask to broadcast across heads and queries
    #
    # Current shape: (B, T)
    # Target shape: (B, 1, 1, T)
    #   - Dimension 1 (1): broadcasts across num_heads
    #   - Dimension 2 (1): broadcasts across query positions
    #   - Dimension 3 (T): matches key positions
    #
    # When added to scores (B, H, T_q, T_k):
    #   - Each head sees same padding mask ✓
    #   - Each query sees same padding mask ✓
    #   - Padding keys are masked for all queries ✓
    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, max_len) → (B, 1, 1, max_len)

    # Alternative reshaping (equivalent):
    # mask = mask.view(batch_size, 1, 1, max_len)
    # mask = mask[:, None, None, :]
    # All produce the same result

    return mask


def solution_10_create_padding_mask_compact(
    sequence_lengths: torch.Tensor,
    max_len: int
) -> torch.Tensor:
    """
    More compact implementation of padding mask (same result).

    This version chains operations for brevity.
    """
    batch_size = sequence_lengths.shape[0]
    positions = torch.arange(max_len, device=sequence_lengths.device)

    # Chain all operations
    mask = (
        (positions.unsqueeze(0) >= sequence_lengths.unsqueeze(1))  # Boolean mask
        .float()  # Convert to float (True→1.0, False→0.0)
        * float('-inf')  # Multiply: 1.0→-inf, 0.0→0.0
        .unsqueeze(1)  # Add head dimension
        .unsqueeze(2)  # Add query dimension
    )

    return mask


def demonstrate_padding_mask():
    """
    Demonstrate padding mask creation and usage.

    Educational code showing how padding masks work in practice.
    """
    print("\n" + "=" * 70)
    print("PADDING MASK DEMONSTRATION")
    print("=" * 70)

    # Create example batch with different sequence lengths
    lengths = torch.tensor([3, 2, 4])
    max_len = 4
    batch_size = 3

    print(f"\nBatch size: {batch_size}")
    print(f"Max length: {max_len}")
    print(f"Sequence lengths: {lengths.tolist()}")

    # Create padding mask
    mask = solution_10_create_padding_mask(lengths, max_len)

    print(f"\nMask shape: {mask.shape}")
    print("Mask values (B=3, 1, 1, T=4):")
    for b in range(batch_size):
        print(f"  Sequence {b} (length {lengths[b]}): {mask[b, 0, 0].tolist()}")

    # Show what happens when applied to attention scores
    print("\nExample: Applying mask to attention scores")
    scores = torch.randn(batch_size, 1, max_len, max_len)  # (B, 1 head, T, T)
    print(f"Scores shape: {scores.shape}")

    masked_scores = scores + mask
    print(f"Masked scores shape: {masked_scores.shape}")

    print("\nSequence 0 scores (length=3):")
    print("  Before masking:")
    print(f"    {scores[0, 0, 0, :]}")
    print("  After masking (position 3 should be -inf):")
    print(f"    {masked_scores[0, 0, 0, :]}")

    # Apply softmax to show padding positions get 0 weight
    attention = F.softmax(masked_scores, dim=-1)
    print("\nAfter softmax (padding positions should be 0.0):")
    print(f"  Sequence 0: {attention[0, 0, 0, :]}")
    print(f"  Sum: {attention[0, 0, 0, :].sum():.4f} (should be 1.0)")


# ==============================================================================
# BONUS SOLUTION: Optimized Attention
# ==============================================================================

def solution_bonus_optimized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_flash: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Optimized attention implementation with Flash Attention support.

    PyTorch 2.0+ includes torch.nn.functional.scaled_dot_product_attention,
    which uses highly optimized kernels (including Flash Attention on capable GPUs).

    Benefits of Flash Attention:
        - 2-4× faster than naive implementation
        - Uses less memory (doesn't materialize full attention matrix)
        - Same mathematical result, just more efficient

    Args:
        query, key, value: Standard attention inputs
        mask: Optional attention mask
        use_flash: Whether to use optimized kernels if available

    Returns:
        output: Attention output
        attention_weights: Attention probabilities (None if using Flash Attention)

    Note: Flash Attention doesn't return attention weights (for efficiency)
    """
    # Check if PyTorch has the optimized function (PyTorch 2.0+)
    has_sdpa = hasattr(F, 'scaled_dot_product_attention')

    if use_flash and has_sdpa:
        # Use PyTorch's optimized implementation
        # This may use Flash Attention on capable hardware
        # Note: mask format is different (boolean True = keep, False = mask)
        # We need to convert our mask format (-inf/0) to boolean

        attn_mask = None
        if mask is not None:
            # Convert from (-inf, 0) format to boolean format
            # -inf → False (mask out)
            # 0 → True (keep)
            attn_mask = (mask == 0)

        # Call optimized function
        # Returns only output, not attention weights (for efficiency)
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is None)  # Use causal mask if no custom mask
        )

        return output, None  # No attention weights with Flash Attention

    else:
        # Fall back to our implementation
        # Use solution_07_full_attention for reference implementation
        return solution_07_full_attention(query, key, value, mask)


# ==============================================================================
# COMPREHENSIVE TEST SUITE
# ==============================================================================

def test_all_solutions():
    """
    Comprehensive test suite for all solutions.

    Tests correctness, shape handling, edge cases, and numerical properties.
    """
    print("\n" + "=" * 70)
    print("TESTING ALL SOLUTIONS")
    print("=" * 70)

    # Test parameters
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8

    # Test Exercise 1: Compute Attention Scores
    print("\n" + "-" * 70)
    print("Exercise 1: Compute Attention Scores")
    print("-" * 70)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    scores = solution_01_compute_attention_scores(Q, K)
    assert scores.shape == (batch_size, seq_len, seq_len), \
        f"Expected {(batch_size, seq_len, seq_len)}, got {scores.shape}"
    print(f"✓ Shape correct: {scores.shape}")
    print(f"✓ Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # Test Exercise 2: Scale Attention Scores
    print("\n" + "-" * 70)
    print("Exercise 2: Scale Attention Scores")
    print("-" * 70)
    scaled = solution_02_scale_attention_scores(scores, d_k)
    assert scaled.shape == scores.shape, "Shape should not change"
    assert scaled.std() < scores.std(), "Scaling should reduce variance"
    print(f"✓ Shape correct: {scaled.shape}")
    print(f"✓ Original std: {scores.std():.3f}")
    print(f"✓ Scaled std: {scaled.std():.3f}")

    # Test Exercise 3: Create Causal Mask
    print("\n" + "-" * 70)
    print("Exercise 3: Create Causal Mask")
    print("-" * 70)
    mask = solution_03_create_causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len), f"Expected {(seq_len, seq_len)}, got {mask.shape}"
    assert mask[0, 0] == 0, "Diagonal should be 0"
    assert mask[0, 1] == float('-inf'), "Upper triangle should be -inf"
    assert mask[seq_len-1, 0] == 0, "Lower triangle should be 0"
    print(f"✓ Shape correct: {mask.shape}")
    print(f"✓ Diagonal: {mask[0, 0]}")
    print(f"✓ Upper triangle: {mask[0, 1]}")
    print("✓ Lower triangular structure verified")

    # Test Exercise 4: Apply Mask
    print("\n" + "-" * 70)
    print("Exercise 4: Apply Mask")
    print("-" * 70)
    masked = solution_04_apply_mask(scaled, mask)
    assert masked.shape == scaled.shape, "Shape should not change"
    assert torch.isinf(masked[0, 0, 1]), "Masked positions should be -inf"
    print(f"✓ Shape correct: {masked.shape}")
    print(f"✓ Masking applied correctly")

    # Test Exercise 5: Compute Attention Weights
    print("\n" + "-" * 70)
    print("Exercise 5: Compute Attention Weights")
    print("-" * 70)
    weights = solution_05_compute_attention_weights(masked)
    row_sums = weights.sum(dim=-1)
    assert weights.shape == masked.shape, "Shape should not change"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Rows should sum to 1"
    assert (weights >= 0).all(), "Weights should be non-negative"
    assert (weights <= 1).all(), "Weights should be <= 1"
    print(f"✓ Shape correct: {weights.shape}")
    print(f"✓ Row sums: {row_sums[0]}")
    print(f"✓ Range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Test Exercise 6: Apply Attention to Values
    print("\n" + "-" * 70)
    print("Exercise 6: Apply Attention to Values")
    print("-" * 70)
    V = torch.randn(batch_size, seq_len, d_v)
    output = solution_06_apply_attention_to_values(weights, V)
    assert output.shape == V.shape, f"Expected {V.shape}, got {output.shape}"
    print(f"✓ Shape correct: {output.shape}")

    # Test Exercise 7: Full Attention
    print("\n" + "-" * 70)
    print("Exercise 7: Full Attention")
    print("-" * 70)
    output_full, weights_full = solution_07_full_attention(Q, K, V, mask)
    assert output_full.shape == V.shape, "Output shape should match value shape"
    assert weights_full.shape == (batch_size, seq_len, seq_len), \
        "Weights shape incorrect"
    row_sums = weights_full.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Rows should sum to 1"
    print(f"✓ Output shape: {output_full.shape}")
    print(f"✓ Weights shape: {weights_full.shape}")
    print(f"✓ Row sums verified")

    # Test Exercise 8: Debug Attention
    print("\n" + "-" * 70)
    print("Exercise 8: Debug Attention (Fixed)")
    print("-" * 70)
    output_debug, weights_debug = solution_08_debug_attention_fixed(Q, K, V)
    assert output_debug.shape == V.shape, "Output shape should match value shape"
    assert weights_debug.shape == (batch_size, seq_len, seq_len), \
        "Weights shape incorrect"
    row_sums = weights_debug.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Rows should sum to 1"
    print(f"✓ All bugs fixed")
    print(f"✓ Output shape: {output_debug.shape}")
    print(f"✓ Weights normalized correctly")

    # Test Exercise 9: Reshape for Multi-Head
    print("\n" + "-" * 70)
    print("Exercise 9: Reshape for Multi-Head")
    print("-" * 70)
    d_model = 64
    num_heads = 8
    x = torch.randn(batch_size, seq_len, d_model)
    reshaped = solution_09_reshape_for_multi_head(x, num_heads)
    expected_shape = (batch_size, num_heads, seq_len, d_model // num_heads)
    assert reshaped.shape == expected_shape, \
        f"Expected {expected_shape}, got {reshaped.shape}"
    print(f"✓ Shape correct: {reshaped.shape}")
    print(f"✓ d_k per head: {d_model // num_heads}")

    # Test Exercise 10: Padding Mask
    print("\n" + "-" * 70)
    print("Exercise 10: Padding Mask")
    print("-" * 70)
    lengths = torch.tensor([3, 2, 4])
    max_len = 4
    padding_mask = solution_10_create_padding_mask(lengths, max_len)
    assert padding_mask.shape == (3, 1, 1, max_len), \
        f"Expected {(3, 1, 1, max_len)}, got {padding_mask.shape}"
    assert padding_mask[0, 0, 0, 2] == 0.0, "Real token should be 0"
    assert padding_mask[0, 0, 0, 3] == float('-inf'), "Padding should be -inf"
    assert padding_mask[1, 0, 0, 1] == 0.0, "Real token should be 0"
    assert padding_mask[1, 0, 0, 2] == float('-inf'), "Padding should be -inf"
    print(f"✓ Shape correct: {padding_mask.shape}")
    print(f"✓ Padding mask values correct")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 70)


# ==============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# ==============================================================================

def demonstrate_full_pipeline():
    """
    Complete walkthrough of attention mechanism with detailed output.

    This demonstrates how all pieces fit together in practice.
    """
    print("\n" + "=" * 70)
    print("COMPLETE ATTENTION PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)  # For reproducibility
    B, T, d_k = 2, 4, 8

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  Key/Query dimension: {d_k}")

    # Create inputs
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    print(f"\nInput shapes:")
    print(f"  Q: {tuple(Q.shape)}")
    print(f"  K: {tuple(K.shape)}")
    print(f"  V: {tuple(V.shape)}")

    # Step 1: Compute scores
    print("\n" + "-" * 70)
    print("Step 1: Compute attention scores (QK^T)")
    print("-" * 70)
    scores = solution_01_compute_attention_scores(Q, K)
    print(f"Scores shape: {tuple(scores.shape)}")
    print(f"Scores for batch 0:\n{scores[0]}")
    print(f"Score statistics: mean={scores.mean():.2f}, std={scores.std():.2f}")

    # Step 2: Scale
    print("\n" + "-" * 70)
    print("Step 2: Scale by sqrt(d_k)")
    print("-" * 70)
    scaled = solution_02_scale_attention_scores(scores, d_k)
    print(f"Scaling factor: sqrt({d_k}) = {math.sqrt(d_k):.3f}")
    print(f"Scaled scores for batch 0:\n{scaled[0]}")
    print(f"Scaled statistics: mean={scaled.mean():.2f}, std={scaled.std():.2f}")

    # Step 3: Apply causal mask
    print("\n" + "-" * 70)
    print("Step 3: Apply causal mask")
    print("-" * 70)
    mask = solution_03_create_causal_mask(T)
    print(f"Mask:\n{mask}")
    masked = solution_04_apply_mask(scaled, mask)
    print(f"Masked scores for batch 0:\n{masked[0]}")

    # Step 4: Softmax
    print("\n" + "-" * 70)
    print("Step 4: Apply softmax to get attention weights")
    print("-" * 70)
    weights = solution_05_compute_attention_weights(masked)
    print(f"Attention weights for batch 0:\n{weights[0]}")
    print(f"Row sums: {weights[0].sum(dim=-1)}")

    # Step 5: Apply to values
    print("\n" + "-" * 70)
    print("Step 5: Apply attention to values")
    print("-" * 70)
    output = solution_06_apply_attention_to_values(weights, V)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output for batch 0, position 0:\n{output[0, 0]}")

    # Compare with full implementation
    print("\n" + "-" * 70)
    print("Verification: Compare with full implementation")
    print("-" * 70)
    output_full, weights_full = solution_07_full_attention(Q, K, V, mask)
    assert torch.allclose(output, output_full, atol=1e-6), "Outputs should match!"
    assert torch.allclose(weights, weights_full, atol=1e-6), "Weights should match!"
    print("✓ Step-by-step and full implementation produce identical results!")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


# ==============================================================================
# MAIN: Run everything
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ATTENTION MECHANISM - COMPLETE SOLUTIONS")
    print("=" * 70)
    print("\nThis file demonstrates comprehensive solutions to all exercises.")
    print("Study the code, comments, and output to deepen your understanding.")

    # Run comprehensive test suite
    test_all_solutions()

    # Run full pipeline demonstration
    demonstrate_full_pipeline()

    # Run debugging demonstrations
    demonstrate_bugs()

    # Run padding mask demonstration
    demonstrate_padding_mask()

    print("\n" + "=" * 70)
    print("LEARNING RECOMMENDATIONS")
    print("=" * 70)
    print("\n1. Read each solution's docstring and inline comments")
    print("2. Compare solutions with exercises.py to see what was needed")
    print("3. Try modifying parameters and observe changes")
    print("4. Implement your own version without looking at solutions")
    print("5. Read theory.md sections referenced in docstrings")
    print("6. Experiment with visualizing attention weights")
    print("\nNext steps:")
    print("- Understand multi-head attention (next module)")
    print("- Learn about positional encodings")
    print("- Study the full transformer architecture")
    print("\n" + "=" * 70)
