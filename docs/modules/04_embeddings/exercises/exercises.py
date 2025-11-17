"""
Module 04: Embeddings & Positional Encoding - Exercises

This module contains hands-on exercises for learning:
1. Token embeddings with √d_model scaling
2. Sinusoidal positional encoding from first principles
3. Learned positional embeddings
4. Comparing positional encoding approaches
5. Combining token and positional embeddings
6. Analyzing embedding properties

Each exercise includes:
- Clear docstrings with learning objectives
- Type hints and shape annotations
- TODO sections for implementation
- Test assertions for self-assessment
- Progressive difficulty

Time estimate: 3-4 hours for all exercises
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt


# ============================================================================
# EXERCISE 1: Basic Token Embedding with Scaling
# ============================================================================

def exercise_01_token_embedding_basics() -> Tuple[torch.Tensor, float]:
    """
    EXERCISE 01: Implement basic token embedding with √d_model scaling.

    Learning Objectives:
    - Understand how token embeddings work
    - Learn why we scale embeddings by √d_model
    - Implement a simple embedding lookup table

    TODO Tasks:
    1. Create an embedding layer for a vocabulary of 5000 tokens, d_model=128
    2. Create a batch of 16 token sequences, each with 32 tokens
    3. Embed the tokens and scale by √d_model
    4. Return the embedded output and the scaling factor

    Shape Requirements:
    - Input tokens: (batch_size=16, seq_len=32) - type: LongTensor
    - Output embeddings: (batch_size=16, seq_len=32, d_model=128)
    - Scaling factor: scalar float

    Example:
        >>> embedded, scale = exercise_01_token_embedding_basics()
        >>> print(embedded.shape)  # torch.Size([16, 32, 128])
        >>> print(scale)  # 11.313... (sqrt(128))

    Hints:
    - Use torch.nn.Embedding for the lookup table
    - Don't forget to initialize weights (randn_)
    - Remember: sqrt is available via math.sqrt
    - Token IDs should be in range [0, vocab_size)
    """
    # TODO: Implement this exercise
    # Step 1: Define embedding parameters
    vocab_size = 5000
    d_model = 128
    batch_size = 16
    seq_len = 32

    # Step 2: Create embedding layer
    # embedding = TODO: Create nn.Embedding layer

    # Step 3: Create random token IDs
    # tokens = TODO: Create tensor of shape (batch_size, seq_len) with values in [0, vocab_size)

    # Step 4: Embed tokens
    # embedded = TODO: Look up tokens in embedding table -> shape (batch_size, seq_len, d_model)

    # Step 5: Compute and apply scaling factor
    # scale = TODO: Calculate sqrt(d_model)
    # embedded = TODO: Multiply embedded by scale

    # Step 6: Verify shapes and return
    # assert TODO: embedded.shape == (batch_size, seq_len, d_model)

    # return embedded, scale
    pass


def test_exercise_01():
    """Test Exercise 01 solution."""
    result = exercise_01_token_embedding_basics()

    if result is None:
        print("[SKIP] Exercise 01 not implemented yet")
        return

    embedded, scale = result

    assert embedded.shape == (16, 32, 128), \
        f"Expected shape (16, 32, 128), got {embedded.shape}"
    assert isinstance(scale, float), \
        f"Expected scale to be float, got {type(scale)}"
    assert abs(scale - math.sqrt(128)) < 0.01, \
        f"Expected scale ≈ {math.sqrt(128):.2f}, got {scale:.2f}"

    # Verify that embeddings have been scaled
    avg_norm = embedded.norm(dim=-1).mean().item()
    assert 5.0 < avg_norm < 15.0, \
        f"Embedded norm should be ~{math.sqrt(128):.1f}, got {avg_norm:.2f}"

    print("[PASS] Exercise 01: Token Embedding Basics")


# ============================================================================
# EXERCISE 2: Sinusoidal Positional Encoding from Scratch
# ============================================================================

def exercise_02_sinusoidal_positional_encoding() -> torch.Tensor:
    """
    EXERCISE 02: Generate sinusoidal positional encoding from scratch.

    Learning Objectives:
    - Understand the sinusoidal encoding formula
    - Learn why different frequencies are used
    - Implement PE(pos, i) computation

    The sinusoidal encoding uses:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos = position in sequence [0, max_len)
    - 2i, 2i+1 = even and odd dimension indices
    - 10000 = fixed base for frequency scaling

    TODO Tasks:
    1. Create position indices [0, 1, 2, ..., max_len-1]
    2. Create dimension indices for frequency division [0, 2, 4, ..., d_model-2]
    3. Compute the frequency scale: exp(-(ln(10000) / d_model) * i)
    4. Apply sine to even dimensions: PE[:, 0::2] = sin(pos * freq_scale)
    5. Apply cosine to odd dimensions: PE[:, 1::2] = cos(pos * freq_scale)

    Shape Requirements:
    - max_len: 100
    - d_model: 64 (must be even!)
    - Output shape: (max_len=100, d_model=64)

    Example:
        >>> pe = exercise_02_sinusoidal_positional_encoding()
        >>> print(pe.shape)  # torch.Size([100, 64])
        >>> print(pe[0, :10])  # First position, first 10 dimensions
        >>> print(pe[1, :10])  # Second position should be different

    Hints:
    - Position tensor shape: (max_len, 1)
    - Dimension tensor shape: (d_model/2,)
    - torch.arange() and unsqueeze() are your friends
    - torch.exp() and torch.log() for frequency computation
    - Broadcasting will handle multiplication
    """
    # TODO: Implement this exercise
    max_len = 100
    d_model = 64

    # Step 1: Create position indices
    # position = TODO: Shape (max_len, 1)

    # Step 2: Create frequency scales
    # div_term = TODO: Compute exp(-(ln(10000) / d_model) * [0, 2, 4, ..., d_model-2])

    # Step 3: Create PE matrix
    # pe = TODO: zeros(max_len, d_model)

    # Step 4: Fill in sine values for even indices
    # pe[:, 0::2] = TODO: sin(position * div_term)

    # Step 5: Fill in cosine values for odd indices
    # pe[:, 1::2] = TODO: cos(position * div_term)

    # Step 6: Verify and return
    # assert TODO: pe.shape == (max_len, d_model)
    # assert TODO: all values are in [-1, 1]

    # return pe
    pass


def test_exercise_02():
    """Test Exercise 02 solution."""
    result = exercise_02_sinusoidal_positional_encoding()

    if result is None:
        print("[SKIP] Exercise 02 not implemented yet")
        return

    pe = result

    assert pe.shape == (100, 64), \
        f"Expected shape (100, 64), got {pe.shape}"
    assert pe.min() >= -1.1 and pe.max() <= 1.1, \
        f"PE values should be in [-1, 1], got range [{pe.min():.2f}, {pe.max():.2f}]"

    # Check that different positions are different
    assert not torch.allclose(pe[0], pe[1]), \
        "Different positions should have different encodings"

    # Check that pattern follows sine/cosine (values alternate between sin/cos)
    # This is harder to test directly, but we can check variance
    even_variance = pe[:, 0::2].var()
    odd_variance = pe[:, 1::2].var()
    assert even_variance > 0.1 and odd_variance > 0.1, \
        "Both sine and cosine dimensions should have reasonable variance"

    print("[PASS] Exercise 02: Sinusoidal Positional Encoding")


# ============================================================================
# EXERCISE 3: Learned Positional Embeddings
# ============================================================================

def exercise_03_learned_positional_embeddings() -> nn.Module:
    """
    EXERCISE 03: Implement learned positional embeddings.

    Learning Objectives:
    - Understand difference between fixed vs learned encodings
    - Learn how to use nn.Embedding for positions
    - Understand limitations of learned embeddings

    Unlike sinusoidal encodings which are fixed, learned positional embeddings
    are just regular embeddings where the "token ID" is the position.

    TODO Tasks:
    1. Create an nn.Embedding with:
       - num_embeddings: 512 (max sequence length)
       - embedding_dim: 64 (d_model)
    2. Create a forward method that:
       - Takes input of shape (batch_size, seq_len, d_model)
       - Gets position indices [0, 1, ..., seq_len-1]
       - Looks up position embeddings
       - Adds them to the input
    3. Handle the case where seq_len > max_len (should raise error)

    Shape Requirements:
    - max_len: 512
    - d_model: 64
    - Input: (batch_size, seq_len, d_model)
    - Output: (batch_size, seq_len, d_model)

    Example:
        >>> pos_emb = exercise_03_learned_positional_embeddings()
        >>> x = torch.randn(4, 100, 64)
        >>> output = pos_emb(x)
        >>> print(output.shape)  # torch.Size([4, 100, 64])

    Hints:
    - Subclass nn.Module
    - Use nn.Embedding for position embeddings
    - Use torch.arange(seq_len) to get position indices
    - Broadcasting: (seq_len, d_model) + (batch_size, seq_len, d_model)
    - Device matters! Make sure positions tensor is on same device as input
    """
    # TODO: Implement this exercise
    # You should define a class and return an instance

    class LearnedPositionalEmbedding(nn.Module):
        def __init__(self, max_len: int, d_model: int):
            super().__init__()
            # TODO: Store max_len and d_model
            # TODO: Create nn.Embedding layer for positions

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Add learned positional embeddings to input.

            Args:
                x: Input of shape (batch_size, seq_len, d_model)

            Returns:
                Output with positional embeddings added, same shape as input
            """
            # TODO: Get batch_size and seq_len from x
            # TODO: Check that seq_len <= max_len (raise AssertionError if not)
            # TODO: Create position indices [0, 1, ..., seq_len-1] on correct device
            # TODO: Look up position embeddings
            # TODO: Add position embeddings to input (broadcasting)
            # TODO: Return result
            pass

    # TODO: Return an instance with max_len=512, d_model=64
    return LearnedPositionalEmbedding(512, 64)


def test_exercise_03():
    """Test Exercise 03 solution."""
    result = exercise_03_learned_positional_embeddings()

    if result is None or not hasattr(result, 'forward'):
        print("[SKIP] Exercise 03 not implemented yet")
        return

    pos_emb = result

    # Test 1: Basic shape preservation
    x = torch.randn(4, 100, 64)
    output = pos_emb(x)
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"

    # Test 2: Different positions should affect output
    x1 = torch.zeros(1, 5, 64)
    output1 = pos_emb(x1)
    assert not torch.allclose(output1[0, 0], output1[0, 1]), \
        "Different positions should produce different embeddings"

    # Test 3: Exceeding max_len should fail
    try:
        x_long = torch.randn(2, 513, 64)
        pos_emb(x_long)
        assert False, "Should raise error for seq_len > max_len"
    except AssertionError:
        pass  # Expected

    # Test 4: Parameters are learnable
    assert pos_emb.position_embeddings.weight.requires_grad, \
        "Position embeddings should be learnable"

    print("[PASS] Exercise 03: Learned Positional Embeddings")


# ============================================================================
# EXERCISE 4: Sinusoidal vs Learned - Interpolation Comparison
# ============================================================================

def exercise_04_compare_interpolation() -> dict:
    """
    EXERCISE 04: Compare sinusoidal vs learned positional encodings on interpolation.

    Learning Objectives:
    - Understand generalization properties of different encoding methods
    - Learn about interpolation: extrapolating to positions between training positions
    - Compare fixed vs learned approaches experimentally

    Sinusoidal encodings can theoretically handle ANY length through interpolation.
    Learned embeddings CANNOT extrapolate (new positions are out of vocabulary).

    TODO Tasks:
    1. Create both sinusoidal PE and learned PE (max_len=64, d_model=32)
    2. Create embeddings for sequences of length [10, 20, 30, ..., 64]
    3. Compare the learned PE outputs when:
       - Training saw max positions [10, 20, 30, ..., 60]
       - Now testing on position 55 (was in training range)
    4. Compare sinusoidal PE outputs for the same positions
    5. Measure how "smooth" the interpolation is using cosine similarity

    Shape Requirements:
    - max_len: 64
    - d_model: 32
    - Batch size for testing: 1
    - seq_len values: [10, 20, 30, 40, 50, 60]

    Return dictionary with keys:
    - 'sinusoidal_pe': (6, 32) tensor of sinusoidal encodings at seq_len values
    - 'learned_pe': (6, 32) tensor of learned embeddings at seq_len values
    - 'sinusoidal_smoothness': list of cosine similarities between consecutive PEs
    - 'learned_smoothness': list of cosine similarities between consecutive PEs
    - 'interpolation_analysis': dict with comparison results

    Example:
        >>> results = exercise_04_compare_interpolation()
        >>> print(f"Sinusoidal smoothness: {results['sinusoidal_smoothness']}")
        >>> print(f"Learned smoothness: {results['learned_smoothness']}")

    Hints:
    - Cosine similarity: torch.nn.CosineSimilarity
    - Extract encodings at position 0 for each seq_len
    - Compare how similar consecutive positions are
    - Sinusoidal should be smoother than learned
    """
    # TODO: Implement this exercise
    max_len = 64
    d_model = 32
    seq_len_values = [10, 20, 30, 40, 50, 60]

    # Step 1: Create positional encodings
    # sinusoidal_pe = TODO: SinusoidalPositionalEncoding(d_model, max_len)
    # learned_pe = TODO: LearnedPositionalEmbedding(max_len, d_model)

    # Step 2: Evaluate at different sequence lengths
    # sinusoidal_encodings = TODO: List of shape (1, seq_len, d_model) for each seq_len
    # learned_encodings = TODO: List of shape (1, seq_len, d_model) for each seq_len

    # Step 3: Extract position 0 encoding for each
    # sin_pe_values = TODO: (6, d_model) tensor
    # learned_pe_values = TODO: (6, d_model) tensor

    # Step 4: Compute smoothness (cosine similarity between consecutive positions)
    # sin_smoothness = TODO: [cosine_sim(sin_pe_values[i], sin_pe_values[i+1]) for i in range(5)]
    # learned_smoothness = TODO: [cosine_sim(learned_pe_values[i], learned_pe_values[i+1]) for i in range(5)]

    # Step 5: Create results dictionary
    # return {
    #     'sinusoidal_pe': sin_pe_values,
    #     'learned_pe': learned_pe_values,
    #     'sinusoidal_smoothness': sin_smoothness,
    #     'learned_smoothness': learned_smoothness,
    #     'interpolation_analysis': {
    #         'mean_sinusoidal_similarity': TODO: average of sin_smoothness,
    #         'mean_learned_similarity': TODO: average of learned_smoothness,
    #         'conclusion': 'TODO: Is sinusoidal smoother?'
    #     }
    # }

    # Note: Import reference implementations if needed:
    # from tiny_transformer.embeddings import SinusoidalPositionalEncoding, LearnedPositionalEmbedding

    pass


def test_exercise_04():
    """Test Exercise 04 solution."""
    result = exercise_04_compare_interpolation()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 04 not implemented yet")
        return

    assert 'sinusoidal_pe' in result, "Missing 'sinusoidal_pe' in result"
    assert 'learned_pe' in result, "Missing 'learned_pe' in result"
    assert 'sinusoidal_smoothness' in result, "Missing 'sinusoidal_smoothness'"
    assert 'learned_smoothness' in result, "Missing 'learned_smoothness'"

    # Shapes
    assert result['sinusoidal_pe'].shape == (6, 32), \
        f"Expected sinusoidal_pe shape (6, 32), got {result['sinusoidal_pe'].shape}"
    assert result['learned_pe'].shape == (6, 32), \
        f"Expected learned_pe shape (6, 32), got {result['learned_pe'].shape}"

    # Smoothness lists
    assert len(result['sinusoidal_smoothness']) == 5, \
        f"Expected 5 smoothness values, got {len(result['sinusoidal_smoothness'])}"
    assert len(result['learned_smoothness']) == 5, \
        f"Expected 5 smoothness values, got {len(result['learned_smoothness'])}"

    print("[PASS] Exercise 04: Interpolation Comparison")


# ============================================================================
# EXERCISE 5: Visualize Positional Encoding Patterns
# ============================================================================

def exercise_05_visualize_positional_encodings() -> dict:
    """
    EXERCISE 05: Visualize and analyze positional encoding patterns.

    Learning Objectives:
    - Understand the structure of sinusoidal encodings
    - Visualize how different frequency bands work
    - See the difference between sinusoidal and learned encodings

    Positional encodings use different frequencies to encode position information.
    Higher frequency components capture fine-grained details, while lower
    frequencies capture coarse position information.

    TODO Tasks:
    1. Generate sinusoidal PE matrix (max_len=200, d_model=128)
    2. Generate learned PE matrix (same dimensions)
    3. Create heatmap visualizations:
       - Row = position, Column = dimension
       - Show how the pattern changes with position
    4. Analyze the frequency content:
       - Compute variance of even vs odd dimensions
       - Check periodicity by computing autocorrelation
    5. Return visualization data and analysis

    Return dictionary with keys:
    - 'sinusoidal_matrix': (200, 128) tensor
    - 'learned_matrix': (200, 128) tensor
    - 'sinusoidal_visualization': (200, 128) numpy array for heatmap
    - 'learned_visualization': (200, 128) numpy array for heatmap
    - 'analysis': dict with statistical properties

    Example:
        >>> results = exercise_05_visualize_positional_encodings()
        >>> sin_matrix = results['sinusoidal_matrix']
        >>> learned_matrix = results['learned_matrix']
        >>>
        >>> # Visualize (if matplotlib available)
        >>> plt.figure(figsize=(12, 5))
        >>> plt.subplot(1, 2, 1)
        >>> plt.imshow(results['sinusoidal_visualization'], aspect='auto', cmap='viridis')
        >>> plt.title('Sinusoidal PE')
        >>> plt.subplot(1, 2, 2)
        >>> plt.imshow(results['learned_visualization'], aspect='auto', cmap='viridis')
        >>> plt.title('Learned PE')
        >>> plt.tight_layout()
        >>> plt.show()

    Hints:
    - For sinusoidal: Use the same formula from Exercise 02
    - For learned: Create embeddings and extract the matrix
    - Convert tensors to numpy for visualization
    - Clamp/normalize values to [0, 1] for better visualization
    - Stats to compute: mean, std, variance per dimension
    """
    # TODO: Implement this exercise
    max_len = 200
    d_model = 128

    # Step 1: Generate sinusoidal positional encoding
    # sin_pe = TODO: Create sinusoidal PE matrix

    # Step 2: Generate learned positional embedding
    # learned_pe_module = TODO: Create LearnedPositionalEmbedding
    # Get the weight matrix directly: learned_pe = TODO: learned_pe_module.position_embeddings.weight[:max_len]

    # Step 3: Prepare visualization
    # sinusoidal_vis = TODO: Convert sin_pe to numpy and normalize to [0, 1]
    # learned_vis = TODO: Convert learned_pe to numpy and normalize to [0, 1]

    # Step 4: Analyze properties
    # analysis = {
    #     'sinusoidal_mean': TODO: mean value of sinusoidal PE,
    #     'sinusoidal_std': TODO: std value of sinusoidal PE,
    #     'learned_mean': TODO: mean value of learned PE,
    #     'learned_std': TODO: std value of learned PE,
    #     'sinusoidal_even_var': TODO: variance of even columns,
    #     'sinusoidal_odd_var': TODO: variance of odd columns,
    #     'learned_dim_variance': TODO: variance across each dimension,
    # }

    # Step 5: Return results
    # return {
    #     'sinusoidal_matrix': sin_pe,
    #     'learned_matrix': learned_pe,
    #     'sinusoidal_visualization': sinusoidal_vis,
    #     'learned_visualization': learned_vis,
    #     'analysis': analysis,
    # }

    pass


def test_exercise_05():
    """Test Exercise 05 solution."""
    result = exercise_05_visualize_positional_encodings()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 05 not implemented yet")
        return

    assert 'sinusoidal_matrix' in result, "Missing 'sinusoidal_matrix'"
    assert 'learned_matrix' in result, "Missing 'learned_matrix'"
    assert 'sinusoidal_visualization' in result, "Missing 'sinusoidal_visualization'"
    assert 'learned_visualization' in result, "Missing 'learned_visualization'"
    assert 'analysis' in result, "Missing 'analysis'"

    # Check shapes
    assert result['sinusoidal_matrix'].shape == (200, 128), \
        f"Expected shape (200, 128), got {result['sinusoidal_matrix'].shape}"
    assert result['learned_matrix'].shape == (200, 128), \
        f"Expected shape (200, 128), got {result['learned_matrix'].shape}"

    # Check visualization is numpy array
    assert isinstance(result['sinusoidal_visualization'], np.ndarray), \
        "sinusoidal_visualization should be numpy array"

    # Check values are normalized
    assert result['sinusoidal_visualization'].min() >= 0, \
        "Visualization min should be >= 0"
    assert result['sinusoidal_visualization'].max() <= 1, \
        "Visualization max should be <= 1"

    print("[PASS] Exercise 05: Visualize Positional Encodings")


# ============================================================================
# EXERCISE 6: Test Extrapolation Capabilities
# ============================================================================

def exercise_06_test_extrapolation() -> dict:
    """
    EXERCISE 06: Test extrapolation capabilities of different encodings.

    Learning Objectives:
    - Understand why sinusoidal encodings can extrapolate
    - Learn why learned embeddings cannot extrapolate
    - Analyze what happens beyond training length

    Sinusoidal encodings have mathematical structure that allows them to
    generate valid encodings for ANY position, even beyond max_len.
    Learned embeddings are just table lookups - they fail beyond max_len.

    TODO Tasks:
    1. Create sinusoidal PE with max_len=100
    2. Create learned PE with max_len=100
    3. Try to generate encodings for positions beyond 100:
       - For sinusoidal: manually compute encodings for pos=150, 200, etc
       - For learned: attempt will fail with AssertionError (expected!)
    4. Compare encodings at relative positions:
       - Position 50 from sinusoidal (within max_len)
       - Position 150 from sinusoidal (extrapolated)
       - Check if they have similar structure
    5. Return analysis of extrapolation

    Return dictionary with keys:
    - 'can_extrapolate_sinusoidal': bool (True)
    - 'can_extrapolate_learned': bool (False)
    - 'extrapolated_sinusoidal': tensor of extrapolated encoding at pos=150
    - 'training_sinusoidal': tensor of encoding at pos=50
    - 'extrapolation_analysis': dict with findings

    Example:
        >>> results = exercise_06_test_extrapolation()
        >>> print(f"Sinusoidal can extrapolate: {results['can_extrapolate_sinusoidal']}")
        >>> print(f"Learned can extrapolate: {results['can_extrapolate_learned']}")

    Hints:
    - For sinusoidal extrapolation, manually compute:
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    - You don't need to modify the PE object, just compute values
    - Learned embedding will raise AssertionError - catch it and document
    - Compare smoothness: do extrapolated values make sense?
    """
    # TODO: Implement this exercise
    max_len = 100
    d_model = 64

    # Step 1: Create both types of PE
    # sinusoidal_pe = TODO: Create SinusoidalPositionalEncoding
    # learned_pe_module = TODO: Create LearnedPositionalEmbedding

    # Step 2: Test sinusoidal extrapolation
    # For position 50 (within max_len)
    # x_50 = TODO: torch.zeros(1, 1, d_model) at position 50
    # out_50 = TODO: Get encoding using PE
    # sin_50 = TODO: Extract the encoding

    # For position 150 (beyond max_len)
    # Manually compute sinusoidal encoding:
    # position = torch.tensor([150.0]).unsqueeze(1)
    # div_term = TODO: torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    # sin_150_even = TODO: torch.sin(position * div_term)
    # sin_150_odd = TODO: torch.cos(position * div_term)
    # sin_150 = TODO: Interleave even and odd

    # Step 3: Test learned extrapolation (will fail)
    # try:
    #     x_long = torch.zeros(1, 101, d_model)
    #     learned_pe_module(x_long)
    #     learned_can_extrapolate = False
    # except AssertionError:
    #     learned_can_extrapolate = False

    # Step 4: Analyze similarity between pos 50 and extrapolated
    # similarity = TODO: torch.nn.functional.cosine_similarity(sin_50, sin_150)

    # Step 5: Return results
    # return {
    #     'can_extrapolate_sinusoidal': True,
    #     'can_extrapolate_learned': False,
    #     'training_sinusoidal': sin_50,
    #     'extrapolated_sinusoidal': sin_150,
    #     'extrapolation_analysis': {
    #         'position_50_encoding': 'TODO: shape and stats',
    #         'position_150_encoding': 'TODO: shape and stats',
    #         'similarity': TODO: cosine similarity value,
    #         'finding': 'TODO: Are extrapolations reasonable?',
    #     }
    # }

    pass


def test_exercise_06():
    """Test Exercise 06 solution."""
    result = exercise_06_test_extrapolation()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 06 not implemented yet")
        return

    assert result['can_extrapolate_sinusoidal'] == True, \
        "Sinusoidal should be able to extrapolate"
    assert result['can_extrapolate_learned'] == False, \
        "Learned should NOT be able to extrapolate"

    # Check shapes
    assert result['training_sinusoidal'].shape[-1] == 64, \
        f"Expected d_model=64, got shape {result['training_sinusoidal'].shape}"
    assert result['extrapolated_sinusoidal'].shape[-1] == 64, \
        f"Expected d_model=64, got shape {result['extrapolated_sinusoidal'].shape}"

    print("[PASS] Exercise 06: Extrapolation Test")


# ============================================================================
# EXERCISE 7: Combine Token + Positional Embeddings
# ============================================================================

def exercise_07_combined_embeddings() -> torch.Tensor:
    """
    EXERCISE 07: Combine token embeddings with positional encoding.

    Learning Objectives:
    - Understand the complete embedding pipeline
    - Learn the importance of √d_model scaling for token embeddings
    - Practice combining multiple embedding types

    In transformers, we:
    1. Embed tokens using an embedding table
    2. Scale by √d_model (this balances with positional encoding)
    3. Add positional encoding
    4. Apply dropout

    TODO Tasks:
    1. Create token embeddings (vocab=5000, d_model=256)
    2. Create positional encoding (sinusoidal, max_len=512)
    3. Create a batch of tokens: (batch_size=8, seq_len=64)
    4. Embed tokens and scale by √d_model
    5. Add positional encoding
    6. Apply dropout with p=0.1 in training mode
    7. Return the final embeddings

    Shape Requirements:
    - vocab_size: 5000
    - d_model: 256
    - max_len: 512
    - Input tokens: (8, 64)
    - Output: (8, 64, 256)

    Example:
        >>> output = exercise_07_combined_embeddings()
        >>> print(output.shape)  # torch.Size([8, 64, 256])
        >>> print(output.mean())  # Should be small but non-zero

    Hints:
    - Token embedding component:
      - Create nn.Embedding layer
      - Scale by math.sqrt(d_model)
    - Positional encoding:
      - Can use the implementation from Exercise 02
      - Or reference: tiny_transformer.embeddings.SinusoidalPositionalEncoding
    - Dropout:
      - Should only be active during training
      - Uses inplace operations are okay here
    - The combination is additive: x = token_emb + positional_enc + dropout
    """
    # TODO: Implement this exercise
    vocab_size = 5000
    d_model = 256
    max_len = 512
    batch_size = 8
    seq_len = 64
    dropout_p = 0.1

    # Step 1: Create token embedding layer
    # token_embedding = TODO: nn.Embedding(vocab_size, d_model)

    # Step 2: Create positional encoding
    # This could be:
    # a) Your implementation from Exercise 02
    # b) The reference implementation
    # For now, assume you have a function that returns a (max_len, d_model) tensor
    # positional_encoding = TODO: Create it somehow

    # Step 3: Create random tokens
    # tokens = TODO: torch.randint(0, vocab_size, (batch_size, seq_len))

    # Step 4: Embed tokens
    # embedded_tokens = TODO: token_embedding(tokens)  -> shape (B, T, d_model)

    # Step 5: Scale by √d_model
    # scale = TODO: math.sqrt(d_model)
    # embedded_tokens = TODO: embedded_tokens * scale

    # Step 6: Add positional encoding
    # # Extract positional encoding for this sequence length
    # pos_enc = TODO: Get positional_encoding[:seq_len]  -> shape (seq_len, d_model)
    # embedded_tokens = TODO: embedded_tokens + pos_enc  -> broadcasting

    # Step 7: Apply dropout
    # dropout = TODO: nn.Dropout(dropout_p)
    # dropout.train()  # Set to training mode
    # output = TODO: dropout(embedded_tokens)

    # Step 8: Verify and return
    # assert TODO: output.shape == (batch_size, seq_len, d_model)

    # return output
    pass


def test_exercise_07():
    """Test Exercise 07 solution."""
    result = exercise_07_combined_embeddings()

    if result is None:
        print("[SKIP] Exercise 07 not implemented yet")
        return

    assert result.shape == (8, 64, 256), \
        f"Expected shape (8, 64, 256), got {result.shape}"

    # Check that values are reasonable
    output_norm = result.norm(dim=-1).mean()
    assert 10.0 < output_norm < 25.0, \
        f"Expected norm ~16, got {output_norm:.2f}"

    # Check that not all values are the same
    assert result.std() > 0.1, \
        f"Expected non-zero variance, got std={result.std():.4f}"

    print("[PASS] Exercise 07: Combined Embeddings")


# ============================================================================
# EXERCISE 8: Debug Embedding Issues
# ============================================================================

def exercise_08_debug_embedding_issues() -> dict:
    """
    EXERCISE 08: Debug and fix common embedding issues.

    Learning Objectives:
    - Learn common mistakes when implementing embeddings
    - Practice debugging tensor shape mismatches
    - Understand dimension requirements (even d_model, vocab bounds, etc.)

    You will be given broken embedding code and must identify and fix issues:
    1. Dimension mismatch: d_model is odd (sinusoidal needs even)
    2. Shape mismatch: Incorrect tensor reshaping
    3. Device mismatch: Tensors on different devices
    4. Value bounds: Token IDs out of vocabulary range
    5. Scaling issues: Forgot to scale by √d_model

    TODO Tasks:
    1. For each broken scenario, identify the bug
    2. Fix the code
    3. Verify the fix works
    4. Document what went wrong

    Return dictionary with keys:
    - 'issue_1': dict with 'bug_description', 'fixed_code', 'test_passed'
    - 'issue_2': dict with same structure
    - ... (5 issues total)
    - 'summary': Overall analysis

    Example:
        >>> results = exercise_08_debug_embedding_issues()
        >>> print(results['issue_1']['bug_description'])
        >>> print(results['issue_1']['fixed_code'])

    Hints:
    - Create functions that intentionally have bugs
    - Try to run them and catch exceptions
    - Document what error occurs
    - Fix by modifying the code
    - Re-run to verify
    - Learn from each mistake!
    """
    # TODO: Implement this exercise
    # This is more of a guided exploration than a single function

    # Issue 1: Odd d_model with sinusoidal encoding
    # Bug: SinusoidalPositionalEncoding requires d_model to be even
    # Test: Try d_model=129
    # Fix: Use d_model=128 instead

    # Issue 2: Token ID out of vocabulary
    # Bug: Token ID >= vocab_size
    # Test: TokenEmbedding(vocab_size=1000, d_model=128) with tokens >= 1000
    # Fix: Ensure all token IDs are < vocab_size

    # Issue 3: Missing √d_model scaling
    # Bug: Forgot to multiply by √d_model
    # Test: Embedding norms are too small (~1 instead of ~11 for d_model=128)
    # Fix: Multiply by math.sqrt(d_model)

    # Issue 4: Sequence length exceeds max_len
    # Bug: seq_len > max_len in positional encoding
    # Test: LearnedPositionalEmbedding(max_len=64) with seq_len=100
    # Fix: Ensure seq_len <= max_len or increase max_len

    # Issue 5: Broadcasting error in combination
    # Bug: Adding (B, T, d_model) + (seq_len,) instead of (seq_len, d_model)
    # Test: Try broadcasting a 1D tensor
    # Fix: Use correct shape (seq_len, d_model)

    results = {
        # TODO: Build this dictionary step by step
        'issue_1': {
            'bug_description': 'TODO: Describe the bug',
            'fixed_code': 'TODO: Show the fix',
            'test_passed': False,
        },
        # ... add more issues
        'summary': 'TODO: Overall findings',
    }

    # return results
    pass


def test_exercise_08():
    """Test Exercise 08 solution."""
    result = exercise_08_debug_embedding_issues()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 08 not implemented yet")
        return

    # Check that we have at least 5 issues
    issue_count = sum(1 for k in result.keys() if k.startswith('issue_'))
    assert issue_count >= 5, f"Expected at least 5 issues, found {issue_count}"

    # Each issue should be a dict
    for i in range(1, issue_count + 1):
        assert 'bug_description' in result[f'issue_{i}'], \
            f"Missing bug_description for issue_{i}"
        assert 'fixed_code' in result[f'issue_{i}'], \
            f"Missing fixed_code for issue_{i}"

    print("[PASS] Exercise 08: Debug Issues")


# ============================================================================
# EXERCISE 9: Count Parameters in Embeddings
# ============================================================================

def exercise_09_parameter_counting() -> dict:
    """
    EXERCISE 09: Count and analyze parameters in embedding layers.

    Learning Objectives:
    - Understand memory cost of embeddings
    - Compare parameter efficiency of different approaches
    - Learn how to count parameters in neural networks

    Embedding layers can use many parameters, especially with large vocabularies.

    TODO Tasks:
    1. Create embeddings with different configs:
       - vocab=50000, d_model=512 (large vocab, large model)
       - vocab=1000, d_model=128 (small vocab, small model)
       - vocab=100000, d_model=1024 (very large)
    2. For each, count:
       - Token embedding parameters
       - Sinusoidal positional encoding parameters (should be 0!)
       - Learned positional embedding parameters
       - Total for combined embedding
    3. Calculate memory usage:
       - Assume float32 = 4 bytes per parameter
       - Count memory in MB
    4. Compare different encodings

    Return dictionary with keys:
    - 'config_1', 'config_2', 'config_3': dicts with parameter counts
    - 'parameter_analysis': dict comparing all configs
    - 'efficiency_comparison': which is most/least efficient?

    Example:
        >>> results = exercise_09_parameter_counting()
        >>> print(results['config_1'])
        >>> print(f"Memory usage: {results['config_1']['memory_mb']:.2f} MB")

    Hints:
    - To count parameters: sum(p.numel() for p in module.parameters())
    - Or use module.parameter() and .numel()
    - Sinusoidal has no learnable parameters
    - Learned embeddings have: max_len * d_model parameters
    - Memory = num_params * 4 bytes / (1024*1024) for MB
    - Compare token vs positional embedding costs
    """
    # TODO: Implement this exercise
    configs = [
        {'vocab_size': 50000, 'd_model': 512, 'max_len': 2048, 'name': 'Large'},
        {'vocab_size': 1000, 'd_model': 128, 'max_len': 512, 'name': 'Small'},
        {'vocab_size': 100000, 'd_model': 1024, 'max_len': 4096, 'name': 'XL'},
    ]

    results = {}

    for i, config in enumerate(configs, 1):
        # For each config, create embeddings and count parameters
        # TODO: Implement calculation for each config
        # config_result = {
        #     'vocab_size': config['vocab_size'],
        #     'd_model': config['d_model'],
        #     'max_len': config['max_len'],
        #     'token_embedding_params': TODO: vocab_size * d_model,
        #     'sinusoidal_params': TODO: 0,
        #     'learned_positional_params': TODO: max_len * d_model,
        #     'combined_total': TODO: sum of above,
        #     'memory_mb': TODO: params * 4 / (1024*1024),
        # }
        # results[f'config_{i}'] = config_result
        pass  # TODO: Replace with actual implementation

    # TODO: Add comparison analysis
    # results['parameter_analysis'] = {
    #     'largest_token_embedding': TODO: which config,
    #     'largest_positional_embedding': TODO: which config,
    #     'total_memory_all_configs': TODO: sum,
    # }

    # TODO: return results
    return results


def test_exercise_09():
    """Test Exercise 09 solution."""
    result = exercise_09_parameter_counting()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 09 not implemented yet")
        return

    # Check structure
    assert 'config_1' in result, "Missing config_1"
    assert 'config_2' in result, "Missing config_2"
    assert 'config_3' in result, "Missing config_3"

    # Check each config has required keys
    for i in range(1, 4):
        config = result[f'config_{i}']
        assert 'token_embedding_params' in config, \
            f"Missing token_embedding_params in config_{i}"
        assert 'memory_mb' in config, \
            f"Missing memory_mb in config_{i}"

    print("[PASS] Exercise 09: Parameter Counting")


# ============================================================================
# EXERCISE 10: Analyze Embedding Similarities
# ============================================================================

def exercise_10_embedding_similarities() -> dict:
    """
    EXERCISE 10: Analyze similarity between learned and sinusoidal encodings.

    Learning Objectives:
    - Understand what patterns learned embeddings learn
    - Compare learned vs sinusoidal encodings quantitatively
    - Explore correlation and similarity metrics

    After training on a task, do learned positional embeddings learn patterns
    similar to sinusoidal encodings? Or are they completely different?

    TODO Tasks:
    1. Create both sinusoidal and learned positional encodings
       (max_len=256, d_model=128)
    2. Generate position embeddings for all positions
    3. Compute similarity metrics:
       - Cosine similarity between each position in sinusoidal vs learned
       - Correlation of position vectors
       - Distance metrics (L2, L1)
    4. Analyze patterns:
       - Are earlier positions more similar than later positions?
       - Do both capture position monotonically?
       - How different are they really?
    5. Return detailed analysis

    Return dictionary with keys:
    - 'sinusoidal_embeddings': (256, 128) tensor
    - 'learned_embeddings': (256, 128) tensor
    - 'cosine_similarities': (256,) tensor of position-wise cosine sims
    - 'l2_distances': (256,) tensor of L2 distances
    - 'similarity_analysis': dict with statistical findings

    Example:
        >>> results = exercise_10_embedding_similarities()
        >>> cos_sims = results['cosine_similarities']
        >>> print(f"Mean cosine similarity: {cos_sims.mean():.3f}")
        >>> print(f"Std: {cos_sims.std():.3f}")

    Hints:
    - Use torch.nn.CosineSimilarity for cosine similarity
    - Or use torch.nn.functional.cosine_similarity
    - L2 distance: torch.norm(a - b, p=2)
    - Correlation: use numpy.corrcoef or torch correlations
    - Plot similarity vs position to see trends
    - Analyze if learned embeddings discovered similar structure
    """
    # TODO: Implement this exercise
    max_len = 256
    d_model = 128

    # Step 1: Create both positional encodings
    # sinusoidal_pe = TODO: Create SinusoidalPositionalEncoding
    # learned_pe = TODO: Create LearnedPositionalEmbedding

    # Step 2: Extract embeddings for all positions
    # sinusoidal_embeddings = TODO: (max_len, d_model) tensor
    # learned_embeddings = TODO: (max_len, d_model) tensor

    # Step 3: Compute cosine similarities position by position
    # cos_sim = torch.nn.CosineSimilarity(dim=-1)
    # cosine_similarities = TODO: [cos_sim(sinusoidal_embeddings[i], learned_embeddings[i]) for i in range(max_len)]
    # cosine_similarities = TODO: torch.stack(cosine_similarities)

    # Step 4: Compute L2 distances
    # l2_distances = TODO: torch.norm(sinusoidal_embeddings - learned_embeddings, p=2, dim=-1)

    # Step 5: Statistical analysis
    # analysis = {
    #     'mean_cosine_similarity': TODO: cosine_similarities.mean(),
    #     'std_cosine_similarity': TODO: cosine_similarities.std(),
    #     'min_cosine_similarity': TODO: cosine_similarities.min(),
    #     'max_cosine_similarity': TODO: cosine_similarities.max(),
    #     'mean_l2_distance': TODO: l2_distances.mean(),
    #     'std_l2_distance': TODO: l2_distances.std(),
    #     'finding': 'TODO: Are they similar or different?',
    # }

    # TODO: return {
    #     'sinusoidal_embeddings': sinusoidal_embeddings,
    #     'learned_embeddings': learned_embeddings,
    #     'cosine_similarities': cosine_similarities,
    #     'l2_distances': l2_distances,
    #     'similarity_analysis': analysis,
    # }

    pass


def test_exercise_10():
    """Test Exercise 10 solution."""
    result = exercise_10_embedding_similarities()

    if result is None or not isinstance(result, dict):
        print("[SKIP] Exercise 10 not implemented yet")
        return

    assert 'sinusoidal_embeddings' in result, "Missing sinusoidal_embeddings"
    assert 'learned_embeddings' in result, "Missing learned_embeddings"
    assert 'cosine_similarities' in result, "Missing cosine_similarities"
    assert 'l2_distances' in result, "Missing l2_distances"
    assert 'similarity_analysis' in result, "Missing similarity_analysis"

    # Check shapes
    assert result['sinusoidal_embeddings'].shape == (256, 128), \
        f"Expected (256, 128), got {result['sinusoidal_embeddings'].shape}"
    assert result['cosine_similarities'].shape[0] == 256, \
        f"Expected 256 similarities, got {result['cosine_similarities'].shape}"

    # Check values
    assert result['cosine_similarities'].min() >= -1.1, \
        "Cosine similarity should be >= -1"
    assert result['cosine_similarities'].max() <= 1.1, \
        "Cosine similarity should be <= 1"

    print("[PASS] Exercise 10: Embedding Similarities")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all exercise tests."""
    print("\n" + "=" * 70)
    print("MODULE 04: EMBEDDINGS & POSITIONAL ENCODING - EXERCISE TESTS")
    print("=" * 70 + "\n")

    tests = [
        ("Exercise 01: Token Embedding Basics", test_exercise_01),
        ("Exercise 02: Sinusoidal Positional Encoding", test_exercise_02),
        ("Exercise 03: Learned Positional Embeddings", test_exercise_03),
        ("Exercise 04: Interpolation Comparison", test_exercise_04),
        ("Exercise 05: Visualize Positional Encodings", test_exercise_05),
        ("Exercise 06: Extrapolation Test", test_exercise_06),
        ("Exercise 07: Combined Embeddings", test_exercise_07),
        ("Exercise 08: Debug Issues", test_exercise_08),
        ("Exercise 09: Parameter Counting", test_exercise_09),
        ("Exercise 10: Embedding Similarities", test_exercise_10),
    ]

    passed = 0
    skipped = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
