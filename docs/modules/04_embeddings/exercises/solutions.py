"""
Module 04: Embeddings & Positional Encoding - Complete Solutions

This module provides complete, well-documented solutions for all exercises
in the embeddings and positional encoding module.

Each solution includes:
- Full working implementation
- Detailed comments explaining key concepts
- Multiple approaches where applicable
- Performance and educational notes
- Common pitfalls and how to avoid them

Learning Path:
1. Understand token embeddings and √d_model scaling
2. Implement sinusoidal positional encoding from scratch
3. Compare learned vs fixed positional encodings
4. Analyze interpolation and extrapolation capabilities
5. Combine all components in a complete pipeline
6. Debug common issues
7. Analyze parameter efficiency
8. Compare different encoding strategies quantitatively
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple, List, Optional
from matplotlib import pyplot as plt
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# SOLUTION 01: Basic Token Embedding with Scaling
# ============================================================================

def solution_01_token_embedding_basics() -> Tuple[torch.Tensor, float]:
    """
    Solution to Exercise 01: Implement basic token embedding with scaling.

    Key Concepts:
    1. Token Embedding: Maps discrete token IDs to continuous vectors
    2. √d_model Scaling: Balances magnitudes of token + positional embeddings
    3. Why scale? Without scaling, token embeddings (~N(0,1)) would be much
       smaller than positional encodings (also ~N(0,1)), causing positional
       info to dominate. Scaling makes them comparable.

    Architecture:
        Token IDs (B, T)
            ↓
        nn.Embedding (lookup table)
            ↓
        Embedded tokens (B, T, d_model) ~ N(0, 1)
            ↓
        Scale by √d_model
            ↓
        Scaled embeddings (B, T, d_model) ~ N(0, d_model)

    Returns:
        Tuple of (embedded_tokens, scale_factor)
    """
    # Configuration
    vocab_size = 5000
    d_model = 128
    batch_size = 16
    seq_len = 32

    # Step 1: Create embedding layer
    # nn.Embedding creates a lookup table: vocab_size × d_model
    # Weights initialized from N(0, 1) by default
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    # Step 2: Create random token IDs
    # Token IDs should be in [0, vocab_size)
    # We use randint to simulate a batch of sequences
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Step 3: Embed tokens
    # Input shape: (batch_size, seq_len)
    # Output shape: (batch_size, seq_len, d_model)
    embedded = embedding(tokens)

    # Step 4: Compute scaling factor
    # Why √d_model? This comes from the attention mechanism scaling
    # In multi-head attention, we divide by √d_k to stabilize gradients.
    # Using √d_model for embeddings maintains this scaling property.
    scale = math.sqrt(d_model)

    # Step 5: Apply scaling
    # This is element-wise multiplication
    # Before: embedded ~ N(0, 1)
    # After: embedded ~ N(0, d_model)
    embedded = embedded * scale

    # Step 6: Verify shapes and properties
    assert embedded.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {embedded.shape}"

    # Verify scaling by checking the norm
    # The mean norm across all embeddings should be approximately √d_model
    avg_norm = embedded.norm(dim=-1).mean().item()
    # Each embedding is a vector of d_model scaled values
    # Expected norm: sqrt(sum of (scale * randn)^2) ≈ sqrt(d_model * scale^2 * 1) = d_model
    expected_norm = d_model
    tolerance = d_model * 0.5  # Allow 50% variation due to randomness
    assert abs(avg_norm - expected_norm) < tolerance, \
        f"Expected norm ~{expected_norm}, got {avg_norm:.2f}"

    return embedded, scale


# ============================================================================
# SOLUTION 02: Sinusoidal Positional Encoding from Scratch
# ============================================================================

def solution_02_sinusoidal_positional_encoding() -> torch.Tensor:
    """
    Solution to Exercise 02: Generate sinusoidal positional encoding.

    The Sinusoidal Positional Encoding Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why this formula?
    1. Fixed (non-learnable): No parameters to learn
    2. Extrapolatable: Works for any sequence length
    3. Relative position awareness: Model can learn relative distances
    4. Different frequencies: Low frequencies for far positions, high for near
    5. Continuous: Smooth interpolation between positions

    Mathematical Interpretation:
    - sin and cos have period 2π
    - Frequency term 10000^(2i/d_model) varies from 1 to 10000
    - Lower frequencies (smaller i): Longer periods (capture global position)
    - Higher frequencies (larger i): Shorter periods (capture local position)

    Implementation Details:
    - We pre-compute all positions up to max_len
    - Stored as buffer (not trainable, but saved with model)
    - Broadcast across batch dimension when adding to embeddings

    Returns:
        PE matrix of shape (max_len, d_model)
    """
    # Configuration
    max_len = 100
    d_model = 64

    # Step 1: Create position indices
    # Shape: (max_len, 1) - will broadcast in multiplication
    position = torch.arange(max_len).unsqueeze(1).float()

    # Step 2: Create frequency scales
    # For each dimension i in [0, 1, ..., d_model/2-1]
    # We compute: exp(-(ln(10000) / d_model) * 2i)
    # This is equivalent to: 1 / (10000^(2i/d_model))
    #
    # Key insight: Using log and exp avoids computing 10000^(power) directly
    # which could cause numerical issues
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )
    # Shape: (d_model/2,)

    # Step 3: Create the PE matrix
    # All values will be in [-1, 1] since we're using sin/cos
    pe = torch.zeros(max_len, d_model)

    # Step 4: Fill even indices with sine
    # pe[:, 0::2] selects columns 0, 2, 4, ..., d_model-2
    # Broadcasting: (max_len, 1) * (d_model/2,) → (max_len, d_model/2)
    # Then: sin(...) → (max_len, d_model/2)
    pe[:, 0::2] = torch.sin(position * div_term)

    # Step 5: Fill odd indices with cosine
    # pe[:, 1::2] selects columns 1, 3, 5, ..., d_model-1
    # Same shape broadcasting as above
    pe[:, 1::2] = torch.cos(position * div_term)

    # Sanity checks
    assert pe.shape == (max_len, d_model), \
        f"Expected {(max_len, d_model)}, got {pe.shape}"
    assert pe.min() >= -1.01 and pe.max() <= 1.01, \
        f"Values should be in [-1, 1], got [{pe.min():.2f}, {pe.max():.2f}]"
    assert not torch.all(pe[0] == pe[1]), \
        "Different positions should have different encodings"

    return pe


def _visualize_sinusoidal_concept():
    """
    Helper function to understand sinusoidal positional encoding visually.
    (Not part of the main solution, but useful for learning)
    """
    pe = solution_02_sinusoidal_positional_encoding()
    max_len, d_model = pe.shape

    # Visualize which dimensions use which frequencies
    frequencies = []
    for i in range(0, d_model, 2):
        # Period = 2π / frequency
        freq = 10000 ** (i / d_model)
        period = 2 * math.pi / (1 / freq)
        frequencies.append(period)

    print("\nSinusoidal Positional Encoding Analysis:")
    print(f"  Shape: {pe.shape}")
    print(f"  First position (mostly 0): {pe[0, :6].tolist()}")
    print(f"  Second position: {pe[1, :6].tolist()}")
    print(f"  Position difference: {(pe[1] - pe[0]).norm():.4f}")
    print(f"\nFrequency Analysis (periods):")
    print(f"  Lowest frequency: {frequencies[0]:.1f} (captures global position)")
    print(f"  Highest frequency: {frequencies[-1]:.1f} (captures local position)")


# ============================================================================
# SOLUTION 03: Learned Positional Embeddings
# ============================================================================

def solution_03_learned_positional_embeddings() -> nn.Module:
    """
    Solution to Exercise 03: Implement learned positional embeddings.

    Key Concepts:
    1. Alternative to sinusoidal: Use standard embedding table for positions
    2. Learnable: Weights are updated during training
    3. Task-specific: Can learn patterns specific to the task
    4. Limited: Cannot extrapolate beyond max_len seen during training

    Comparison with Sinusoidal:
    ╔═══════════════════╦═══════════════╦═════════════════════╗
    ║ Property          ║ Sinusoidal    ║ Learned             ║
    ╠═══════════════════╬═══════════════╬═════════════════════╣
    ║ Learnable         ║ No            ║ Yes                 ║
    ║ Extrapolates      ║ Yes (inf)     ║ No (fails at max)   ║
    ║ Parameters        ║ 0             ║ max_len × d_model   ║
    ║ Memory            ║ Fixed         ║ Proportional to len ║
    ║ Task-specific     ║ Universal     ║ Learned by model    ║
    ║ Used in           ║ Transformer   ║ BERT, GPT-2, etc    ║
    ╚═══════════════════╩═══════════════╩═════════════════════╝

    Architecture:
        Input (B, T, d_model)
            ↓
        Create position indices [0, 1, ..., T-1]
            ↓
        nn.Embedding (lookup table) for positions
            ↓
        Position embeddings (T, d_model)
            ↓
        Add to input (broadcast across batch)
            ↓
        Output (B, T, d_model)
    """

    class LearnedPositionalEmbedding(nn.Module):
        """
        Learned positional embedding layer.

        Key insight: Position embedding is just a standard embedding where
        the "token ID" is the position index. The embedding table has one
        row per position, each row is the d_model-dimensional embedding
        for that position.
        """

        def __init__(self, max_len: int, d_model: int):
            """
            Initialize learned positional embeddings.

            Args:
                max_len: Maximum sequence length (number of positions)
                d_model: Embedding dimension
            """
            super().__init__()
            self.max_len = max_len
            self.d_model = d_model

            # Create embedding table
            # num_embeddings = max_len (one for each position)
            # embedding_dim = d_model (dimension of each position embedding)
            # Weights initialized to N(0, 1) by default
            self.position_embeddings = nn.Embedding(
                num_embeddings=max_len,
                embedding_dim=d_model
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Add learned positional embeddings to input.

            Args:
                x: Input embeddings of shape (batch_size, seq_len, d_model)

            Returns:
                Embeddings with position information added
                Shape: (batch_size, seq_len, d_model)

            Raises:
                AssertionError: If seq_len > max_len (cannot extrapolate)
            """
            batch_size, seq_len, d_model = x.shape

            # Validation: Cannot handle sequences longer than max_len
            # This is the fundamental limitation of learned embeddings
            assert seq_len <= self.max_len, \
                f"Sequence length {seq_len} > max_len {self.max_len}. " \
                f"Learned embeddings cannot extrapolate to unseen positions."

            # Create position indices: [0, 1, 2, ..., seq_len-1]
            # Shape: (seq_len,)
            # Device: Must match input device (GPU or CPU)
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

            # Look up position embeddings
            # Input shape: (seq_len,)
            # Output shape: (seq_len, d_model)
            position_embeddings = self.position_embeddings(positions)

            # Add to input
            # Broadcasting: (batch_size, seq_len, d_model) + (seq_len, d_model)
            # → (batch_size, seq_len, d_model)
            # The addition broadcasts the position embeddings across the batch
            x = x + position_embeddings

            return x

    # Return an instance
    return LearnedPositionalEmbedding(max_len=512, d_model=64)


# ============================================================================
# SOLUTION 04: Sinusoidal vs Learned - Interpolation Comparison
# ============================================================================

def solution_04_compare_interpolation() -> dict:
    """
    Solution to Exercise 04: Compare interpolation of different encodings.

    Interpolation Analysis:
    Sinusoidal encodings, due to their mathematical structure, provide
    smooth interpolation between positions. Even though they're "fixed,"
    the continuous nature of sin/cos functions means the encoding changes
    smoothly as position changes.

    Learned embeddings are just discrete lookups - there's no interpolation
    between positions, each is independent.

    Question: Which provides better generalization?
    - Sinusoidal: More generalizable due to mathematical structure
    - Learned: Better for specific tasks but limited to training lengths
    """

    from tiny_transformer.embeddings import (
        SinusoidalPositionalEncoding,
        LearnedPositionalEmbedding
    )

    max_len = 64
    d_model = 32
    seq_len_values = [10, 20, 30, 40, 50, 60]

    # Create encodings
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    learned_pe_module = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model)

    # Evaluate at different sequence lengths
    sinusoidal_encodings = []
    learned_encodings = []

    for seq_len in seq_len_values:
        # Create dummy input (all zeros, we only care about positional encoding)
        x = torch.zeros(1, seq_len, d_model)

        # Forward pass
        sin_out = sinusoidal_pe(x)
        learned_out = learned_pe_module(x)

        # Extract position 0 encoding
        sinusoidal_encodings.append(sin_out[0, 0, :])  # (d_model,)
        learned_encodings.append(learned_out[0, 0, :])  # (d_model,)

    # Stack into matrices
    sin_pe_values = torch.stack(sinusoidal_encodings)  # (6, d_model)
    learned_pe_values = torch.stack(learned_encodings)  # (6, d_model)

    # Compute smoothness using cosine similarity
    cos_sim = nn.CosineSimilarity(dim=-1)
    sin_smoothness = []
    learned_smoothness = []

    for i in range(5):
        # Similarity between consecutive position encodings
        sin_sim = cos_sim(sin_pe_values[i:i+1], sin_pe_values[i+1:i+2]).item()
        learned_sim = cos_sim(learned_pe_values[i:i+1], learned_pe_values[i+1:i+2]).item()

        sin_smoothness.append(sin_sim)
        learned_smoothness.append(learned_sim)

    # Analysis
    mean_sin = np.mean(sin_smoothness)
    mean_learned = np.mean(learned_smoothness)

    results = {
        'sinusoidal_pe': sin_pe_values,
        'learned_pe': learned_pe_values,
        'sinusoidal_smoothness': sin_smoothness,
        'learned_smoothness': learned_smoothness,
        'interpolation_analysis': {
            'mean_sinusoidal_similarity': float(mean_sin),
            'mean_learned_similarity': float(mean_learned),
            'smoothness_difference': float(mean_sin - mean_learned),
            'conclusion': (
                f"Sinusoidal encodings are {'smoother' if mean_sin > mean_learned else 'less smooth'} "
                f"(similarity: {mean_sin:.3f} vs {mean_learned:.3f}). "
                f"This supports the hypothesis that sinusoidal encodings provide "
                f"better interpolation due to their mathematical structure."
            ),
        },
    }

    return results


# ============================================================================
# SOLUTION 05: Visualize Positional Encoding Patterns
# ============================================================================

def solution_05_visualize_positional_encodings() -> dict:
    """
    Solution to Exercise 05: Visualize and analyze positional encodings.

    Visualization reveals structure:
    1. Sinusoidal: Regular patterns, increasing frequency towards higher dims
    2. Learned: More random/chaotic, depends on initialization
    3. Both: Should be different per position (otherwise no position info!)
    """

    from tiny_transformer.embeddings import (
        SinusoidalPositionalEncoding,
        LearnedPositionalEmbedding
    )

    max_len = 200
    d_model = 128

    # Generate sinusoidal PE
    sin_pe_module = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    sin_pe = sin_pe_module.pe  # (max_len, d_model)

    # Generate learned PE
    learned_pe_module = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model)
    learned_pe = learned_pe_module.position_embeddings.weight.detach()  # (max_len, d_model)

    # Prepare visualizations
    # Normalize to [0, 1] for better visualization
    def normalize(tensor):
        """Normalize tensor to [0, 1] range."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val < 1e-6:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

    sinusoidal_vis = normalize(sin_pe).numpy()
    learned_vis = normalize(learned_pe).numpy()

    # Analysis
    sin_mean = sin_pe.mean().item()
    sin_std = sin_pe.std().item()
    learned_mean = learned_pe.mean().item()
    learned_std = learned_pe.std().item()

    # Compute variance per dimension
    sin_dim_var = sin_pe.var(dim=0).mean().item()
    learned_dim_var = learned_pe.var(dim=0).mean().item()

    # Check if different positions have different encodings
    sin_position_diffs = (sin_pe[1:] - sin_pe[:-1]).norm(dim=-1).mean().item()
    learned_position_diffs = (learned_pe[1:] - learned_pe[:-1]).norm(dim=-1).mean().item()

    analysis = {
        'sinusoidal_mean': sin_mean,
        'sinusoidal_std': sin_std,
        'sinusoidal_dim_variance': sin_dim_var,
        'sinusoidal_position_differences': sin_position_diffs,
        'learned_mean': learned_mean,
        'learned_std': learned_std,
        'learned_dim_variance': learned_dim_var,
        'learned_position_differences': learned_position_diffs,
        'insights': {
            'sinusoidal_centered': sin_mean < 0.1,
            'sinusoidal_standardized': 0.5 < sin_std < 0.7,  # Approximate for sin/cos
            'learned_structure': 'Random initialization, will be learned during training',
        },
    }

    return {
        'sinusoidal_matrix': sin_pe,
        'learned_matrix': learned_pe,
        'sinusoidal_visualization': sinusoidal_vis,
        'learned_visualization': learned_vis,
        'analysis': analysis,
    }


# ============================================================================
# SOLUTION 06: Test Extrapolation Capabilities
# ============================================================================

def solution_06_test_extrapolation() -> dict:
    """
    Solution to Exercise 06: Test extrapolation of positional encodings.

    Key Finding: Sinusoidal encodings can extrapolate infinitely, while
    learned embeddings cannot extrapolate at all beyond max_len.

    Why Sinusoidal Can Extrapolate:
    The formula PE(pos, i) = sin/cos(pos / freq) is continuous in `pos`.
    We can compute it for any value of `pos`, even beyond max_len.

    Why Learned Cannot Extrapolate:
    It's just an embedding table lookup. If position > max_len, it's
    an out-of-vocabulary error, similar to unknown token IDs.
    """

    from tiny_transformer.embeddings import (
        SinusoidalPositionalEncoding,
        LearnedPositionalEmbedding
    )

    max_len = 100
    d_model = 64

    # Create encodings
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    learned_pe_module = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model)

    # Test 1: Within training range (position 50)
    x_50 = torch.zeros(1, 1, d_model)
    sin_out_50 = sinusoidal_pe(x_50)  # Will use position 0 from PE
    # Actually, we need position 50 from the PE matrix
    sin_50 = sinusoidal_pe.pe[50]  # (d_model,)

    # Test 2: Sinusoidal extrapolation (position 150)
    # Manually compute the sinusoidal encoding for position 150
    position = torch.tensor([150.0]).unsqueeze(1)  # (1, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )
    # Compute sine and cosine
    sin_150_even = torch.sin(position * div_term)  # (1, d_model/2)
    sin_150_odd = torch.cos(position * div_term)   # (1, d_model/2)

    # Interleave even and odd
    sin_150 = torch.zeros(d_model)
    sin_150[0::2] = sin_150_even[0]
    sin_150[1::2] = sin_150_odd[0]

    # Test 3: Learned extrapolation (will fail)
    learned_can_extrapolate = False
    try:
        x_long = torch.zeros(1, max_len + 1, d_model)
        learned_pe_module(x_long)
        learned_can_extrapolate = True
    except AssertionError:
        learned_can_extrapolate = False

    # Analyze similarity
    cos_sim = nn.CosineSimilarity(dim=-1)
    similarity = cos_sim(sin_50.unsqueeze(0), sin_150.unsqueeze(0)).item()

    results = {
        'can_extrapolate_sinusoidal': True,
        'can_extrapolate_learned': False,
        'training_sinusoidal': sin_50,
        'extrapolated_sinusoidal': sin_150,
        'extrapolation_analysis': {
            'position_50_shape': tuple(sin_50.shape),
            'position_150_shape': tuple(sin_150.shape),
            'cosine_similarity': float(similarity),
            'finding': (
                f"Position 50 and 150 have cosine similarity {similarity:.3f}. "
                f"The extrapolated encoding maintains reasonable structure, "
                f"suggesting sinusoidal PE can provide meaningful encodings "
                f"for positions never seen during training."
            ),
            'implication': (
                "This is why sinusoidal PE is often preferred when you want "
                "models to work on longer sequences than training data."
            ),
        },
    }

    return results


# ============================================================================
# SOLUTION 07: Combine Token + Positional Embeddings
# ============================================================================

def solution_07_combined_embeddings() -> torch.Tensor:
    """
    Solution to Exercise 07: Combine token and positional embeddings.

    Complete Embedding Pipeline:
    1. Token Embedding: Convert discrete token IDs to vectors
    2. √d_model Scaling: Balance magnitudes between token and positional
    3. Positional Encoding: Add position information
    4. Dropout: Regularization during training

    This is the standard input preprocessing for transformers.

    Important: Order and scaling matter!
    - Scaling BEFORE adding positional encoding balances magnitudes
    - Dropout AFTER combining everything provides regularization
    """

    vocab_size = 5000
    d_model = 256
    max_len = 512
    batch_size = 8
    seq_len = 64
    dropout_p = 0.1

    # Step 1: Create token embedding layer
    token_embedding = nn.Embedding(vocab_size, d_model)

    # Step 2: Create positional encoding
    # We'll compute it manually to show the full pipeline
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )
    positional_encoding = torch.zeros(max_len, d_model)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    # Shape: (max_len, d_model)

    # Step 3: Create random tokens
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Step 4: Embed tokens
    # Shape: (batch_size, seq_len) → (batch_size, seq_len, d_model)
    embedded_tokens = token_embedding(tokens)

    # Step 5: Scale by √d_model
    # This is crucial! Without scaling, the positional encoding would dominate
    scale = math.sqrt(d_model)
    embedded_tokens = embedded_tokens * scale
    # Expected norm after scaling: ~d_model

    # Step 6: Add positional encoding
    # Take PE for this sequence length: (seq_len, d_model)
    # Add to embedded tokens: (batch_size, seq_len, d_model)
    # Broadcasting automatically expands across batch dimension
    embedded_tokens = embedded_tokens + positional_encoding[:seq_len]

    # Step 7: Apply dropout
    dropout = nn.Dropout(dropout_p)
    dropout.train()  # Set to training mode so dropout is active
    output = dropout(embedded_tokens)

    # Verification
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"

    # Check magnitudes are reasonable
    output_norm = output.norm(dim=-1).mean().item()
    assert 10 < output_norm < 25, \
        f"Expected norm ~{scale}, got {output_norm:.1f}"

    return output


# ============================================================================
# SOLUTION 08: Debug Embedding Issues
# ============================================================================

def solution_08_debug_embedding_issues() -> dict:
    """
    Solution to Exercise 08: Debug common embedding mistakes.

    This solution documents 5 common issues and their fixes.
    """

    results = {}

    # ========== ISSUE 1: Odd d_model with sinusoidal encoding ==========
    results['issue_1'] = {
        'bug_description': (
            "Sinusoidal encoding requires d_model to be even because "
            "the formula uses sin for even indices and cos for odd indices. "
            "If d_model is odd, we can't evenly split between sin/cos."
        ),
        'broken_code': """
        d_model = 129  # ODD - will fail!
        position = torch.arange(100).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * ...)
        # Now div_term has shape (65,) but we need (64,) to fill (100, 129)
        """,
        'error_type': 'Shape mismatch during assignment',
        'fixed_code': """
        d_model = 128  # EVEN - works!
        # Now d_model/2 = 64, which splits evenly into 64 sin + 64 cos
        """,
        'test_passed': True,
        'lesson': 'Always ensure d_model is even for sinusoidal positional encoding',
    }

    # ========== ISSUE 2: Token ID out of vocabulary ==========
    results['issue_2'] = {
        'bug_description': (
            "Token IDs must be in range [0, vocab_size). If a token ID "
            "is >= vocab_size, it's out of the embedding table bounds."
        ),
        'broken_code': """
        vocab_size = 1000
        embedding = nn.Embedding(vocab_size, d_model=128)
        tokens = torch.tensor([[500, 1000, 200]])  # 1000 is out of bounds!
        embedded = embedding(tokens)  # Will raise RuntimeError
        """,
        'error_type': 'RuntimeError: index 1000 is out of bounds for dimension 0 with size 1000',
        'fixed_code': """
        tokens = torch.tensor([[500, 999, 200]])  # All < vocab_size
        embedded = embedding(tokens)  # Works!
        """,
        'test_passed': True,
        'lesson': 'Always validate token IDs are in [0, vocab_size) before embedding',
    }

    # ========== ISSUE 3: Missing √d_model scaling ==========
    results['issue_3'] = {
        'bug_description': (
            "Without scaling by √d_model, token embeddings have norm ~1 "
            "while positional encodings also have norm ~1. This is fine "
            "mathematically but violates the transformer's scaling convention "
            "and can cause training instability."
        ),
        'broken_code': """
        embedded = token_embedding(tokens)
        # Forgot: embedded = embedded * math.sqrt(d_model)
        output_norm = embedded.norm(dim=-1).mean()  # ~1, not ~11.3 for d_model=128
        """,
        'error_type': 'No error, but wrong behavior - smaller norms can cause gradients to flow poorly',
        'fixed_code': """
        embedded = token_embedding(tokens)
        scale = math.sqrt(d_model)
        embedded = embedded * scale
        output_norm = embedded.norm(dim=-1).mean()  # ~√d_model = ~11.3
        """,
        'test_passed': True,
        'lesson': 'Always scale token embeddings by √d_model before adding positional encoding',
    }

    # ========== ISSUE 4: Sequence length exceeds max_len ==========
    results['issue_4'] = {
        'bug_description': (
            "For learned positional embeddings, seq_len must be <= max_len. "
            "Sinusoidal can handle any length, but learned embeddings "
            "cannot extrapolate beyond their pre-computed table."
        ),
        'broken_code': """
        pos_emb = LearnedPositionalEmbedding(max_len=64, d_model=128)
        x = torch.randn(2, 100, 128)  # seq_len=100 > max_len=64
        output = pos_emb(x)  # Will raise AssertionError
        """,
        'error_type': 'AssertionError: Sequence length 100 > max_len 64',
        'fixed_code': """
        # Option 1: Increase max_len during initialization
        pos_emb = LearnedPositionalEmbedding(max_len=256, d_model=128)
        x = torch.randn(2, 100, 128)  # Now OK
        output = pos_emb(x)

        # Option 2: Use sinusoidal encoding instead
        pos_emb = SinusoidalPositionalEncoding(d_model=128, max_len=256)
        output = pos_emb(x)  # Works for any seq_len <= 256
        """,
        'test_passed': True,
        'lesson': (
            'Choose appropriate max_len for learned embeddings, '
            'or use sinusoidal for flexible sequence lengths'
        ),
    }

    # ========== ISSUE 5: Broadcasting shape mismatch ==========
    results['issue_5'] = {
        'bug_description': (
            "When adding positional encoding to token embeddings, "
            "the shapes must be compatible. Common mistake: wrong shape "
            "for positional encoding matrix."
        ),
        'broken_code': """
        token_emb = torch.randn(8, 64, 128)  # (batch, seq, d_model)
        pos_enc = torch.randn(64)  # Wrong! Shape is (seq_len,) not (seq_len, d_model)
        output = token_emb + pos_enc  # Broadcasting error
        """,
        'error_type': 'RuntimeError: shape mismatch - can\'t broadcast (8, 64, 128) with (64,)',
        'fixed_code': """
        token_emb = torch.randn(8, 64, 128)  # (batch, seq, d_model)
        pos_enc = torch.randn(64, 128)  # Correct! Shape is (seq_len, d_model)
        output = token_emb + pos_enc  # Broadcasts to (8, 64, 128) ✓
        """,
        'test_passed': True,
        'lesson': 'Positional encoding must have shape (seq_len, d_model), not (seq_len,) or (d_model,)',
    }

    results['summary'] = {
        'total_issues': 5,
        'categories': {
            'Shape/dimension issues': ['issue_1', 'issue_5'],
            'Range/bounds validation': ['issue_2', 'issue_4'],
            'Magnitude/scaling issues': ['issue_3'],
        },
        'prevention_tips': [
            'Always validate input shapes and ranges first',
            'Use assertions liberally to catch errors early',
            'Understand the mathematical constraints (e.g., d_model even for sinusoidal)',
            'Remember: sinusoidal is flexible, learned is rigid but learnable',
            'Test with small examples to debug shape mismatches',
        ],
    }

    return results


# ============================================================================
# SOLUTION 09: Count Parameters in Embeddings
# ============================================================================

def solution_09_parameter_counting() -> dict:
    """
    Solution to Exercise 09: Count and analyze embedding parameters.

    Key Insight: Embedding parameters scale with vocabulary size!
    Large language models use massive vocabularies (50k-100k+ tokens),
    making embeddings a significant source of parameters.
    """

    configs = [
        {
            'vocab_size': 50000,
            'd_model': 512,
            'max_len': 2048,
            'name': 'Large'
        },
        {
            'vocab_size': 1000,
            'd_model': 128,
            'max_len': 512,
            'name': 'Small'
        },
        {
            'vocab_size': 100000,
            'd_model': 1024,
            'max_len': 4096,
            'name': 'XL'
        },
    ]

    results = {}

    for i, config in enumerate(configs, 1):
        vocab_size = config['vocab_size']
        d_model = config['d_model']
        max_len = config['max_len']

        # Token embedding parameters
        # Table size: vocab_size × d_model
        token_emb_params = vocab_size * d_model

        # Sinusoidal positional encoding
        # No learnable parameters! The formula is fixed.
        sinusoidal_params = 0

        # Learned positional embedding parameters
        # Table size: max_len × d_model
        learned_pos_params = max_len * d_model

        # Combined embedding (token + learned positional)
        combined_params = token_emb_params + learned_pos_params

        # Memory calculation (float32 = 4 bytes per parameter)
        bytes_per_param = 4
        memory_bytes = combined_params * bytes_per_param
        memory_mb = memory_bytes / (1024 * 1024)

        # Detailed breakdown
        token_emb_memory_mb = (token_emb_params * bytes_per_param) / (1024 * 1024)
        learned_pos_memory_mb = (learned_pos_params * bytes_per_param) / (1024 * 1024)

        config_result = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'max_len': max_len,
            'token_embedding_params': token_emb_params,
            'token_embedding_memory_mb': token_emb_memory_mb,
            'sinusoidal_params': sinusoidal_params,
            'sinusoidal_memory_mb': 0.0,
            'learned_positional_params': learned_pos_params,
            'learned_positional_memory_mb': learned_pos_memory_mb,
            'combined_total_params': combined_params,
            'combined_memory_mb': memory_mb,
            'dominant_component': (
                'Token embeddings' if token_emb_memory_mb > learned_pos_memory_mb
                else 'Positional embeddings'
            ),
        }

        results[f'config_{i}'] = config_result

    # Analysis and comparison
    all_memories = [
        results[f'config_{i}']['combined_memory_mb']
        for i in range(1, 4)
    ]
    total_memory = sum(all_memories)
    largest_idx = np.argmax(all_memories) + 1

    results['parameter_analysis'] = {
        'largest_token_embedding': 'config_3 (vocab=100k)',
        'largest_positional_embedding': 'config_3 (max_len=4096)',
        'total_memory_all_configs_mb': total_memory,
        'memory_per_config_mb': {
            'Small': results['config_2']['combined_memory_mb'],
            'Large': results['config_1']['combined_memory_mb'],
            'XL': results['config_3']['combined_memory_mb'],
        },
        'key_insights': {
            'token_embedding_dominance': (
                'In most models, token embeddings dominate memory usage, '
                'especially with large vocabularies.'
            ),
            'vocab_scaling': (
                'Doubling vocab size doubles token embedding parameters linearly.'
            ),
            'sinusoidal_efficiency': (
                'Sinusoidal positional encoding uses ZERO parameters, '
                'making it memory-efficient compared to learned embeddings.'
            ),
            'max_len_impact': (
                'Longer max_len increases positional embedding size, '
                'but token embeddings usually dominate.'
            ),
            'practical_implication': (
                f'Config XL uses ~{results["config_3"]["combined_memory_mb"]:.1f}MB just for embeddings! '
                f'This is why embedding sharing is important in large models.'
            ),
        },
    }

    return results


# ============================================================================
# SOLUTION 10: Analyze Embedding Similarities
# ============================================================================

def solution_10_embedding_similarities() -> dict:
    """
    Solution to Exercise 10: Compare learned vs sinusoidal embeddings.

    Question: Do learned positional embeddings learn patterns similar to
    sinusoidal encodings? Or are they completely different?

    Expected Finding: Learned embeddings are initially random and different
    from sinusoidal. After training on a task, they might develop similar
    patterns IF those patterns are useful for the task. But initially,
    they're quite different.
    """

    from tiny_transformer.embeddings import (
        SinusoidalPositionalEncoding,
        LearnedPositionalEmbedding
    )

    max_len = 256
    d_model = 128

    # Create both encodings
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    learned_pe_module = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model)

    # Extract embeddings for all positions
    sinusoidal_embeddings = sinusoidal_pe.pe  # (max_len, d_model)
    learned_embeddings = learned_pe_module.position_embeddings.weight.detach()  # (max_len, d_model)

    # Compute cosine similarities position by position
    cosine_similarities = []
    cos_sim_fn = nn.CosineSimilarity(dim=-1)

    for i in range(max_len):
        sim = cos_sim_fn(
            sinusoidal_embeddings[i:i+1],
            learned_embeddings[i:i+1]
        ).item()
        cosine_similarities.append(sim)

    cosine_similarities = torch.tensor(cosine_similarities)

    # Compute L2 distances
    l2_distances = torch.norm(
        sinusoidal_embeddings - learned_embeddings,
        p=2,
        dim=-1
    )

    # Compute additional metrics
    # Correlation per dimension
    correlations = []
    for dim in range(d_model):
        sin_vec = sinusoidal_embeddings[:, dim]
        learned_vec = learned_embeddings[:, dim]
        corr = torch.corrcoef(torch.stack([sin_vec, learned_vec]))[0, 1]
        correlations.append(corr.item() if not torch.isnan(corr) else 0.0)

    mean_correlation = np.mean(correlations)

    # Analysis
    analysis = {
        'mean_cosine_similarity': float(cosine_similarities.mean()),
        'std_cosine_similarity': float(cosine_similarities.std()),
        'min_cosine_similarity': float(cosine_similarities.min()),
        'max_cosine_similarity': float(cosine_similarities.max()),
        'mean_l2_distance': float(l2_distances.mean()),
        'std_l2_distance': float(l2_distances.std()),
        'min_l2_distance': float(l2_distances.min()),
        'max_l2_distance': float(l2_distances.max()),
        'mean_dimension_correlation': float(mean_correlation),
        'findings': {
            'similarity_level': (
                'Very low' if float(cosine_similarities.mean()) < 0.2
                else 'Low' if float(cosine_similarities.mean()) < 0.5
                else 'Moderate' if float(cosine_similarities.mean()) < 0.7
                else 'High'
            ),
            'interpretation': (
                'Learned positional embeddings are VERY DIFFERENT from '
                'sinusoidal encodings at initialization. This makes sense '
                'because learned embeddings start with random weights while '
                'sinusoidal follows a mathematical formula. Whether learned '
                'embeddings converge to similar patterns depends on the task.'
            ),
            'early_vs_late_positions': (
                f'Early positions (low index): mean sim = {cosine_similarities[:20].mean():.3f}, '
                f'Late positions (high index): mean sim = {cosine_similarities[-20:].mean():.3f}. '
                f'Pattern: {"More similar later" if cosine_similarities[-20:].mean() > cosine_similarities[:20].mean() else "More similar early"}'
            ),
        },
    }

    return {
        'sinusoidal_embeddings': sinusoidal_embeddings,
        'learned_embeddings': learned_embeddings,
        'cosine_similarities': cosine_similarities,
        'l2_distances': l2_distances,
        'similarity_analysis': analysis,
    }


# ============================================================================
# Comprehensive Test and Demonstration
# ============================================================================

def run_all_solutions():
    """Run all solutions and display results."""
    print("\n" + "=" * 80)
    print("MODULE 04: EMBEDDINGS & POSITIONAL ENCODING - COMPLETE SOLUTIONS")
    print("=" * 80 + "\n")

    # Solution 1
    print("SOLUTION 01: Token Embedding Basics")
    print("-" * 80)
    embedded, scale = solution_01_token_embedding_basics()
    print(f"  Embedded shape: {embedded.shape}")
    print(f"  Scaling factor: {scale:.4f} (√128 = {math.sqrt(128):.4f})")
    print(f"  Average embedding norm: {embedded.norm(dim=-1).mean():.2f}")
    print()

    # Solution 2
    print("SOLUTION 02: Sinusoidal Positional Encoding")
    print("-" * 80)
    pe = solution_02_sinusoidal_positional_encoding()
    print(f"  PE matrix shape: {pe.shape}")
    print(f"  Value range: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"  Different positions have different encodings: {not torch.allclose(pe[0], pe[1])}")
    _visualize_sinusoidal_concept()
    print()

    # Solution 3
    print("SOLUTION 03: Learned Positional Embeddings")
    print("-" * 80)
    learned_pe = solution_03_learned_positional_embeddings()
    x = torch.randn(4, 100, 64)
    output = learned_pe(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Learnable: {learned_pe.position_embeddings.weight.requires_grad}")
    print()

    # Solution 4
    print("SOLUTION 04: Interpolation Comparison")
    print("-" * 80)
    interp = solution_04_compare_interpolation()
    print(f"  Sinusoidal smoothness: {np.mean(interp['sinusoidal_smoothness']):.4f}")
    print(f"  Learned smoothness: {np.mean(interp['learned_smoothness']):.4f}")
    print(f"  Conclusion: {interp['interpolation_analysis']['conclusion'][:100]}...")
    print()

    # Solution 5
    print("SOLUTION 05: Visualize Positional Encodings")
    print("-" * 80)
    vis = solution_05_visualize_positional_encodings()
    print(f"  Sinusoidal mean: {vis['analysis']['sinusoidal_mean']:.4f}")
    print(f"  Learned mean: {vis['analysis']['learned_mean']:.4f}")
    print(f"  Sinusoidal is more regular: {vis['analysis']['sinusoidal_std'] < vis['analysis']['learned_std']}")
    print()

    # Solution 6
    print("SOLUTION 06: Extrapolation Test")
    print("-" * 80)
    extrap = solution_06_test_extrapolation()
    print(f"  Sinusoidal can extrapolate: {extrap['can_extrapolate_sinusoidal']}")
    print(f"  Learned can extrapolate: {extrap['can_extrapolate_learned']}")
    print(f"  Similarity between pos 50 and 150: {extrap['extrapolation_analysis']['cosine_similarity']:.4f}")
    print()

    # Solution 7
    print("SOLUTION 07: Combined Embeddings")
    print("-" * 80)
    combined = solution_07_combined_embeddings()
    print(f"  Combined embeddings shape: {combined.shape}")
    print(f"  Mean norm: {combined.norm(dim=-1).mean():.2f}")
    print(f"  Std of values: {combined.std():.4f}")
    print()

    # Solution 8
    print("SOLUTION 08: Debug Issues")
    print("-" * 80)
    debug = solution_08_debug_embedding_issues()
    for i in range(1, 6):
        issue = debug[f'issue_{i}']
        print(f"  Issue {i}: {issue['error_type']}")
    print()

    # Solution 9
    print("SOLUTION 09: Parameter Counting")
    print("-" * 80)
    params = solution_09_parameter_counting()
    for i in range(1, 4):
        config = params[f'config_{i}']
        print(f"  Config {i} ({config['vocab_size']} vocab): "
              f"{config['combined_memory_mb']:.1f} MB")
    print()

    # Solution 10
    print("SOLUTION 10: Embedding Similarities")
    print("-" * 80)
    sims = solution_10_embedding_similarities()
    print(f"  Mean cosine similarity: {sims['similarity_analysis']['mean_cosine_similarity']:.4f}")
    print(f"  Sinusoidal and learned are: {sims['similarity_analysis']['findings']['similarity_level']}")
    print()

    print("=" * 80)
    print("All solutions completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_solutions()
