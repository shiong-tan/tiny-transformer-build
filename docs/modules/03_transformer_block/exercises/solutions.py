"""
Transformer Block Practice Exercises - COMPLETE SOLUTIONS

This file provides comprehensive, production-quality solutions to all exercises
from exercises.py. Each solution includes:

1. Complete implementation with all code filled in
2. Detailed explanatory comments (WHY, not just WHAT)
3. Shape annotations showing tensor dimensions at each step
4. Inline educational notes about key concepts and common mistakes
5. References to theory.md for deeper understanding
6. Alternative approaches where applicable

Study these solutions to:
- Understand best practices for implementing transformer blocks
- Learn defensive programming (shape checking, assertions)
- See how theory translates to code
- Recognize common pitfalls and how to avoid them

Reference implementations:
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/feedforward.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/transformer_block.py

Theory reference:
- /Users/shiongtan/projects/tiny-transformer-build/docs/modules/03_transformer_block/theory.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from collections import defaultdict


# ==============================================================================
# EXERCISE 1 SOLUTION: Basic Feed-Forward Network
# ==============================================================================

class Solution01_FeedForward(nn.Module):
    """
    Complete feed-forward network implementation.

    This is the core transformation layer in transformers, providing:
    - Non-linear transformation capabilities
    - Increased model capacity through expansion
    - Position-wise processing (each position processed independently)

    Theory reference: theory.md, Section "Feed-Forward Networks"
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer: expand from d_model to d_ff
        # Why expand? More dimensions = more capacity for feature transformation
        # Standard practice: d_ff = 4 × d_model
        self.fc1 = nn.Linear(d_model, d_ff)  # (d_model, d_ff)

        # ReLU activation: introduces non-linearity
        # Without this, stacking multiple linear layers would still be linear!
        # f(f(x)) = W₂(W₁x) = (W₂W₁)x = Wx, still linear
        # With ReLU: f(f(x)) can learn non-linear functions
        self.relu = nn.ReLU()

        # Second linear layer: project back to d_model
        # Output must match input dimension for residual connections
        self.fc2 = nn.Linear(d_ff, d_model)  # (d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Shape flow:
            (B, T, d_model) → fc1 → (B, T, d_ff)
                           → relu → (B, T, d_ff)
                           → fc2 → (B, T, d_model)

        Key insight:
            This operates on each position independently!
            For position i: output[:, i, :] = FFN(input[:, i, :])
            No information exchange between positions (that's attention's job)
        """
        # Validate input dimensions
        assert x.dim() == 3, f"Expected 3D input (B, T, d_model), got {x.dim()}D"
        assert x.size(-1) == self.d_model, \
            f"Expected d_model={self.d_model}, got {x.size(-1)}"

        # Step 1: Expand to d_ff dimensions
        # (B, T, d_model) → (B, T, d_ff)
        hidden = self.fc1(x)

        # Step 2: Apply non-linear activation
        # (B, T, d_ff) → (B, T, d_ff)
        # ReLU(x) = max(0, x): zeros out negative values
        # Creates sparse activations (many zeros) which can be beneficial
        hidden = self.relu(hidden)

        # Step 3: Project back to d_model
        # (B, T, d_ff) → (B, T, d_model)
        output = self.fc2(hidden)

        # Verify output shape matches input shape (required for residual connections)
        assert output.shape == x.shape, \
            f"Output shape {output.shape} != input shape {x.shape}"

        return output


# ==============================================================================
# EXERCISE 2 SOLUTION: Feed-Forward with GELU Activation
# ==============================================================================

class Solution02_FeedForwardGELU(nn.Module):
    """
    Modern feed-forward network with GELU activation and dropout.

    GELU (Gaussian Error Linear Unit) is preferred over ReLU because:
    - Smoother gradients (differentiable everywhere)
    - Better empirical performance on language tasks
    - No "dead neuron" problem (ReLU neurons can die if always negative)
    - Stochastic regularization interpretation

    Theory reference: theory.md, Section "GELU vs ReLU Activation"
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer: d_model → d_ff
        self.fc1 = nn.Linear(d_model, d_ff)

        # GELU activation: more sophisticated than ReLU
        # GELU(x) = x * Φ(x) where Φ is standard normal CDF
        # Smooth curve, allows small negative values (unlike ReLU's hard zero)
        self.activation = nn.GELU()

        # Dropout for regularization
        # Randomly zeros elements with probability p during training
        # Scaled by 1/(1-p) to maintain expected value
        # Automatically disabled during eval() mode
        self.dropout = nn.Dropout(dropout)

        # Second linear layer: d_ff → d_model
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU and dropout.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Shape flow:
            (B, T, d_model) → fc1 → (B, T, d_ff)
                           → GELU → (B, T, d_ff)
                           → dropout → (B, T, d_ff)
                           → fc2 → (B, T, d_model)

        Why dropout after activation?
            - Regularizes the network by randomly dropping features
            - Forces network to learn redundant representations
            - Prevents co-adaptation of neurons
            - Applied to activations, not to weight gradients directly
        """
        # Step 1: Expand to d_ff
        # (B, T, d_model) → (B, T, d_ff)
        x = self.fc1(x)

        # Step 2: Apply GELU activation
        # (B, T, d_ff) → (B, T, d_ff)
        # GELU is smooth, allowing gradient flow even for negative inputs
        # Compare to ReLU which has zero gradient for x < 0
        x = self.activation(x)

        # Step 3: Apply dropout
        # (B, T, d_ff) → (B, T, d_ff)
        # During training: randomly zeros elements with probability p
        # During eval: identity operation (no dropout)
        # PyTorch handles train/eval mode automatically
        x = self.dropout(x)

        # Step 4: Project back to d_model
        # (B, T, d_ff) → (B, T, d_model)
        x = self.fc2(x)

        return x

        # Educational note: Dropout placement
        # Option 1 (used here): After activation, before second linear
        #   - Regularizes activations
        #   - Standard in modern transformers
        #
        # Option 2: After second linear (in residual path)
        #   - Regularizes the residual contribution
        #   - Also common practice
        #
        # Option 3: Both locations
        #   - Maximum regularization
        #   - Can be too aggressive (underfitting)


# ==============================================================================
# EXERCISE 3 SOLUTION: Layer Normalization
# ==============================================================================

class Solution03_LayerNorm(nn.Module):
    """
    Layer Normalization implementation from scratch.

    LayerNorm normalizes across features (last dimension), making it:
    - Independent of batch size (unlike BatchNorm)
    - Independent of sequence length
    - Ideal for sequences with variable lengths
    - Consistent between training and inference

    Formula:
        y = γ ⊙ (x - μ) / √(σ² + ε) + β

    Where:
        μ: mean across features (d_model dimension)
        σ²: variance across features
        γ: learnable scale (initialized to 1)
        β: learnable shift (initialized to 0)
        ε: small constant for numerical stability

    Theory reference: theory.md, Section "Layer Normalization"
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable scale parameter (gamma)
        # Initialized to ones → initially acts as identity
        # Allows model to learn optimal scale for each feature
        # Shape: (d_model,) - one scale value per feature dimension
        self.gamma = nn.Parameter(torch.ones(d_model))

        # Learnable shift parameter (beta)
        # Initialized to zeros → initially acts as identity
        # Allows model to learn optimal shift for each feature
        # Shape: (d_model,) - one shift value per feature dimension
        self.beta = nn.Parameter(torch.zeros(d_model))

        # Why learnable parameters?
        # Pure normalization forces mean=0, var=1, which may be too restrictive
        # γ and β give the model freedom to:
        # - Undo normalization if needed (γ=σ, β=μ recovers original)
        # - Learn feature-specific normalization strengths
        # - Adapt to the optimal activation distribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            normalized: Normalized tensor of same shape

        For each position (b, t), we:
        1. Compute mean and variance across d_model features
        2. Normalize to mean=0, var=1
        3. Apply learnable scale and shift

        Example:
            Input:  [[0.5, 1.2, -0.3, 0.8], ...]  (d_model=4)
            Mean:   0.55
            Var:    0.38
            Norm:   [[-0.08, 1.05, -1.37, 0.40], ...]
            Output: γ ⊙ Norm + β (after learning)
        """
        # Validate input
        assert x.size(-1) == self.d_model, \
            f"Expected last dim {self.d_model}, got {x.size(-1)}"

        # Step 1: Compute mean across last dimension (features)
        # keepdim=True maintains the dimension for broadcasting
        # Shape: (B, T, 1) - one mean per position
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)

        # Step 2: Compute variance across last dimension
        # unbiased=False: use N in denominator (not N-1)
        # Why unbiased=False? We're normalizing the same samples we computed
        # statistics from, so N is appropriate (not a sample estimate)
        # Shape: (B, T, 1) - one variance per position
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)

        # Step 3: Normalize to mean=0, std=1
        # (x - mean) centers the distribution
        # / sqrt(var + eps) scales to unit variance
        # eps prevents division by zero when variance is tiny
        # Shape: (B, T, d_model)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Step 4: Apply learnable affine transformation
        # gamma and beta are (d_model,) and broadcast to (B, T, d_model)
        # This allows the model to learn feature-specific scales and shifts
        # If γ=1 and β=0 (initialization), this is identity
        # Shape: (B, T, d_model)
        output = self.gamma * x_normalized + self.beta

        # Verification (optional, for debugging):
        # Check that normalization worked (mean ≈ 0 before affine transform)
        # normalized_mean = x_normalized.mean(dim=-1)
        # assert normalized_mean.abs().max() < 0.01, "Normalization failed"

        return output

        # Common mistakes:
        # 1. Using dim=0 (batch dimension) instead of dim=-1 (features)
        #    → This would be BatchNorm, not LayerNorm!
        # 2. Forgetting keepdim=True
        #    → Broadcasting won't work correctly
        # 3. Using unbiased=True
        #    → Not technically wrong, but inconsistent with standard practice
        # 4. Forgetting eps in sqrt
        #    → NaN/Inf when variance is zero


# ==============================================================================
# EXERCISE 4 SOLUTION: Residual Connection
# ==============================================================================

def solution_04_residual_connection(
    x: torch.Tensor,
    sublayer_output: torch.Tensor
) -> torch.Tensor:
    """
    Implement residual connection (skip connection).

    This simple addition is one of the most important innovations in deep learning!

    Why residual connections work:
    1. Gradient highway: ∂y/∂x = 1 + ∂f/∂x (the "+1" provides direct gradient flow)
    2. Identity mapping: Easy to learn f(x) = 0 (identity), hard to learn f(x) = x
    3. Ensemble effect: Network becomes ensemble of 2^n paths (n = number of layers)

    Theory reference: theory.md, Section "Residual Connections"

    Args:
        x: Original input of shape (batch_size, seq_len, d_model)
        sublayer_output: Output from sublayer of same shape

    Returns:
        output: Sum of input and sublayer output

    Mathematical formulation:
        y = x + f(x)

    Where:
        x: Input (identity path)
        f(x): Sublayer transformation
        y: Output

    Gradient flow:
        ∂L/∂x = ∂L/∂y · ∂y/∂x
              = ∂L/∂y · (1 + ∂f/∂x)

    The "+1" ensures gradients flow directly backward, even if ∂f/∂x is small!
    """
    # Verify shapes match (required for addition)
    assert x.shape == sublayer_output.shape, \
        f"Shape mismatch: {x.shape} vs {sublayer_output.shape}"

    # Simply add the tensors
    # This is element-wise addition, creating a residual (skip) connection
    output = x + sublayer_output

    # That's it! Despite its simplicity, this enables:
    # - Training networks with 100+ layers
    # - Stable gradient flow
    # - Better optimization landscape
    # - Faster convergence

    # Educational note: Why addition and not concatenation?
    # Concatenation: y = concat(x, f(x))
    #   - Doubles the dimension
    #   - Requires projection to match dimensions
    #   - More parameters, more computation
    #
    # Addition: y = x + f(x)
    #   - No dimension change
    #   - No extra parameters
    #   - Simple, elegant, effective

    return output


# ==============================================================================
# EXERCISE 5 SOLUTION: Pre-LN Transformer Block
# ==============================================================================

class Solution05_TransformerBlockPreLN(nn.Module):
    """
    Complete Pre-LN transformer block implementation.

    Pre-LN architecture (modern standard):
        x₁ = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        x₂ = x₁ + Dropout(FeedForward(LayerNorm(x₁)))

    Key insight: Normalization BEFORE sublayers
    - More stable gradient flow
    - Easier to train deep networks
    - No warm-up required

    Theory reference: theory.md, Section "Pre-LN vs Post-LN Architecture"
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Multi-head self-attention
        # For self-attention: query = key = value = input
        # Allows positions to communicate and aggregate information
        # Import from tiny_transformer for production-quality implementation
        from tiny_transformer.multi_head import MultiHeadAttention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # Position-wise feed-forward network
        # Applies same transformation to each position independently
        # Provides non-linear transformation and increased capacity
        self.feed_forward = Solution02_FeedForwardGELU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # Layer normalization for attention sublayer (Pre-LN: before attention)
        self.ln1 = nn.LayerNorm(d_model)

        # Layer normalization for FFN sublayer (Pre-LN: before FFN)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout for attention residual path
        # Applied after attention, before residual addition
        # Regularizes the contribution from attention
        self.dropout1 = nn.Dropout(dropout)

        # Dropout for FFN residual path
        # Applied after FFN, before residual addition
        # Regularizes the contribution from FFN
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Pre-LN transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask for causal/padding masking

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Pre-LN flow (modern standard):
            Input x
              ↓
              ├────────────────→ Add (residual)
              ↓                  ↑
            LayerNorm            |
              ↓                  |
            Attention ───────────┘
              ↓
            Dropout
              ↓
            Intermediate x₁
              ↓
              ├────────────────→ Add (residual)
              ↓                  ↑
            LayerNorm            |
              ↓                  |
            FeedForward ─────────┘
              ↓
            Dropout
              ↓
            Output x₂
        """
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        assert x.size(-1) == self.d_model, \
            f"Expected d_model={self.d_model}, got {x.size(-1)}"

        # Sublayer 1: Multi-Head Self-Attention with Pre-LN
        # ===================================================

        # Pre-LN: Normalize BEFORE sublayer
        # This keeps inputs to attention in a consistent range
        # Improves training stability
        normed_x = self.ln1(x)  # (B, T, d_model)

        # Self-attention: query = key = value = normed input
        # Allows each position to attend to all other positions
        # mask can prevent attending to future positions (causal) or padding
        attn_output = self.attention(
            query=normed_x,
            key=normed_x,
            value=normed_x,
            mask=mask
        )  # (B, T, d_model)

        # Apply dropout to attention output
        # Regularizes the attention contribution
        # Automatically disabled in eval mode
        attn_output = self.dropout1(attn_output)  # (B, T, d_model)

        # Residual connection: add original input
        # This is the "highway" for gradient flow
        # x₁ = x + attention(norm(x))
        x = x + attn_output  # (B, T, d_model)

        # Sublayer 2: Feed-Forward Network with Pre-LN
        # =============================================

        # Pre-LN: Normalize BEFORE sublayer
        # Again, keeps inputs in consistent range
        normed_x = self.ln2(x)  # (B, T, d_model)

        # Feed-forward transformation
        # Expands to d_ff, applies GELU, projects back to d_model
        # Position-wise: no communication between positions
        ffn_output = self.feed_forward(normed_x)  # (B, T, d_model)

        # Apply dropout to FFN output
        # Regularizes the FFN contribution
        ffn_output = self.dropout2(ffn_output)  # (B, T, d_model)

        # Residual connection: add input to this sublayer
        # x₂ = x₁ + ffn(norm(x₁))
        x = x + ffn_output  # (B, T, d_model)

        # Final output
        # Shape is preserved: (B, T, d_model)
        return x

        # Why Pre-LN is better than Post-LN:
        #
        # Gradient flow in Pre-LN:
        #   ∂L/∂x has direct path through residuals (not through LayerNorm)
        #   More stable, especially for deep networks
        #
        # Gradient flow in Post-LN:
        #   ∂L/∂x must pass through LayerNorm
        #   Can cause gradient explosion in early layers
        #   Requires learning rate warm-up
        #
        # Training stability:
        #   Pre-LN: Stable from initialization
        #   Post-LN: Often requires careful tuning


# ==============================================================================
# EXERCISE 6 SOLUTION: Post-LN Transformer Block
# ==============================================================================

class Solution06_TransformerBlockPostLN(nn.Module):
    """
    Transformer block with Post-LN architecture (original design).

    Post-LN architecture (original "Attention Is All You Need"):
        x₁ = LayerNorm(x + MultiHeadAttention(x))
        x₂ = LayerNorm(x₁ + FeedForward(x₁))

    Key insight: Normalization AFTER residual addition
    - Harder to train than Pre-LN
    - Often requires learning rate warm-up
    - Used in original transformer paper

    Theory reference: theory.md, Section "Pre-LN vs Post-LN Architecture"
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Import multi-head attention
        from tiny_transformer.multi_head import MultiHeadAttention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = Solution02_FeedForwardGELU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # Layer normalization (Post-LN: after residual)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout for residual paths
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Post-LN transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Post-LN flow (original design):
            Input x
              ↓
              ├────────────────→ Add (residual)
              ↓                  ↑
            Attention ───────────┘
              ↓
            Dropout
              ↓
            LayerNorm
              ↓
            Intermediate x₁
              ↓
              ├────────────────→ Add (residual)
              ↓                  ↑
            FeedForward ─────────┘
              ↓
            Dropout
              ↓
            LayerNorm
              ↓
            Output x₂
        """
        # Sublayer 1: Multi-Head Self-Attention with Post-LN
        # ===================================================

        # Self-attention on original input (NOT normalized)
        # This is the key difference from Pre-LN
        attn_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=mask
        )  # (B, T, d_model)

        # Apply dropout
        attn_output = self.dropout1(attn_output)  # (B, T, d_model)

        # Residual connection: add original input
        x = x + attn_output  # (B, T, d_model)

        # Post-LN: Normalize AFTER residual addition
        # This can cause gradient issues because gradients must pass through norm
        x = self.ln1(x)  # (B, T, d_model)

        # Sublayer 2: Feed-Forward Network with Post-LN
        # ==============================================

        # Feed-forward on normalized output (from previous sublayer)
        ffn_output = self.feed_forward(x)  # (B, T, d_model)

        # Apply dropout
        ffn_output = self.dropout2(ffn_output)  # (B, T, d_model)

        # Residual connection: add input to this sublayer
        x = x + ffn_output  # (B, T, d_model)

        # Post-LN: Normalize AFTER residual addition
        x = self.ln2(x)  # (B, T, d_model)

        return x

        # Comparison: Post-LN vs Pre-LN
        #
        # Post-LN:
        #   x = norm(x + sublayer(x))
        #   - Gradients pass through norm (can be unstable)
        #   - Often needs learning rate warm-up
        #   - Final output is normalized (mean≈0, var≈1)
        #
        # Pre-LN:
        #   x = x + sublayer(norm(x))
        #   - Gradients have direct path (more stable)
        #   - No warm-up needed
        #   - Final output NOT normalized (may grow with depth)
        #
        # Modern practice: Use Pre-LN unless replicating specific papers


# ==============================================================================
# EXERCISE 7 SOLUTION: Stack Multiple Transformer Blocks
# ==============================================================================

class Solution07_StackedTransformer(nn.Module):
    """
    Stack multiple transformer blocks to create a deep model.

    Deep transformers enable hierarchical feature learning:
    - Early layers: Low-level features (syntax, word relations)
    - Middle layers: Mid-level features (phrases, local context)
    - Late layers: High-level features (semantics, long-range dependencies)

    Theory reference: theory.md, Section "Why This Architecture?"
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_pre_ln: bool = True
    ):
        super().__init__()

        self.n_layers = n_layers
        self.use_pre_ln = use_pre_ln

        # Create a list of transformer blocks
        # Use nn.ModuleList to properly register all parameters
        # ModuleList allows PyTorch to track all sub-modules for:
        # - Parameter collection (for optimizer)
        # - Device movement (.to(device))
        # - Train/eval mode switching
        self.layers = nn.ModuleList([
            Solution05_TransformerBlockPreLN(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) if use_pre_ln else Solution06_TransformerBlockPostLN(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization (for Pre-LN only)
        # Why needed for Pre-LN?
        # - Pre-LN doesn't normalize the final output
        # - Without this, output magnitude can grow with depth
        # - Final norm ensures consistent output scale
        #
        # For Post-LN, the last block already normalizes, so this isn't needed
        if use_pre_ln:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through stacked transformer blocks.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask (shared across all layers)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Flow:
            x → Block₁ → Block₂ → ... → Blockₙ → [Final Norm] → output

        Each block:
        - Processes input sequentially
        - Maintains shape: (B, T, d_model)
        - Can attend to all positions (modified by mask)
        """
        # Pass input through each transformer block sequentially
        for layer in self.layers:
            # Each layer receives output from previous layer
            # All layers share the same mask (if provided)
            x = layer(x, mask=mask)  # (B, T, d_model)

        # Apply final normalization for Pre-LN architecture
        # Ensures final output has normalized distribution
        if self.final_norm is not None:
            x = self.final_norm(x)  # (B, T, d_model)

        return x

        # Educational notes:
        #
        # Why sequential processing?
        # - Each layer builds on representations from previous layer
        # - Enables hierarchical feature learning
        # - Earlier layers: syntax, Later layers: semantics
        #
        # Memory considerations:
        # - Forward pass stores activations for backward pass
        # - Memory usage: O(n_layers × batch_size × seq_len × d_model)
        # - Use gradient checkpointing for very deep models
        #
        # Depth vs Width trade-off:
        # - More layers: Better compositionality, hierarchical features
        # - Wider layers: More capacity per layer, easier to train
        # - Standard: 12-24 layers for large models


# ==============================================================================
# EXERCISE 8 SOLUTION: Debug Gradient Flow
# ==============================================================================

def solution_08_debug_gradient_flow(
    model: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Debug gradient flow through transformer model.

    This is crucial for diagnosing training issues:
    - Vanishing gradients: Very small gradient norms
    - Exploding gradients: Very large gradient norms
    - Dead layers: No gradient flow through certain parameters
    - NaN/Inf: Numerical instability

    Theory reference: theory.md, Section "Common Mistakes"

    Returns comprehensive gradient statistics for analysis.
    """
    # Ensure model is in training mode (needed for proper gradient computation)
    model.train()

    # Ensure input requires gradients
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    # Forward pass
    # During forward pass, PyTorch builds computation graph
    # This graph is used for backward pass to compute gradients
    output = model(x, mask=mask)  # (B, T, d_model)

    # Create a simple loss
    # For debugging, we just sum all outputs
    # In real training, this would be cross-entropy, etc.
    loss = output.sum()

    # Backward pass
    # Computes gradients for all parameters with requires_grad=True
    # Uses chain rule to propagate gradients backward through graph
    loss.backward()

    # Collect gradient statistics
    gradient_stats = {
        'input_grad_norm': 0.0,
        'layer_grad_norms': [],
        'max_grad': float('-inf'),
        'min_grad': float('inf'),
        'has_nan': False,
        'has_inf': False,
        'param_grad_norms': {},
        'zero_grad_params': []
    }

    # Check input gradients
    if x.grad is not None:
        gradient_stats['input_grad_norm'] = x.grad.norm().item()
        gradient_stats['has_nan'] = gradient_stats['has_nan'] or torch.isnan(x.grad).any().item()
        gradient_stats['has_inf'] = gradient_stats['has_inf'] or torch.isinf(x.grad).any().item()
    else:
        print("Warning: Input has no gradient!")

    # Iterate through all parameters and collect gradient info
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute gradient norm for this parameter
            grad_norm = param.grad.norm().item()
            gradient_stats['param_grad_norms'][name] = grad_norm

            # Track max and min gradients (excluding zeros)
            max_val = param.grad.abs().max().item()
            min_val = param.grad[param.grad != 0].abs().min().item() if (param.grad != 0).any() else 0.0

            gradient_stats['max_grad'] = max(gradient_stats['max_grad'], max_val)
            if min_val > 0:  # Only update if non-zero
                gradient_stats['min_grad'] = min(gradient_stats['min_grad'], min_val)

            # Check for NaN or Inf
            gradient_stats['has_nan'] = gradient_stats['has_nan'] or torch.isnan(param.grad).any().item()
            gradient_stats['has_inf'] = gradient_stats['has_inf'] or torch.isinf(param.grad).any().item()

            # Check for zero gradients (dead parameters)
            if grad_norm == 0.0 or grad_norm < 1e-10:
                gradient_stats['zero_grad_params'].append(name)

        else:
            print(f"Warning: Parameter {name} has no gradient!")
            gradient_stats['zero_grad_params'].append(name)

    # Compute per-layer statistics (for stacked transformers)
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layer_grad_norms = []
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    layer_grad_norms.append(param.grad.norm().item())

            if layer_grad_norms:
                avg_layer_norm = sum(layer_grad_norms) / len(layer_grad_norms)
                gradient_stats['layer_grad_norms'].append(avg_layer_norm)

    # Handle edge case where no gradients were found
    if gradient_stats['min_grad'] == float('inf'):
        gradient_stats['min_grad'] = 0.0

    return gradient_stats


def demonstrate_gradient_flow():
    """
    Demonstrate gradient flow debugging with examples.

    Shows how to interpret gradient statistics.
    """
    print("\n" + "=" * 70)
    print("GRADIENT FLOW DEBUGGING DEMONSTRATION")
    print("=" * 70)

    # Create a model
    model = Solution07_StackedTransformer(
        n_layers=6,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        dropout=0.1,
        use_pre_ln=True
    )

    # Create input
    x = torch.randn(4, 32, 256, requires_grad=True)

    # Debug gradients
    stats = solution_08_debug_gradient_flow(model, x)

    print(f"\nGradient Statistics:")
    print(f"  Input gradient norm: {stats['input_grad_norm']:.6f}")
    print(f"  Max gradient: {stats['max_grad']:.6f}")
    print(f"  Min gradient: {stats['min_grad']:.6f}")
    print(f"  Has NaN: {stats['has_nan']}")
    print(f"  Has Inf: {stats['has_inf']}")

    if stats['layer_grad_norms']:
        print(f"\n  Per-layer average gradient norms:")
        for i, norm in enumerate(stats['layer_grad_norms']):
            print(f"    Layer {i}: {norm:.6f}")

    if stats['zero_grad_params']:
        print(f"\n  Warning: Parameters with zero gradients:")
        for param_name in stats['zero_grad_params'][:5]:  # Show first 5
            print(f"    - {param_name}")

    # Interpretation guide
    print("\n" + "-" * 70)
    print("INTERPRETATION GUIDE:")
    print("-" * 70)
    print("Healthy gradients:")
    print("  - Input grad norm: 0.01 - 1.0")
    print("  - Layer grad norms: Similar across layers (within 10×)")
    print("  - No NaN or Inf")
    print("  - No zero gradient parameters")
    print("\nVanishing gradients:")
    print("  - Very small norms (< 1e-6)")
    print("  - Gradients decrease in earlier layers")
    print("  → Solution: Pre-LN, gradient clipping, better initialization")
    print("\nExploding gradients:")
    print("  - Very large norms (> 100)")
    print("  - May lead to NaN/Inf")
    print("  → Solution: Gradient clipping, lower learning rate, Pre-LN")


# ==============================================================================
# EXERCISE 9 SOLUTION: Analyze Attention Patterns Through Blocks
# ==============================================================================

def solution_09_analyze_attention_patterns(
    model: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Analyze how attention patterns evolve through transformer layers.

    This reveals:
    - Which positions each layer focuses on
    - How attention becomes more specialized in deeper layers
    - Whether attention is too diffuse or too focused

    Returns attention statistics for visualization and analysis.
    """
    model.eval()  # Set to eval mode for consistent behavior

    # Storage for attention weights from each layer
    attention_weights_per_layer = []

    # Hook function to capture attention weights
    # Hooks are PyTorch's way to intercept forward/backward pass
    def attention_hook(module, input, output):
        """Hook to capture attention weights from MultiHeadAttention."""
        # MultiHeadAttention returns (output, attention_weights)
        if isinstance(output, tuple) and len(output) == 2:
            _, attn_weights = output
            if attn_weights is not None:
                # Detach and clone to avoid affecting computation graph
                attention_weights_per_layer.append(attn_weights.detach().clone())

    # Register hooks on all attention modules
    hooks = []
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'attention'):
                # Register forward hook to capture attention weights
                hook = layer.attention.register_forward_hook(attention_hook)
                hooks.append(hook)

    # Forward pass (triggers hooks)
    with torch.no_grad():  # No need to compute gradients
        _ = model(x, mask=mask)

    # Remove hooks (cleanup)
    for hook in hooks:
        hook.remove()

    # Analyze attention patterns
    attention_info = {
        'attention_weights': attention_weights_per_layer,
        'attention_entropy': [],
        'top_attended_positions': [],
        'attention_sparsity': [],
        'layer_stats': []
    }

    # Compute statistics for each layer
    for layer_idx, attn_weights in enumerate(attention_weights_per_layer):
        # attn_weights shape: (B, n_heads, T, T)
        B, H, T, _ = attn_weights.shape

        # Compute entropy of attention distributions
        # Entropy measures how "spread out" the attention is
        # High entropy = attending to many positions equally (diffuse)
        # Low entropy = attending to few positions (focused)
        #
        # H(p) = -Σ p(i) * log(p(i))
        # For uniform distribution over T positions: H = log(T)
        # For single position: H = 0

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attn_safe = attn_weights + eps

        # Compute entropy: -Σ p*log(p)
        entropy = -(attn_weights * torch.log(attn_safe)).sum(dim=-1)  # (B, H, T)
        avg_entropy = entropy.mean().item()
        attention_info['attention_entropy'].append(avg_entropy)

        # Find top attended positions for each query
        # For each query position, which key positions get the most attention?
        top_k = min(5, T)  # Top 5 positions (or fewer if sequence is short)
        top_indices = torch.topk(attn_weights, k=top_k, dim=-1).indices  # (B, H, T, k)

        # Average across batch and heads to get representative patterns
        # Convert to list for each query position
        top_positions_per_query = []
        for t in range(T):
            # Get top positions for this query across batch and heads
            query_tops = top_indices[:, :, t, :].flatten().tolist()
            # Count frequency of each position
            from collections import Counter
            position_counts = Counter(query_tops)
            # Get most common positions
            most_common = position_counts.most_common(top_k)
            top_positions_per_query.append([pos for pos, _ in most_common])

        attention_info['top_attended_positions'].append(top_positions_per_query)

        # Compute sparsity: what fraction of attention weights are near zero?
        # Sparsity indicates how many positions are effectively ignored
        threshold = 0.01  # Consider weights < 0.01 as "effectively zero"
        sparsity = (attn_weights < threshold).float().mean().item()
        attention_info['attention_sparsity'].append(sparsity)

        # Additional statistics
        layer_stats = {
            'mean_attention': attn_weights.mean().item(),
            'std_attention': attn_weights.std().item(),
            'max_attention': attn_weights.max().item(),
            'min_attention': attn_weights.min().item(),
            'entropy': avg_entropy,
            'sparsity': sparsity
        }
        attention_info['layer_stats'].append(layer_stats)

    return attention_info


def demonstrate_attention_analysis():
    """
    Demonstrate attention pattern analysis.

    Shows how to interpret attention statistics.
    """
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN ANALYSIS DEMONSTRATION")
    print("=" * 70)

    # Create a model
    model = Solution07_StackedTransformer(
        n_layers=4,
        d_model=128,
        n_heads=4,
        d_ff=512,
        dropout=0.0,  # No dropout for consistent analysis
        use_pre_ln=True
    )

    # Create input
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 128)

    # Create causal mask (for autoregressive attention)
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # Analyze attention
    # Note: This requires modifying the model to return attention weights
    # For this demonstration, we'll show the expected output format
    print("\nExpected output format:")
    print("  - attention_weights: List of attention tensors per layer")
    print("  - attention_entropy: Entropy of attention distributions")
    print("  - top_attended_positions: Most attended positions per query")
    print("  - attention_sparsity: Fraction of near-zero attention weights")

    print("\nInterpretation:")
    print("  High entropy → Diffuse attention (attending to many positions)")
    print("  Low entropy → Focused attention (attending to few positions)")
    print("  High sparsity → Sparse attention (many ignored positions)")
    print("  Low sparsity → Dense attention (attending broadly)")

    print("\nTypical patterns in transformers:")
    print("  - Early layers: More diffuse (gathering broad context)")
    print("  - Middle layers: Mix of diffuse and focused")
    print("  - Late layers: More focused (specific information extraction)")


# ==============================================================================
# EXERCISE 10 SOLUTION: Count Parameters in Transformer Block
# ==============================================================================

def solution_10_count_parameters(
    d_model: int,
    n_heads: int,
    d_ff: int
) -> dict:
    """
    Count parameters in a transformer block.

    Provides detailed breakdown of where parameters are allocated,
    helping with model sizing and memory estimation.

    Theory reference: theory.md, Section "Parameter Count"
    """
    # Multi-Head Attention Parameters
    # ================================

    # Query projection: d_model → d_model
    # W_q matrix: (d_model, d_model)
    w_q_params = d_model * d_model

    # Key projection: d_model → d_model
    # W_k matrix: (d_model, d_model)
    w_k_params = d_model * d_model

    # Value projection: d_model → d_model
    # W_v matrix: (d_model, d_model)
    w_v_params = d_model * d_model

    # Output projection: d_model → d_model
    # W_o matrix: (d_model, d_model)
    w_o_params = d_model * d_model

    # Total attention parameters
    # (4 matrices, each d_model × d_model)
    attention_params = w_q_params + w_k_params + w_v_params + w_o_params

    # Detailed breakdown
    attention_breakdown = {
        'W_q': w_q_params,
        'W_k': w_k_params,
        'W_v': w_v_params,
        'W_o': w_o_params,
        'total': attention_params
    }

    # Feed-Forward Network Parameters
    # ================================

    # First linear layer: d_model → d_ff
    # W1 matrix: (d_ff, d_model)
    # b1 bias: (d_ff,)
    w1_params = d_model * d_ff
    b1_params = d_ff

    # Second linear layer: d_ff → d_model
    # W2 matrix: (d_model, d_ff)
    # b2 bias: (d_model,)
    w2_params = d_ff * d_model
    b2_params = d_model

    # Total FFN parameters
    ffn_params = w1_params + b1_params + w2_params + b2_params

    # Detailed breakdown
    ffn_breakdown = {
        'W1': w1_params,
        'b1': b1_params,
        'W2': w2_params,
        'b2': b2_params,
        'total': ffn_params
    }

    # Layer Normalization Parameters
    # ================================

    # Two LayerNorm layers (one for attention, one for FFN)
    # Each LayerNorm has:
    #   - gamma (scale): (d_model,)
    #   - beta (shift): (d_model,)

    # First LayerNorm (before attention)
    ln1_params = 2 * d_model  # gamma + beta

    # Second LayerNorm (before FFN)
    ln2_params = 2 * d_model  # gamma + beta

    # Total LayerNorm parameters
    layernorm_params = ln1_params + ln2_params

    # Detailed breakdown
    layernorm_breakdown = {
        'ln1_gamma': d_model,
        'ln1_beta': d_model,
        'ln2_gamma': d_model,
        'ln2_beta': d_model,
        'total': layernorm_params
    }

    # Total Parameters in One Block
    # ==============================

    total_params = attention_params + ffn_params + layernorm_params

    # Parameter Distribution Analysis
    # ================================

    attention_percentage = (attention_params / total_params) * 100
    ffn_percentage = (ffn_params / total_params) * 100
    layernorm_percentage = (layernorm_params / total_params) * 100

    # Create comprehensive result dictionary
    param_counts = {
        'attention_params': attention_params,
        'ffn_params': ffn_params,
        'layernorm_params': layernorm_params,
        'total_params': total_params,
        'attention_breakdown': attention_breakdown,
        'ffn_breakdown': ffn_breakdown,
        'layernorm_breakdown': layernorm_breakdown,
        'distribution': {
            'attention_percentage': attention_percentage,
            'ffn_percentage': ffn_percentage,
            'layernorm_percentage': layernorm_percentage
        }
    }

    return param_counts


def demonstrate_parameter_counting():
    """
    Demonstrate parameter counting with common configurations.

    Shows parameter counts for different model sizes.
    """
    print("\n" + "=" * 70)
    print("PARAMETER COUNTING DEMONSTRATION")
    print("=" * 70)

    # Common configurations
    configurations = [
        ("Tiny", 128, 4, 512),
        ("Small", 256, 4, 1024),
        ("Base (GPT-2 small)", 768, 12, 3072),
        ("Large (GPT-2 medium)", 1024, 16, 4096),
    ]

    for name, d_model, n_heads, d_ff in configurations:
        counts = solution_10_count_parameters(d_model, n_heads, d_ff)

        print(f"\n{name} Configuration:")
        print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
        print(f"\n  Parameter Breakdown:")
        print(f"    Multi-Head Attention: {counts['attention_params']:,} "
              f"({counts['distribution']['attention_percentage']:.1f}%)")
        print(f"    Feed-Forward Network: {counts['ffn_params']:,} "
              f"({counts['distribution']['ffn_percentage']:.1f}%)")
        print(f"    Layer Normalization:  {counts['layernorm_params']:,} "
              f"({counts['distribution']['layernorm_percentage']:.1f}%)")
        print(f"    Total per block:      {counts['total_params']:,}")

    print("\n" + "-" * 70)
    print("KEY OBSERVATIONS:")
    print("-" * 70)
    print("1. FFN contains ~2/3 of parameters (due to d_ff = 4 × d_model)")
    print("2. LayerNorm has negligible parameters compared to attention/FFN")
    print("3. Attention parameters scale as O(d_model²)")
    print("4. FFN parameters scale as O(d_model × d_ff)")


# ==============================================================================
# COMPREHENSIVE TEST SUITE
# ==============================================================================

def test_all_solutions():
    """
    Comprehensive test suite for all solutions.

    Tests correctness, shapes, edge cases, and numerical properties.
    """
    print("\n" + "=" * 70)
    print("TESTING ALL SOLUTIONS")
    print("=" * 70)

    # Test Solution 1: Basic FeedForward
    print("\n" + "-" * 70)
    print("Solution 1: Basic Feed-Forward Network")
    print("-" * 70)
    ffn = Solution01_FeedForward(d_model=512, d_ff=2048)
    x = torch.randn(32, 128, 512)
    output = ffn(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test Solution 2: FeedForward with GELU
    print("\n" + "-" * 70)
    print("Solution 2: Feed-Forward with GELU")
    print("-" * 70)
    ffn = Solution02_FeedForwardGELU(d_model=768, d_ff=3072, dropout=0.1)
    x = torch.randn(32, 128, 768)
    output = ffn(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")

    # Test dropout behavior
    ffn.eval()
    out1 = ffn(x)
    out2 = ffn(x)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
    print("✓ Dropout disabled in eval mode")

    # Test Solution 3: LayerNorm
    print("\n" + "-" * 70)
    print("Solution 3: Layer Normalization")
    print("-" * 70)
    ln = Solution03_LayerNorm(d_model=768)
    x = torch.randn(32, 128, 768)
    output = ln(x)
    assert output.shape == x.shape

    # Check normalization (per position)
    mean = output[0, 0].mean().item()
    std = output[0, 0].std().item()
    assert abs(mean) < 0.01, f"Mean should be ≈ 0, got {mean}"
    assert abs(std - 1.0) < 0.1, f"Std should be ≈ 1, got {std}"
    print(f"✓ Normalization working: mean={mean:.6f}, std={std:.3f}")

    # Test Solution 4: Residual Connection
    print("\n" + "-" * 70)
    print("Solution 4: Residual Connection")
    print("-" * 70)
    x = torch.randn(32, 128, 768)
    sublayer_out = torch.randn(32, 128, 768)
    output = solution_04_residual_connection(x, sublayer_out)
    assert output.shape == x.shape
    assert torch.allclose(output, x + sublayer_out)
    print("✓ Residual connection correct")

    # Test Solution 5: Pre-LN Transformer Block
    print("\n" + "-" * 70)
    print("Solution 5: Pre-LN Transformer Block")
    print("-" * 70)
    block = Solution05_TransformerBlockPreLN(
        d_model=256, n_heads=4, d_ff=1024, dropout=0.1
    )
    x = torch.randn(16, 32, 256)
    output = block(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")

    # Test with mask
    mask = torch.tril(torch.ones(32, 32))
    output = block(x, mask=mask)
    assert output.shape == x.shape
    print("✓ Works with causal mask")

    # Test Solution 6: Post-LN Transformer Block
    print("\n" + "-" * 70)
    print("Solution 6: Post-LN Transformer Block")
    print("-" * 70)
    block = Solution06_TransformerBlockPostLN(
        d_model=256, n_heads=4, d_ff=1024, dropout=0.1
    )
    x = torch.randn(16, 32, 256)
    output = block(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved: {x.shape} → {output.shape}")

    # Test Solution 7: Stacked Transformer
    print("\n" + "-" * 70)
    print("Solution 7: Stacked Transformer")
    print("-" * 70)
    model = Solution07_StackedTransformer(
        n_layers=6, d_model=256, n_heads=4, d_ff=1024, use_pre_ln=True
    )
    x = torch.randn(8, 32, 256)
    output = model(x)
    assert output.shape == x.shape
    print(f"✓ Shape preserved through {model.n_layers} layers")

    # Test gradient flow
    x_grad = torch.randn(4, 16, 256, requires_grad=True)
    output = model(x_grad)
    loss = output.sum()
    loss.backward()
    assert x_grad.grad is not None
    print(f"✓ Gradients flow through all {model.n_layers} layers")

    # Test Solution 8: Gradient Flow Debugging
    print("\n" + "-" * 70)
    print("Solution 8: Gradient Flow Debugging")
    print("-" * 70)
    model = Solution05_TransformerBlockPreLN(
        d_model=128, n_heads=4, d_ff=512, dropout=0.0
    )
    x = torch.randn(4, 16, 128, requires_grad=True)
    stats = solution_08_debug_gradient_flow(model, x)

    assert 'input_grad_norm' in stats
    assert stats['input_grad_norm'] > 0, "Gradients should flow to input"
    assert not stats['has_nan'], "Should not have NaN gradients"
    assert not stats['has_inf'], "Should not have Inf gradients"
    print(f"✓ Gradient flow healthy: norm={stats['input_grad_norm']:.6f}")

    # Test Solution 10: Parameter Counting
    print("\n" + "-" * 70)
    print("Solution 10: Parameter Counting")
    print("-" * 70)
    counts = solution_10_count_parameters(d_model=768, n_heads=12, d_ff=3072)

    # Verify calculations
    expected_attention = 4 * 768 * 768  # 4 matrices of d_model × d_model
    expected_ffn = 768 * 3072 + 3072 + 3072 * 768 + 768  # W1, b1, W2, b2
    expected_ln = 4 * 768  # 2 LayerNorms, each with gamma and beta

    assert counts['attention_params'] == expected_attention
    assert counts['ffn_params'] == expected_ffn
    assert counts['layernorm_params'] == expected_ln
    print(f"✓ Parameter counting correct:")
    print(f"  Attention: {counts['attention_params']:,}")
    print(f"  FFN: {counts['ffn_params']:,}")
    print(f"  LayerNorm: {counts['layernorm_params']:,}")
    print(f"  Total: {counts['total_params']:,}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 70)


# ==============================================================================
# MAIN: Run everything
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRANSFORMER BLOCK - COMPLETE SOLUTIONS")
    print("=" * 70)
    print("\nThis file demonstrates comprehensive solutions to all exercises.")
    print("Study the code, comments, and output to deepen your understanding.")

    # Run comprehensive test suite
    test_all_solutions()

    # Run demonstrations
    demonstrate_gradient_flow()
    demonstrate_attention_analysis()
    demonstrate_parameter_counting()

    print("\n" + "=" * 70)
    print("LEARNING RECOMMENDATIONS")
    print("=" * 70)
    print("\n1. Read each solution's docstring and inline comments")
    print("2. Compare solutions with exercises.py to see what was needed")
    print("3. Try modifying parameters and observe changes")
    print("4. Implement your own version without looking at solutions")
    print("5. Read theory.md sections referenced in docstrings")
    print("6. Experiment with different architectures (Pre-LN vs Post-LN)")
    print("\nNext steps:")
    print("- Understand positional encodings (next module)")
    print("- Study full transformer architecture")
    print("- Learn about training dynamics and optimization")
    print("\n" + "=" * 70)
