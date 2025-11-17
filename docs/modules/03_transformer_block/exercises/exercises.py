"""
Transformer Block Practice Exercises

These exercises will help you understand the complete transformer block by implementing
each component from scratch. Work through them in order, building from feedforward
networks to the full transformer block.

Each exercise has:
- Clear docstring with task description
- Type hints for all parameters
- Expected input/output shapes
- Example usage
- TODO comments marking where to write code
- Test assertions (commented out - uncomment to verify your solution)

Topics covered:
- Feed-forward networks
- GELU activation
- Layer normalization
- Residual connections
- Pre-LN vs Post-LN architecture
- Stacking transformer blocks
- Gradient flow debugging
- Parameter counting

Prerequisites:
- Completed Module 01 (Attention Mechanism)
- Completed Module 02 (Multi-Head Attention)

Reference implementations:
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/feedforward.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/transformer_block.py

Theory reference:
- /Users/shiongtan/projects/tiny-transformer-build/docs/modules/03_transformer_block/theory.md

Good luck! 🚀
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# EXERCISE 1: Basic Feed-Forward Network
# Difficulty: Easy
# ==============================================================================

class Exercise01_FeedForward(nn.Module):
    """
    Implement a basic position-wise feed-forward network.

    Architecture:
        Input (d_model) → Linear → ReLU → Linear → Output (d_model)

    This is simpler than the production version (uses ReLU instead of GELU,
    no dropout). Focus on understanding the structure.

    Args:
        d_model: Model dimension (input and output)
        d_ff: Hidden dimension for the feed-forward network

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> ffn = Exercise01_FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(32, 128, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([32, 128, 512])

    Theory reference: theory.md, Section "Feed-Forward Networks"
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # TODO: Implement initialization
        # Step 1: Create first linear layer (d_model → d_ff)
        # Step 2: Create ReLU activation
        # Step 3: Create second linear layer (d_ff → d_model)

        pass  # Remove this and add your implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Implement forward pass
        # Step 1: Apply first linear layer: (B, T, d_model) → (B, T, d_ff)
        # Step 2: Apply ReLU activation
        # Step 3: Apply second linear layer: (B, T, d_ff) → (B, T, d_model)

        pass  # Remove this and add your implementation

        # Uncomment to test:
        # assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        # assert output.shape == x.shape, f"Shape changed: {x.shape} → {output.shape}"


# ==============================================================================
# EXERCISE 2: Feed-Forward with GELU Activation
# Difficulty: Easy
# ==============================================================================

class Exercise02_FeedForwardGELU(nn.Module):
    """
    Implement feed-forward network with GELU activation.

    GELU is preferred over ReLU in modern transformers for:
    - Smoother gradients
    - Better empirical performance
    - No "dead neuron" problem

    Architecture:
        Input (d_model) → Linear → GELU → Dropout → Linear → Output (d_model)

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (typically 4 × d_model)
        dropout: Dropout probability (default: 0.1)

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> ffn = Exercise02_FeedForwardGELU(d_model=768, d_ff=3072, dropout=0.1)
        >>> x = torch.randn(32, 128, 768)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([32, 128, 768])

    Hints:
        - Use nn.GELU() for activation
        - Apply dropout after the activation, before second linear layer
        - Remember to check self.training for dropout behavior

    Theory reference: theory.md, Section "GELU vs ReLU Activation"
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # TODO: Implement initialization
        # Create layers: fc1, gelu, dropout, fc2

        pass  # Remove this and add your implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation and dropout.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Shape flow:
            (B, T, d_model) → fc1 → (B, T, d_ff)
                           → gelu → (B, T, d_ff)
                           → dropout → (B, T, d_ff)
                           → fc2 → (B, T, d_model)
        """
        # TODO: Implement forward pass

        pass  # Remove this and add your implementation

        # Uncomment to test:
        # assert output.shape == x.shape


# ==============================================================================
# EXERCISE 3: Layer Normalization
# Difficulty: Medium
# ==============================================================================

class Exercise03_LayerNorm(nn.Module):
    """
    Implement Layer Normalization from scratch.

    LayerNorm normalizes across the feature dimension, making it perfect
    for sequences (unlike BatchNorm which normalizes across the batch).

    Formula:
        y = γ ⊙ (x - μ) / √(σ² + ε) + β

    Where:
        μ: Mean across features for each position
        σ²: Variance across features for each position
        γ: Learnable scale (initialized to 1)
        β: Learnable shift (initialized to 0)
        ε: Small constant for numerical stability (1e-5)

    Args:
        d_model: Dimension to normalize over
        eps: Epsilon for numerical stability

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> ln = Exercise03_LayerNorm(d_model=768)
        >>> x = torch.randn(32, 128, 768)
        >>> output = ln(x)
        >>> # Check normalization (per position)
        >>> print(output[0, 0].mean())  # Should be ≈ 0
        >>> print(output[0, 0].std())   # Should be ≈ 1

    Theory reference: theory.md, Section "Layer Normalization"
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

        # TODO: Create learnable parameters
        # gamma: scale parameter, initialized to ones
        # beta: shift parameter, initialized to zeros
        # Use nn.Parameter to make them learnable

        pass  # Remove this and add your implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            normalized: Normalized tensor of same shape

        Steps:
            1. Compute mean across last dimension (d_model)
            2. Compute variance across last dimension
            3. Normalize: (x - mean) / sqrt(var + eps)
            4. Scale and shift: gamma * normalized + beta
        """
        # TODO: Implement forward pass
        # Step 1: Compute mean (use keepdim=True for broadcasting)
        # Step 2: Compute variance (use keepdim=True, unbiased=False)
        # Step 3: Normalize
        # Step 4: Apply learnable scale and shift

        pass  # Remove this and add your implementation

        # Uncomment to test:
        # assert output.shape == x.shape
        # # Check normalization (mean ≈ 0, std ≈ 1)
        # mean_check = output.mean(dim=-1).abs().mean()
        # assert mean_check < 0.01, f"Mean should be ≈ 0, got {mean_check}"


# ==============================================================================
# EXERCISE 4: Residual Connection
# Difficulty: Easy
# ==============================================================================

def exercise_04_residual_connection(
    x: torch.Tensor,
    sublayer_output: torch.Tensor
) -> torch.Tensor:
    """
    Implement a residual connection.

    Residual connections are crucial for:
    - Gradient flow in deep networks
    - Enabling easier optimization
    - Creating an ensemble of paths

    Formula:
        output = x + sublayer_output

    Args:
        x: Original input of shape (batch_size, seq_len, d_model)
        sublayer_output: Output from sublayer of same shape

    Returns:
        output: Sum of input and sublayer output

    Example:
        >>> x = torch.randn(32, 128, 768)
        >>> sublayer_out = torch.randn(32, 128, 768)
        >>> output = exercise_04_residual_connection(x, sublayer_out)
        >>> output.shape
        torch.Size([32, 128, 768])

    Theory reference: theory.md, Section "Residual Connections"
    """
    # TODO: Implement residual connection
    # This is deceptively simple but critically important!
    # Just add the two tensors together

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # assert output.shape == x.shape == sublayer_output.shape
    # # Verify it's actually doing addition
    # expected = x + sublayer_output
    # assert torch.allclose(output, expected)


# ==============================================================================
# EXERCISE 5: Pre-LN Transformer Block
# Difficulty: Hard
# ==============================================================================

class Exercise05_TransformerBlockPreLN(nn.Module):
    """
    Implement a complete transformer block with Pre-LN architecture.

    Pre-LN (Layer Normalization before sublayers) is the modern standard because:
    - More stable gradients
    - Easier to train
    - No warm-up needed

    Architecture:
        x₁ = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        x₂ = x₁ + Dropout(FeedForward(LayerNorm(x₁)))

    Components:
        - Multi-head self-attention
        - Position-wise feed-forward network
        - Two layer normalizations (Pre-LN: before sublayers)
        - Two residual connections
        - Dropout for regularization

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> from tiny_transformer.multi_head import MultiHeadAttention
        >>> block = Exercise05_TransformerBlockPreLN(
        ...     d_model=768, n_heads=12, d_ff=3072, dropout=0.1
        ... )
        >>> x = torch.randn(32, 128, 768)
        >>> mask = torch.tril(torch.ones(128, 128))  # Causal mask
        >>> output = block(x, mask=mask)
        >>> output.shape
        torch.Size([32, 128, 768])

    Theory reference: theory.md, Section "Complete Transformer Block"
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # TODO: Initialize all components
        # You'll need:
        # 1. MultiHeadAttention (from tiny_transformer.multi_head)
        # 2. FeedForward (use Exercise02_FeedForwardGELU or your implementation)
        # 3. Two LayerNorm layers (ln1 and ln2)
        # 4. Two Dropout layers (dropout1 and dropout2)

        pass  # Remove this and add your implementation

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block (Pre-LN).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Pre-LN flow:
            Input
              ↓
              ├────────────────→ Add
              ↓                  ↑
            LayerNorm            |
              ↓                  |
            Attention ───────────┘
              ↓
            Dropout
              ↓
              ├────────────────→ Add
              ↓                  ↑
            LayerNorm            |
              ↓                  |
            FeedForward ─────────┘
              ↓
            Dropout
              ↓
            Output
        """
        # TODO: Implement Pre-LN forward pass
        # Sublayer 1: Attention with Pre-LN
        #   Step 1: Normalize input
        #   Step 2: Apply self-attention (query=key=value=normalized)
        #   Step 3: Apply dropout
        #   Step 4: Residual connection: x = x + dropout(attention_out)

        # Sublayer 2: Feed-Forward with Pre-LN
        #   Step 1: Normalize x
        #   Step 2: Apply feed-forward
        #   Step 3: Apply dropout
        #   Step 4: Residual connection: x = x + dropout(ffn_out)

        pass  # Remove this and add your implementation

        # Uncomment to test:
        # assert output.shape == x.shape


# ==============================================================================
# EXERCISE 6: Post-LN Transformer Block
# Difficulty: Medium
# ==============================================================================

class Exercise06_TransformerBlockPostLN(nn.Module):
    """
    Implement transformer block with Post-LN architecture.

    Post-LN (original architecture) normalizes AFTER residual addition.
    Harder to train than Pre-LN, but used in original "Attention Is All You Need".

    Architecture:
        x₁ = LayerNorm(x + MultiHeadAttention(x))
        x₂ = LayerNorm(x₁ + FeedForward(x₁))

    Key difference from Pre-LN:
        Post-LN: x = LayerNorm(x + sublayer(x))
        Pre-LN:  x = x + sublayer(LayerNorm(x))

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> block = Exercise06_TransformerBlockPostLN(
        ...     d_model=768, n_heads=12, d_ff=3072, dropout=0.1
        ... )
        >>> x = torch.randn(32, 128, 768)
        >>> output = block(x)
        >>> output.shape
        torch.Size([32, 128, 768])

    Note:
        Post-LN is more challenging to train (may require learning rate warm-up).
        Use Pre-LN for new projects unless you have specific reasons.

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

        # TODO: Initialize components (same as Pre-LN)

        pass  # Remove this and add your implementation

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block (Post-LN).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)

        Post-LN flow:
            Input
              ↓
              ├──────────────→ Add
              ↓               ↑
            Attention ────────┘
              ↓
            Dropout
              ↓
            LayerNorm
              ↓
              ├──────────────→ Add
              ↓               ↑
            FeedForward ──────┘
              ↓
            Dropout
              ↓
            LayerNorm
              ↓
            Output
        """
        # TODO: Implement Post-LN forward pass
        # Sublayer 1: Attention with Post-LN
        #   Step 1: Apply self-attention
        #   Step 2: Apply dropout
        #   Step 3: Residual connection and normalize: x = LayerNorm(x + dropout(attn_out))

        # Sublayer 2: Feed-Forward with Post-LN
        #   Step 1: Apply feed-forward
        #   Step 2: Apply dropout
        #   Step 3: Residual connection and normalize: x = LayerNorm(x + dropout(ffn_out))

        pass  # Remove this and add your implementation


# ==============================================================================
# EXERCISE 7: Stack Multiple Transformer Blocks
# Difficulty: Medium
# ==============================================================================

class Exercise07_StackedTransformer(nn.Module):
    """
    Stack multiple transformer blocks to create a deep model.

    Deep transformers enable:
    - Hierarchical feature learning
    - More complex transformations
    - Better performance (with proper training)

    Architecture:
        Input → Block₁ → Block₂ → ... → Blockₙ → Output

    Common configurations:
        - Small: 6 layers (BERT-base, GPT-2 small)
        - Large: 12 layers (GPT-2 medium)
        - XL: 24+ layers (BERT-large, GPT-2 large)

    Args:
        n_layers: Number of transformer blocks to stack
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        use_pre_ln: If True, use Pre-LN; else Post-LN

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> model = Exercise07_StackedTransformer(
        ...     n_layers=12, d_model=768, n_heads=12, d_ff=3072
        ... )
        >>> x = torch.randn(32, 128, 768)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 128, 768])

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

        # TODO: Create a list/ModuleList of transformer blocks
        # Use nn.ModuleList to properly register all parameters
        # Choose Pre-LN or Post-LN based on use_pre_ln flag

        # Hint: Consider adding a final LayerNorm for Pre-LN architecture
        # (to normalize the final output)

        pass  # Remove this and add your implementation

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through stacked transformer blocks.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Pass input through each layer sequentially
        # For Pre-LN, consider applying final LayerNorm at the end

        pass  # Remove this and add your implementation


# ==============================================================================
# EXERCISE 8: Debug Gradient Flow
# Difficulty: Hard
# ==============================================================================

def exercise_08_debug_gradient_flow(
    model: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Debug gradient flow through a transformer model.

    This exercise helps you understand how gradients flow through the network
    and identify potential issues like vanishing or exploding gradients.

    Args:
        model: A transformer model (block or stacked blocks)
        x: Input tensor with requires_grad=True
        mask: Optional attention mask

    Returns:
        gradient_stats: Dictionary containing:
            - 'input_grad_norm': Gradient norm at input
            - 'layer_grad_norms': List of gradient norms for each layer
            - 'max_grad': Maximum gradient value
            - 'min_grad': Minimum gradient value (excluding zeros)
            - 'has_nan': Whether any gradients are NaN
            - 'has_inf': Whether any gradients are infinite

    Example:
        >>> from tiny_transformer.transformer_block import TransformerBlock
        >>> block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        >>> x = torch.randn(16, 32, 256, requires_grad=True)
        >>> stats = exercise_08_debug_gradient_flow(block, x)
        >>> print(f"Input gradient norm: {stats['input_grad_norm']:.6f}")
        >>> print(f"Has NaN: {stats['has_nan']}")

    Steps:
        1. Forward pass
        2. Compute loss (e.g., sum of outputs)
        3. Backward pass
        4. Collect gradient statistics

    Theory reference: theory.md, Section "Common Mistakes"
    """
    # TODO: Implement gradient flow debugging
    # Step 1: Forward pass
    # Step 2: Create a simple loss (e.g., output.sum())
    # Step 3: Backward pass (loss.backward())
    # Step 4: Collect statistics:
    #   - Input gradient norm
    #   - Gradient norms for each parameter
    #   - Check for NaN/Inf
    #   - Max/min gradient values

    pass  # Remove this and add your implementation


# ==============================================================================
# EXERCISE 9: Analyze Attention Patterns Through Blocks
# Difficulty: Medium
# ==============================================================================

def exercise_09_analyze_attention_patterns(
    model: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Analyze how attention patterns change through transformer layers.

    This helps understand:
    - What different layers focus on
    - How attention evolves through depth
    - Whether the model is learning useful patterns

    Args:
        model: Stacked transformer model
        x: Input tensor of shape (batch_size, seq_len, d_model)
        mask: Optional attention mask

    Returns:
        attention_info: Dictionary containing:
            - 'attention_weights': List of attention weights per layer
              Each entry has shape (batch_size, n_heads, seq_len, seq_len)
            - 'attention_entropy': Entropy of attention distributions per layer
              High entropy = diffuse attention, Low entropy = focused attention
            - 'top_attended_positions': Most attended positions per layer

    Example:
        >>> model = Exercise07_StackedTransformer(n_layers=6, d_model=256, ...)
        >>> x = torch.randn(4, 32, 256)
        >>> info = exercise_09_analyze_attention_patterns(model, x)
        >>> print(f"Number of layers: {len(info['attention_weights'])}")
        >>> print(f"Attention entropy in layer 0: {info['attention_entropy'][0]:.3f}")

    Hints:
        - You'll need to modify forward pass to return attention weights
        - Entropy = -Σ(p * log(p)) for probability distribution p
        - Use torch.distributions.Categorical for entropy computation

    Theory reference: theory.md, Section "Complete Transformer Block"
    """
    # TODO: Implement attention pattern analysis
    # This is challenging because you need to:
    # 1. Extract attention weights from each layer
    # 2. Compute entropy of attention distributions
    # 3. Identify most attended positions

    # Note: You may need to modify the forward pass or add hooks
    # to extract intermediate attention weights

    pass  # Remove this and add your implementation


# ==============================================================================
# EXERCISE 10: Count Parameters in Transformer Block
# Difficulty: Easy
# ==============================================================================

def exercise_10_count_parameters(
    d_model: int,
    n_heads: int,
    d_ff: int
) -> dict:
    """
    Count parameters in a transformer block.

    Understanding parameter distribution helps with:
    - Model sizing
    - Memory estimation
    - Identifying computational bottlenecks

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension

    Returns:
        param_counts: Dictionary containing:
            - 'attention_params': Parameters in multi-head attention
            - 'ffn_params': Parameters in feed-forward network
            - 'layernorm_params': Parameters in layer normalizations
            - 'total_params': Total parameters in one block
            - 'attention_breakdown': Detailed breakdown of attention params
              (W_q, W_k, W_v, W_o)
            - 'ffn_breakdown': Detailed breakdown of FFN params
              (W1, b1, W2, b2)

    Example:
        >>> counts = exercise_10_count_parameters(d_model=768, n_heads=12, d_ff=3072)
        >>> print(f"Total parameters: {counts['total_params']:,}")
        >>> print(f"Attention: {counts['attention_params']:,}")
        >>> print(f"FFN: {counts['ffn_params']:,}")

    Formulas:
        Multi-Head Attention:
          W_q: d_model × d_model
          W_k: d_model × d_model
          W_v: d_model × d_model
          W_o: d_model × d_model
          Total: 4 × d_model²

        Feed-Forward:
          W1: d_model × d_ff
          b1: d_ff
          W2: d_ff × d_model
          b2: d_model
          Total: 2 × (d_model × d_ff) + d_ff + d_model

        LayerNorm (×2):
          gamma: d_model
          beta: d_model
          Total: 2 × 2 × d_model = 4 × d_model

    Theory reference: theory.md, Section "Parameter Count"
    """
    # TODO: Calculate parameter counts
    # Multi-head attention parameters

    # Feed-forward network parameters

    # Layer normalization parameters (2 LayerNorms)

    # Create detailed breakdown

    pass  # Remove this and add your implementation

    # Uncomment to test:
    # # Test with GPT-2 dimensions
    # counts = exercise_10_count_parameters(d_model=768, n_heads=12, d_ff=3072)
    # assert counts['total_params'] == 7087872  # ~7.1M
    # assert counts['attention_params'] == 2359296  # ~2.4M
    # assert counts['ffn_params'] == 4722432  # ~4.7M


# ==============================================================================
# BONUS EXERCISE: Compare Pre-LN vs Post-LN Training Dynamics
# Difficulty: Advanced
# ==============================================================================

def bonus_compare_ln_architectures(
    n_layers: int = 6,
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 1024,
    n_steps: int = 100
) -> dict:
    """
    Compare Pre-LN vs Post-LN training stability.

    This demonstrates why Pre-LN is the modern standard:
    - More stable gradients
    - Faster convergence
    - No warm-up needed

    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_steps: Number of training steps to simulate

    Returns:
        comparison: Dictionary containing:
            - 'pre_ln_losses': Training losses for Pre-LN
            - 'post_ln_losses': Training losses for Post-LN
            - 'pre_ln_grad_norms': Gradient norms for Pre-LN
            - 'post_ln_grad_norms': Gradient norms for Post-LN
            - 'pre_ln_stable': Whether Pre-LN training was stable
            - 'post_ln_stable': Whether Post-LN training was stable

    Example:
        >>> results = bonus_compare_ln_architectures(n_layers=12)
        >>> print(f"Pre-LN stable: {results['pre_ln_stable']}")
        >>> print(f"Post-LN stable: {results['post_ln_stable']}")
        >>> # Plot losses
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results['pre_ln_losses'], label='Pre-LN')
        >>> plt.plot(results['post_ln_losses'], label='Post-LN')
        >>> plt.legend()
        >>> plt.show()

    This is a research-oriented exercise. Feel free to experiment!

    Theory reference: theory.md, Section "Pre-LN vs Post-LN Architecture"
    """
    # TODO: Implement training comparison
    # 1. Create two models (Pre-LN and Post-LN)
    # 2. Create dummy data and optimizer
    # 3. Run training loop for n_steps
    # 4. Track losses and gradient norms
    # 5. Determine stability (no NaN/Inf, reasonable gradient norms)

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
    # print("\nExercise 1: Basic Feed-Forward Network")
    # ffn = Exercise01_FeedForward(d_model=512, d_ff=2048)
    # x = torch.randn(32, 128, 512)
    # output = ffn(x)
    # assert output.shape == x.shape
    # print("✓ Exercise 1 passed!")

    # Test Exercise 2
    # print("\nExercise 2: Feed-Forward with GELU")
    # ffn = Exercise02_FeedForwardGELU(d_model=768, d_ff=3072, dropout=0.1)
    # x = torch.randn(32, 128, 768)
    # output = ffn(x)
    # assert output.shape == x.shape
    # print("✓ Exercise 2 passed!")

    # Test Exercise 3
    # print("\nExercise 3: Layer Normalization")
    # ln = Exercise03_LayerNorm(d_model=768)
    # x = torch.randn(32, 128, 768)
    # output = ln(x)
    # assert output.shape == x.shape
    # # Check normalization
    # mean = output[0, 0].mean().item()
    # std = output[0, 0].std().item()
    # assert abs(mean) < 0.01, f"Mean should be ≈ 0, got {mean}"
    # assert abs(std - 1.0) < 0.1, f"Std should be ≈ 1, got {std}"
    # print("✓ Exercise 3 passed!")

    # Test Exercise 4
    # print("\nExercise 4: Residual Connection")
    # x = torch.randn(32, 128, 768)
    # sublayer_out = torch.randn(32, 128, 768)
    # output = exercise_04_residual_connection(x, sublayer_out)
    # assert output.shape == x.shape
    # assert torch.allclose(output, x + sublayer_out)
    # print("✓ Exercise 4 passed!")

    # Test Exercise 5
    # print("\nExercise 5: Pre-LN Transformer Block")
    # block = Exercise05_TransformerBlockPreLN(
    #     d_model=256, n_heads=4, d_ff=1024, dropout=0.1
    # )
    # x = torch.randn(16, 32, 256)
    # output = block(x)
    # assert output.shape == x.shape
    # print("✓ Exercise 5 passed!")

    # Test Exercise 6
    # print("\nExercise 6: Post-LN Transformer Block")
    # block = Exercise06_TransformerBlockPostLN(
    #     d_model=256, n_heads=4, d_ff=1024, dropout=0.1
    # )
    # x = torch.randn(16, 32, 256)
    # output = block(x)
    # assert output.shape == x.shape
    # print("✓ Exercise 6 passed!")

    # Test Exercise 7
    # print("\nExercise 7: Stacked Transformer")
    # model = Exercise07_StackedTransformer(
    #     n_layers=6, d_model=256, n_heads=4, d_ff=1024
    # )
    # x = torch.randn(8, 32, 256)
    # output = model(x)
    # assert output.shape == x.shape
    # print("✓ Exercise 7 passed!")

    # Test Exercise 10
    # print("\nExercise 10: Count Parameters")
    # counts = exercise_10_count_parameters(d_model=768, n_heads=12, d_ff=3072)
    # assert 'total_params' in counts
    # assert counts['total_params'] > 0
    # print(f"  Total parameters: {counts['total_params']:,}")
    # print("✓ Exercise 10 passed!")

    print("\n" + "=" * 70)
    print("All tests passed! Great work!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Transformer Block - Practice Exercises")
    print("=" * 70)
    print("\nWelcome! These exercises will help you master transformer blocks.")
    print("\nInstructions:")
    print("1. Work through exercises 1-10 in order")
    print("2. Read each docstring carefully")
    print("3. Implement the TODO sections")
    print("4. Uncomment the assertions to test your code")
    print("5. Run this file to see if your solutions work")
    print("\nTips:")
    print("- Use print() to debug tensor shapes")
    print("- Refer to tiny_transformer/ implementations for guidance")
    print("- Read theory.md sections referenced in docstrings")
    print("- Solutions are in solutions.py (try not to peek!)")
    print("=" * 70)

    # Uncomment this when you've completed some exercises:
    # run_all_tests()

    # Example: Test Exercise 1 individually
    print("\nExample: Testing Exercise 1")
    print("-" * 70)
    print("After implementing Exercise01_FeedForward, uncomment this:")
    print("# ffn = Exercise01_FeedForward(d_model=512, d_ff=2048)")
    print("# x = torch.randn(32, 128, 512)")
    print("# output = ffn(x)")
    print("# print(f'Input shape: {x.shape}')")
    print("# print(f'Output shape: {output.shape}')")
    print("# assert output.shape == x.shape")
    print("# print('✓ Exercise 1 working!')")

    print("\nGood luck! 🚀")
