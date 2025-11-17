"""
Module 05: Full Model (TinyTransformerLM) - Solutions

This file contains complete, working solutions for all Module 05 exercises.
Each solution includes:
- Full implementation with detailed comments
- Explanation of design choices
- Performance notes
- Links to theory sections
- Visualization helpers where appropriate

Use these solutions to:
- Check your work after attempting exercises
- Learn best practices and patterns
- Understand implementation details
- Debug your own solutions

Try to solve exercises yourself first before looking at solutions!

Theory reference: /Users/shiongtan/projects/tiny-transformer-build/docs/modules/05_full_model/theory.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


# ==============================================================================
# SOLUTION 1: Language Modeling Head
# ==============================================================================

class Solution01_LanguageModelingHead(nn.Module):
    """
    SOLUTION 01: Language modeling head implementation.

    Design choices:
    - No bias: Standard practice for LM heads, especially with weight tying
    - Normal initialization (0.02 std): Matches GPT-2 initialization
    - Simple linear projection: Logits go directly to loss function

    The LM head converts hidden states to vocabulary logits. No activation
    function is applied because cross-entropy loss expects raw logits.

    Implementation notes:
    - We use bias=False to save parameters and improve numerical stability
    - The std=0.02 initialization is empirically proven for transformers
    - This layer can share weights with token embedding (weight tying)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Linear projection: d_model → vocab_size
        # No bias because it's redundant with weight tying and saves memory
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights with small normal distribution
        # This prevents extreme initial logits and helps training stability
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            x: Hidden states of shape (batch_size, seq_len, d_model)

        Returns:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)

        Shape flow:
            (B, T, d_model) → Linear → (B, T, vocab_size)

        Note: These are raw logits, not probabilities. Use with cross_entropy loss.
        """
        # Simple linear projection
        # Each position independently projects to vocab space
        logits = self.projection(x)

        return logits


# ==============================================================================
# SOLUTION 2: Manual Model Stacking
# ==============================================================================

def solution_02_manual_model_stacking(
    vocab_size: int = 1000,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    batch_size: int = 8,
    seq_len: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SOLUTION 02: Manually stack all components.

    This demonstrates the complete forward pass by explicitly creating
    and connecting each component. Understanding this helps you see
    how the complete model works before using the integrated version.

    Implementation notes:
    - We import components from tiny_transformer
    - Each component maintains shape (batch_size, seq_len, d_model)
    - Final layer norm stabilizes representations before projection
    - No causal mask needed for this demo (but required for training)
    """
    from tiny_transformer.embeddings import TransformerEmbedding
    from tiny_transformer.transformer_block import TransformerBlock

    # Step 1: Create input tokens
    # Random integers in range [0, vocab_size) simulate real token IDs
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {tokens.shape}")

    # Step 2: Create and apply embedding layer
    # Combines token embeddings + positional encoding + dropout
    embedding = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=seq_len,
        positional='sinusoidal',  # Use sinusoidal (no learned params)
        dropout=0.1
    )

    # (batch_size, seq_len) → (batch_size, seq_len, d_model)
    x = embedding(tokens)
    print(f"After embedding: {x.shape}")

    # Step 3: Apply transformer blocks sequentially
    # Each block: self-attention + feed-forward + residuals + layer norms
    blocks = nn.ModuleList([
        TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.1
        )
        for _ in range(n_layers)
    ])

    for i, block in enumerate(blocks):
        # Each block maintains shape: (batch_size, seq_len, d_model)
        x, _ = block(x)
        print(f"After block {i}: {x.shape}")

    # Step 4: Final layer normalization
    # Pre-LN architecture needs normalization after last block
    ln_f = nn.LayerNorm(d_model)
    x = ln_f(x)
    print(f"After final LayerNorm: {x.shape}")

    # Step 5: Language modeling head
    # Projects d_model → vocab_size
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)

    # (batch_size, seq_len, d_model) → (batch_size, seq_len, vocab_size)
    logits = lm_head(x)
    print(f"After LM head: {logits.shape}")

    # Verify shapes
    assert tokens.shape == (batch_size, seq_len)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    return tokens, logits


# ==============================================================================
# SOLUTION 3: Weight Tying Implementation
# ==============================================================================

def solution_03_weight_tying(
    vocab_size: int = 5000,
    d_model: int = 256
) -> Tuple[bool, int]:
    """
    SOLUTION 03: Weight tying demonstration.

    Weight tying shares the embedding matrix with the output projection.
    This is done by making lm_head.weight a reference (not copy) to
    the embedding weight tensor.

    Implementation notes:
    - Use assignment (=) not clone() to create reference
    - Verify with 'is' operator (object identity)
    - Check data_ptr() to confirm same memory location
    - Parameters saved = vocab_size × d_model

    Why it works:
    - Input: Look up row i of embedding matrix
    - Output: Dot product with each row of same matrix
    - Semantically consistent and parameter-efficient
    """

    # Step 1: Create token embedding layer
    token_embedding = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(token_embedding.weight, mean=0.0, std=0.02)

    print(f"Token embedding parameters: {token_embedding.weight.numel():,}")

    # Step 2: Create LM head (output projection)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)

    print(f"LM head parameters (before tying): {lm_head.weight.numel():,}")

    # Step 3: Count parameters BEFORE tying
    params_before = (
        token_embedding.weight.numel() +
        lm_head.weight.numel()
    )
    print(f"Total parameters before tying: {params_before:,}")

    # Step 4: Tie weights
    # CRITICAL: Use assignment to create reference, NOT clone()
    # This makes lm_head.weight point to the same tensor as token_embedding.weight
    lm_head.weight = token_embedding.weight

    print("Weight tying applied!")

    # Step 5: Verify they're actually shared (same object)
    weights_are_shared = lm_head.weight is token_embedding.weight
    print(f"Weights are same object: {weights_are_shared}")

    # Also check memory address
    same_memory = (
        lm_head.weight.data_ptr() == token_embedding.weight.data_ptr()
    )
    print(f"Weights at same memory address: {same_memory}")

    # Step 6: Count unique parameters AFTER tying
    # Now we only count the shared weight once
    # We can't just sum .numel() because that would double-count
    unique_params = {id(p): p.numel() for p in [token_embedding.weight, lm_head.weight]}
    params_after = sum(unique_params.values())

    print(f"Total parameters after tying: {params_after:,}")

    # Step 7: Calculate savings
    params_saved = params_before - params_after
    print(f"Parameters saved: {params_saved:,}")

    # Expected savings: exactly vocab_size × d_model
    expected_savings = vocab_size * d_model
    assert params_saved == expected_savings, \
        f"Expected to save {expected_savings:,}, but saved {params_saved:,}"

    return weights_are_shared, params_saved


# ==============================================================================
# SOLUTION 4: Parameter Counting with Weight Tying
# ==============================================================================

def solution_04_parameter_counting(
    vocab_size: int = 10000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    max_len: int = 1024,
    tie_weights: bool = True
) -> Dict[str, int]:
    """
    SOLUTION 04: Complete parameter counting.

    This implements the analytical formulas for counting parameters
    in each component. Understanding this helps you estimate model
    size before training and optimize configurations.

    Formulas (from theory.md):
    - Token embedding: vocab_size × d_model
    - Positional (learned): max_len × d_model
    - Attention per layer: 4 × d_model² + 4 × d_model
    - FFN per layer: 2 × d_model × d_ff + d_ff + d_model
    - LayerNorm per layer (×2): 4 × d_model
    - Final LayerNorm: 2 × d_model
    - LM head: d_model × vocab_size (or 0 if tied)

    Implementation notes:
    - We count each component separately for clarity
    - Weight tying subtracts the shared embedding from total
    - We return detailed breakdown for analysis
    """

    # Step 1: Token embedding parameters
    # Each token gets a d_model-dimensional vector
    token_emb_params = vocab_size * d_model
    print(f"Token embedding: {token_emb_params:,}")

    # Step 2: Positional embedding parameters (assuming learned, not sinusoidal)
    # Each position up to max_len gets a d_model-dimensional vector
    # Note: Sinusoidal has 0 parameters (formula-based)
    pos_emb_params = max_len * d_model
    print(f"Positional embedding: {pos_emb_params:,}")

    # Step 3: Single transformer block parameters

    # Multi-head attention:
    # - Q, K, V, O projections: each is d_model × d_model
    # - Each projection has bias: d_model
    # - Total: 4 × (d_model × d_model + d_model)
    attention_weights = 4 * d_model * d_model  # 4 weight matrices
    attention_biases = 4 * d_model             # 4 bias vectors
    attention_params = attention_weights + attention_biases
    print(f"Attention per block: {attention_params:,}")

    # Feed-forward network:
    # - fc1: d_model → d_ff (weight + bias)
    # - fc2: d_ff → d_model (weight + bias)
    ffn_fc1 = d_model * d_ff + d_ff    # W1 + b1
    ffn_fc2 = d_ff * d_model + d_model  # W2 + b2
    ffn_params = ffn_fc1 + ffn_fc2
    print(f"FFN per block: {ffn_params:,}")

    # Layer normalization (2 per block):
    # Each LayerNorm has gamma (scale) and beta (shift)
    # Total: 2 × (2 × d_model) = 4 × d_model
    ln_params_per_block = 4 * d_model
    print(f"LayerNorm per block: {ln_params_per_block:,}")

    # Total per block
    block_params = attention_params + ffn_params + ln_params_per_block
    print(f"Total per block: {block_params:,}")

    # Step 4: All transformer blocks
    all_blocks_params = n_layers * block_params
    print(f"All {n_layers} blocks: {all_blocks_params:,}")

    # Step 5: Final layer normalization
    # One final LayerNorm after all blocks (Pre-LN style)
    final_ln_params = 2 * d_model  # gamma + beta
    print(f"Final LayerNorm: {final_ln_params:,}")

    # Step 6: Language modeling head
    # Linear projection: d_model → vocab_size (no bias)
    lm_head_params = d_model * vocab_size
    print(f"LM head (without tying): {lm_head_params:,}")

    # Step 7: Calculate totals
    total_without_tying = (
        token_emb_params +
        pos_emb_params +
        all_blocks_params +
        final_ln_params +
        lm_head_params
    )
    print(f"\nTotal without weight tying: {total_without_tying:,}")

    # If weight tying, embedding and LM head share weights
    # So we subtract the LM head parameters (they're the same as embedding)
    total_with_tying = total_without_tying - (
        vocab_size * d_model if tie_weights else 0
    )
    print(f"Total with weight tying: {total_with_tying:,}")

    # Calculate savings
    savings = total_without_tying - total_with_tying
    savings_percent = (savings / total_without_tying) * 100
    print(f"Savings from tying: {savings:,} ({savings_percent:.1f}%)")

    # Step 8: Return detailed breakdown
    return {
        'token_embedding': token_emb_params,
        'positional_embedding': pos_emb_params,
        'transformer_blocks': all_blocks_params,
        'attention_per_block': attention_params,
        'ffn_per_block': ffn_params,
        'ln_per_block': ln_params_per_block,
        'final_ln': final_ln_params,
        'lm_head': lm_head_params if not tie_weights else 0,
        'total': total_without_tying,
        'total_with_tying': total_with_tying,
        'savings_from_tying': savings,
        'savings_percent': savings_percent
    }


# ==============================================================================
# SOLUTION 5: Basic Causal Generation
# ==============================================================================

@torch.no_grad()
def solution_05_causal_generation(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    SOLUTION 05: Autoregressive text generation.

    This implements the basic generation loop used by GPT models:
    1. Forward pass to get logits
    2. Extract logits for last position
    3. Apply temperature scaling
    4. Sample next token
    5. Append and repeat

    Implementation notes:
    - @torch.no_grad() disables gradient computation (faster, less memory)
    - Temperature controls randomness: <1 = more deterministic, >1 = more random
    - We use multinomial sampling for diversity (not greedy argmax)
    - Each iteration appends one token to the sequence

    Temperature effects:
    - temperature = 0.1: Very focused, repetitive (near-greedy)
    - temperature = 1.0: Normal sampling from model distribution
    - temperature = 2.0: More diverse, potentially incoherent

    Performance notes:
    - This is O(T²) in sequence length (recomputes attention for all positions)
    - KV-caching can reduce to O(T) but not implemented here
    - See Module 07 for advanced sampling strategies
    """

    # Set model to evaluation mode
    # This disables dropout and uses deterministic behavior
    model.eval()

    # Clone to avoid modifying input
    current = start_tokens.clone()

    print(f"Starting with {current.size(1)} tokens")

    # Generation loop
    for step in range(max_new_tokens):
        # Step 1: Forward pass
        # Get logits for all positions
        # Shape: (batch_size, current_seq_len, vocab_size)
        logits, _ = model(current)

        # Step 2: Extract logits for last position only
        # We only care about predicting the next token
        # Shape: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Step 3: Apply temperature scaling
        # Temperature < 1: Sharpen distribution (more confident)
        # Temperature > 1: Flatten distribution (more random)
        # Temperature = 1: No change
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Step 4: Convert logits to probabilities
        # Softmax normalizes across vocabulary dimension
        probs = F.softmax(next_token_logits, dim=-1)

        # Step 5: Sample next token from distribution
        # multinomial samples according to probabilities
        # Shape: (batch_size, 1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Step 6: Append to sequence
        # Concatenate along sequence dimension
        # current goes from (B, T) to (B, T+1)
        current = torch.cat([current, next_token], dim=1)

        # Optional: Print progress
        if (step + 1) % 5 == 0:
            print(f"Generated {step + 1}/{max_new_tokens} tokens")

    print(f"Final sequence length: {current.size(1)}")

    return current


# ==============================================================================
# SOLUTION 6: Hidden State Extraction
# ==============================================================================

def solution_06_hidden_state_extraction(
    vocab_size: int = 1000,
    d_model: int = 256,
    n_layers: int = 4
) -> Tuple[List[torch.Tensor], Dict[str, any]]:
    """
    SOLUTION 06: Extract and analyze hidden states.

    This demonstrates how to collect intermediate representations
    and analyze how they evolve through the network. Useful for:
    - Understanding what each layer learns
    - Debugging training issues
    - Visualization and interpretation

    Implementation notes:
    - We use return_hidden_states=True to collect activations
    - Cosine similarity measures how much representations change
    - Norms indicate representation magnitude
    - High similarity might indicate vanishing gradients or redundancy

    Analysis insights:
    - Early layers: Often learn syntax and local patterns
    - Middle layers: Learn semantic relationships
    - Late layers: Task-specific features
    - Norms should remain stable (not explode or vanish)
    """
    from tiny_transformer.model import TinyTransformerLM

    # Create small model for testing
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_model * 4,
        tie_weights=True
    )

    # Create random input tokens
    batch_size, seq_len = 4, 32
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {tokens.shape}")

    # Forward pass with hidden state collection
    # return_hidden_states=True returns list of tensors
    logits, hidden_states = model(tokens, return_hidden_states=True)

    print(f"Collected {len(hidden_states)} hidden states")
    print(f"Each state shape: {hidden_states[0].shape}")

    # Analyze hidden states

    # 1. Calculate norm for each layer
    # Norm indicates overall magnitude of representations
    layer_norms = []
    for i, h in enumerate(hidden_states):
        # Average norm across batch and sequence
        avg_norm = h.norm(dim=-1).mean().item()
        layer_norms.append(avg_norm)
        print(f"Layer {i} average norm: {avg_norm:.4f}")

    # 2. Calculate cosine similarity between consecutive layers
    # This tells us how much each layer changes the representation
    def cosine_similarity_layers(h1, h2):
        """Compute average cosine similarity between two hidden states."""
        # Flatten batch and sequence dimensions
        h1_flat = h1.reshape(-1, h1.size(-1))  # (B*T, d_model)
        h2_flat = h2.reshape(-1, h2.size(-1))  # (B*T, d_model)

        # Compute cosine similarity for each position
        # F.cosine_similarity computes along dim=-1
        cos_sim = F.cosine_similarity(h1_flat, h2_flat, dim=-1)

        # Average across all positions
        return cos_sim.mean().item()

    layer_similarities = []
    for i in range(len(hidden_states) - 1):
        sim = cosine_similarity_layers(hidden_states[i], hidden_states[i+1])
        layer_similarities.append(sim)
        print(f"Similarity between layer {i} and {i+1}: {sim:.4f}")

    # 3. First to last similarity
    # Measures total transformation through the network
    first_to_last_sim = cosine_similarity_layers(
        hidden_states[0],
        hidden_states[-1]
    )
    print(f"First to last similarity: {first_to_last_sim:.4f}")

    # 4. Additional analysis: Variance per layer
    layer_variances = []
    for i, h in enumerate(hidden_states):
        var = h.var().item()
        layer_variances.append(var)
        print(f"Layer {i} variance: {var:.4f}")

    # Compile analysis results
    analysis = {
        'layer_norms': layer_norms,
        'layer_similarities': layer_similarities,
        'first_to_last_similarity': first_to_last_sim,
        'layer_variances': layer_variances,
        'num_layers': len(hidden_states)
    }

    return hidden_states, analysis


# ==============================================================================
# SOLUTION 7: Model Configuration System
# ==============================================================================

class Solution07_ModelConfig:
    """
    SOLUTION 07: Model configuration presets.

    This provides reusable configurations for different model sizes.
    Configurations follow established patterns from GPT/BERT literature.

    Design principles:
    - d_ff = 4 × d_model (standard ratio)
    - n_heads must divide d_model evenly
    - max_len scales with model size
    - dropout = 0.1 is standard

    Configurations roughly target:
    - tiny: ~1-2M params (quick testing)
    - small: ~10M params (prototyping)
    - medium: ~50M params (small-scale training)
    - large: ~100M+ params (research)
    """

    @staticmethod
    def get_config(size: str = "tiny") -> Dict[str, any]:
        """
        Get preset model configuration.

        Args:
            size: One of "tiny", "small", "medium", "large"

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If size is not recognized
        """

        configs = {
            # Tiny: For rapid iteration and testing
            # ~1-2M parameters with 1000 vocab
            "tiny": {
                "d_model": 128,
                "n_heads": 4,        # 128 / 4 = 32 dims per head
                "n_layers": 2,
                "d_ff": 512,         # 4 × d_model
                "max_len": 256,
                "dropout": 0.1,
            },

            # Small: For prototyping and debugging
            # ~5-10M parameters with 5000 vocab
            "small": {
                "d_model": 256,
                "n_heads": 8,        # 256 / 8 = 32 dims per head
                "n_layers": 4,
                "d_ff": 1024,        # 4 × d_model
                "max_len": 512,
                "dropout": 0.1,
            },

            # Medium: For small-scale training
            # ~30-50M parameters with 10000 vocab
            "medium": {
                "d_model": 512,
                "n_heads": 8,        # 512 / 8 = 64 dims per head
                "n_layers": 6,
                "d_ff": 2048,        # 4 × d_model
                "max_len": 1024,
                "dropout": 0.1,
            },

            # Large: Similar to GPT-2 small / BERT-base
            # ~100M+ parameters with 30000 vocab
            "large": {
                "d_model": 768,
                "n_heads": 12,       # 768 / 12 = 64 dims per head
                "n_layers": 12,
                "d_ff": 3072,        # 4 × d_model
                "max_len": 2048,
                "dropout": 0.1,
            },
        }

        if size not in configs:
            available = list(configs.keys())
            raise ValueError(
                f"Unknown size '{size}'. Choose from: {available}"
            )

        config = configs[size]
        print(f"Configuration '{size}':")
        for key, value in config.items():
            print(f"  {key}: {value}")

        return config

    @staticmethod
    def estimate_parameters(config: Dict[str, any], vocab_size: int) -> int:
        """
        Estimate total parameters from config.

        Args:
            config: Model configuration dictionary
            vocab_size: Vocabulary size

        Returns:
            Estimated total parameters (with weight tying)
        """
        # Use the formula from solution_04
        result = solution_04_parameter_counting(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_len=config['max_len'],
            tie_weights=True
        )

        total = result['total_with_tying']
        print(f"\nEstimated parameters: {total:,}")
        return total

    @staticmethod
    def create_model_from_config(
        size: str,
        vocab_size: int
    ) -> 'TinyTransformerLM':
        """
        Create a model from preset configuration.

        Args:
            size: Configuration size ("tiny", "small", etc.)
            vocab_size: Vocabulary size

        Returns:
            Initialized TinyTransformerLM model
        """
        from tiny_transformer.model import TinyTransformerLM

        config = Solution07_ModelConfig.get_config(size)
        estimated = Solution07_ModelConfig.estimate_parameters(config, vocab_size)

        model = TinyTransformerLM(
            vocab_size=vocab_size,
            **config,
            tie_weights=True
        )

        actual = model.count_parameters()
        print(f"Actual parameters: {actual:,}")
        print(f"Estimate accuracy: {(actual/estimated)*100:.1f}%")

        return model


# ==============================================================================
# SOLUTION 8: Model Scaling Analysis
# ==============================================================================

def solution_08_model_scaling_analysis() -> Dict[str, any]:
    """
    SOLUTION 08: Analyze parameter scaling with different dimensions.

    This systematically varies each hyperparameter to understand
    how parameter count scales. Key insights:

    - d_model: Quadratic growth (appears in attention: 4×d²)
    - n_layers: Linear growth (each layer adds same amount)
    - vocab_size: Linear growth (with tying, otherwise quadratic in some components)
    - d_ff: Linear growth within layers

    Understanding scaling helps you:
    - Choose appropriate configurations for compute budget
    - Understand where parameters are concentrated
    - Make informed trade-offs (depth vs width)
    """

    # Base configuration for comparison
    base_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'vocab_size': 10000,
        'max_len': 512
    }

    print("Base configuration:")
    for k, v in base_config.items():
        print(f"  {k}: {v}")

    # Helper function to count parameters
    def count_params(config):
        result = solution_04_parameter_counting(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_len=config['max_len'],
            tie_weights=True
        )
        return result['total_with_tying']

    # === EXPERIMENT 1: Vary d_model ===
    print("\n" + "="*60)
    print("EXPERIMENT 1: Varying d_model (keeping others constant)")
    print("="*60)

    d_model_values = [128, 256, 512, 768, 1024]
    d_model_params = []

    for d in d_model_values:
        config = base_config.copy()
        config['d_model'] = d
        config['d_ff'] = d * 4  # Maintain ratio
        config['n_heads'] = min(8, d // 32)  # Adjust heads to fit

        params = count_params(config)
        d_model_params.append(params)

        print(f"d_model={d:4d}: {params:>12,} params")

    # Analyze growth rate
    print("\nGrowth analysis:")
    for i in range(1, len(d_model_values)):
        ratio = d_model_params[i] / d_model_params[i-1]
        dim_ratio = d_model_values[i] / d_model_values[i-1]
        print(f"  {d_model_values[i-1]}→{d_model_values[i]}: "
              f"{ratio:.2f}x params (dim ratio: {dim_ratio:.2f}x)")

    # === EXPERIMENT 2: Vary n_layers ===
    print("\n" + "="*60)
    print("EXPERIMENT 2: Varying n_layers (keeping others constant)")
    print("="*60)

    n_layers_values = [2, 4, 6, 8, 12]
    n_layers_params = []

    for n in n_layers_values:
        config = base_config.copy()
        config['n_layers'] = n

        params = count_params(config)
        n_layers_params.append(params)

        print(f"n_layers={n:2d}: {params:>12,} params")

    print("\nGrowth analysis:")
    for i in range(1, len(n_layers_values)):
        ratio = n_layers_params[i] / n_layers_params[i-1]
        layer_ratio = n_layers_values[i] / n_layers_values[i-1]
        print(f"  {n_layers_values[i-1]}→{n_layers_values[i]} layers: "
              f"{ratio:.2f}x params (layer ratio: {layer_ratio:.2f}x)")

    # === EXPERIMENT 3: Vary vocab_size ===
    print("\n" + "="*60)
    print("EXPERIMENT 3: Varying vocab_size (with weight tying)")
    print("="*60)

    vocab_values = [1000, 5000, 10000, 20000, 50000]
    vocab_params = []

    for v in vocab_values:
        config = base_config.copy()
        config['vocab_size'] = v

        params = count_params(config)
        vocab_params.append(params)

        print(f"vocab={v:5d}: {params:>12,} params")

    print("\nGrowth analysis:")
    for i in range(1, len(vocab_values)):
        ratio = vocab_params[i] / vocab_params[i-1]
        vocab_ratio = vocab_values[i] / vocab_values[i-1]
        print(f"  {vocab_values[i-1]}→{vocab_values[i]}: "
              f"{ratio:.2f}x params (vocab ratio: {vocab_ratio:.2f}x)")

    # === GENERATE OBSERVATIONS ===
    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)

    observations = {
        'd_model_growth': "Approximately quadratic (attention: 4×d²)",
        'n_layers_growth': "Linear (each layer adds constant amount)",
        'vocab_growth': "Linear with tying (only embedding affected)",
        'bottleneck': "d_model has strongest impact on parameter count",
        'recommendation': "Prefer depth over width for efficiency"
    }

    for key, obs in observations.items():
        print(f"{key}: {obs}")

    return {
        'd_model_values': d_model_values,
        'd_model_scaling': d_model_params,
        'n_layers_values': n_layers_values,
        'n_layers_scaling': n_layers_params,
        'vocab_values': vocab_values,
        'vocab_scaling': vocab_params,
        'observations': observations
    }


# ==============================================================================
# SOLUTION 9: Memory Estimation
# ==============================================================================

def solution_09_memory_estimation(
    vocab_size: int = 10000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    batch_size: int = 32,
    seq_len: int = 512
) -> Dict[str, float]:
    """
    SOLUTION 09: Estimate GPU memory requirements.

    Memory breakdown for training a transformer:

    1. Model parameters: Model weights
    2. Gradients: Same size as parameters
    3. Optimizer states: For Adam, 2× parameters (momentum + variance)
    4. Activations: Intermediate values saved for backward pass
    5. Attention matrices: O(n²) memory with sequence length

    Implementation notes:
    - All calculations assume float32 (4 bytes per value)
    - Activations are approximate (depends on implementation details)
    - Attention memory scales quadratically with seq_len
    - Actual memory may vary due to PyTorch internals

    Memory optimization strategies:
    - Gradient checkpointing: Recompute activations during backward
    - Mixed precision: Use float16 (2 bytes) for some operations
    - Gradient accumulation: Smaller batches, accumulate gradients
    - Reduce sequence length: Biggest impact on attention memory
    """

    print("="*60)
    print("MEMORY ESTIMATION")
    print("="*60)
    print(f"\nModel configuration:")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_layers: {n_layers}")
    print(f"\nTraining configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")

    # Step 1: Count model parameters
    param_counts = solution_04_parameter_counting(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        tie_weights=True
    )
    param_count = param_counts['total_with_tying']

    print(f"\nTotal parameters: {param_count:,}")

    # Step 2: Model parameters memory (float32 = 4 bytes)
    parameters_bytes = param_count * 4
    parameters_mb = parameters_bytes / (1024 * 1024)
    print(f"\n1. Model Parameters: {parameters_mb:.2f} MB")

    # Step 3: Gradients (same size as parameters)
    # During training, each parameter gets a gradient
    gradients_mb = parameters_mb
    print(f"2. Gradients: {gradients_mb:.2f} MB")

    # Step 4: Optimizer states (Adam)
    # Adam maintains:
    # - First moment (momentum): same size as parameters
    # - Second moment (variance): same size as parameters
    # Total: 2× parameters
    optimizer_states_mb = parameters_mb * 2
    print(f"3. Optimizer States (Adam): {optimizer_states_mb:.2f} MB")

    # Step 5: Activations
    # Activations are intermediate values saved during forward pass
    # for use in backward pass. Rough estimate:
    # - Each layer stores: batch × seq_len × d_model
    # - Multiple intermediate values per layer (~10 tensors)
    # - This is approximate; actual depends on implementation
    activations_elements = batch_size * seq_len * d_model * n_layers * 10
    activations_bytes = activations_elements * 4
    activations_mb = activations_bytes / (1024 * 1024)
    print(f"4. Activations (estimated): {activations_mb:.2f} MB")

    # Step 6: Attention matrices
    # Each attention layer stores:
    # - Attention weights: (batch, n_heads, seq_len, seq_len)
    # - This is stored for each layer
    # Total: batch × n_heads × seq_len² × n_layers
    attention_elements = batch_size * n_heads * seq_len * seq_len * n_layers
    attention_bytes = attention_elements * 4
    attention_mb = attention_bytes / (1024 * 1024)
    print(f"5. Attention Matrices: {attention_mb:.2f} MB")

    # Note: Attention memory is O(n²) with sequence length!
    print(f"\n   Note: Attention memory scales with seq_len²")
    print(f"   Current: {seq_len}² = {seq_len*seq_len:,} per head")

    # Step 7: Total memory
    total_mb = (
        parameters_mb +
        gradients_mb +
        optimizer_states_mb +
        activations_mb +
        attention_mb
    )
    total_gb = total_mb / 1024

    print(f"\n{'='*60}")
    print(f"TOTAL ESTIMATED MEMORY: {total_mb:.2f} MB = {total_gb:.2f} GB")
    print(f"{'='*60}")

    # Additional analysis: Memory breakdown percentages
    print(f"\nMemory breakdown:")
    components = {
        'Parameters': parameters_mb,
        'Gradients': gradients_mb,
        'Optimizer': optimizer_states_mb,
        'Activations': activations_mb,
        'Attention': attention_mb
    }

    for name, mb in components.items():
        pct = (mb / total_mb) * 100
        print(f"  {name:15s}: {mb:8.2f} MB ({pct:5.1f}%)")

    # Memory optimization suggestions
    print(f"\nOptimization suggestions:")
    if attention_mb / total_mb > 0.3:
        print("  • Attention uses >30% of memory. Consider:")
        print("    - Reduce sequence length")
        print("    - Use gradient checkpointing")
    if activations_mb / total_mb > 0.3:
        print("  • Activations use >30% of memory. Consider:")
        print("    - Gradient checkpointing (recompute activations)")
        print("    - Reduce batch size")
    if total_gb > 8:
        print("  • Total memory >8GB. Consider:")
        print("    - Mixed precision training (FP16)")
        print("    - Gradient accumulation")

    return {
        'parameters_mb': parameters_mb,
        'gradients_mb': gradients_mb,
        'optimizer_states_mb': optimizer_states_mb,
        'activations_mb': activations_mb,
        'attention_mb': attention_mb,
        'total_mb': total_mb,
        'total_gb': total_gb,
        'breakdown': components
    }


# ==============================================================================
# SOLUTION 10: End-to-End Training and Generation
# ==============================================================================

def solution_10_integration_test(
    vocab_size: int = 1000,
    n_steps: int = 100
) -> Dict[str, any]:
    """
    SOLUTION 10: Complete training and generation pipeline.

    This brings everything together:
    1. Model creation with proper config
    2. Training loop with loss computation
    3. Validation
    4. Text generation
    5. Analysis and visualization

    Implementation notes:
    - We use a small model for quick testing
    - Random data is sufficient for testing the pipeline
    - Real training would use actual text data and tokenizer
    - Loss should decrease (even with random data due to memorization)

    This demonstrates:
    - Complete forward pass (model(input))
    - Loss computation (cross_entropy)
    - Backward pass (loss.backward())
    - Optimization (optimizer.step())
    - Generation (@torch.no_grad())
    """
    from tiny_transformer.model import TinyTransformerLM

    print("="*70)
    print("INTEGRATION TEST: Full Training + Generation Pipeline")
    print("="*70)

    # Step 1: Create model with small config
    print("\n[Step 1] Creating model...")
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_len=256,
        tie_weights=True
    )

    param_count = model.count_parameters()
    print(f"Model created with {param_count:,} parameters")

    # Print parameter breakdown
    breakdown = model.get_parameter_breakdown()
    print("\nParameter breakdown:")
    for name, count in breakdown.items():
        if name != 'total':
            pct = (count / breakdown['total']) * 100
            print(f"  {name:20s}: {count:>10,} ({pct:>5.1f}%)")

    # Step 2: Create optimizer
    print("\n[Step 2] Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Using Adam optimizer with lr=1e-3")

    # Step 3: Training loop
    print(f"\n[Step 3] Training for {n_steps} steps...")
    train_losses = []
    model.train()

    batch_size = 8
    seq_len = 64

    for step in range(n_steps):
        # Generate dummy data (random tokens)
        # In real training, this would be actual text
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # For language modeling, target is input shifted by 1
        # Here we use random targets for simplicity
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, _ = model(input_ids)

        # Compute loss
        # Reshape: (B, T, V) → (B*T, V) and (B, T) → (B*T,)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Track loss
        train_losses.append(loss.item())

        # Print progress
        if (step + 1) % 20 == 0:
            avg_loss = sum(train_losses[-20:]) / 20
            print(f"  Step {step+1:3d}/{n_steps}: loss = {loss.item():.4f}, "
                  f"avg_loss = {avg_loss:.4f}")

    print(f"\nTraining complete!")
    print(f"  Initial loss: {train_losses[0]:.4f}")
    print(f"  Final loss: {train_losses[-1]:.4f}")
    print(f"  Improvement: {train_losses[0] - train_losses[-1]:.4f}")

    # Step 4: Validation
    print("\n[Step 4] Running validation...")
    model.eval()

    with torch.no_grad():
        val_input = torch.randint(0, vocab_size, (4, 64))
        val_target = torch.randint(0, vocab_size, (4, 64))

        val_logits, _ = model(val_input)
        val_loss = F.cross_entropy(
            val_logits.view(-1, vocab_size),
            val_target.view(-1)
        ).item()

    print(f"Validation loss: {val_loss:.4f}")

    # Step 5: Text generation
    print("\n[Step 5] Generating text...")

    # Create starting sequence
    start_tokens = torch.randint(0, vocab_size, (2, 10))
    print(f"Starting with {start_tokens.size(1)} tokens")

    # Generate with different temperatures
    temperatures = [0.5, 1.0, 2.0]
    generated_outputs = {}

    for temp in temperatures:
        generated = solution_05_causal_generation(
            model,
            start_tokens,
            max_new_tokens=20,
            temperature=temp
        )
        generated_outputs[temp] = generated
        print(f"  Temperature {temp}: generated {generated.size(1)} tokens")

    # Step 6: Analysis
    print("\n[Step 6] Analysis...")

    # Loss curve analysis
    print("\nLoss curve:")
    windows = [10, 20, 50]
    for w in windows:
        if len(train_losses) >= w:
            avg = sum(train_losses[-w:]) / w
            print(f"  Last {w:2d} steps average: {avg:.4f}")

    # Check if loss decreased
    loss_decreased = train_losses[-1] < train_losses[0]
    print(f"\nLoss decreased: {loss_decreased}")

    # Gradient check
    print("\nGradient statistics:")
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    if grad_norms:
        print(f"  Mean gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  Max gradient norm: {max(grad_norms):.6f}")
        print(f"  Min gradient norm: {min(grad_norms):.6f}")

    # Generation validation
    print("\nGeneration validation:")
    for temp, gen in generated_outputs.items():
        # Check all tokens are in valid range
        valid = (gen >= 0).all() and (gen < vocab_size).all()
        print(f"  Temperature {temp}: all tokens valid = {valid.item()}")

    # Step 7: Return results
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_loss': val_loss,
        'generated_tokens': generated_outputs[1.0],  # Return temp=1.0
        'all_generations': generated_outputs,
        'final_params': param_count,
        'loss_decreased': loss_decreased
    }

    print("\n" + "="*70)
    print("INTEGRATION TEST COMPLETE")
    print("="*70)

    return results


# ==============================================================================
# Additional Helper Functions
# ==============================================================================

def visualize_model_scaling():
    """
    Bonus: Visualize scaling analysis results.

    Creates plots showing how parameters scale with different dimensions.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Run scaling analysis
    results = solution_08_model_scaling_analysis()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: d_model scaling
    axes[0].plot(
        results['d_model_values'],
        [p / 1e6 for p in results['d_model_scaling']],
        'o-', linewidth=2, markersize=8
    )
    axes[0].set_xlabel('d_model', fontsize=12)
    axes[0].set_ylabel('Parameters (millions)', fontsize=12)
    axes[0].set_title('Parameter Scaling with d_model', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: n_layers scaling
    axes[1].plot(
        results['n_layers_values'],
        [p / 1e6 for p in results['n_layers_scaling']],
        'o-', linewidth=2, markersize=8, color='orange'
    )
    axes[1].set_xlabel('n_layers', fontsize=12)
    axes[1].set_ylabel('Parameters (millions)', fontsize=12)
    axes[1].set_title('Parameter Scaling with n_layers', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: vocab_size scaling
    axes[2].plot(
        results['vocab_values'],
        [p / 1e6 for p in results['vocab_scaling']],
        'o-', linewidth=2, markersize=8, color='green'
    )
    axes[2].set_xlabel('vocab_size', fontsize=12)
    axes[2].set_ylabel('Parameters (millions)', fontsize=12)
    axes[2].set_title('Parameter Scaling with vocab_size', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_scaling_analysis.png', dpi=150)
    print("\nVisualization saved to: model_scaling_analysis.png")
    plt.show()


def compare_weight_tying_impact():
    """
    Bonus: Compare models with and without weight tying.

    Shows parameter count and memory savings.
    """
    print("="*60)
    print("WEIGHT TYING IMPACT ANALYSIS")
    print("="*60)

    configs = [
        ("Small", 5000, 256, 4, 4),
        ("Medium", 10000, 512, 8, 6),
        ("Large", 30000, 768, 12, 12),
    ]

    for name, vocab, d_model, n_heads, n_layers in configs:
        print(f"\n{name} model (vocab={vocab:,}, d_model={d_model}):")

        # Without tying
        without = solution_04_parameter_counting(
            vocab_size=vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tie_weights=False
        )

        # With tying
        with_tying = solution_04_parameter_counting(
            vocab_size=vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tie_weights=True
        )

        savings = without['total'] - with_tying['total_with_tying']
        savings_pct = (savings / without['total']) * 100

        print(f"  Without tying: {without['total']:>12,} params")
        print(f"  With tying:    {with_tying['total_with_tying']:>12,} params")
        print(f"  Savings:       {savings:>12,} params ({savings_pct:.1f}%)")


# ==============================================================================
# Testing All Solutions
# ==============================================================================

def run_all_solution_tests():
    """
    Run all solution tests to verify correctness.
    """
    print("\n" + "="*70)
    print("RUNNING ALL SOLUTION TESTS")
    print("="*70)

    # Test Solution 1
    print("\n[TEST 1] Language Modeling Head")
    lm_head = Solution01_LanguageModelingHead(vocab_size=10000, d_model=512)
    x = torch.randn(32, 128, 512)
    logits = lm_head(x)
    assert logits.shape == (32, 128, 10000)
    print("✓ Solution 1 passed!")

    # Test Solution 2
    print("\n[TEST 2] Manual Model Stacking")
    tokens, logits = solution_02_manual_model_stacking()
    assert tokens.shape == (8, 32)
    assert logits.shape == (8, 32, 1000)
    print("✓ Solution 2 passed!")

    # Test Solution 3
    print("\n[TEST 3] Weight Tying")
    shared, saved = solution_03_weight_tying(vocab_size=5000, d_model=256)
    assert shared == True
    assert saved == 5000 * 256
    print("✓ Solution 3 passed!")

    # Test Solution 4
    print("\n[TEST 4] Parameter Counting")
    counts = solution_04_parameter_counting()
    assert counts['total'] > 0
    assert counts['savings_from_tying'] > 0
    print("✓ Solution 4 passed!")

    # Test Solution 5
    print("\n[TEST 5] Causal Generation")
    from tiny_transformer.model import TinyTransformerLM
    model = TinyTransformerLM(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
    start = torch.randint(0, 1000, (2, 10))
    generated = solution_05_causal_generation(model, start, max_new_tokens=20)
    assert generated.shape == (2, 30)
    print("✓ Solution 5 passed!")

    # Test Solution 6
    print("\n[TEST 6] Hidden State Extraction")
    states, analysis = solution_06_hidden_state_extraction()
    assert len(states) > 0
    assert 'layer_norms' in analysis
    print("✓ Solution 6 passed!")

    # Test Solution 7
    print("\n[TEST 7] Model Configuration")
    config = Solution07_ModelConfig.get_config("small")
    assert 'd_model' in config
    model = Solution07_ModelConfig.create_model_from_config("tiny", vocab_size=1000)
    assert model is not None
    print("✓ Solution 7 passed!")

    # Test Solution 8
    print("\n[TEST 8] Model Scaling Analysis")
    results = solution_08_model_scaling_analysis()
    assert 'd_model_scaling' in results
    print("✓ Solution 8 passed!")

    # Test Solution 9
    print("\n[TEST 9] Memory Estimation")
    mem = solution_09_memory_estimation(batch_size=32, seq_len=512)
    assert mem['total_gb'] > 0
    print("✓ Solution 9 passed!")

    # Test Solution 10
    print("\n[TEST 10] Integration Test")
    results = solution_10_integration_test(vocab_size=1000, n_steps=50)
    assert len(results['train_losses']) == 50
    print("✓ Solution 10 passed!")

    print("\n" + "="*70)
    print("ALL SOLUTION TESTS PASSED!")
    print("="*70)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Module 05: Full Model - Solutions")
    print("="*70)
    print("\nThis file contains complete solutions for all exercises.")
    print("Use it to check your work or learn implementation details.")
    print("\nAvailable functions:")
    print("  - run_all_solution_tests(): Test all solutions")
    print("  - solution_XX_...: Individual solution functions")
    print("  - visualize_model_scaling(): Create scaling plots")
    print("  - compare_weight_tying_impact(): Analyze weight tying")
    print("="*70)

    # Uncomment to run tests:
    # run_all_solution_tests()

    # Uncomment to run individual solutions:
    # solution_03_weight_tying()
    # solution_08_model_scaling_analysis()
    # solution_10_integration_test()

    print("\nTo test solutions, uncomment the function calls in __main__")
