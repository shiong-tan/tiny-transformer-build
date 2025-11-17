"""
Module 05: Full Model (TinyTransformerLM) - Exercises

This module contains hands-on exercises for learning how to assemble a complete
transformer language model from individual components. You'll practice:

1. Building a language modeling head
2. Stacking components (embeddings + blocks + head)
3. Implementing weight tying
4. Counting parameters correctly
5. Basic text generation
6. Extracting hidden states
7. Creating model configurations
8. Analyzing model scaling
9. Estimating memory requirements
10. End-to-end training and generation

Each exercise includes:
- Clear docstrings with learning objectives
- Type hints and shape annotations
- TODO sections for implementation
- Test assertions for self-assessment
- Progressive difficulty (Easy → Medium → Hard)

Prerequisites:
- Completed Modules 01-04 (attention, multi-head, blocks, embeddings)
- Understanding of cross-entropy loss
- Basic knowledge of autoregressive generation

Reference implementations:
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/model.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/embeddings.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/transformer_block.py

Theory reference:
- /Users/shiongtan/projects/tiny-transformer-build/docs/modules/05_full_model/theory.md

Time estimate: 4-5 hours for all exercises

Good luck! Let's build a complete language model!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


# ==============================================================================
# EXERCISE 1: Language Modeling Head
# Difficulty: Easy
# ==============================================================================

class Exercise01_LanguageModelingHead(nn.Module):
    """
    EXERCISE 01: Implement a simple language modeling head.

    Learning Objectives:
    - Understand the purpose of the LM head (projecting to vocab space)
    - Learn why we don't use bias in the output projection
    - Practice shape transformations for next-token prediction

    The LM head is a simple linear layer that projects from d_model dimensions
    to vocab_size dimensions. Each position gets a distribution over the vocabulary.

    Architecture:
        Input: (batch_size, seq_len, d_model)
        → Linear(d_model, vocab_size, bias=False)
        → Output: (batch_size, seq_len, vocab_size)

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, vocab_size) - logits (not probabilities!)

    Example:
        >>> lm_head = Exercise01_LanguageModelingHead(vocab_size=10000, d_model=512)
        >>> hidden_states = torch.randn(32, 128, 512)
        >>> logits = lm_head(hidden_states)
        >>> logits.shape
        torch.Size([32, 128, 10000])

    Theory reference: theory.md, Section "Complete Model Design"

    Self-Assessment Questions:
    1. Why don't we use bias in the LM head?
    2. Why are these called "logits" and not "probabilities"?
    3. What loss function will we use with these logits?
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        # TODO: Implement initialization
        # Step 1: Create a linear layer: d_model → vocab_size
        # Step 2: Set bias=False (standard practice for LM heads)
        # Step 3: Initialize weights with normal distribution (mean=0, std=0.02)

        pass  # Remove this and add your implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            x: Hidden states of shape (batch_size, seq_len, d_model)

        Returns:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)
        """
        # TODO: Implement forward pass
        # Apply the linear projection

        pass  # Remove this and add your implementation

        # Uncomment to test:
        # assert logits.dim() == 3, f"Expected 3D output, got {logits.dim()}D"
        # assert logits.shape[-1] == self.vocab_size


# ==============================================================================
# EXERCISE 2: Manual Model Stacking
# Difficulty: Easy
# ==============================================================================

def exercise_02_manual_model_stacking(
    vocab_size: int = 1000,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    batch_size: int = 8,
    seq_len: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EXERCISE 02: Manually stack all components to create a forward pass.

    Learning Objectives:
    - Understand the complete forward pass flow
    - Practice combining embeddings → blocks → head
    - Learn shape transformations at each stage

    You'll manually create and connect:
    1. Token + Positional Embeddings
    2. Multiple Transformer Blocks
    3. Final Layer Norm
    4. Language Modeling Head

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        batch_size: Batch size for test
        seq_len: Sequence length for test

    Returns:
        input_tokens: Input token IDs of shape (batch_size, seq_len)
        logits: Output logits of shape (batch_size, seq_len, vocab_size)

    Example:
        >>> tokens, logits = exercise_02_manual_model_stacking()
        >>> print(tokens.shape)  # torch.Size([8, 32])
        >>> print(logits.shape)  # torch.Size([8, 32, 1000])

    Theory reference: theory.md, Section "Forward Pass Step-by-Step"

    Self-Assessment Questions:
    1. What's the shape after embeddings? After each block? After LM head?
    2. Why do we need LayerNorm before the LM head?
    3. What happens to the shapes through residual connections?
    """
    # TODO: Implement this exercise

    # Step 1: Import necessary components
    # from tiny_transformer.embeddings import TransformerEmbedding
    # from tiny_transformer.transformer_block import TransformerBlock

    # Step 2: Create input tokens (random integers in [0, vocab_size))
    # tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Step 3: Create and apply embedding layer
    # embedding = TransformerEmbedding(...)
    # x = embedding(tokens)  # Shape: (batch_size, seq_len, d_model)

    # Step 4: Create and apply transformer blocks (n_layers times)
    # for i in range(n_layers):
    #     block = TransformerBlock(...)
    #     x, _ = block(x)  # Shape: (batch_size, seq_len, d_model)

    # Step 5: Create and apply final layer norm
    # ln_f = nn.LayerNorm(d_model)
    # x = ln_f(x)

    # Step 6: Create and apply LM head
    # lm_head = nn.Linear(d_model, vocab_size, bias=False)
    # logits = lm_head(x)  # Shape: (batch_size, seq_len, vocab_size)

    # Step 7: Verify shapes and return
    # assert tokens.shape == (batch_size, seq_len)
    # assert logits.shape == (batch_size, seq_len, vocab_size)

    # return tokens, logits
    pass


# ==============================================================================
# EXERCISE 3: Weight Tying Implementation
# Difficulty: Medium
# ==============================================================================

def exercise_03_weight_tying(
    vocab_size: int = 5000,
    d_model: int = 256
) -> Tuple[bool, int]:
    """
    EXERCISE 03: Implement weight tying between embeddings and LM head.

    Learning Objectives:
    - Understand what weight tying means
    - Learn how to share parameters in PyTorch
    - Calculate parameter savings from weight tying

    Weight tying means the embedding matrix and output projection share
    the same weight tensor. This:
    - Reduces parameters significantly
    - Improves generalization
    - Creates semantic consistency

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension

    Returns:
        weights_are_shared: True if weights are properly tied (same object)
        params_saved: Number of parameters saved by tying

    Example:
        >>> shared, saved = exercise_03_weight_tying(vocab_size=5000, d_model=256)
        >>> print(f"Weights shared: {shared}")
        >>> print(f"Parameters saved: {saved:,}")

    Theory reference: theory.md, Section "Weight Tying"

    Self-Assessment Questions:
    1. How do you check if two tensors share the same memory?
    2. What's the formula for parameters saved?
    3. When should you NOT use weight tying?
    """
    # TODO: Implement this exercise

    # Step 1: Create token embedding layer
    # token_embedding = nn.Embedding(vocab_size, d_model)

    # Step 2: Create LM head (linear layer without bias)
    # lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # Step 3: Count parameters BEFORE tying
    # params_before = sum(p.numel() for p in [token_embedding, lm_head])

    # Step 4: Tie weights (make lm_head.weight reference token_embedding.weight)
    # lm_head.weight = token_embedding.weight

    # Step 5: Count parameters AFTER tying
    # params_after = ... (need to count unique parameters only)

    # Step 6: Verify they're actually shared (same object, same memory address)
    # weights_are_shared = (lm_head.weight is token_embedding.weight)
    # also_check = (lm_head.weight.data_ptr() == token_embedding.weight.data_ptr())

    # Step 7: Calculate savings
    # params_saved = params_before - params_after

    # return weights_are_shared, params_saved
    pass


# ==============================================================================
# EXERCISE 4: Parameter Counting with Weight Tying
# Difficulty: Medium
# ==============================================================================

def exercise_04_parameter_counting(
    vocab_size: int = 10000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    max_len: int = 1024,
    tie_weights: bool = True
) -> Dict[str, int]:
    """
    EXERCISE 04: Count parameters in a complete transformer model.

    Learning Objectives:
    - Understand parameter distribution across components
    - Learn to calculate parameter counts analytically
    - See the impact of weight tying on total parameters

    You'll count parameters for:
    - Token embeddings: vocab_size × d_model
    - Positional embeddings: max_len × d_model (if learned)
    - Transformer blocks: n_layers × (attention + ffn + layer_norms)
    - Final layer norm: 2 × d_model (gamma + beta)
    - LM head: d_model × vocab_size (unless tied)

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        tie_weights: Whether to tie embedding/output weights

    Returns:
        Dictionary with parameter counts:
        {
            'token_embedding': int,
            'positional_embedding': int,
            'transformer_blocks': int,
            'final_ln': int,
            'lm_head': int,
            'total': int,
            'total_with_tying': int,
            'savings_from_tying': int
        }

    Example:
        >>> counts = exercise_04_parameter_counting(
        ...     vocab_size=10000, d_model=512, n_heads=8, n_layers=6
        ... )
        >>> print(f"Total params: {counts['total']:,}")
        >>> print(f"Savings: {counts['savings_from_tying']:,}")

    Theory reference: theory.md, Section "Parameter Counting Methodology"

    Formulas:
        Token embedding: vocab_size × d_model
        Positional (learned): max_len × d_model
        Single attention: 4 × d_model² + 4 × d_model (Q, K, V, O + biases)
        Single FFN: 2 × d_model × d_ff + d_ff + d_model (fc1, fc2 + biases)
        Single LayerNorm: 2 × d_model (gamma + beta)
        Transformer block: attention + ffn + 2×layernorm
        LM head: d_model × vocab_size (no bias)

    Self-Assessment Questions:
    1. Which component has the most parameters?
    2. How does parameter count scale with n_layers?
    3. What percentage savings does weight tying provide?
    """
    # TODO: Implement this exercise

    # Step 1: Calculate token embedding parameters
    # token_emb_params = vocab_size * d_model

    # Step 2: Calculate positional embedding parameters (assume learned)
    # pos_emb_params = max_len * d_model

    # Step 3: Calculate single transformer block parameters
    # Attention: 4 weight matrices (Q, K, V, O) + 4 biases
    # attention_params = ...

    # FFN: 2 weight matrices + 2 biases
    # ffn_params = ...

    # Layer norms: 2 layer norms per block
    # ln_params_per_block = ...

    # Total per block
    # block_params = attention_params + ffn_params + ln_params_per_block

    # Step 4: Calculate all transformer blocks
    # all_blocks_params = n_layers * block_params

    # Step 5: Calculate final layer norm
    # final_ln_params = 2 * d_model

    # Step 6: Calculate LM head
    # lm_head_params = d_model * vocab_size

    # Step 7: Calculate totals
    # total_without_tying = (token_emb_params + pos_emb_params +
    #                        all_blocks_params + final_ln_params + lm_head_params)

    # If weight tying, subtract the shared parameters
    # total_with_tying = total_without_tying - (vocab_size * d_model if tie_weights else 0)

    # savings = total_without_tying - total_with_tying

    # Step 8: Return breakdown
    # return {
    #     'token_embedding': token_emb_params,
    #     'positional_embedding': pos_emb_params,
    #     'transformer_blocks': all_blocks_params,
    #     'final_ln': final_ln_params,
    #     'lm_head': lm_head_params if not tie_weights else 0,
    #     'total': total_without_tying,
    #     'total_with_tying': total_with_tying,
    #     'savings_from_tying': savings
    # }
    pass


# ==============================================================================
# EXERCISE 5: Basic Causal Generation
# Difficulty: Medium
# ==============================================================================

@torch.no_grad()
def exercise_05_causal_generation(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    EXERCISE 05: Implement basic greedy/temperature-based generation.

    Learning Objectives:
    - Understand autoregressive text generation
    - Learn temperature sampling
    - Practice the generation loop pattern

    Generation process:
    1. Start with initial tokens
    2. Forward pass to get logits
    3. Extract logits for last position
    4. Apply temperature scaling
    5. Sample next token
    6. Append to sequence
    7. Repeat until max_new_tokens

    Args:
        model: A TinyTransformerLM model (or similar)
        start_tokens: Initial tokens of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)

    Returns:
        generated: Full sequence of shape (batch_size, seq_len + max_new_tokens)

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=1000, d_model=128, ...)
        >>> start = torch.randint(0, 1000, (1, 10))
        >>> generated = exercise_05_causal_generation(model, start, max_new_tokens=20)
        >>> print(generated.shape)  # torch.Size([1, 30])

    Theory reference: theory.md, Section "Generation Interface"

    Self-Assessment Questions:
    1. Why do we use @torch.no_grad()?
    2. What does temperature do to the distribution?
    3. How is this different from training?
    """
    # TODO: Implement this exercise

    # Step 1: Set model to eval mode
    # model.eval()

    # Step 2: Initialize current sequence
    # current = start_tokens.clone()

    # Step 3: Generation loop
    # for _ in range(max_new_tokens):
    #     # Step 3a: Forward pass
    #     logits, _ = model(current)
    #
    #     # Step 3b: Get logits for last position
    #     next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
    #
    #     # Step 3c: Apply temperature
    #     if temperature != 1.0:
    #         next_logits = next_logits / temperature
    #
    #     # Step 3d: Convert to probabilities
    #     probs = F.softmax(next_logits, dim=-1)
    #
    #     # Step 3e: Sample next token
    #     next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
    #
    #     # Step 3f: Append to sequence
    #     current = torch.cat([current, next_token], dim=1)

    # Step 4: Return generated sequence
    # return current
    pass


# ==============================================================================
# EXERCISE 6: Hidden State Extraction
# Difficulty: Medium
# ==============================================================================

def exercise_06_hidden_state_extraction(
    vocab_size: int = 1000,
    d_model: int = 256,
    n_layers: int = 4
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    EXERCISE 06: Extract and analyze hidden states from each layer.

    Learning Objectives:
    - Learn to extract intermediate representations
    - Understand how representations evolve through layers
    - Analyze representation norms and similarities

    You'll modify the forward pass to collect hidden states from:
    - After embeddings
    - After each transformer block
    - After final layer norm

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer blocks

    Returns:
        hidden_states: List of tensors, one per layer
        analysis: Dictionary with:
            - 'layer_norms': List of average norms per layer
            - 'layer_similarities': Cosine similarities between consecutive layers
            - 'first_to_last_similarity': Similarity between input and output

    Example:
        >>> states, analysis = exercise_06_hidden_state_extraction()
        >>> print(f"Number of layers: {len(states)}")
        >>> print(f"Norms per layer: {analysis['layer_norms']}")

    Theory reference: theory.md, Section "Forward Pass Step-by-Step"

    Self-Assessment Questions:
    1. How do norms change through layers?
    2. Why do we care about cosine similarity?
    3. What does low similarity between layers indicate?
    """
    # TODO: Implement this exercise

    # Step 1: Create a simple model (or use TinyTransformerLM)
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(
    #     vocab_size=vocab_size,
    #     d_model=d_model,
    #     n_heads=4,
    #     n_layers=n_layers,
    #     d_ff=d_model * 4
    # )

    # Step 2: Create input tokens
    # tokens = torch.randint(0, vocab_size, (4, 32))

    # Step 3: Forward pass with hidden state collection
    # Use return_hidden_states=True if available, or manually collect
    # logits, hidden_states = model(tokens, return_hidden_states=True)

    # Step 4: Analyze hidden states
    # Calculate norm for each layer
    # layer_norms = [h.norm(dim=-1).mean().item() for h in hidden_states]

    # Step 5: Calculate cosine similarities between consecutive layers
    # def cosine_sim(a, b):
    #     a_flat = a.reshape(-1, a.size(-1))
    #     b_flat = b.reshape(-1, b.size(-1))
    #     return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()

    # layer_similarities = [
    #     cosine_sim(hidden_states[i], hidden_states[i+1])
    #     for i in range(len(hidden_states) - 1)
    # ]

    # Step 6: First to last similarity
    # first_to_last = cosine_sim(hidden_states[0], hidden_states[-1])

    # analysis = {
    #     'layer_norms': layer_norms,
    #     'layer_similarities': layer_similarities,
    #     'first_to_last_similarity': first_to_last
    # }

    # return hidden_states, analysis
    pass


# ==============================================================================
# EXERCISE 7: Model Configuration System
# Difficulty: Medium
# ==============================================================================

class Exercise07_ModelConfig:
    """
    EXERCISE 07: Create a configuration system for models.

    Learning Objectives:
    - Understand common model configuration patterns
    - Learn to create reusable configs
    - Practice model instantiation from configs

    You'll create preset configurations like:
    - "tiny": For quick testing (~1M params)
    - "small": For prototyping (~10M params)
    - "medium": For training (~50M params)

    Example:
        >>> config = Exercise07_ModelConfig.get_config("small")
        >>> print(config)
        >>> model = TinyTransformerLM(**config, vocab_size=10000)

    Theory reference: theory.md, Section "Model Size Configurations"

    Self-Assessment Questions:
    1. What's the relationship between d_model and d_ff?
    2. Why does n_heads need to divide d_model?
    3. How do you scale a model while maintaining proportions?
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
        # TODO: Implement this method

        # Define configurations
        # configs = {
        #     "tiny": {
        #         "d_model": 128,
        #         "n_heads": 4,
        #         "n_layers": 2,
        #         "d_ff": 512,
        #         "max_len": 256,
        #         "dropout": 0.1,
        #     },
        #     "small": {
        #         "d_model": 256,
        #         "n_heads": 8,
        #         "n_layers": 4,
        #         "d_ff": 1024,
        #         "max_len": 512,
        #         "dropout": 0.1,
        #     },
        #     "medium": {
        #         "d_model": 512,
        #         "n_heads": 8,
        #         "n_layers": 6,
        #         "d_ff": 2048,
        #         "max_len": 1024,
        #         "dropout": 0.1,
        #     },
        # }

        # if size not in configs:
        #     raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

        # return configs[size]
        pass

    @staticmethod
    def estimate_parameters(config: Dict[str, any], vocab_size: int) -> int:
        """
        Estimate total parameters from config.

        Args:
            config: Model configuration dictionary
            vocab_size: Vocabulary size

        Returns:
            Estimated total parameters
        """
        # TODO: Use the formula from Exercise 4
        # Extract config values
        # Calculate and return total parameters
        pass


# ==============================================================================
# EXERCISE 8: Model Scaling Analysis
# Difficulty: Hard
# ==============================================================================

def exercise_08_model_scaling_analysis() -> Dict[str, List[int]]:
    """
    EXERCISE 08: Analyze how parameter count scales with model dimensions.

    Learning Objectives:
    - Understand parameter scaling laws
    - Learn which dimensions have biggest impact
    - Practice systematic model comparison

    You'll vary each dimension independently and measure parameter count:
    - Vary d_model: [128, 256, 512, 768, 1024]
    - Vary n_layers: [2, 4, 6, 8, 12]
    - Vary vocab_size: [1000, 5000, 10000, 20000, 50000]

    Returns:
        Dictionary with scaling results:
        {
            'd_model_scaling': List[int],  # params for each d_model
            'n_layers_scaling': List[int],  # params for each n_layers
            'vocab_scaling': List[int],  # params for each vocab_size
            'observations': Dict[str, str]  # key insights
        }

    Example:
        >>> results = exercise_08_model_scaling_analysis()
        >>> print(f"Doubling d_model: {results['d_model_scaling'][1] / results['d_model_scaling'][0]:.1f}x")

    Theory reference: theory.md, Section "Parameter Counting Methodology"

    Self-Assessment Questions:
    1. Which dimension causes quadratic growth?
    2. Which dimension causes linear growth?
    3. How does weight tying affect vocab_size scaling?
    """
    # TODO: Implement this exercise

    # Base configuration
    # base_config = {
    #     'd_model': 256,
    #     'n_heads': 8,
    #     'n_layers': 4,
    #     'd_ff': 1024,
    #     'vocab_size': 10000,
    #     'max_len': 512
    # }

    # Step 1: Vary d_model (keep others constant)
    # d_model_values = [128, 256, 512, 768, 1024]
    # d_model_params = []
    # for d in d_model_values:
    #     config = base_config.copy()
    #     config['d_model'] = d
    #     config['d_ff'] = d * 4  # Maintain ratio
    #     params = exercise_04_parameter_counting(**config)['total_with_tying']
    #     d_model_params.append(params)

    # Step 2: Vary n_layers (keep others constant)
    # n_layers_values = [2, 4, 6, 8, 12]
    # n_layers_params = []
    # for n in n_layers_values:
    #     ...

    # Step 3: Vary vocab_size (keep others constant)
    # vocab_values = [1000, 5000, 10000, 20000, 50000]
    # vocab_params = []
    # for v in vocab_values:
    #     ...

    # Step 4: Generate observations
    # observations = {
    #     'd_model_growth': '...',  # e.g., "quadratic"
    #     'n_layers_growth': '...',  # e.g., "linear"
    #     'vocab_growth': '...',  # e.g., "linear with tying"
    # }

    # return {
    #     'd_model_scaling': d_model_params,
    #     'n_layers_scaling': n_layers_params,
    #     'vocab_scaling': vocab_params,
    #     'observations': observations
    # }
    pass


# ==============================================================================
# EXERCISE 9: Memory Estimation
# Difficulty: Hard
# ==============================================================================

def exercise_09_memory_estimation(
    vocab_size: int = 10000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    batch_size: int = 32,
    seq_len: int = 512
) -> Dict[str, float]:
    """
    EXERCISE 09: Estimate memory requirements for training.

    Learning Objectives:
    - Understand memory consumption breakdown
    - Learn to estimate GPU memory needs
    - Practice memory optimization thinking

    Memory components:
    1. Model parameters (weights)
    2. Gradients (same size as parameters)
    3. Optimizer states (Adam: 2x parameters)
    4. Activations (depends on batch_size and seq_len)
    5. Attention matrices (O(n²) with seq_len)

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        batch_size: Training batch size
        seq_len: Sequence length

    Returns:
        Dictionary with memory estimates (in MB):
        {
            'parameters_mb': float,
            'gradients_mb': float,
            'optimizer_states_mb': float,
            'activations_mb': float,
            'attention_mb': float,
            'total_mb': float,
            'total_gb': float
        }

    Example:
        >>> mem = exercise_09_memory_estimation(batch_size=32, seq_len=512)
        >>> print(f"Total memory: {mem['total_gb']:.2f} GB")

    Theory reference: theory.md, Section "Practical Tips for Training"

    Formulas:
        - Float32: 4 bytes per parameter
        - Parameters: Use exercise_04_parameter_counting
        - Gradients: Same as parameters
        - Adam states: 2x parameters (momentum + variance)
        - Activations: Approximate as batch_size × seq_len × d_model × n_layers × 10
        - Attention: batch_size × n_heads × seq_len² × n_layers × 4

    Self-Assessment Questions:
    1. What's the biggest memory consumer?
    2. How does sequence length affect memory?
    3. What can you do to reduce memory usage?
    """
    # TODO: Implement this exercise

    # Step 1: Count parameters
    # param_count = exercise_04_parameter_counting(
    #     vocab_size=vocab_size,
    #     d_model=d_model,
    #     n_heads=n_heads,
    #     n_layers=n_layers,
    #     tie_weights=True
    # )['total_with_tying']

    # Step 2: Calculate memory for parameters (float32 = 4 bytes)
    # parameters_bytes = param_count * 4
    # parameters_mb = parameters_bytes / (1024 * 1024)

    # Step 3: Gradients (same size as parameters)
    # gradients_mb = parameters_mb

    # Step 4: Optimizer states (Adam: 2x parameters)
    # optimizer_states_mb = parameters_mb * 2

    # Step 5: Activations (rough estimate)
    # Intermediate activations stored for backward pass
    # activations_elements = batch_size * seq_len * d_model * n_layers * 10
    # activations_mb = (activations_elements * 4) / (1024 * 1024)

    # Step 6: Attention matrices (batch × heads × seq_len × seq_len × layers)
    # attention_elements = batch_size * n_heads * seq_len * seq_len * n_layers
    # attention_mb = (attention_elements * 4) / (1024 * 1024)

    # Step 7: Total
    # total_mb = (parameters_mb + gradients_mb + optimizer_states_mb +
    #             activations_mb + attention_mb)
    # total_gb = total_mb / 1024

    # return {
    #     'parameters_mb': parameters_mb,
    #     'gradients_mb': gradients_mb,
    #     'optimizer_states_mb': optimizer_states_mb,
    #     'activations_mb': activations_mb,
    #     'attention_mb': attention_mb,
    #     'total_mb': total_mb,
    #     'total_gb': total_gb
    # }
    pass


# ==============================================================================
# EXERCISE 10: End-to-End Training and Generation
# Difficulty: Hard
# ==============================================================================

def exercise_10_integration_test(
    vocab_size: int = 1000,
    n_steps: int = 100
) -> Dict[str, any]:
    """
    EXERCISE 10: Complete training and generation pipeline.

    Learning Objectives:
    - Integrate all components into working system
    - Implement basic training loop
    - Perform generation after training
    - Validate the complete pipeline

    You'll implement:
    1. Model creation
    2. Dummy data generation
    3. Training loop with loss computation
    4. Validation
    5. Text generation
    6. Loss tracking

    Args:
        vocab_size: Vocabulary size
        n_steps: Number of training steps

    Returns:
        Dictionary with results:
        {
            'model': trained model,
            'train_losses': List[float],
            'val_loss': float,
            'generated_tokens': torch.Tensor,
            'final_params': int
        }

    Example:
        >>> results = exercise_10_integration_test(vocab_size=1000, n_steps=100)
        >>> print(f"Final loss: {results['train_losses'][-1]:.4f}")
        >>> print(f"Generated: {results['generated_tokens'].shape}")

    Theory reference: theory.md, Section "Causal Language Modeling Objective"

    Self-Assessment Questions:
    1. Does the loss decrease over training?
    2. Are the generated tokens valid (in vocab range)?
    3. What's the relationship between training and generation?
    """
    # TODO: Implement this exercise

    # Step 1: Create model
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(
    #     vocab_size=vocab_size,
    #     d_model=128,
    #     n_heads=4,
    #     n_layers=2,
    #     d_ff=512,
    #     tie_weights=True
    # )

    # Step 2: Create optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 3: Training loop
    # train_losses = []
    # model.train()
    #
    # for step in range(n_steps):
    #     # Generate dummy data (random tokens)
    #     input_ids = torch.randint(0, vocab_size, (8, 64))
    #     target_ids = torch.randint(0, vocab_size, (8, 64))
    #
    #     # Forward pass
    #     logits, _ = model(input_ids)
    #
    #     # Compute loss
    #     loss = F.cross_entropy(
    #         logits.view(-1, vocab_size),
    #         target_ids.view(-1)
    #     )
    #
    #     # Backward pass
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     # Track loss
    #     train_losses.append(loss.item())

    # Step 4: Validation
    # model.eval()
    # with torch.no_grad():
    #     val_input = torch.randint(0, vocab_size, (4, 64))
    #     val_target = torch.randint(0, vocab_size, (4, 64))
    #     val_logits, _ = model(val_input)
    #     val_loss = F.cross_entropy(
    #         val_logits.view(-1, vocab_size),
    #         val_target.view(-1)
    #     ).item()

    # Step 5: Generation
    # start_tokens = torch.randint(0, vocab_size, (1, 10))
    # generated = exercise_05_causal_generation(model, start_tokens, max_new_tokens=20)

    # Step 6: Count final parameters
    # final_params = model.count_parameters()

    # return {
    #     'model': model,
    #     'train_losses': train_losses,
    #     'val_loss': val_loss,
    #     'generated_tokens': generated,
    #     'final_params': final_params
    # }
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
    print("Module 05: Full Model - Exercise Tests")
    print("=" * 70)

    # Test Exercise 1
    # print("\n[TEST 1] Language Modeling Head")
    # lm_head = Exercise01_LanguageModelingHead(vocab_size=10000, d_model=512)
    # x = torch.randn(32, 128, 512)
    # logits = lm_head(x)
    # assert logits.shape == (32, 128, 10000), f"Expected (32, 128, 10000), got {logits.shape}"
    # print("✓ Exercise 1 passed!")

    # Test Exercise 2
    # print("\n[TEST 2] Manual Model Stacking")
    # tokens, logits = exercise_02_manual_model_stacking()
    # assert tokens.shape == (8, 32), f"Expected tokens (8, 32), got {tokens.shape}"
    # assert logits.shape == (8, 32, 1000), f"Expected logits (8, 32, 1000), got {logits.shape}"
    # print("✓ Exercise 2 passed!")

    # Test Exercise 3
    # print("\n[TEST 3] Weight Tying")
    # shared, saved = exercise_03_weight_tying(vocab_size=5000, d_model=256)
    # assert shared == True, "Weights should be shared!"
    # assert saved == 5000 * 256, f"Should save 5000×256 params, got {saved}"
    # print(f"✓ Exercise 3 passed! Saved {saved:,} parameters")

    # Test Exercise 4
    # print("\n[TEST 4] Parameter Counting")
    # counts = exercise_04_parameter_counting(vocab_size=10000, d_model=512)
    # assert 'total' in counts, "Missing 'total' in output"
    # assert counts['savings_from_tying'] > 0, "Should have savings from tying"
    # print(f"✓ Exercise 4 passed! Total: {counts['total_with_tying']:,} params")

    # Test Exercise 5
    # print("\n[TEST 5] Causal Generation")
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
    # start = torch.randint(0, 1000, (2, 10))
    # generated = exercise_05_causal_generation(model, start, max_new_tokens=20)
    # assert generated.shape == (2, 30), f"Expected (2, 30), got {generated.shape}"
    # print("✓ Exercise 5 passed!")

    # Test Exercise 6
    # print("\n[TEST 6] Hidden State Extraction")
    # states, analysis = exercise_06_hidden_state_extraction()
    # assert len(states) > 0, "Should have hidden states"
    # assert 'layer_norms' in analysis, "Missing layer_norms in analysis"
    # print(f"✓ Exercise 6 passed! Analyzed {len(states)} layers")

    # Test Exercise 7
    # print("\n[TEST 7] Model Configuration")
    # config = Exercise07_ModelConfig.get_config("small")
    # assert 'd_model' in config, "Config missing d_model"
    # assert config['n_heads'] > 0, "n_heads should be positive"
    # print(f"✓ Exercise 7 passed! Config: {config}")

    # Test Exercise 8
    # print("\n[TEST 8] Model Scaling Analysis")
    # results = exercise_08_model_scaling_analysis()
    # assert 'd_model_scaling' in results, "Missing d_model_scaling"
    # assert len(results['d_model_scaling']) == 5, "Should have 5 d_model values"
    # print("✓ Exercise 8 passed!")

    # Test Exercise 9
    # print("\n[TEST 9] Memory Estimation")
    # mem = exercise_09_memory_estimation(batch_size=32, seq_len=512)
    # assert 'total_gb' in mem, "Missing total_gb"
    # assert mem['total_gb'] > 0, "Memory should be positive"
    # print(f"✓ Exercise 9 passed! Estimated: {mem['total_gb']:.2f} GB")

    # Test Exercise 10
    # print("\n[TEST 10] Integration Test")
    # results = exercise_10_integration_test(vocab_size=1000, n_steps=50)
    # assert len(results['train_losses']) == 50, "Should have 50 losses"
    # assert results['train_losses'][-1] < results['train_losses'][0], "Loss should decrease"
    # print("✓ Exercise 10 passed!")

    print("\n" + "=" * 70)
    print("All tests passed! Excellent work!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Module 05: Full Model (TinyTransformerLM) - Exercises")
    print("=" * 70)
    print("\nWelcome! These exercises teach you to build complete language models.")
    print("\nExercise Structure:")
    print("  01. Language Modeling Head (Easy)")
    print("  02. Manual Model Stacking (Easy)")
    print("  03. Weight Tying (Medium)")
    print("  04. Parameter Counting (Medium)")
    print("  05. Causal Generation (Medium)")
    print("  06. Hidden State Extraction (Medium)")
    print("  07. Model Configuration (Medium)")
    print("  08. Model Scaling Analysis (Hard)")
    print("  09. Memory Estimation (Hard)")
    print("  10. Integration Test (Hard)")
    print("\nInstructions:")
    print("1. Work through exercises 1-10 in order")
    print("2. Read each docstring carefully")
    print("3. Implement the TODO sections")
    print("4. Uncomment test assertions to verify")
    print("5. Run this file to test your solutions")
    print("\nTips:")
    print("- Refer to theory.md for concepts")
    print("- Check tiny_transformer/model.py for reference")
    print("- Use print() to debug shapes and values")
    print("- Solutions available in solutions.py")
    print("=" * 70)

    # Uncomment when ready to test:
    # run_all_tests()

    print("\nGood luck building your transformer! 🚀")
