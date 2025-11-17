"""
Complete Transformer Language Model.

This module implements TinyTransformerLM, a complete transformer-based
language model that combines all components:
- Token and positional embeddings
- Stack of transformer blocks
- Language modeling head

Supports:
- Weight tying between embeddings and output layer
- Causal language modeling
- Text generation (basic interface, full implementation in Module 07)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from tiny_transformer.embeddings import TransformerEmbedding
from tiny_transformer.transformer_block import TransformerBlock


class TinyTransformerLM(nn.Module):
    """
    Complete Transformer Language Model.

    Architecture:
        Input tokens (B, T)
        ↓
        Embedding Layer: TransformerEmbedding
        ↓
        Transformer Blocks (n_layers)
        ↓
        Final Layer Norm
        ↓
        Language Modeling Head
        ↓
        Logits (B, T, vocab_size)

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        positional: Type of positional encoding ("sinusoidal" or "learned")
        tie_weights: Whether to tie embedding and output weights
        padding_idx: Optional padding token index

    Shape:
        Input: (batch_size, seq_len) - Token IDs
        Output: (batch_size, seq_len, vocab_size) - Logits

    Example:
        >>> model = TinyTransformerLM(
        ...     vocab_size=10000,
        ...     d_model=512,
        ...     n_heads=8,
        ...     n_layers=6,
        ...     d_ff=2048
        ... )
        >>> tokens = torch.randint(0, 10000, (32, 128))
        >>> logits = model(tokens)  # (32, 128, 10000)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 1024,
        dropout: float = 0.1,
        positional: str = "sinusoidal",
        tie_weights: bool = True,
        padding_idx: Optional[int] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.tie_weights = tie_weights

        # Embedding layer (token + positional)
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            positional=positional,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization (Pre-LN style)
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head: d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between embedding and output layer
        # This reduces parameters and often improves performance
        if tie_weights:
            # The embedding layer scales by √d_model, so we need to account for this
            # when tying weights with the output layer
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights.

        Uses:
        - Normal(0, 0.02) for embeddings and linear layers
        - Ones for layer norm gamma
        - Zeros for layer norm beta and biases
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through the transformer language model.

        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            mask: Optional causal mask for attention
                  If None, uses causal mask (lower triangular)
            return_hidden_states: Whether to return intermediate activations

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Optional list of hidden states if return_hidden_states=True

        Shape flow:
            (B, T) tokens
            → Embedding → (B, T, d_model)
            → Transformer Blocks (×n_layers) → (B, T, d_model)
            → Layer Norm → (B, T, d_model)
            → LM Head → (B, T, vocab_size)
        """
        # Validate input
        assert tokens.dim() == 2, \
            f"Expected 2D input (B, T), got shape {tokens.shape}"

        batch_size, seq_len = tokens.shape

        # Create default causal mask if not provided
        if mask is None:
            # Lower triangular matrix: position i can attend to positions ≤ i
            mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))

        # Step 1: Embed tokens (includes positional encoding and dropout)
        # (B, T) → (B, T, d_model)
        x = self.embedding(tokens)

        # Track hidden states if requested
        hidden_states = [x] if return_hidden_states else None

        # Step 2: Pass through transformer blocks
        for block in self.blocks:
            # Each block: (B, T, d_model) → (B, T, d_model)
            x, _ = block(x, mask=mask)

            if return_hidden_states:
                hidden_states.append(x)

        # Step 3: Final layer normalization
        # (B, T, d_model) → (B, T, d_model)
        x = self.ln_f(x)

        # Step 4: Language modeling head
        # (B, T, d_model) → (B, T, vocab_size)
        logits = self.lm_head(x)

        # Note: If weight tying is enabled, the lm_head uses the same weights
        # as the embedding layer. The embedding layer scales by √d_model,
        # but the lm_head does not, which is intentional.

        return logits, hidden_states

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count total parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Total parameter count
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> dict:
        """
        Get detailed parameter count breakdown by component.

        Returns:
            Dictionary mapping component names to parameter counts
        """
        breakdown = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'transformer_blocks': sum(p.numel() for p in self.blocks.parameters()),
            'ln_f': sum(p.numel() for p in self.ln_f.parameters()),
            'lm_head': sum(p.numel() for p in self.lm_head.parameters()),
        }

        # Adjust for weight tying
        if self.tie_weights:
            # Embedding weights are shared with lm_head, so don't double count
            # The lm_head weight is a reference to embedding weight
            shared_params = self.embedding.token_embedding.embedding.weight.numel()
            breakdown['total'] = sum(breakdown.values()) - shared_params
        else:
            breakdown['total'] = sum(breakdown.values())

        return breakdown

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Basic generation method (stub for Module 07).

        This is a simple greedy/temperature-based generation.
        Full implementation with top-k, top-p, etc. in Module 07.

        Args:
            start_tokens: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)

        Returns:
            Generated tokens of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        tokens = start_tokens

        for _ in range(max_new_tokens):
            # Truncate to max_len if needed
            tokens_input = tokens if tokens.size(1) <= self.max_len else tokens[:, -self.max_len:]

            # Forward pass
            logits, _ = self(tokens_input)

            # Get logits for last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, n_layers={self.n_layers}, "
                f"d_ff={self.d_ff}, tie_weights={self.tie_weights}")


def get_model_config(size: str = "tiny") -> dict:
    """
    Get preset model configuration.

    Args:
        size: One of "tiny", "small", "medium", "large"

    Returns:
        Configuration dictionary

    Configurations:
        - tiny: ~1M params, for quick testing
        - small: ~10M params, for prototyping
        - medium: ~50M params, reasonable performance
        - large: ~100M params, good performance
    """
    configs = {
        "tiny": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "d_ff": 512,
            "max_len": 512,
            "dropout": 0.1,
        },
        "small": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "max_len": 1024,
            "dropout": 0.1,
        },
        "medium": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 8,
            "d_ff": 2048,
            "max_len": 1024,
            "dropout": 0.1,
        },
        "large": {
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "d_ff": 3072,
            "max_len": 2048,
            "dropout": 0.1,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    return configs[size]


if __name__ == "__main__":
    print("=" * 70)
    print("TinyTransformerLM - Demo")
    print("=" * 70)
    print()

    # Create tiny model
    config = get_model_config("tiny")
    model = TinyTransformerLM(
        vocab_size=1000,
        **config
    )

    print(f"Model configuration: {config}")
    print(f"Total parameters: {model.count_parameters():,}")
    print()

    # Parameter breakdown
    breakdown = model.get_parameter_breakdown()
    print("Parameter breakdown:")
    for name, count in breakdown.items():
        pct = (count / breakdown['total']) * 100
        print(f"  {name:20s}: {count:>10,} ({pct:>5.1f}%)")
    print()

    # Test forward pass
    tokens = torch.randint(0, 1000, (4, 64))
    logits, _ = model(tokens)

    print(f"Input shape:  {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print()

    # Test generation
    start_tokens = torch.randint(0, 1000, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20)

    print(f"Generation:")
    print(f"  Start length: {start_tokens.size(1)}")
    print(f"  Generated length: {generated.size(1)}")
    print()

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
