# Module 05: Full Model Assembly

Assembling all components - embeddings, transformer blocks, and output projection - into a complete transformer language model ready for training and generation.

## What You'll Learn

1. **Model Architecture** - Stacking transformer blocks
2. **Output Projection** - Converting hidden states to logits
3. **Weight Tying** - Sharing embedding and output weights
4. **Model Initialization** - Proper weight initialization strategies
5. **Forward Pass** - Complete end-to-end computation

## Prerequisites

- ✓ Completed Modules 01-04
- ✓ Understanding of all transformer components
- ✓ Ready to see it all come together!

## Complete Architecture

```
Token IDs: [5, 12, 8, ...]  (batch, seq_len)
    ↓
┌─────────────────────────────────────┐
│ Embedding Layer                     │
│  - Token Embedding (vocab → d_model)│
│  - Positional Encoding              │
│  - Dropout                           │
└─────────────────────────────────────┘
    ↓ (batch, seq_len, d_model)
┌─────────────────────────────────────┐
│ Transformer Block 1                 │
│  - Multi-Head Attention             │
│  - Feed-Forward Network             │
│  - Layer Norms & Residuals          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 2                 │
└─────────────────────────────────────┘
    ↓
      ... (N blocks total)
    ↓
┌─────────────────────────────────────┐
│ Transformer Block N                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Final Layer Norm                    │
└─────────────────────────────────────┘
    ↓ (batch, seq_len, d_model)
┌─────────────────────────────────────┐
│ Output Projection (LM Head)         │
│  d_model → vocab_size               │
└─────────────────────────────────────┘
    ↓
Logits: (batch, seq_len, vocab_size)
```

## Key Concepts

### 1. Model Stacking
- **Depth**: N transformer blocks (6-96+ layers)
- **Same d_model throughout**: Residual connections require this
- **Sequential processing**: Output of block n → Input of block n+1

### 2. Language Modeling Head
- **Linear projection**: d_model → vocab_size
- **No activation**: Raw logits for cross-entropy loss
- **Weight tying**: Often shares weights with token embedding

### 3. Weight Tying
```python
# Share weights between embedding and output projection
self.lm_head.weight = self.embedding.token_embedding.embedding.weight
```

**Benefits**:
- Fewer parameters (vocab_size × d_model saved)
- Better generalization (empirical finding)
- Used in GPT, BERT, etc.

### 4. Causal Language Modeling
- **Objective**: Predict next token given previous tokens
- **Causal mask**: Prevent attending to future
- **Autoregressive**: Generate one token at a time

## What You'll Implement

```python
class TinyTransformerLM(nn.Module):
    """
    Complete transformer language model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        tie_weights: Whether to tie embedding and output weights
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
        tie_weights: bool = True
    ):
        super().__init__()

        # Embedding layer
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len, dropout
        )

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, tokens, mask=None):
        """
        Forward pass through complete model.

        Args:
            tokens: (batch_size, seq_len) - Token IDs
            mask: Optional causal mask

        Returns:
            logits: (batch_size, seq_len, vocab_size) - Next token predictions
        """
        # Embeddings
        x = self.embedding(tokens)  # (B, T, D)

        # Through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)  # (B, T, D)

        # Final norm
        x = self.ln_f(x)  # (B, T, D)

        # Output projection
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """
        Generate text autoregressively.

        Args:
            start_tokens: (batch_size, seq_len) - Initial tokens
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature

        Returns:
            generated: (batch_size, seq_len + max_new_tokens)
        """
        # Implementation in Module 07 (Sampling)
        pass
```

## Model Configurations

| Config | vocab_size | d_model | n_heads | n_layers | d_ff | Params |
|--------|-----------|---------|---------|----------|------|--------|
| Tiny   | 256       | 128     | 4       | 4        | 512  | ~500K  |
| Small  | 1000      | 256     | 8       | 6        | 1024 | ~5M    |
| Medium | 5000      | 512     | 8       | 12       | 2048 | ~40M   |
| Base   | 10000     | 768     | 12      | 12       | 3072 | ~125M  |

## Module Contents

- `theory.md` - Architecture deep dive and design decisions
- `../../tiny_transformer/model.py` - Complete TinyTransformerLM
- `../../tests/test_model.py` - End-to-end tests
- `../../examples/model_demo.py` - Usage examples

## Success Criteria

- [ ] Understand the complete model architecture
- [ ] Implement full transformer language model
- [ ] Implement weight tying correctly
- [ ] Understand causal language modeling objective
- [ ] Pass shape tests for complete forward pass
- [ ] Run model on sample inputs
- [ ] Count parameters correctly

## Next Module

**Module 06: Training** - Implementing the training loop, loss computation, and optimization for your transformer.
