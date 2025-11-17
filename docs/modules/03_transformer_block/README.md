# Module 03: Transformer Block

Combining multi-head attention with feed-forward networks, layer normalization, and residual connections to create the complete transformer block - the fundamental building block of transformer models.

## What You'll Learn

1. **Feed-Forward Networks** - Position-wise fully connected layers
2. **Layer Normalization** - Stabilizing training with normalization
3. **Residual Connections** - Enabling deep networks with skip connections
4. **Complete Transformer Block** - Assembling all components
5. **Pre-LN vs Post-LN** - Different normalization architectures

## Prerequisites

- ✓ Completed Module 02 (Multi-Head Attention)
- ✓ Understanding of neural network architectures
- ✓ Familiarity with normalization techniques

## The Architecture

A complete transformer block consists of:

```
Input (batch, seq_len, d_model)
    ↓
┌─────────────────────────────────┐
│ Multi-Head Self-Attention       │
└─────────────────────────────────┘
    ↓ (residual connection)
┌─────────────────────────────────┐
│ Layer Normalization             │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Feed-Forward Network (FFN)      │
│   - Linear: d_model → d_ff      │
│   - Activation (GELU/ReLU)      │
│   - Linear: d_ff → d_model      │
└─────────────────────────────────┘
    ↓ (residual connection)
┌─────────────────────────────────┐
│ Layer Normalization             │
└─────────────────────────────────┘
    ↓
Output (batch, seq_len, d_model)
```

## Key Concepts

### 1. Feed-Forward Network (FFN)
- **Position-wise**: Applied independently to each position
- **Expansion**: d_model → d_ff (typically 4× larger)
- **Activation**: GELU (modern) or ReLU (original)
- **Contraction**: d_ff → d_model

### 2. Layer Normalization
- **Per-token normalization**: Normalizes across d_model dimension
- **Trainable parameters**: Gain (γ) and bias (β)
- **Stability**: Prevents internal covariate shift

### 3. Residual Connections
- **Skip connections**: x + SubLayer(x)
- **Gradient flow**: Enables deep networks (100+ layers)
- **Identity mapping**: Easier optimization

### 4. Architecture Variants
- **Post-LN** (Original): LN after each sub-layer
- **Pre-LN** (Modern): LN before each sub-layer - more stable training

## What You'll Implement

```python
# 1. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        # Linear layers with activation

# 2. Transformer Block (Pre-LN architecture)
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN: normalize before each sub-layer
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)[0]
        x = x + self.feed_forward(self.norm2(x))
        return x
```

## Hyperparameters

| Parameter | Tiny | Small | Base | Large |
|-----------|------|-------|------|-------|
| d_model   | 128  | 256   | 512  | 1024  |
| n_heads   | 4    | 8     | 8    | 16    |
| d_ff      | 512  | 1024  | 2048 | 4096  |
| dropout   | 0.1  | 0.1   | 0.1  | 0.1   |

Note: d_ff is typically 4× d_model

## Module Contents

- `theory.md` - Deep dive into transformer block architecture
- `../../tiny_transformer/feedforward.py` - FFN implementation
- `../../tiny_transformer/transformer_block.py` - Complete block
- `../../tests/test_transformer_block.py` - Comprehensive tests

## Success Criteria

- [ ] Understand why we need FFN in addition to attention
- [ ] Implement position-wise feed-forward network
- [ ] Understand layer normalization vs batch normalization
- [ ] Implement Pre-LN transformer block with residuals
- [ ] Pass all tests
- [ ] Explain gradient flow through residual connections

## Next Module

**Module 04: Embeddings** - Converting tokens to vectors and adding positional information.
