# Module 04: Embeddings & Positional Encoding

Converting discrete tokens into continuous vectors and adding positional information - the input layer of transformer models.

## What You'll Learn

1. **Token Embeddings** - Mapping discrete tokens to dense vectors
2. **Positional Encoding** - Injecting position information
3. **Sinusoidal Encoding** - Fixed positional patterns (original)
4. **Learned Positional Embeddings** - Trainable positions (modern)
5. **Combined Input** - Token + position embeddings

## Prerequisites

- ✓ Completed Module 03 (Transformer Blocks)
- ✓ Understanding of embedding layers
- ✓ Basic trigonometry (for sinusoidal encoding)

## The Problem

**Transformers have no inherent notion of sequence order!**

Unlike RNNs which process sequentially, attention is permutation-invariant:
- `Attention([A, B, C])` = `Attention([C, A, B])` ❌

We need to explicitly inject positional information.

## Architecture

```
Token IDs: [5, 12, 8, 42, ...]  (integers)
    ↓
┌──────────────────────────────┐
│ Token Embedding Lookup       │
│ vocab_size → d_model         │
└──────────────────────────────┘
    ↓
Token Embeddings: (batch, seq_len, d_model)
    +  (element-wise addition)
┌──────────────────────────────┐
│ Positional Encoding          │
│ Sinusoidal OR Learned        │
└──────────────────────────────┘
    ↓
Positional Embeddings: (seq_len, d_model) or (batch, seq_len, d_model)
    ↓
Combined: (batch, seq_len, d_model)
    ↓
[To Transformer Blocks...]
```

## Key Concepts

### 1. Token Embeddings
- **Learnable lookup table**: `vocab_size × d_model`
- **Dense representation**: Each token → d_model-dimensional vector
- **Shared across positions**: Same token ID → same embedding (before adding position)

### 2. Sinusoidal Positional Encoding (Original Transformer)
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Advantages**:
- No learnable parameters
- Can extrapolate to longer sequences
- Smooth, periodic patterns

### 3. Learned Positional Embeddings (GPT, BERT)
- **Learnable lookup table**: `max_seq_len × d_model`
- **More flexible**: Can learn task-specific patterns
- **Limited**: Can't handle sequences longer than `max_seq_len`

### 4. Embedding Scaling
- **Common practice**: Scale embeddings by √d_model
- **Reason**: Prevents embeddings from dominating after position addition

## What You'll Implement

```python
# 1. Token Embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        # Scale by sqrt(d_model) - common practice
        return self.embedding(tokens) * math.sqrt(self.d_model)

# 2. Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        # Pre-compute sinusoidal patterns

    def forward(self, x):
        # Add positional encoding to input

# 3. Learned Positional Embeddings
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(positions)

# 4. Combined Embedding Layer
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int,
                 positional: str = "sinusoidal"):
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        if positional == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len)
        else:
            self.pos_embedding = LearnedPositionalEmbedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        tok_emb = self.token_embedding(tokens)  # (B, T, D)
        pos_emb = self.pos_embedding(tok_emb)   # (T, D) or (B, T, D)
        return self.dropout(tok_emb + pos_emb)  # (B, T, D)
```

## Sinusoidal Encoding Intuition

The sinusoidal encoding uses different frequencies for each dimension:
- **Low dimensions**: Fast-changing patterns (sin/cos with high frequency)
- **High dimensions**: Slow-changing patterns (sin/cos with low frequency)

This creates a unique "signature" for each position!

```
Position:  0    1    2    3    4    5
Dim 0:    [0.0, 0.8, 1.0, 0.8, 0.0, -0.8] ← High frequency
Dim 1:    [1.0, 0.5, 0.0, -0.5, -1.0, -0.5]
...
Dim 510:  [0.0, 0.001, 0.002, ...] ← Low frequency
Dim 511:  [1.0, 0.999, 0.998, ...]
```

## Module Contents

- `theory.md` - Deep dive into embeddings and positional encoding
- `../../tiny_transformer/embeddings/token_embedding.py` - Token embedding
- `../../tiny_transformer/embeddings/positional_encoding.py` - Both sinusoidal and learned
- `../../tests/test_embeddings.py` - Comprehensive tests

## Success Criteria

- [ ] Understand why transformers need positional information
- [ ] Implement token embedding with scaling
- [ ] Implement sinusoidal positional encoding from scratch
- [ ] Implement learned positional embeddings
- [ ] Understand trade-offs between sinusoidal vs learned
- [ ] Pass all tests
- [ ] Visualize positional encoding patterns

## Next Module

**Module 05: Full Model Assembly** - Combining all components into a complete transformer language model.
