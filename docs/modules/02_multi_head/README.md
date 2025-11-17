# Module 02: Multi-Head Attention

Building on the attention mechanism from Module 01, we now explore **multi-head attention** - the key innovation that makes transformers powerful.

## What You'll Learn

1. **Why multiple heads?** - The motivation behind parallel attention mechanisms
2. **Head splitting** - How to partition d_model into multiple heads
3. **Parallel computation** - Running multiple attention heads simultaneously
4. **Head concatenation** - Combining outputs from all heads
5. **Output projection** - Final linear transformation
6. **Implementation** - Building MultiHeadAttention from scratch

## Prerequisites

- ✓ Completed Module 01 (Attention Mechanism)
- ✓ Understanding of scaled dot-product attention
- ✓ Comfortable with tensor reshaping and transposing

## The Big Idea

**Single-head attention** is like looking at a sentence through one lens - you capture one type of relationship.

**Multi-head attention** is like looking through multiple lenses simultaneously:
- Head 1 might focus on **syntax** (subject-verb agreement)
- Head 2 might capture **semantics** (word meanings)
- Head 3 might learn **long-range dependencies** (pronouns to antecedents)
- Head 4 might identify **positional patterns**

By running attention in parallel with different learned transformations, the model can capture **multiple types of relationships simultaneously**.

## Architecture Overview

```
Input: (batch, seq_len, d_model)
    ↓
Split into n_heads:
    Q: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
    K: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
    V: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_v)
    ↓
Parallel Attention (per head):
    Each head: (batch, seq_len, d_k) → (batch, seq_len, d_v)
    ↓
Concatenate heads:
    (batch, n_heads, seq_len, d_v) → (batch, seq_len, n_heads * d_v)
    ↓
Output projection:
    (batch, seq_len, n_heads * d_v) → (batch, seq_len, d_model)
```

## Key Concepts

### 1. Dimension Partitioning

Instead of one large attention with dimension `d_model`, we split into `n_heads` smaller attentions:

```python
d_k = d_model // n_heads  # Dimension per head
d_v = d_model // n_heads  # Usually d_k = d_v

# Example: d_model=512, n_heads=8
# → d_k = d_v = 64 per head
```

### 2. Linear Projections

Each head gets its own learned projection matrices:

```python
# For n_heads=8, d_model=512:
W_q: (d_model, d_model) = (512, 512)  # Projects to all heads at once
W_k: (d_model, d_model) = (512, 512)
W_v: (d_model, d_model) = (512, 512)
W_o: (d_model, d_model) = (512, 512)  # Output projection
```

### 3. Reshape Operations

The key to multi-head attention is reshaping:

```python
# Start: (batch, seq_len, d_model)
x = x @ W_q  # (batch, seq_len, d_model)

# Reshape to separate heads
x = x.view(batch, seq_len, n_heads, d_k)  # (batch, seq_len, n_heads, d_k)

# Transpose for parallel computation
x = x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
```

## Why Multi-Head Attention Works

### 1. **Representational Power**
- Different heads can learn different attention patterns
- One head might focus on local context, another on global
- Increases model capacity without increasing sequence length computation

### 2. **Ensemble Learning**
- Multiple "experts" (heads) vote on what to attend to
- Reduces reliance on any single attention pattern
- More robust to different types of inputs

### 3. **Computational Efficiency**
- Total computation is similar to single-head with same d_model
- But parallelism allows efficient GPU utilization
- n_heads small attentions ≈ same cost as 1 large attention

## Module Structure

```
docs/modules/02_multi_head/
├── README.md              ← You are here
├── theory.md             ← Deep dive into multi-head attention
└── exercises/
    ├── exercises.py      ← Practice problems
    └── solutions.py      ← Detailed solutions
```

## Learning Path

### 1. Read Theory (45 minutes)
**File**: `theory.md`

Deep dive into:
- Mathematical formulation
- Shape transformations at each step
- Comparison to single-head attention
- Implementation details

### 2. Study Implementation (30 minutes)
**File**: `../../tiny_transformer/multi_head.py`

Examine the `MultiHeadAttention` class:
- Linear projections for Q, K, V
- Head splitting logic
- Attention computation
- Head concatenation
- Output projection

### 3. Complete Exercises (60 minutes)
**File**: `exercises/exercises.py`

Practice:
- Reshaping for multi-head
- Computing multi-head attention manually
- Debugging shape errors
- Implementing from scratch

### 4. Review Solutions (30 minutes)
**File**: `exercises/solutions.py`

Study detailed implementations with explanations.

### 5. Run Tests (15 minutes)

```bash
pytest tests/test_multi_head.py -v
```

Verify your understanding with comprehensive test suite.

## Key Formulas

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

### With Dimensions

```
Input:  (batch, seq_len, d_model)

Q W^Q → (batch, seq_len, d_model) @ (d_model, d_model)
      = (batch, seq_len, d_model)
      → reshape → (batch, n_heads, seq_len, d_k)

Attention per head:
      (batch, n_heads, seq_len, d_k) → (batch, n_heads, seq_len, d_v)

Concatenate:
      (batch, n_heads, seq_len, d_v) → (batch, seq_len, n_heads * d_v)

Output projection:
      (batch, seq_len, n_heads * d_v) @ (n_heads * d_v, d_model)
      = (batch, seq_len, d_model)
```

## Common Hyperparameters

| Model Size | d_model | n_heads | d_k (per head) | Total params |
|------------|---------|---------|----------------|--------------|
| Tiny       | 128     | 4       | 32             | ~66K         |
| Small      | 256     | 8       | 32             | ~263K        |
| Base       | 512     | 8       | 64             | ~1M          |
| Large      | 1024    | 16      | 64             | ~4M          |

Note: d_k = d_model / n_heads (typically)

## Success Criteria

You're ready for Module 03 when you can:

- [ ] Explain why we use multiple attention heads
- [ ] Trace tensor shapes through all multi-head operations
- [ ] Implement head splitting and concatenation
- [ ] Understand the role of each projection matrix (W^Q, W^K, W^V, W^O)
- [ ] Debug multi-head attention shape errors
- [ ] Pass all tests in `test_multi_head.py`

## Common Pitfalls

### 1. **Forgetting to transpose for parallel computation**
```python
# Wrong: (batch, seq_len, n_heads, d_k)
# Right: (batch, n_heads, seq_len, d_k)
x = x.transpose(1, 2)
```

### 2. **Incorrect concatenation**
```python
# After transpose: (batch, n_heads, seq_len, d_v)
# Need: (batch, seq_len, n_heads * d_v)

# Wrong:
x = x.view(batch, seq_len, -1)  # Lost head dimension!

# Right:
x = x.transpose(1, 2)  # First swap back to (batch, seq_len, n_heads, d_v)
x = x.contiguous().view(batch, seq_len, -1)  # Then flatten
```

### 3. **Dimension mismatch in projections**
```python
# W_o should map from (n_heads * d_v) back to d_model
# If d_v = d_k = d_model // n_heads, then n_heads * d_v = d_model

self.W_o = nn.Linear(n_heads * d_v, d_model)  # Correct
```

## Visualization

Multi-head attention creates **multiple attention patterns** simultaneously:

```
Input: "The cat sat on the mat"

Head 1 (Local):         Head 2 (Syntax):        Head 3 (Semantics):
The → The              The → cat              The → the, mat
cat → The, cat, sat    cat → sat              cat → cat, mat
sat → cat, sat, on     sat → cat, on          sat → sat
on  → sat, on, the     on → the, mat          on  → on
the → on, the, mat     the → mat              the → The, the
mat → the, mat         mat → sat, mat         mat → cat, mat

Combined output: Rich representation incorporating all patterns!
```

## Next Module

**Module 03: Transformer Blocks** - Combining multi-head attention with feed-forward networks, layer normalization, and residual connections to build complete transformer blocks.

---

**Ready to learn multi-head attention?** Start with `theory.md` →
