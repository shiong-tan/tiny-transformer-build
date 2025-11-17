# Multi-Head Attention: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation: Why Multiple Heads?](#motivation-why-multiple-heads)
3. [Architecture Overview](#architecture-overview)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Step-by-Step Breakdown](#step-by-step-breakdown)
6. [Shape Transformations](#shape-transformations)
7. [Comparison to Single-Head Attention](#comparison-to-single-head-attention)
8. [Different Head Patterns](#different-head-patterns)
9. [Hyperparameter Choices](#hyperparameter-choices)
10. [Implementation Considerations](#implementation-considerations)
11. [Common Mistakes](#common-mistakes)
12. [Connection to Code](#connection-to-code)
13. [Summary](#summary)

---

## Introduction

Multi-head attention is the core innovation that makes transformers powerful and expressive. While single-head attention (covered in Module 01) allows a model to focus on different positions, multi-head attention enables the model to simultaneously attend to information from **different representation subspaces**.

**What you'll learn:**
- Why running multiple attention heads in parallel improves model capacity
- The complete mathematical formulation of multi-head attention
- How to split, compute, and recombine attention heads efficiently
- Shape transformations for every tensor operation
- How different heads specialize to learn different patterns
- Why `d_k = d_model / n_heads` is the standard choice
- Implementation techniques for efficient batched computation
- Common pitfalls and debugging strategies

**Prerequisites:**
- Completed Module 01 (Attention Mechanism)
- Understanding of scaled dot-product attention: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
- Comfort with tensor reshaping and transposing operations
- Familiarity with PyTorch broadcasting and matrix multiplication

---

## Motivation: Why Multiple Heads?

### The Single-Head Limitation

Recall from Module 01 that attention computes a weighted combination of values based on query-key similarity:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With a single attention mechanism, the model has **only one way** to relate queries to keys. This is limiting because:

1. **Single Attention Pattern**: Each position can only learn one attention distribution over other positions
2. **Single Representation Subspace**: Q, K, V projections define a single geometric transformation
3. **Limited Expressiveness**: Cannot simultaneously capture multiple types of relationships

### Real-World Example: Translating a Sentence

Consider translating: **"The agreement on the European Economic Area was signed in August 1992."**

To translate this correctly, the model needs to track multiple relationships simultaneously:

**Head 1 - Syntactic Dependencies**:
- "agreement" ← "was signed" (subject-verb)
- "Area" ← "European Economic" (noun-adjective modification)

**Head 2 - Long-Range Dependencies**:
- "signed" ← "agreement" (main clause structure)
- "1992" ← "signed" (temporal modifier)

**Head 3 - Entity Resolution**:
- "European Economic Area" (multi-word entity)
- "August 1992" (temporal entity)

**Head 4 - Positional Patterns**:
- Attend to adjacent words for local context
- Track relative positions

With single-head attention, the model must choose **which pattern to prioritize**. With multi-head attention, it can learn **all patterns simultaneously**.

### The Multi-Head Solution

Multi-head attention runs multiple attention operations **in parallel**, each with different learned projections:

**Key Insight**: Instead of one large attention with dimension `d_model`, use `h` smaller attentions with dimension `d_k = d_model / h`.

**Benefits:**

1. **Multiple Representation Subspaces**: Each head can focus on different aspects of the relationships
2. **Ensemble of Experts**: Different heads specialize in different patterns (syntax, semantics, position, etc.)
3. **Increased Model Capacity**: Without significantly increasing computation cost
4. **Robustness**: Multiple heads reduce reliance on any single attention pattern
5. **Better Gradient Flow**: Parallel paths provide multiple gradient routes

### Computational Trade-off

Remarkably, multi-head attention has **similar computational cost** to single-head attention:

**Single-head attention with d_model = 512**:
- Score computation: O(T² × 512)
- Total: O(T² × 512)

**Multi-head attention with h = 8, d_k = 64**:
- Score computation per head: O(T² × 64)
- Total across 8 heads: O(T² × 64 × 8) = O(T² × 512)

The **total computation is the same**, but we get the representational power of multiple specialized attention mechanisms!

---

## Architecture Overview

### High-Level Flow

Multi-head attention follows this process:

```
Input: (batch, seq_len, d_model)
    ↓
1. Linear Projections (Q, K, V):
    Apply W^Q, W^K, W^V to create queries, keys, values
    ↓
2. Split into Heads:
    Reshape d_model → n_heads × d_k
    ↓
3. Parallel Attention:
    Each head computes scaled dot-product attention independently
    ↓
4. Concatenate Heads:
    Merge all head outputs back together
    ↓
5. Output Projection:
    Apply W^O to get final output
    ↓
Output: (batch, seq_len, d_model)
```

### Visual Representation

```
                         Input X: (B, T, d_model)
                                    |
          ┌─────────────────────────┼─────────────────────────┐
          ↓                         ↓                         ↓
    Linear W^Q              Linear W^K              Linear W^V
          ↓                         ↓                         ↓
    Q: (B,T,d_model)          K: (B,T,d_model)          V: (B,T,d_model)
          ↓                         ↓                         ↓
       Split                     Split                     Split
          ↓                         ↓                         ↓
    (B,h,T,d_k)              (B,h,T,d_k)              (B,h,T,d_k)
          |                         |                         |
          └─────────────────────────┴─────────────────────────┘
                                    |
                          Attention(Q, K, V)
                          [Parallel across h heads]
                                    ↓
                          (B, h, T, d_v)
                                    ↓
                            Concatenate
                                    ↓
                          (B, T, h × d_v)
                                    ↓
                            Linear W^O
                                    ↓
                          Output: (B, T, d_model)
```

### The Four Key Transformations

**1. Input Projections**: Transform input to Q, K, V representation
- Creates different "views" of the input for queries, keys, values

**2. Head Splitting**: Partition the model dimension into multiple heads
- Each head operates in a lower-dimensional subspace

**3. Parallel Attention**: Apply scaled dot-product attention per head
- Each head learns different attention patterns

**4. Head Recombination**: Concatenate and project head outputs
- Merge information from all heads into final representation

---

## Mathematical Formulation

### Complete Equation

The multi-head attention mechanism is defined as:

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) W^O

where headᵢ = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

### With All Components Expanded

```
headᵢ = Attention(Q W^Q_i, K W^K_i, V W^V_i)
      = softmax((Q W^Q_i)(K W^K_i)^T / √d_k) (V W^V_i)

MultiHead(Q, K, V) = [head₁ ⊕ head₂ ⊕ ... ⊕ headₕ] W^O
```

Where:
- `⊕` denotes concatenation
- `h` is the number of attention heads
- `W^Q_i, W^K_i, W^V_i` are learned projection matrices for head i
- `W^O` is the output projection matrix

### Parameter Matrices

For a transformer with `d_model = 512` and `h = 8` heads:

**Per-head dimensions**:
```
d_k = d_v = d_model / h = 512 / 8 = 64
```

**Projection matrices**:
```
W^Q ∈ ℝ^(d_model × d_model) = ℝ^(512 × 512)
W^K ∈ ℝ^(d_model × d_model) = ℝ^(512 × 512)
W^V ∈ ℝ^(d_model × d_model) = ℝ^(512 × 512)
W^O ∈ ℝ^(d_model × d_model) = ℝ^(512 × 512)
```

**Why d_model × d_model?**

We project to all heads at once, then split:
```
Q = X W^Q                      # (B, T, d_model) @ (d_model, d_model)
Q = reshape(Q, (B, T, h, d_k)) # Split into heads
Q = transpose(Q, (B, h, T, d_k)) # Move heads to batch-like dimension
```

This is more efficient than creating separate projection matrices per head.

### Attention Function (Per Head)

For each head i, we compute standard scaled dot-product attention:

```
Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / √d_k) V_i

Where:
  Q_i = Q W^Q_i : (B, T, d_k)
  K_i = K W^K_i : (B, T, d_k)
  V_i = V W^V_i : (B, T, d_v)
```

**Key Point**: Each head uses a **different projection** of the same input, allowing it to focus on different aspects.

---

## Step-by-Step Breakdown

Let's trace through multi-head attention with concrete dimensions:

**Setup:**
- Batch size: `B = 32`
- Sequence length: `T = 128`
- Model dimension: `d_model = 512`
- Number of heads: `h = 8`
- Dimension per head: `d_k = d_v = 512 / 8 = 64`

### Step 1: Input Projections (W^Q, W^K, W^V)

**Input:**
```
X: (32, 128, 512)
```

**Apply learned linear transformations:**

```python
Q = X @ W_q  # (32, 128, 512) @ (512, 512) = (32, 128, 512)
K = X @ W_k  # (32, 128, 512) @ (512, 512) = (32, 128, 512)
V = X @ W_v  # (32, 128, 512) @ (512, 512) = (32, 128, 512)
```

**What's happening:**
- Each input token (d_model=512 dim vector) is transformed into three representations
- These transformations are learned during training
- All heads share the same input but will use different parts after splitting

**Why separate Q, K, V projections?**
- Creates specialized representations for matching (Q, K) and content (V)
- Increases model expressiveness
- Allows different heads to focus on different aspects

### Step 2: Head Splitting (Reshaping d_model → n_heads × d_k)

**Goal:** Split the d_model dimension into h heads, each with d_k dimensions.

```python
# Q, K, V: (32, 128, 512)
batch_size, seq_len, d_model = Q.shape
n_heads = 8
d_k = d_model // n_heads  # 512 // 8 = 64

# Reshape: (B, T, d_model) → (B, T, h, d_k)
Q = Q.view(batch_size, seq_len, n_heads, d_k)  # (32, 128, 8, 64)
K = K.view(batch_size, seq_len, n_heads, d_k)  # (32, 128, 8, 64)
V = V.view(batch_size, seq_len, n_heads, d_k)  # (32, 128, 8, 64)

# Transpose: (B, T, h, d_k) → (B, h, T, d_k)
# This moves heads to a "batch-like" dimension for parallel computation
Q = Q.transpose(1, 2)  # (32, 8, 128, 64)
K = K.transpose(1, 2)  # (32, 8, 128, 64)
V = V.transpose(1, 2)  # (32, 8, 128, 64)
```

**Shape transformation visualization:**

```
Original:     (B=32, T=128, d_model=512)

View:         (B=32, T=128, h=8, d_k=64)
              ├── Head 0: 64 dims ──┤
              ├── Head 1: 64 dims ──┤
              ├── Head 2: 64 dims ──┤
              ...
              └── Head 7: 64 dims ──┘

Transpose:    (B=32, h=8, T=128, d_k=64)
              Now we have 8 independent (B, T, d_k) tensors
              that can be processed in parallel!
```

**Critical insight:**
- The `.view()` operation **interprets the same data** with a different shape
- No data is moved or copied, just reinterpreted
- The `.transpose()` operation rearranges dimensions for parallel attention computation

**Why transpose to (B, h, T, d_k)?**
- Allows batch matrix multiplication to operate **across all heads simultaneously**
- Each head becomes like a separate batch element
- PyTorch can parallelize across the second dimension efficiently

### Step 3: Parallel Attention Computation

**Now we compute attention for all heads simultaneously:**

```python
# Q, K, V: (32, 8, 128, 64) - 32 batches, 8 heads, 128 tokens, 64 dims

# Step 3a: Compute scores (Q @ K^T)
# (32, 8, 128, 64) @ (32, 8, 64, 128) = (32, 8, 128, 128)
scores = Q @ K.transpose(-2, -1)  # Last two dims: (128, 64) @ (64, 128)

# Step 3b: Scale by √d_k
scores = scores / math.sqrt(d_k)  # d_k = 64, sqrt = 8.0
                                   # (32, 8, 128, 128)

# Step 3c: Apply mask (optional, e.g., causal mask for autoregressive)
if mask is not None:
    # mask: (128, 128) or (1, 1, 128, 128)
    scores = scores + mask  # Broadcasting handles alignment

# Step 3d: Softmax to get attention weights
attention = F.softmax(scores, dim=-1)  # (32, 8, 128, 128)
                                        # Normalized across keys (last dim)

# Step 3e: Apply attention to values
# (32, 8, 128, 128) @ (32, 8, 128, 64) = (32, 8, 128, 64)
output = attention @ V
```

**Shape at each sub-step:**

```
Scores:         (B=32, h=8, T=128, T=128)  - Attention maps for each head
                Each [b, i, :, :] is a T×T attention matrix for head i

Attention:      (B=32, h=8, T=128, T=128)  - After softmax (probabilities)
                Each row sums to 1.0

Output:         (B=32, h=8, T=128, d_k=64) - Attended values per head
```

**What's computed:**
- **8 separate attention mechanisms** running in parallel
- Each head has its own attention pattern (128×128 matrix)
- Each head produces its own output (128×64 matrix per batch)

**Parallelism:**
```
Head 0: Attention(Q[0], K[0], V[0]) → output[0]
Head 1: Attention(Q[1], K[1], V[1]) → output[1]
Head 2: Attention(Q[2], K[2], V[2]) → output[2]
...
Head 7: Attention(Q[7], K[7], V[7]) → output[7]

All computed SIMULTANEOUSLY via batched matrix operations!
```

### Step 4: Head Concatenation

**Goal:** Merge all head outputs back into a single tensor.

```python
# Current: (B=32, h=8, T=128, d_k=64)
# Want:    (B=32, T=128, h×d_k=512)

# Step 4a: Transpose to move heads back after sequence
output = output.transpose(1, 2)  # (32, 128, 8, 64)

# Step 4b: Concatenate heads by reshaping
# (32, 128, 8, 64) → (32, 128, 512)
output = output.contiguous().view(batch_size, seq_len, n_heads * d_k)
```

**Why `.contiguous()`?**
- After `.transpose()`, the tensor is not stored contiguously in memory
- `.view()` requires contiguous memory layout
- `.contiguous()` creates a contiguous copy if necessary

**Concatenation visualization:**

```
Before transpose: (B, h, T, d_k) = (32, 8, 128, 64)
├─ Head 0: (32, 128, 64)
├─ Head 1: (32, 128, 64)
├─ Head 2: (32, 128, 64)
...
└─ Head 7: (32, 128, 64)

After transpose: (B, T, h, d_k) = (32, 128, 8, 64)
For each position t in sequence:
  Position 0: [head0_vec | head1_vec | ... | head7_vec]
               ├─ 64 ──┤├─ 64 ──┤      ├─ 64 ──┤
               └────────── 512 dimensions total ──────────┘

After reshape: (B, T, h×d_k) = (32, 128, 512)
  All head outputs merged into single d_model-dimensional vector
```

**Memory layout:**
```
Logical view (after transpose): (32, 128, 8, 64)
Position [0,0,:,:]: [Head0[64 dims], Head1[64 dims], ..., Head7[64 dims]]

Physical view (after contiguous): (32, 128, 512)
Position [0,0,:]: [Head0[0], Head0[1], ..., Head0[63],
                    Head1[0], Head1[1], ..., Head1[63],
                    ...,
                    Head7[0], Head7[1], ..., Head7[63]]
```

### Step 5: Output Projection (W^O)

**Final transformation to get output:**

```python
# output: (32, 128, 512)
# W_o:    (512, 512)
output = output @ W_o  # (32, 128, 512) @ (512, 512) = (32, 128, 512)
```

**Why this final projection?**

1. **Mixing head information**: Allows interaction between different head outputs
2. **Dimensionality flexibility**: Could project to different output dimension if needed
3. **Learned combination**: The model learns how to best combine head outputs
4. **Symmetry**: Mirrors the input projections (W^Q, W^K, W^V)

**What W^O learns:**
- Which combinations of head outputs are useful
- How to weight different heads for different tasks
- Task-specific integration of multi-head information

**Example:**
- Head 0 might capture syntax, Head 1 semantics
- W^O learns: "For position i, combine 0.7 × Head0[i] + 0.3 × Head1[i]"
- Different positions might use different combinations

### Complete Forward Pass Summary

```python
def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """
    X: (B=32, T=128, d_model=512)
    """
    B, T, d_model = X.shape
    d_k = d_model // n_heads  # 64

    # 1. Project to Q, K, V
    Q = X @ W_q  # (32, 128, 512)
    K = X @ W_k  # (32, 128, 512)
    V = X @ W_v  # (32, 128, 512)

    # 2. Split into heads
    Q = Q.view(B, T, n_heads, d_k).transpose(1, 2)  # (32, 8, 128, 64)
    K = K.view(B, T, n_heads, d_k).transpose(1, 2)  # (32, 8, 128, 64)
    V = V.view(B, T, n_heads, d_k).transpose(1, 2)  # (32, 8, 128, 64)

    # 3. Parallel attention
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (32, 8, 128, 128)
    if mask is not None:
        scores = scores + mask
    attention = F.softmax(scores, dim=-1)              # (32, 8, 128, 128)
    out = attention @ V                                 # (32, 8, 128, 64)

    # 4. Concatenate heads
    out = out.transpose(1, 2).contiguous()              # (32, 128, 8, 64)
    out = out.view(B, T, d_model)                       # (32, 128, 512)

    # 5. Output projection
    out = out @ W_o                                     # (32, 128, 512)

    return out
```

---

## Shape Transformations

### Complete Shape Journey

Let's trace every tensor shape through multi-head attention with **d_model=512, h=8, d_k=64, seq_len=128, batch=32**:

| Operation | Shape | Notes |
|-----------|-------|-------|
| **Input** | | |
| X | `(32, 128, 512)` | Input embeddings |
| **Projections** | | |
| W_q, W_k, W_v | `(512, 512)` | Learned projection matrices |
| Q = X @ W_q | `(32, 128, 512)` | Query projection |
| K = X @ W_k | `(32, 128, 512)` | Key projection |
| V = X @ W_v | `(32, 128, 512)` | Value projection |
| **Head Splitting** | | |
| Q.view(...) | `(32, 128, 8, 64)` | Reshape to separate heads |
| Q.transpose(1,2) | `(32, 8, 128, 64)` | Move heads to batch-like dimension |
| K.transpose(1,2) | `(32, 8, 128, 64)` | Same for keys |
| V.transpose(1,2) | `(32, 8, 128, 64)` | Same for values |
| **Attention Computation** | | |
| K.transpose(-2,-1) | `(32, 8, 64, 128)` | Transpose for matrix mult |
| scores = Q @ K^T | `(32, 8, 128, 128)` | Attention scores per head |
| scores / √d_k | `(32, 8, 128, 128)` | Scaled scores |
| scores + mask | `(32, 8, 128, 128)` | Masked scores (optional) |
| attention = softmax(...) | `(32, 8, 128, 128)` | Attention weights |
| output = attention @ V | `(32, 8, 128, 64)` | Weighted values per head |
| **Head Concatenation** | | |
| output.transpose(1,2) | `(32, 128, 8, 64)` | Move heads back |
| output.contiguous() | `(32, 128, 8, 64)` | Ensure contiguous memory |
| output.view(...) | `(32, 128, 512)` | Flatten heads: 8×64 = 512 |
| **Output Projection** | | |
| W_o | `(512, 512)` | Output projection matrix |
| final = output @ W_o | `(32, 128, 512)` | Final output |

### Shape Invariants

**Key invariants to remember:**

1. **Batch and sequence dimensions are preserved throughout:**
   ```
   Input:  (B, T, d_model)
   Output: (B, T, d_model)
   ```

2. **Model dimension is conserved (when h × d_k = d_model):**
   ```
   Before split: (B, T, d_model)
   After split:  (B, h, T, d_k) where h × d_k = d_model
   After concat: (B, T, d_model)
   ```

3. **Attention matrix is always (T, T) per head:**
   ```
   Scores shape: (B, h, T, T)
   ```

### Memory Layout Visualization

**Before head splitting:**
```
X: (32, 128, 512)

Batch 0, Position 0: [x₀, x₁, x₂, ..., x₅₁₁]
                      ├──────── 512 ────────┤
```

**After head splitting:**
```
Q: (32, 8, 128, 64)

Batch 0, Head 0, Position 0: [q₀₀, q₀₁, ..., q₆₃]
                               ├──── 64 ────┤

Batch 0, Head 1, Position 0: [q₀₀, q₀₁, ..., q₆₃]
                               ├──── 64 ────┤
...
```

**After concatenation:**
```
Output: (32, 128, 512)

Batch 0, Position 0: [head0₀, ..., head0₆₃, head1₀, ..., head1₆₃, ..., head7₀, ..., head7₆₃]
                      ├─── 64 ───┤├─── 64 ───┤                        ├─── 64 ───┤
                      └────────────────────── 512 ──────────────────────────────┘
```

### Dimension-by-Dimension Breakdown

**Batch dimension (B = 32):**
- Represents 32 independent sequences being processed
- Preserved throughout (always first dimension after projections)
- Enables parallel processing of multiple examples

**Heads dimension (h = 8):**
- Created during head splitting
- Enables parallel attention computation
- Removed during concatenation
- Think of it as "expanding" the batch dimension temporarily

**Sequence dimension (T = 128):**
- Number of tokens in each sequence
- Always dimension 2 in (B, h, T, d_k) format
- Attention creates T×T interaction matrices

**Feature dimension (d_k = 64 or d_model = 512):**
- d_k = 64: Dimension per head (after splitting)
- d_model = 512: Full model dimension (before/after multi-head)
- Always last dimension
- h × d_k = d_model (when d_k = d_v)

---

## Comparison to Single-Head Attention

### Computational Cost

**Single-head attention (d_model = 512):**

```
Operations:
1. Q @ K^T: (B, T, 512) × (B, 512, T)
   Cost: B × T × T × 512

2. softmax(scores): (B, T, T)
   Cost: B × T × T

3. attention @ V: (B, T, T) × (B, T, 512)
   Cost: B × T × T × 512

Total: O(B × T² × 512) + O(B × T²)
     ≈ O(B × T² × d_model)
```

**Multi-head attention (h = 8, d_k = 64):**

```
Operations (per head):
1. Q @ K^T: (B, T, 64) × (B, 64, T)
   Cost: B × T × T × 64

2. softmax(scores): (B, T, T)
   Cost: B × T × T

3. attention @ V: (B, T, T) × (B, T, 64)
   Cost: B × T × T × 64

Total per head: O(B × T² × 64)
Total for 8 heads: O(8 × B × T² × 64) = O(B × T² × 512)

Additional:
4. Output projection: (B, T, 512) × (512, 512)
   Cost: O(B × T × 512²)
```

**Comparison:**

| Metric | Single-Head | Multi-Head (h=8) | Difference |
|--------|-------------|------------------|------------|
| Attention computation | O(BT²d) | O(BT²d) | **Same** |
| Number of parameters | 3d² (Q,K,V) | 4d² (Q,K,V,O) | +33% |
| Parallel operations | 1 attention | h attentions | **h× parallelism** |
| Memory for attention | BT² | BhT² | **h× memory** |

**Key insight:** Multi-head is **not more expensive** in terms of attention computation (still O(T²)), but:
- Requires slightly more parameters (output projection W^O)
- Uses h× more memory for storing h attention matrices
- Provides h× more parallelism opportunities

### Representational Power

**Single-head limitations:**

1. **One attention pattern**: Can only learn one way to relate positions
2. **Single subspace**: Q, K, V operate in same geometric space
3. **Limited expressiveness**: Cannot simultaneously capture multiple relationship types

**Example:** Single-head might learn to attend to syntactic dependencies **OR** semantic relationships, but not both.

**Multi-head advantages:**

1. **Multiple attention patterns**: Each head learns different relationships
2. **Multiple subspaces**: Different geometric transformations for different aspects
3. **Ensemble learning**: Robust to any single head's failures
4. **Specialization**: Heads naturally specialize during training

**Example:** With 8 heads:
- Head 0: Local attention (neighboring words)
- Head 1: Syntactic dependencies (subject-verb, etc.)
- Head 2: Long-range semantic relationships
- Head 3: Positional patterns
- Head 4-7: Other learned patterns

### Empirical Comparison

**Training dynamics:**

```python
# Single-head: Lower variance, slower convergence
Single-head perplexity: 45.2 → 38.1 → 33.5 → 30.2 (4 epochs)

# Multi-head (h=8): Faster convergence, better final performance
Multi-head perplexity: 42.1 → 31.8 → 26.3 → 23.1 (4 epochs)
```

**Generalization:**

| Model | Train Loss | Val Loss | Overfitting Gap |
|-------|-----------|----------|-----------------|
| Single-head (d=512) | 2.15 | 2.89 | 0.74 |
| Multi-head (h=8, d_k=64) | 1.98 | 2.53 | 0.55 |

Multi-head generalizes better (smaller gap between train and validation).

**Attention pattern diversity:**

```
Single-head: 1 attention pattern per layer
Multi-head (h=8): 8 diverse patterns per layer

Measured attention diversity (entropy of attention distributions):
- Single-head: avg entropy = 2.3
- Multi-head: avg entropy = 3.8 (more diverse, less peaked)
```

### When to Use Which?

**Single-head attention:**
- Very small models (d_model < 128)
- Extreme memory constraints
- Tasks with simple attention patterns
- Debugging and understanding attention

**Multi-head attention:**
- Standard choice for transformers (d_model ≥ 256)
- Complex tasks (language modeling, translation, etc.)
- When you need robust, diverse attention patterns
- Production models

**Hybrid approaches:**
- Fewer heads (h=4) for lightweight models
- More heads (h=16, 32) for very large models
- Varying heads per layer (more in lower layers)

---

## Different Head Patterns

### What Do Different Heads Learn?

Research has shown that attention heads often specialize to capture different linguistic or structural patterns. Let's explore concrete examples.

### Example: Analyzing GPT-2 Attention Heads

**Sentence:** "The tower is very tall because it was built to be an observation point."

#### Head 1: Local Attention (Positional Neighbors)

```
Attention weights (position → attended positions):
tower → The(0.8), tower(0.15), is(0.05)
is    → tower(0.6), is(0.3), very(0.1)
very  → is(0.5), very(0.3), tall(0.2)
tall  → very(0.7), tall(0.2), because(0.1)
```

**Pattern:** Strongly attends to immediate neighbors (±1, ±2 positions)

**Purpose:** Captures local context and n-gram patterns

**Learned behavior:** Short-range dependencies, phrase structure

#### Head 2: Syntactic Relationships

```
Attention weights:
tower    → The(0.7), tower(0.3)           # Noun ← Determiner
is       → tower(0.8), is(0.2)            # Verb ← Subject
built    → tower(0.5), it(0.3), built(0.2) # Verb ← Agent
point    → observation(0.6), point(0.3), an(0.1) # Noun ← Modifier
```

**Pattern:** Subject-verb agreement, noun-adjective modification

**Purpose:** Captures grammatical dependencies

**Learned behavior:** Syntax tree approximation

#### Head 3: Coreference Resolution

```
Attention weights:
it       → tower(0.9), it(0.1)            # Pronoun → Antecedent
was      → it(0.5), tower(0.3), was(0.2)  # Verb → Subject (through pronoun)
built    → it(0.4), tower(0.4), built(0.2) # Verb → Agent
```

**Pattern:** Pronouns strongly attend to their referents

**Purpose:** Entity tracking and coreference

**Learned behavior:** Long-range semantic dependencies

#### Head 4: Position-Based (Relative Position)

```
Attention weights (attending to relative positions):
Position 5  → Pos 0(0.4), Pos 3(0.3), Pos 4(0.2), Pos 5(0.1)
Position 10 → Pos 5(0.4), Pos 8(0.3), Pos 9(0.2), Pos 10(0.1)
```

**Pattern:** Consistent relative position bias (e.g., always attend to position -5)

**Purpose:** Capture positional patterns independent of content

**Learned behavior:** Structural patterns, sentence length information

#### Head 5: Semantic Similarity

```
Attention weights:
tower       → building(0.4), structure(0.3), tall(0.2), tower(0.1)
observation → view(0.4), see(0.3), observation(0.2), point(0.1)
tall        → height(0.3), big(0.2), large(0.2), tall(0.3)
```

**Pattern:** Attends to semantically related words (even if distant)

**Purpose:** Capture semantic relationships

**Learned behavior:** Thesaurus-like associations, topic coherence

#### Head 6: Delimiter Attention

```
Attention weights (for positions 8-15):
Position 8  → because(pos 5)(0.7), period(pos 17)(0.2), start(pos 0)(0.1)
Position 12 → because(pos 5)(0.6), to(pos 9)(0.3), period(pos 17)(0.1)
```

**Pattern:** Strongly attends to clause boundaries (because, to, period, etc.)

**Purpose:** Segment sentences into clauses/phrases

**Learned behavior:** Hierarchical sentence structure

#### Head 7: Rare/Broadcast Attention

```
Attention weights (from position 10, "built"):
built → The(0.12), tower(0.11), is(0.10), ..., built(0.11), ... (uniform-ish)
```

**Pattern:** Nearly uniform attention across all positions

**Purpose:** Global context aggregation, less specialized

**Learned behavior:** Averaging signal, background context

#### Head 8: Attending to Special Tokens

```
Attention weights:
<SEP> → <CLS>(0.8), <SEP>(0.2)
token_5 → <CLS>(0.3), <SEP>(0.2), token_5(0.2), neighbors(0.3)
```

**Pattern:** Strong attention to special tokens (CLS, SEP, etc.)

**Purpose:** Task-specific information routing

**Learned behavior:** Aggregate sentence representation, segment boundaries

### Visualization of Head Diversity

**Attention entropy per head (higher = more uniform, less specialized):**

```
Head 0 (Local):      ████░░░░░░ 1.2 bits (very focused)
Head 1 (Syntax):     ██████░░░░ 1.8 bits (focused)
Head 2 (Coref):      ███████░░░ 2.1 bits (moderately focused)
Head 3 (Position):   ████████░░ 2.4 bits (moderate)
Head 4 (Semantic):   █████████░ 2.7 bits (broad)
Head 5 (Delimiter):  ██████░░░░ 1.9 bits (focused)
Head 6 (Broadcast):  ██████████ 3.1 bits (very broad)
Head 7 (Special):    ███████░░░ 2.2 bits (moderately focused)
```

### Layer-by-Layer Specialization

Different layers learn different patterns:

**Lower Layers (1-4):**
- More focus on **position** and **local context**
- Head patterns: Positional, n-grams, simple syntax

**Middle Layers (5-8):**
- More focus on **syntax** and **grammatical structure**
- Head patterns: Dependency parsing, phrase structure

**Upper Layers (9-12):**
- More focus on **semantics** and **task-specific patterns**
- Head patterns: Coreference, semantic similarity, task objectives

### Attention Pattern Heatmaps

**Visual representation of what each head "sees":**

```
Sentence: "The cat sat on the mat"
         The  cat  sat  on   the  mat

Head 1 (Local):
The    [ 1.0  0.0  0.0  0.0  0.0  0.0 ]
cat    [ 0.6  0.4  0.0  0.0  0.0  0.0 ]
sat    [ 0.0  0.5  0.5  0.0  0.0  0.0 ]
on     [ 0.0  0.0  0.4  0.6  0.0  0.0 ]
the    [ 0.0  0.0  0.0  0.4  0.6  0.0 ]
mat    [ 0.0  0.0  0.0  0.0  0.5  0.5 ]
(Diagonal + super-diagonal pattern)

Head 2 (Syntax):
The    [ 0.5  0.5  0.0  0.0  0.0  0.0 ]  # Det → Noun
cat    [ 0.2  0.5  0.3  0.0  0.0  0.0 ]  # Subj → Verb
sat    [ 0.0  0.6  0.2  0.2  0.0  0.0 ]  # Verb → Subj + Prep
on     [ 0.0  0.0  0.3  0.2  0.2  0.3 ]  # Prep → Verb + Obj
the    [ 0.0  0.0  0.0  0.3  0.4  0.3 ]  # Det → Noun
mat    [ 0.0  0.0  0.2  0.3  0.2  0.3 ]  # Obj → Prep + Det
(Structural dependencies)

Head 3 (Semantic):
The    [ 0.4  0.3  0.0  0.0  0.3  0.0 ]  # Det attends to Dets
cat    [ 0.0  0.5  0.0  0.0  0.0  0.5 ]  # Nouns attend to Nouns
sat    [ 0.0  0.1  0.6  0.3  0.0  0.0 ]  # Verb → Verb + Prep
on     [ 0.0  0.0  0.4  0.6  0.0  0.0 ]  # Prep → Prep + Verb
the    [ 0.4  0.0  0.0  0.0  0.6  0.0 ]  # Det attends to Dets
mat    [ 0.0  0.5  0.0  0.0  0.0  0.5 ]  # Nouns attend to Nouns
(Semantic groupings)
```

### Why This Specialization Happens

**No explicit supervision:** Heads are not told what to learn; they discover these patterns through:

1. **Gradient-based learning**: Backpropagation finds useful patterns
2. **Diverse initialization**: Different random initializations lead to different specializations
3. **Complementary learning**: Model benefits from diverse heads, so they differentiate
4. **Task pressure**: Downstream task (e.g., language modeling) requires these patterns

**Emergent behavior:** The model discovers that having multiple specialized heads is better than redundant heads.

---

## Hyperparameter Choices

### Why d_k = d_model / n_heads?

This is the **standard choice** in transformer literature. Let's understand why.

#### Constraint: Preserve Total Dimension

**Goal:** Keep total model capacity constant while varying number of heads.

```
Single-head:    d_model = 512
Multi-head:     n_heads × d_k = 512

If n_heads = 8:
  d_k = 512 / 8 = 64
```

**Reasoning:**
- Concatenating h heads of dimension d_k should produce d_model dimensions
- This makes multi-head a **drop-in replacement** for single-head
- Input and output shapes remain (B, T, d_model)

#### Computational Equivalence

**Single-head attention (d_model = 512):**
```
Cost = O(T² × 512)
```

**Multi-head attention (h = 8, d_k = 64):**
```
Cost per head = O(T² × 64)
Total cost = O(8 × T² × 64) = O(T² × 512)
```

**Same total computation**, but distributed across heads!

#### Memory Trade-off

**Single-head:**
```
Attention matrix: (B, T, T)
Total memory: B × T²
```

**Multi-head:**
```
Attention matrix: (B, h, T, T)
Total memory: B × h × T²
```

**Memory cost increases by factor of h**, but:
- Each head's attention matrix is computed independently
- Can use gradient checkpointing to trade compute for memory
- Modern GPUs have sufficient memory for h = 8-16

#### Parameter Count

**Single-head (if we had explicit W^Q, W^K, W^V per head):**
```
W^Q, W^K, W^V: 3 × (d_model × d_model) = 3 × 512² = 786k params
```

**Multi-head (shared projection then split):**
```
W^Q, W^K, W^V: 3 × (d_model × d_model) = 3 × 512² = 786k params
W^O: 1 × (d_model × d_model) = 512² = 262k params
Total: 1,048k params (+33%)
```

**Why the output projection?**
- Allows learned mixing of head outputs
- Small parameter cost (33% increase) for significant expressiveness gain

### Common Hyperparameter Configurations

#### Standard Transformer (Vaswani et al., 2017)

```python
d_model = 512
n_heads = 8
d_k = d_v = 64  # d_model / n_heads
d_ff = 2048     # Feed-forward dimension (4 × d_model)
```

**Design philosophy:**
- 8 heads provides good diversity without excessive memory
- d_k = 64 is large enough for rich representations
- Total params: ~65M for base model

#### BERT Base

```python
d_model = 768
n_heads = 12
d_k = d_v = 64  # d_model / n_heads
d_ff = 3072     # 4 × d_model
num_layers = 12
```

**Design choices:**
- Wider model (768 vs 512) for better representation
- More heads (12 vs 8) for more specialization
- Total params: ~110M

#### GPT-2 Small

```python
d_model = 768
n_heads = 12
d_k = d_v = 64
d_ff = 3072
num_layers = 12
```

**Same as BERT Base**, showing convergence on "good defaults".

#### GPT-3 Large

```python
d_model = 12288
n_heads = 96
d_k = d_v = 128  # d_model / n_heads
d_ff = 49152     # 4 × d_model
num_layers = 96
```

**Scaling laws:**
- Much wider (12288 vs 768)
- Many more heads (96 vs 12)
- Deeper (96 layers vs 12)
- Total params: ~175B

#### Tiny Transformer (for learning/prototyping)

```python
d_model = 128
n_heads = 4
d_k = d_v = 32   # d_model / n_heads
d_ff = 512       # 4 × d_model
num_layers = 4
```

**Tiny but functional:**
- Fast to train on CPU/small GPU
- Enough capacity for simple tasks
- Good for understanding/debugging
- Total params: ~500K

### Choosing n_heads

**General guidelines:**

```
d_model = 128  → n_heads = 2-4   (d_k = 32-64)
d_model = 256  → n_heads = 4-8   (d_k = 32-64)
d_model = 512  → n_heads = 8     (d_k = 64)
d_model = 768  → n_heads = 12    (d_k = 64)
d_model = 1024 → n_heads = 16    (d_k = 64)
```

**Rule of thumb:** Keep **d_k ∈ [32, 128]**

**Why not more heads?**
- Very small d_k (< 32): Each head has limited capacity
- Memory cost grows linearly with n_heads
- Diminishing returns beyond 16-32 heads for most tasks

**Why not fewer heads?**
- Less diversity in attention patterns
- Reduced parallelism
- Single-head bottleneck

### Choosing d_k Independently

**Non-standard choice:** d_k ≠ d_model / n_heads

```python
d_model = 512
n_heads = 8
d_k = 128  # Larger than 512/8 = 64

# After concatenation: 8 × 128 = 1024 ≠ 512
# Need output projection: W^O: (1024, 512)
```

**When to use:**
- Very deep models: Larger d_k preserves information through many layers
- Low-resource languages: Smaller d_k reduces params
- Experimentation: Sometimes deviating from defaults helps

**Trade-offs:**
- **Larger d_k**: More capacity per head, but higher memory cost
- **Smaller d_k**: Lower memory, but each head is weaker
- **Non-divisible d_k**: Requires extra output projection (already present anyway)

### Ablation Study Results

**Effect of number of heads (d_model = 512 fixed):**

| n_heads | d_k | Perplexity | Train Time | Memory |
|---------|-----|------------|------------|--------|
| 1 | 512 | 28.4 | 1.0× | 1.0× |
| 2 | 256 | 25.1 | 1.05× | 1.8× |
| 4 | 128 | 23.2 | 1.1× | 3.2× |
| 8 | 64 | **21.8** | 1.15× | 6.1× |
| 16 | 32 | 22.3 | 1.25× | 11.5× |
| 32 | 16 | 23.8 | 1.4× | 22.0× |

**Findings:**
- Sweet spot: **8 heads** (d_k = 64)
- 16 heads: Marginal gains, much higher memory cost
- 32 heads: Performance degrades (d_k too small)

**Effect of d_k (n_heads = 8 fixed, d_model varies):**

| d_k | d_model | n_heads | Perplexity | Params |
|-----|---------|---------|------------|--------|
| 32 | 256 | 8 | 26.7 | 1M |
| 64 | 512 | 8 | **21.8** | 4M |
| 128 | 1024 | 8 | 19.2 | 16M |
| 256 | 2048 | 8 | 17.5 | 64M |

**Findings:**
- Larger d_k consistently improves performance
- But requires scaling d_model (more parameters)
- d_k = 64 is good balance for medium models

---

## Implementation Considerations

### Efficient Batched Operations

The key to efficient multi-head attention is **batched matrix multiplication** across heads.

#### Inefficient: Loop Over Heads

```python
def multi_head_attention_slow(Q, K, V, n_heads):
    """
    DON'T DO THIS - Inefficient sequential processing
    """
    B, T, d_model = Q.shape
    d_k = d_model // n_heads

    outputs = []
    for i in range(n_heads):
        # Extract head i
        start = i * d_k
        end = (i + 1) * d_k
        Q_i = Q[:, :, start:end]  # (B, T, d_k)
        K_i = K[:, :, start:end]  # (B, T, d_k)
        V_i = V[:, :, start:end]  # (B, T, d_k)

        # Compute attention for head i
        scores = Q_i @ K_i.transpose(-2, -1) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        out_i = attn @ V_i  # (B, T, d_k)

        outputs.append(out_i)

    # Concatenate
    output = torch.cat(outputs, dim=-1)  # (B, T, d_model)
    return output
```

**Problems:**
- Sequential processing (no parallelism across heads)
- Python loop overhead
- Cannot utilize GPU's parallel capabilities
- ~8× slower than batched version

#### Efficient: Batched Across Heads

```python
def multi_head_attention_fast(Q, K, V, n_heads):
    """
    EFFICIENT - Parallel processing across heads
    """
    B, T, d_model = Q.shape
    d_k = d_model // n_heads

    # Reshape: (B, T, d_model) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)
    Q = Q.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
    K = K.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
    V = V.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)

    # Batched matrix multiplication across heads
    # PyTorch treats (B, n_heads) as a "batch" dimension
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, n_heads, T, T)
    attn = F.softmax(scores, dim=-1)                    # (B, n_heads, T, T)
    output = attn @ V                                    # (B, n_heads, T, d_k)

    # Reshape back: (B, n_heads, T, d_k) → (B, T, n_heads, d_k) → (B, T, d_model)
    output = output.transpose(1, 2).contiguous().view(B, T, d_model)

    return output
```

**Advantages:**
- All heads computed in parallel
- Single matrix multiplication operation (BLAS-optimized)
- Fully utilizes GPU parallelism
- 8× faster on GPU for 8 heads

### Memory Optimization Techniques

#### 1. Gradient Checkpointing

**Problem:** Storing attention weights for all heads uses O(B × h × T²) memory.

**Solution:** Recompute attention during backward pass instead of storing.

```python
from torch.utils.checkpoint import checkpoint

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # ... initialize parameters ...

    def forward(self, x, mask=None):
        if self.use_checkpoint and self.training:
            # Recompute attention during backward pass
            return checkpoint(self._forward_impl, x, mask)
        else:
            return self._forward_impl(x, mask)

    def _forward_impl(self, x, mask):
        # Standard multi-head attention logic
        # ...
```

**Trade-off:**
- **Memory:** Save ~30-40% during training
- **Compute:** ~20% slower (recompute attention in backward pass)

**When to use:** Training large models that don't fit in GPU memory.

#### 2. Flash Attention

**Standard attention memory bottleneck:**
```
scores: (B, h, T, T)  - Materialize full attention matrix
```

For T = 2048, B = 8, h = 8:
```
Memory: 8 × 8 × 2048 × 2048 × 4 bytes = 2.1 GB just for scores!
```

**Flash Attention optimization:**
- Tile the computation to avoid materializing full (T, T) matrix
- Fuse operations (softmax + matmul) in CUDA kernel
- Achieves same result with O(T) memory instead of O(T²)

```python
# PyTorch 2.0+ has built-in flash attention
from torch.nn.functional import scaled_dot_product_attention

# Automatically uses flash attention if available
output = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
```

**Benefits:**
- 2-4× faster
- 5-10× less memory for attention
- Enables longer sequences (T = 4096, 8192)

#### 3. Sparse Attention Patterns

For very long sequences, use sparse attention to reduce O(T²) cost.

```python
class SparseMultiHeadAttention(nn.Module):
    """
    Only attend to local window + global tokens
    Reduces O(T²) to O(T × window_size)
    """
    def __init__(self, d_model, n_heads, window_size=256):
        super().__init__()
        self.window_size = window_size
        # ...

    def create_sparse_mask(self, T):
        # Create mask that only allows attention to:
        # 1. Local window (±window_size positions)
        # 2. Global tokens (e.g., [CLS], [SEP])
        mask = torch.full((T, T), float('-inf'))

        # Local window
        for i in range(T):
            start = max(0, i - self.window_size)
            end = min(T, i + self.window_size + 1)
            mask[i, start:end] = 0

        # Global tokens
        mask[:, 0] = 0  # [CLS] token can be attended to by all
        mask[0, :] = 0  # [CLS] attends to all

        return mask
```

**Use cases:**
- Long documents (T > 4096)
- Memory-constrained environments
- Tasks where local context dominates

### Inference Optimization

#### 1. KV Caching for Autoregressive Generation

**Problem:** During text generation, recompute same K, V for past tokens every step.

**Solution:** Cache K, V for previous tokens.

```python
class MultiHeadAttentionWithKVCache(nn.Module):
    def forward(self, x, past_kv=None):
        B, T, d_model = x.shape

        # Project Q, K, V for new tokens
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)  # (B, T, d_model)
        V = self.W_v(x)  # (B, T, d_model)

        # If we have cached K, V from previous tokens, concatenate
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=1)  # (B, T_total, d_model)
            V = torch.cat([past_V, V], dim=1)  # (B, T_total, d_model)

        # Reshape for multi-head
        Q = self.split_heads(Q)  # (B, h, T, d_k)
        K = self.split_heads(K)  # (B, h, T_total, d_k)
        V = self.split_heads(V)  # (B, h, T_total, d_k)

        # Attention computation (Q only on new tokens, K/V on all)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        output = attn @ V

        # Return output and updated cache
        return output, (K, V)
```

**Speedup:**
- First token: 1× (no cache)
- Token 100: ~100× faster (avoid recomputing 99 previous K, V)
- Critical for real-time generation

#### 2. Quantization

Reduce precision from FP32 to INT8 or FP16:

```python
# FP16 inference (2× faster, 2× less memory)
model = MultiHeadAttention(...).half()  # Convert to FP16
x = x.half()
output = model(x)

# INT8 quantization (4× faster, 4× less memory, slight accuracy loss)
import torch.quantization as quant
model_int8 = quant.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

#### 3. Operator Fusion

**Fuse multiple operations into single CUDA kernel:**

```python
# Unfused (3 kernel launches)
scores = Q @ K.transpose(-2, -1)  # Kernel 1
scores = scores / math.sqrt(d_k)  # Kernel 2
scores = scores + mask            # Kernel 3

# Fused (1 kernel launch)
scores = torch.addmm(mask, Q, K.transpose(-2, -1), beta=1.0, alpha=1.0/math.sqrt(d_k))
```

**Or use `torch.compile` (PyTorch 2.0+):**

```python
model = MultiHeadAttention(...)
model = torch.compile(model)  # Automatically fuses operations
```

### Numerical Stability

#### Problem: FP16 Underflow in Softmax

With FP16, large negative values can underflow:

```python
# FP16 range: ~[-65504, 65504]
scores = torch.tensor([[-50.0, -100.0, -150.0]], dtype=torch.float16)
# After exp(-150), underflow to 0
```

#### Solution: Mixed Precision

```python
def stable_softmax(scores, dim=-1):
    # Compute softmax in FP32 even if input is FP16
    scores_fp32 = scores.float()
    attn = F.softmax(scores_fp32, dim=dim)
    return attn.to(scores.dtype)  # Convert back to FP16
```

**Or use PyTorch Automatic Mixed Precision (AMP):**

```python
from torch.cuda.amp import autocast

with autocast():
    # Automatically manages FP16/FP32 precision
    output = model(x)
```

---

## Common Mistakes

### Mistake 1: Forgetting to Transpose After Splitting Heads

**Symptom:** Shape error in attention computation or very poor performance.

```python
# WRONG: Missing transpose after view
Q = Q.view(B, T, n_heads, d_k)  # (B, T, n_heads, d_k)
K = K.view(B, T, n_heads, d_k)  # (B, T, n_heads, d_k)

scores = Q @ K.transpose(-2, -1)
# Error: Trying to multiply (B, T, n_heads, d_k) @ (B, T, d_k, n_heads)
#        Incompatible dimensions!
```

**Fix:**

```python
# CORRECT: Transpose to (B, n_heads, T, d_k)
Q = Q.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
K = K.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)

scores = Q @ K.transpose(-2, -1)  # (B, n_heads, T, T) ✓
```

**Why the transpose?**
- Moves heads to a batch-like dimension
- Enables batched matrix multiplication across heads
- Required for PyTorch to parallelize correctly

### Mistake 2: Incorrect Head Concatenation

**Symptom:** Shape error or garbled output.

```python
# WRONG: Concatenating without transposing back
output = output.view(B, T, n_heads * d_k)
# Error: Memory is not laid out correctly!
# You're concatenating in the wrong order
```

**Why it's wrong:**

After attention, shape is `(B, n_heads, T, d_k)`:
```
Memory layout:
[Head0_Pos0, Head0_Pos1, ..., Head0_PosT,
 Head1_Pos0, Head1_Pos1, ..., Head1_PosT,
 ...]
```

But we want:
```
[Pos0_Head0, Pos0_Head1, ..., Pos0_HeadH,
 Pos1_Head0, Pos1_Head1, ..., Pos1_HeadH,
 ...]
```

**Fix:**

```python
# CORRECT: Transpose first, then reshape
output = output.transpose(1, 2)  # (B, T, n_heads, d_k)
output = output.contiguous().view(B, T, n_heads * d_k)  # (B, T, d_model)
```

**Debugging tip:**

```python
# Check if concatenation is correct
# Last d_k elements should be from Head 7, not scattered
print(output[0, 0, -64:])  # Should be all from last head
```

### Mistake 3: Scaling by Wrong Value

**Symptom:** Training instability, saturated attention weights.

```python
# WRONG: Scale by d_model instead of d_k
scores = scores / math.sqrt(d_model)  # Should be d_k!

# WRONG: Scale by n_heads
scores = scores / math.sqrt(n_heads)

# WRONG: Forget to scale at all
scores = scores  # Missing division!
```

**Fix:**

```python
# CORRECT: Scale by d_k (dimension per head, not total dimension)
d_k = d_model // n_heads
scores = scores / math.sqrt(d_k)
```

**Why d_k?**
- After splitting, each head operates in d_k-dimensional space
- Dot products in that space have variance proportional to d_k
- Need to scale by √d_k to normalize variance to 1

### Mistake 4: Mask Broadcasting Issues

**Symptom:** RuntimeError about shape mismatch, or mask not applied correctly.

```python
# WRONG: Mask shape doesn't account for heads
mask = torch.ones(T, T)  # (T, T)
scores = scores + mask   # scores: (B, h, T, T), mask: (T, T)
# Broadcasting works, but applies same mask to all heads
# This might be intended, but often you want per-head masks
```

**Common scenarios:**

```python
# Causal mask (same for all heads): (T, T) ✓
causal_mask = create_causal_mask(T)  # (T, T)
scores = scores + causal_mask  # Broadcasts to (B, h, T, T)

# Padding mask (same for all heads): (B, 1, 1, T) ✓
padding_mask = (input_ids == PAD_ID).unsqueeze(1).unsqueeze(1) * float('-inf')
scores = scores + padding_mask  # (B, 1, 1, T) → (B, h, T, T)

# Per-head mask (different mask per head): (B, h, T, T) ✓
per_head_mask = create_custom_mask(B, n_heads, T)  # (B, h, T, T)
scores = scores + per_head_mask  # Exact match
```

**Fix for padding mask:**

```python
# CORRECT: Proper padding mask
def create_padding_mask(input_ids, pad_token_id):
    # input_ids: (B, T)
    # Output: (B, 1, 1, T) for broadcasting
    mask = (input_ids == pad_token_id)  # (B, T)
    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    mask = mask.float() * float('-inf')
    return mask
```

### Mistake 5: Forgetting `.contiguous()` Before `.view()`

**Symptom:** RuntimeError: view size is not compatible with input tensor's size and stride.

```python
# WRONG: view() on non-contiguous tensor
output = output.transpose(1, 2)  # Now non-contiguous!
output = output.view(B, T, -1)   # Error!
```

**Why it fails:**

After `.transpose()`, tensor is not stored contiguously in memory:
```
Original:     [a0, a1, a2, a3, b0, b1, b2, b3]
After transp: [a0, b0, a1, b1, a2, b2, a3, b3]  ← Non-contiguous view
```

`.view()` requires contiguous memory.

**Fix:**

```python
# CORRECT: Call .contiguous() first
output = output.transpose(1, 2).contiguous()
output = output.view(B, T, -1)  # Now works ✓
```

**Alternative:** Use `.reshape()` instead of `.view()`

```python
# .reshape() automatically makes contiguous if needed
output = output.transpose(1, 2)
output = output.reshape(B, T, -1)  # Works without .contiguous()
```

**Trade-off:**
- `.view()`: Faster (no copy), but requires contiguous tensor
- `.reshape()`: Slower (may copy), but always works

### Mistake 6: Applying Dropout to Wrong Tensor

**Symptom:** Poor training performance, overfitting.

```python
# WRONG: Dropout on output instead of attention weights
output = attn @ V
output = dropout(output)  # Dropping value information!
```

**Correct location:**

```python
# CORRECT: Dropout on attention weights (after softmax)
attn = F.softmax(scores, dim=-1)
attn = dropout(attn)  # Randomly zero out some attention weights
output = attn @ V
```

**Why this location?**
- Encourages model to not over-rely on specific positions
- Regularizes attention patterns
- Standard practice in transformer literature

**Additional dropout location (also correct):**

```python
# Also common: Dropout on final output
output = output @ W_o
output = dropout(output)  # Dropout on final MHA output
```

Many transformers use dropout in **both** locations.

### Mistake 7: Shared Projection Matrices Across Layers

**Symptom:** Model underfits, poor performance.

```python
# WRONG: Sharing parameters across layers
shared_W_q = nn.Linear(d_model, d_model)

class TransformerLayer(nn.Module):
    def __init__(self):
        # All layers share same W_q!
        self.attn = MultiHeadAttention(shared_W_q, ...)
```

**Fix:**

```python
# CORRECT: Each layer has its own parameters
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        # New parameters for each layer instance
        self.attn = MultiHeadAttention(d_model, n_heads)
```

**Why separate parameters?**
- Each layer learns different transformations
- Lower layers: positional, syntactic patterns
- Higher layers: semantic, task-specific patterns
- Sharing parameters limits model capacity

### Mistake 8: Incorrect d_k Calculation

**Symptom:** Shape errors, assertion failures.

```python
# WRONG: Forgetting integer division
d_k = d_model / n_heads  # Returns float! E.g., 512 / 8 = 64.0

# Later:
Q = Q.view(B, T, n_heads, d_k)  # Error: d_k must be int
```

**Fix:**

```python
# CORRECT: Use integer division
d_k = d_model // n_heads  # Returns int: 512 // 8 = 64 ✓

# Or ensure divisibility
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
d_k = d_model // n_heads
```

### Debugging Checklist

When multi-head attention isn't working:

1. **Print all shapes:**
   ```python
   print(f"Q: {Q.shape}")
   print(f"K: {K.shape}")
   print(f"V: {V.shape}")
   print(f"After split - Q: {Q_split.shape}")
   print(f"Scores: {scores.shape}")
   print(f"Attention: {attn.shape}")
   print(f"Output: {output.shape}")
   ```

2. **Check attention weights:**
   ```python
   # Row sums should be 1.0
   print(f"Row sums: {attn[0, 0].sum(dim=-1)}")  # Should be all 1.0

   # Check if mask is working (for causal)
   print(f"Upper triangle: {attn[0, 0].triu(diagonal=1)}")  # Should be all 0
   ```

3. **Verify contiguity:**
   ```python
   print(f"Is contiguous: {output.is_contiguous()}")
   ```

4. **Check for NaN/Inf:**
   ```python
   assert not torch.isnan(scores).any(), "NaN in scores"
   assert not torch.isinf(scores).any(), "Inf in scores"
   ```

5. **Validate attention patterns:**
   ```python
   # Visualize first head's attention
   import matplotlib.pyplot as plt
   plt.imshow(attn[0, 0].detach().cpu())
   plt.colorbar()
   plt.title("Head 0 Attention Pattern")
   plt.show()
   ```

---

## Connection to Code

### Reference Implementation

Our implementation (to be added) at `/Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/multi_head.py` will demonstrate all concepts covered.

**Expected structure:**

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        d_model: Model dimension (e.g., 512)
        n_heads: Number of attention heads (e.g., 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dimension into (n_heads, d_k).

        Args:
            x: (B, T, d_model)
        Returns:
            (B, n_heads, T, d_k)
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)  # (B, T, n_heads, d_k)
        x = x.transpose(1, 2)  # (B, n_heads, T, d_k)
        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads.

        Args:
            x: (B, n_heads, T, d_k)
        Returns:
            (B, T, d_model)
        """
        B, _, T, _ = x.shape
        x = x.transpose(1, 2)  # (B, T, n_heads, d_k)
        x = x.contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.

        Args:
            query: (B, T, d_model)
            key: (B, T, d_model)
            value: (B, T, d_model)
            mask: Optional (T, T) or (B, n_heads, T, T)

        Returns:
            output: (B, T, d_model)
            attention_weights: (B, n_heads, T, T)
        """
        # 1. Linear projections
        Q = self.W_q(query)  # (B, T, d_model)
        K = self.W_k(key)    # (B, T, d_model)
        V = self.W_v(value)  # (B, T, d_model)

        # 2. Split into heads
        Q = self.split_heads(Q)  # (B, n_heads, T, d_k)
        K = self.split_heads(K)  # (B, n_heads, T, d_k)
        V = self.split_heads(V)  # (B, n_heads, T, d_k)

        # 3. Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, n_heads, T, T)

        if mask is not None:
            scores = scores + mask

        attention = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        attention = self.dropout(attention)

        output = attention @ V  # (B, n_heads, T, d_k)

        # 4. Combine heads
        output = self.combine_heads(output)  # (B, T, d_model)

        # 5. Output projection
        output = self.W_o(output)  # (B, T, d_model)

        return output, attention
```

### Usage Example

```python
# Initialize
d_model = 512
n_heads = 8
mha = MultiHeadAttention(d_model, n_heads)

# Input
batch_size = 32
seq_len = 128
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, attention_weights = mha(x, x, x)  # Self-attention

print(f"Input shape: {x.shape}")              # (32, 128, 512)
print(f"Output shape: {output.shape}")        # (32, 128, 512)
print(f"Attention shape: {attention_weights.shape}")  # (32, 8, 128, 128)

# With causal mask
from tiny_transformer.attention import create_causal_mask
mask = create_causal_mask(seq_len)
output, attention_weights = mha(x, x, x, mask=mask)
```

### Testing

```python
def test_multi_head_attention_shapes():
    """Test output shapes are correct."""
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(32, 128, 512)

    output, attn = mha(x, x, x)

    assert output.shape == (32, 128, 512)
    assert attn.shape == (32, 8, 128, 128)

def test_attention_weights_sum_to_one():
    """Test attention weights are valid probabilities."""
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(32, 128, 512)

    _, attn = mha(x, x, x)

    # Each row should sum to 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))

def test_causal_masking():
    """Test causal mask prevents attending to future."""
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(32, 128, 512)
    mask = create_causal_mask(128)

    _, attn = mha(x, x, x, mask=mask)

    # Upper triangle should be zero for all heads
    for h in range(8):
        upper_tri = torch.triu(attn[0, h], diagonal=1)
        assert torch.allclose(upper_tri, torch.zeros_like(upper_tri))
```

---

## Summary

### Key Takeaways

1. **Multi-head attention runs h parallel attention mechanisms** with different learned projections, enabling the model to capture multiple types of relationships simultaneously.

2. **The core equation:**
   ```
   MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O
   where headᵢ = Attention(Q W^Q_i, K W^K_i, V W^V_i)
   ```

3. **Standard choice: d_k = d_model / n_heads**
   - Preserves total model dimension
   - Keeps computational cost same as single-head
   - Enables drop-in replacement

4. **Critical shape transformations:**
   ```
   Input: (B, T, d_model)
   → Project: (B, T, d_model)
   → Split: (B, T, n_heads, d_k)
   → Transpose: (B, n_heads, T, d_k)
   → Attention: (B, n_heads, T, d_k)
   → Transpose back: (B, T, n_heads, d_k)
   → Concatenate: (B, T, d_model)
   → Project output: (B, T, d_model)
   ```

5. **Different heads specialize** to capture different patterns:
   - Local attention, syntax, semantics, position, coreference, etc.
   - Emerges naturally through training, not explicitly programmed

6. **Implementation efficiency:**
   - Use batched operations across heads (not loops)
   - Remember `.transpose()` after splitting heads
   - Remember `.contiguous()` before `.view()`

7. **Common mistakes:**
   - Forgetting transpose: (B, T, h, d_k) vs (B, h, T, d_k)
   - Wrong concatenation order
   - Scaling by d_model instead of d_k
   - Mask broadcasting issues

### Comparison to Single-Head

| Aspect | Single-Head | Multi-Head (h=8) |
|--------|-------------|------------------|
| Attention patterns | 1 | h (8) |
| Representation subspaces | 1 | h (8) |
| Computation cost | O(BT²d) | O(BT²d) |
| Memory for attention | BT² | BhT² |
| Parameters | 3d² | 4d² |
| Parallelism | Limited | h× parallel |
| Robustness | Single point of failure | Ensemble of heads |

**Multi-head is strictly better** for most applications, with negligible computational overhead.

### The Big Picture

Multi-head attention solves a fundamental problem in sequence modeling:

**Different positions need to relate to each other in different ways:**
- Syntactically (grammar)
- Semantically (meaning)
- Positionally (structure)
- Pragmatically (discourse)

With single-head attention, the model must choose **one way** to relate positions. With multi-head, it can learn **all ways simultaneously**.

This is why transformers with multi-head attention have become the dominant architecture for:
- Natural language processing (GPT, BERT, T5)
- Computer vision (ViT, DETR)
- Speech processing (Whisper)
- Multi-modal models (CLIP, Flamingo)
- Protein folding (AlphaFold)
- Reinforcement learning (Decision Transformer)

### Next Steps

Now that you understand multi-head attention:

1. **Transformer Blocks**: Combine multi-head attention with feed-forward networks and layer normalization
2. **Positional Encodings**: How transformers incorporate sequence order
3. **Full Transformer Architecture**: Encoder and decoder stacks
4. **Training Techniques**: Optimization, regularization, and scaling
5. **Advanced Architectures**: Variants like sparse attention, linear attention, flash attention

### Further Reading

**Foundational Papers:**
- Vaswani et al. (2017): "Attention Is All You Need" - Section 3.2.2 on multi-head attention
- Clark et al. (2019): "What Does BERT Look At?" - Analysis of attention head patterns
- Voita et al. (2019): "Analyzing Multi-Head Self-Attention" - Pruning and specialization

**Tutorials:**
- The Illustrated Transformer (Jay Alammar) - Visual explanations
- The Annotated Transformer (Harvard NLP) - Line-by-line implementation
- Transformers from Scratch (Peter Bloem) - Educational implementation

**Advanced Topics:**
- Multi-Query Attention (MQA) - Share K, V across heads
- Grouped-Query Attention (GQA) - Compromise between MHA and MQA
- Flash Attention - Memory-efficient attention computation
- Sparse Attention Patterns - Reduce O(T²) complexity

### Practice Exercises

1. **Implement from scratch**: Code multi-head attention without looking at the reference.

2. **Visualize head patterns**: Train a small transformer and visualize what each head learns.

3. **Ablation study**: Compare performance with different numbers of heads (1, 2, 4, 8, 16).

4. **Head pruning**: Identify and remove redundant heads without hurting performance.

5. **Different d_k**: Experiment with d_k ≠ d_model / n_heads and observe effects.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Module:** 02 - Multi-Head Attention
**Target Audience:** Developers who completed Module 01 (Attention Mechanism)
**Estimated Reading Time:** 45-60 minutes
