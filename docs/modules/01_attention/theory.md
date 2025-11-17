# Attention Mechanism: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Intuition: What is Attention?](#intuition-what-is-attention)
3. [The Attention Problem](#the-attention-problem)
4. [Scaled Dot-Product Attention](#scaled-dot-product-attention)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Component Deep Dive](#component-deep-dive)
7. [The Scaling Factor](#the-scaling-factor)
8. [Masking in Attention](#masking-in-attention)
9. [Computational Complexity](#computational-complexity)
10. [Implementation Details](#implementation-details)
11. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
12. [Summary](#summary)

---

## Introduction

Attention is the fundamental mechanism that powers modern transformer architectures. Introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017), it revolutionized how neural networks process sequential data.

**What you'll learn:**
- The intuition behind attention mechanisms
- The mathematical foundation of scaled dot-product attention
- How each component (Query, Key, Value) works
- Why we scale by √d_k and what happens if we don't
- How masking enables autoregressive generation
- Practical implementation details and debugging strategies

**Prerequisites:**
- Basic linear algebra (matrix multiplication, dot products)
- Understanding of neural networks and backpropagation
- Familiarity with PyTorch tensors (helpful but not required)

---

## Intuition: What is Attention?

### The Cocktail Party Analogy

Imagine you're at a crowded party with multiple conversations happening simultaneously. Despite the noise, you can focus on one conversation while being vaguely aware of others. If someone mentions your name across the room, your attention immediately shifts there. This selective focus is exactly what attention mechanisms do in neural networks.

**Key insight:** Attention lets a model dynamically decide which parts of the input are relevant for processing the current element.

### Why Do We Need Attention?

Before attention, sequence models like RNNs and LSTMs had a fundamental limitation: they compressed entire sequences into fixed-size hidden states. This created an information bottleneck.

**Problems with previous approaches:**

1. **Information Bottleneck**: Long sequences must compress into a fixed-size vector
2. **Vanishing Gradients**: Gradient signals decay over long distances
3. **Sequential Processing**: Can't parallelize, making training slow
4. **Context Limitation**: Distant dependencies are hard to learn

**Attention's solution:**

Instead of compressing everything into a fixed vector, attention allows each position to directly access information from all other positions. It's like having a direct phone line to every word in the sentence, rather than passing messages through a chain.

### A Simple Example

Consider translating: "The animal didn't cross the street because **it** was too tired."

To translate "it" correctly, the model needs to know what "it" refers to. Attention allows the model to:
1. Look at "it"
2. Compute relevance scores with all other words
3. Focus heavily on "animal" (high attention weight)
4. Ignore less relevant words like "the" and "because"

The result: the model "attends" to "animal" when processing "it", correctly understanding the reference.

---

## The Attention Problem

Let's formalize what we want attention to do:

**Goal:** Given a sequence of inputs, compute a weighted representation where the weights indicate importance.

**Inputs:**
- A sequence of tokens: [x₁, x₂, ..., xₙ]
- Each token represented as a vector

**Desired Output:**
- For each position i, compute output yᵢ as a weighted combination of all inputs
- Weights should reflect how relevant each input is to position i

**Key Question:** How do we compute relevance between positions?

**Answer:** Use learned similarity functions based on three components:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I actually provide?"

---

## Scaled Dot-Product Attention

### The Core Equation

The fundamental equation of attention is beautifully simple:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Let's break this down step by step.

### The Recipe

Think of attention as a 6-step recipe:

1. **Compute Similarity**: Calculate QK^T to measure how relevant each key is to each query
2. **Scale**: Divide by √d_k to keep values in a reasonable range
3. **Mask** (optional): Set certain positions to -∞ to prevent attending to them
4. **Normalize**: Apply softmax to convert scores into probabilities
5. **Weight**: Multiply probabilities by values
6. **Aggregate**: Sum weighted values to get the output

### Visual Flow

```
Query (Q)      Key (K)
  ↓              ↓
  └──── Dot Product ────┘
          ↓
    [Similarity Scores]
          ↓
    Divide by √d_k
          ↓
    [Scaled Scores]
          ↓
    Apply Mask (optional)
          ↓
    [Masked Scores]
          ↓
       Softmax
          ↓
  [Attention Weights]
          ↓
          ×  Value (V)
          ↓
      [Output]
```

---

## Mathematical Formulation

### Notation and Dimensions

Let's establish our notation:

- **B**: Batch size (number of sequences processed together)
- **T**: Sequence length (number of tokens in each sequence)
- **d_k**: Dimension of keys and queries (typically 64-512)
- **d_v**: Dimension of values (often equal to d_k)

### Tensor Shapes Throughout

```
Input:
  Q: (B, T, d_k)  - Queries
  K: (B, T, d_k)  - Keys
  V: (B, T, d_v)  - Values

Step 1 - Similarity Scores:
  QK^T: (B, T, d_k) × (B, d_k, T) → (B, T, T)

Step 2 - Scaled Scores:
  Scores / √d_k: (B, T, T)

Step 3 - Masked Scores (optional):
  Scores + Mask: (B, T, T)

Step 4 - Attention Weights:
  softmax(Scores): (B, T, T)

Step 5 - Output:
  Weights × V: (B, T, T) × (B, T, d_v) → (B, T, d_v)
```

### Detailed Step-by-Step Mathematics

#### Step 1: Compute Similarity Scores

```
Scores = QK^T
```

For each query position i and key position j:
```
Score[i,j] = Query[i] · Key[j]
           = Σ(k=1 to d_k) Query[i,k] × Key[j,k]
```

**What this means:**
- High positive score → Query and Key are similar (point in same direction)
- Score near zero → Query and Key are orthogonal (unrelated)
- Negative score → Query and Key point in opposite directions

**Shape transformation:**
```python
# Q: (B, T, d_k)
# K: (B, T, d_k)
# K.transpose(-2, -1): (B, d_k, T)

scores = Q @ K.transpose(-2, -1)  # (B, T, T)
```

The result is a **T × T attention score matrix** for each batch element:
```
        Key₁  Key₂  Key₃  Key₄
Query₁ [s₁₁  s₁₂  s₁₃  s₁₄]
Query₂ [s₂₁  s₂₂  s₂₃  s₂₄]
Query₃ [s₃₁  s₃₂  s₃₃  s₃₄]
Query₄ [s₄₁  s₄₂  s₄₃  s₄₄]
```

#### Step 2: Scale by √d_k

```
ScaledScores = Scores / √d_k
```

**Why?** See the dedicated section below, but briefly: prevents dot products from becoming too large, which would cause softmax saturation and vanishing gradients.

```python
scores = scores / math.sqrt(d_k)
```

#### Step 3: Apply Mask (Optional)

```
MaskedScores = ScaledScores + Mask
```

The mask contains:
- `0` for positions we want to keep
- `-∞` for positions we want to mask out

After adding `-∞`, softmax will convert those positions to probability 0.

```python
if mask is not None:
    scores = scores + mask  # Broadcasting handles shape alignment
```

**Causal mask example** (for sequence length 4):
```
[[  0,  -∞,  -∞,  -∞],
 [  0,   0,  -∞,  -∞],
 [  0,   0,   0,  -∞],
 [  0,   0,   0,   0]]
```

This ensures position i can only attend to positions ≤ i.

#### Step 4: Apply Softmax

```
AttentionWeights = softmax(MaskedScores)
```

Softmax is applied **row-wise** (over the last dimension):

```
AttentionWeights[i,j] = exp(MaskedScores[i,j]) / Σ(k=1 to T) exp(MaskedScores[i,k])
```

**Properties:**
- All weights are positive: AttentionWeights[i,j] ≥ 0
- Each row sums to 1: Σⱼ AttentionWeights[i,j] = 1
- Each row is a probability distribution over key positions

```python
attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
```

**Example:**
```
Before softmax (scores):
[[ 2.1,  -∞,   -∞,   -∞],
 [ 1.5,  3.2,  -∞,   -∞],
 [ 0.8,  1.1,  2.5,  -∞],
 [ 0.3,  0.9,  1.2,  2.8]]

After softmax (attention weights):
[[1.00, 0.00, 0.00, 0.00],
 [0.15, 0.85, 0.00, 0.00],
 [0.17, 0.23, 0.60, 0.00],
 [0.11, 0.15, 0.21, 0.53]]
```

#### Step 5: Apply Attention to Values

```
Output = AttentionWeights × V
```

For each query position i:
```
Output[i] = Σ(j=1 to T) AttentionWeights[i,j] × Value[j]
```

**What this means:**
The output for position i is a weighted average of all value vectors, where the weights represent how much we "attend" to each position.

```python
output = attention_weights @ value  # (B, T, T) × (B, T, d_v) → (B, T, d_v)
```

**Concrete example** (with d_v = 3):

```
Attention weights (row 2):
[0.17, 0.23, 0.60, 0.00]

Values:
Value₁ = [1.0, 2.0, 3.0]
Value₂ = [4.0, 5.0, 6.0]
Value₃ = [7.0, 8.0, 9.0]
Value₄ = [2.0, 1.0, 0.0]

Output[2] = 0.17 × [1.0, 2.0, 3.0]
          + 0.23 × [4.0, 5.0, 6.0]
          + 0.60 × [7.0, 8.0, 9.0]
          + 0.00 × [2.0, 1.0, 0.0]

          = [0.17, 0.34, 0.51]
          + [0.92, 1.15, 1.38]
          + [4.20, 4.80, 5.40]
          + [0.00, 0.00, 0.00]

          = [5.29, 6.29, 7.29]
```

The output is dominated by Value₃ because attention weight 0.60 is highest for position 3.

---

## Component Deep Dive

### Query, Key, Value: The Trio

The Q, K, V decomposition is inspired by information retrieval systems (like search engines):

#### Query (Q): "What am I looking for?"

- Represents what information the current position needs
- Like typing a search query into Google
- Encoded representation of "what I want to know"

**Example:** When processing the word "it" in "The animal didn't cross the street because it was tired":
- Query encodes: "I'm a pronoun, I need to find what I refer to"

#### Key (K): "What do I contain?"

- Represents what information each position offers
- Like the indexed keywords of a web page
- Encoded representation of "what I can provide"

**Example:** Each word has a key:
- "animal" key might encode: "I'm a noun, I could be a referent"
- "because" key might encode: "I'm a conjunction, probably not a referent"

#### Value (V): "What information do I actually provide?"

- The actual content that gets propagated
- Like the content of a web page (vs. its keywords)
- The representation that gets mixed together in the output

**Why the separation?**

You might wonder: why not just use the same vector for everything?

**Answer:** Separation of concerns:
- **Keys** are optimized for matching (similarity measurement)
- **Queries** are optimized for specifying needs
- **Values** are optimized for information content

This is similar to how databases work:
- **Key**: Index for fast lookup
- **Query**: What you're searching for
- **Value**: The actual data retrieved

In practice, Q, K, and V are created by applying learned linear transformations to the input:

```python
Q = input @ W_q  # (B, T, d_model) × (d_model, d_k) → (B, T, d_k)
K = input @ W_k  # (B, T, d_model) × (d_model, d_k) → (B, T, d_k)
V = input @ W_v  # (B, T, d_model) × (d_model, d_v) → (B, T, d_v)
```

Where W_q, W_k, W_v are learned parameter matrices.

### Why Dot Product for Similarity?

There are many ways to measure similarity. Why dot product?

**Advantages:**
1. **Computationally efficient**: Matrix multiplication is highly optimized on GPUs
2. **Mathematically clean**: Clear geometric interpretation
3. **Differentiable**: Easy to backpropagate through
4. **Parallelizable**: Can compute all similarities simultaneously

**Alternatives considered:**
- **Additive attention**: concat(Q, K) through an MLP
  - More expressive but slower
  - Used in older attention mechanisms
- **Cosine similarity**: Q·K / (||Q|| ||K||)
  - Normalized, but more expensive
- **Euclidean distance**: ||Q - K||
  - Less intuitive for neural networks

**Dot product interpretation:**

```
Q · K = ||Q|| ||K|| cos(θ)
```

Where θ is the angle between Q and K:
- θ = 0° (parallel): Maximum positive similarity
- θ = 90° (orthogonal): Zero similarity
- θ = 180° (opposite): Maximum negative similarity

---

## The Scaling Factor

### Why Divide by √d_k?

This is one of the most subtle but important aspects of attention. Let's understand it deeply.

#### The Problem: Variance Growth

Consider two random vectors Q and K, each with dimension d_k, where each component is sampled from a standard normal distribution (mean 0, variance 1).

The dot product is:
```
Q · K = Σ(i=1 to d_k) Q[i] × K[i]
```

**Statistical analysis:**
- Each product Q[i] × K[i] has mean 0 and variance 1
- The sum of d_k independent terms has mean 0 and variance d_k
- Therefore: Var(Q · K) = d_k

**What this means:**
As d_k increases, the dot products have larger magnitudes.

**Demonstration:**

```
d_k = 64:  Q·K might be in range [-25, 25]
d_k = 512: Q·K might be in range [-70, 70]
```

#### The Softmax Saturation Problem

Softmax is defined as:
```
softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)
```

**Problem:** When inputs to softmax have large magnitudes, it saturates:

**Example:**
```python
import numpy as np

# Small values - softmax works well
small = np.array([1.0, 2.0, 3.0])
print(softmax(small))
# Output: [0.09, 0.24, 0.67] - Nice distribution

# Large values - softmax saturates
large = np.array([10.0, 20.0, 30.0])
print(softmax(large))
# Output: [0.00, 0.00, 1.00] - Almost one-hot!
```

**Why saturation is bad:**

1. **Vanishing Gradients**: When softmax outputs are close to 0 or 1:
   ```
   ∂softmax/∂x ≈ 0
   ```
   Gradients don't flow back effectively.

2. **Loss of Information**: Attention becomes too peaked (almost one-hot), losing the ability to attend to multiple positions.

3. **Training Instability**: Large gradients early in training, then sudden vanishing.

#### The Solution: Scale by √d_k

By dividing by √d_k, we normalize the variance:

```
Var(Q·K / √d_k) = Var(Q·K) / d_k = d_k / d_k = 1
```

Now the dot products have constant variance regardless of d_k!

**Result:**
- Scores stay in a reasonable range
- Softmax doesn't saturate
- Gradients flow nicely
- Training is stable

#### Empirical Demonstration

Let's see what happens with and without scaling:

```python
import torch
import torch.nn.functional as F

# Simulate attention with d_k = 64
d_k = 64
Q = torch.randn(1, 4, d_k)
K = torch.randn(1, 4, d_k)

# Without scaling
scores_unscaled = Q @ K.transpose(-2, -1)
attn_unscaled = F.softmax(scores_unscaled, dim=-1)

# With scaling
scores_scaled = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
attn_scaled = F.softmax(scores_scaled, dim=-1)

print("Unscaled scores:")
print(scores_unscaled[0])
# Output: Large values, e.g., [[-15.2, 8.3, -12.1, 18.7], ...]

print("\nUnscaled attention (saturated):")
print(attn_unscaled[0])
# Output: Nearly one-hot, e.g., [[0.00, 0.00, 0.00, 1.00], ...]

print("\nScaled scores:")
print(scores_scaled[0])
# Output: Reasonable values, e.g., [[-1.9, 1.0, -1.5, 2.3], ...]

print("\nScaled attention (good distribution):")
print(attn_scaled[0])
# Output: Distributed weights, e.g., [[0.03, 0.15, 0.04, 0.78], ...]
```

### Mathematical Intuition

Another way to think about it: we want the pre-softmax scores to be roughly in the range [-3, 3] for good softmax behavior.

Without scaling:
- Standard deviation of Q·K: √d_k
- For d_k=64: std ≈ 8, so values often exceed [-24, 24]

With scaling:
- Standard deviation of Q·K/√d_k: 1
- Values typically stay within [-3, 3]

**Rule of thumb:** If you see attention weights that are almost one-hot during training, you probably forgot to scale!

---

## Masking in Attention

### Why Do We Need Masking?

Masking serves two main purposes:

1. **Causal/Autoregressive Generation**: Prevent attending to future tokens
2. **Padding**: Prevent attending to padding tokens in variable-length sequences

### Causal Masking for Autoregressive Models

**Problem:** When generating text, position i shouldn't "see" future positions (i+1, i+2, ...).

**Example:** Generating "The cat sat on the"...
- When predicting "cat", we can only use "The"
- When predicting "sat", we can use "The cat"
- When predicting "on", we can use "The cat sat"

If we allowed attending to future tokens, the model would cheat during training!

#### The Causal Mask

A causal mask is a lower-triangular matrix:

```
Sequence: ["The", "cat", "sat", "on"]

Causal Mask:
        The   cat   sat   on
The  [  0    -∞    -∞    -∞  ]
cat  [  0     0    -∞    -∞  ]
sat  [  0     0     0    -∞  ]
on   [  0     0     0     0  ]
```

**Reading the mask:**
- Row i, column j: Can position i attend to position j?
- `0`: Yes, allow attention
- `-∞`: No, block attention

**After softmax:**
```
exp(-∞) = 0
```
So masked positions contribute zero to the attention output.

#### Implementation

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal mask for autoregressive generation."""
    # Start with lower triangular matrix of ones
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # Convert: 1 → 0 (allow), 0 → -inf (block)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)

    return mask
```

**Usage in attention:**

```python
# Create mask once
mask = create_causal_mask(seq_len)  # (T, T)

# Apply in attention
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, T, T)
scores = scores + mask  # Broadcasting: (B, T, T) + (T, T) → (B, T, T)
attention_weights = F.softmax(scores, dim=-1)
```

**Key insight:** We add the mask *before* softmax, so `-∞` values become 0 after softmax.

#### Attention Weights with Causal Mask

**Before masking** (can attend anywhere):
```
        Pos0  Pos1  Pos2  Pos3
Pos0  [ 0.25  0.25  0.25  0.25 ]
Pos1  [ 0.30  0.20  0.40  0.10 ]
Pos2  [ 0.15  0.35  0.20  0.30 ]
Pos3  [ 0.10  0.30  0.25  0.35 ]
```

**After causal masking** (lower-triangular):
```
        Pos0  Pos1  Pos2  Pos3
Pos0  [ 1.00  0.00  0.00  0.00 ]
Pos1  [ 0.60  0.40  0.00  0.00 ]
Pos2  [ 0.21  0.49  0.30  0.00 ]
Pos3  [ 0.10  0.30  0.25  0.35 ]
```

Notice how each row is renormalized to sum to 1.0.

### Padding Masking

For variable-length sequences, we pad shorter sequences with special tokens. We don't want to attend to padding.

**Example:**
```
Sequence 1: ["Hello", "world", "<pad>", "<pad>"]
Sequence 2: ["I", "love", "transformers", "!"]

Padding mask for Sequence 1:
        Pos0  Pos1  Pos2  Pos3
All   [  0     0    -∞    -∞   ]  (applied to all rows)
```

This prevents attending to positions 2 and 3 (the padding tokens).

### Combining Masks

You can combine causal and padding masks:

```python
# Causal mask: (T, T)
causal_mask = create_causal_mask(seq_len)

# Padding mask: (B, T) indicating which positions are padding
# Then broadcast to (B, 1, T) or (B, T, T)
padding_mask = (input_ids == PAD_TOKEN_ID).unsqueeze(1).float() * float('-inf')

# Combine
full_mask = causal_mask + padding_mask
```

---

## Computational Complexity

### Time Complexity: O(n²)

The attention mechanism has quadratic time complexity in sequence length.

**Analysis:**

Step 1: Compute QK^T
```
Operation: (B, T, d_k) × (B, d_k, T)
Cost: O(B × T × T × d_k) = O(BT²d_k)
```

Step 2-4: Scale, mask, softmax
```
Cost: O(BT²)
```

Step 5: Multiply by V
```
Operation: (B, T, T) × (B, T, d_v)
Cost: O(B × T × T × d_v) = O(BT²d_v)
```

**Total: O(BT²d)**

Where d = max(d_k, d_v), typically 64-512.

### Space Complexity: O(n²)

The attention weight matrix requires:
```
Space: (B, T, T) floats
```

For T = 1024, B = 32:
- 32 × 1024 × 1024 × 4 bytes = 128 MB just for attention weights!
- This is why long sequences are challenging

### Scaling Challenges

**Problem:** Doubling sequence length quadruples compute and memory!

```
T = 512:   512² = 262,144 operations
T = 1024:  1024² = 1,048,576 operations (4× increase)
T = 2048:  2048² = 4,194,304 operations (16× increase)
```

**This is why:**
- Most transformers are limited to 512-2048 tokens
- Longer contexts require special techniques (sparse attention, linear attention)
- GPU memory becomes the bottleneck

### Efficient Attention Variants

Many modern transformers use modified attention to reduce complexity:

1. **Sparse Attention** (O(n√n) or O(n log n))
   - Only attend to subset of positions
   - Examples: Longformer, BigBird

2. **Linear Attention** (O(n))
   - Approximate attention with kernel methods
   - Examples: Performer, Linear Transformer

3. **Flash Attention** (O(n²) but memory-efficient)
   - Tiling and recomputation to reduce memory
   - Same complexity but 2-4× faster in practice

**Trade-off:** These variants sacrifice some expressiveness for efficiency.

### Practical Implications

When implementing transformers:

**Memory estimation:**
```python
# Rough memory for attention weights
batch_size = 32
seq_len = 512
num_heads = 8
num_layers = 12

attention_memory = batch_size * seq_len**2 * num_heads * num_layers * 4  # bytes
print(f"Attention memory: {attention_memory / 1e9:.2f} GB")
# Output: ~6.4 GB
```

**This doesn't include:**
- Activations for forward/backward pass
- Model parameters
- Gradients
- Optimizer states

**Rule of thumb:** For training, you need ~4-6× the model parameter memory.

---

## Implementation Details

### Reference Implementation

Our implementation in `/Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/attention.py` follows best practices:

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        query: (B, T, d_k)
        key: (B, T, d_k)
        value: (B, T, d_v)
        mask: (T, T) or (B, T, T), values are 0 or -inf
        dropout: Optional dropout layer

    Returns:
        output: (B, T, d_v)
        attention_weights: (B, T, T)
    """
    d_k = query.size(-1)

    # Step 1: Compute scores
    scores = query @ key.transpose(-2, -1)  # (B, T, T)

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask
    if mask is not None:
        scores = scores + mask

    # Step 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)

    # Step 5: Apply dropout (during training)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 6: Apply to values
    output = attention_weights @ value  # (B, T, d_v)

    return output, attention_weights
```

### Key Implementation Choices

#### 1. Returning Attention Weights

We return attention weights for:
- **Visualization**: Understanding what the model attends to
- **Debugging**: Checking if attention makes sense
- **Analysis**: Studying attention patterns

Trade-off: Uses extra memory, but invaluable for development.

#### 2. Dropout on Attention Weights

Applying dropout to attention weights:
```python
attention_weights = dropout(attention_weights)
```

**Why?**
- Regularization: Prevents over-reliance on specific positions
- Robustness: Model learns to use multiple attention patterns

**When to use:**
- Training: Yes (typically 0.1 dropout rate)
- Inference: No (dropout is disabled)

#### 3. Numerical Stability

Softmax can overflow/underflow with large inputs. PyTorch's `F.softmax` handles this internally using the log-sum-exp trick:

```python
# Naive (unstable):
softmax(x) = exp(x) / sum(exp(x))

# Stable:
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

No action needed if using `F.softmax`, but good to know!

#### 4. Broadcasting for Masks

Masks can have different shapes:
```python
mask.shape = (T, T)        # Same mask for all batch elements
mask.shape = (B, T, T)     # Different mask per batch element
mask.shape = (B, 1, T)     # Padding mask (broadcast to (B, T, T))
```

PyTorch's broadcasting handles this automatically:
```python
scores = scores + mask  # Broadcasting handles shape alignment
```

### Practical Example

Let's walk through a complete example:

```python
import torch
import torch.nn.functional as F
import math

# Setup
batch_size = 2
seq_len = 4
d_k = 8

# Random inputs
torch.manual_seed(42)
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

print("Input shapes:")
print(f"Q: {Q.shape}")  # (2, 4, 8)
print(f"K: {K.shape}")  # (2, 4, 8)
print(f"V: {V.shape}")  # (2, 4, 8)

# Step 1: Compute scores
scores = Q @ K.transpose(-2, -1)
print(f"\nScores shape: {scores.shape}")  # (2, 4, 4)
print(f"Scores (batch 0):\n{scores[0]}")

# Step 2: Scale
scores = scores / math.sqrt(d_k)
print(f"\nScaled scores (batch 0):\n{scores[0]}")

# Step 3: Causal mask
mask = create_causal_mask(seq_len)
print(f"\nMask:\n{mask}")
scores = scores + mask

# Step 4: Softmax
attention_weights = F.softmax(scores, dim=-1)
print(f"\nAttention weights (batch 0):\n{attention_weights[0]}")
print(f"Row sums: {attention_weights[0].sum(dim=-1)}")  # Should be [1, 1, 1, 1]

# Step 5: Apply to values
output = attention_weights @ V
print(f"\nOutput shape: {output.shape}")  # (2, 4, 8)
print(f"Output (batch 0, position 0):\n{output[0, 0]}")
```

**Expected output structure:**
```
Input shapes:
Q: torch.Size([2, 4, 8])
K: torch.Size([2, 4, 8])
V: torch.Size([2, 4, 8])

Scores shape: torch.Size([2, 4, 4])
Scores (batch 0):
tensor([[ 2.1, -0.5,  1.3, -1.8],
        [ 0.7,  3.2, -0.9,  2.1],
        ...])

Scaled scores (batch 0):
tensor([[ 0.74, -0.18,  0.46, -0.64],
        [ 0.25,  1.13, -0.32,  0.74],
        ...])

Attention weights (batch 0):
tensor([[1.00, 0.00, 0.00, 0.00],
        [0.28, 0.72, 0.00, 0.00],
        [0.20, 0.15, 0.65, 0.00],
        [0.15, 0.25, 0.18, 0.42]])

Row sums: tensor([1., 1., 1., 1.])
```

### Testing Your Implementation

Essential tests for attention:

```python
def test_attention_output_shape():
    """Test output has correct shape."""
    B, T, d_k = 2, 10, 64
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    output, attn = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (B, T, d_k)
    assert attn.shape == (B, T, T)

def test_attention_weights_sum_to_one():
    """Test attention weights are valid probabilities."""
    B, T, d_k = 2, 10, 64
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    _, attn = scaled_dot_product_attention(Q, K, V)

    # Each row should sum to 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))

def test_causal_mask():
    """Test causal masking prevents attending to future."""
    B, T, d_k = 2, 5, 64
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    mask = create_causal_mask(T)
    _, attn = scaled_dot_product_attention(Q, K, V, mask)

    # Upper triangle should be zero
    for i in range(T):
        for j in range(i+1, T):
            assert attn[:, i, j].item() == 0.0

def test_attention_is_differentiable():
    """Test gradients flow through attention."""
    B, T, d_k = 2, 5, 64
    Q = torch.randn(B, T, d_k, requires_grad=True)
    K = torch.randn(B, T, d_k, requires_grad=True)
    V = torch.randn(B, T, d_k, requires_grad=True)

    output, _ = scaled_dot_product_attention(Q, K, V)
    loss = output.sum()
    loss.backward()

    # All inputs should have gradients
    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
```

---

## Common Pitfalls and Debugging

### Pitfall 1: Forgetting to Scale

**Symptom:** Training is unstable, attention weights are nearly one-hot, gradients vanish.

**Cause:**
```python
# WRONG:
scores = Q @ K.transpose(-2, -1)
attention = F.softmax(scores, dim=-1)  # Scores are too large!
```

**Fix:**
```python
# CORRECT:
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
attention = F.softmax(scores, dim=-1)
```

**How to detect:**
```python
# Print score statistics
print(f"Score mean: {scores.mean():.2f}")
print(f"Score std: {scores.std():.2f}")
print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# For d_k=64, expect std ≈ 1.0 with scaling, ≈ 8.0 without
```

### Pitfall 2: Wrong Mask Values

**Symptom:** Masked positions still have attention weight, or all weights are 0/NaN.

**Cause:**
```python
# WRONG: Mask should be 0 and -inf, not 0 and 1
mask = torch.tril(torch.ones(T, T))  # [1, 0, 0], [1, 1, 0], [1, 1, 1]
scores = scores + mask  # Adding 1s doesn't mask!
```

**Fix:**
```python
# CORRECT:
mask = torch.tril(torch.ones(T, T))
mask = mask.masked_fill(mask == 0, float('-inf'))
mask = mask.masked_fill(mask == 1, 0.0)
```

**How to detect:**
```python
# Visualize attention weights
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0].detach(), cmap='viridis')
plt.colorbar()
plt.title('Attention Weights (should be lower triangular)')
plt.show()
```

### Pitfall 3: Softmax on Wrong Dimension

**Symptom:** Attention weights don't sum to 1, bizarre attention patterns.

**Cause:**
```python
# WRONG: Softmax over rows instead of columns
attention = F.softmax(scores, dim=-2)  # dim=-2 is the query dimension!
```

**Fix:**
```python
# CORRECT: Softmax over the key dimension (last dimension)
attention = F.softmax(scores, dim=-1)
```

**How to detect:**
```python
# Check row sums
row_sums = attention.sum(dim=-1)
print(f"Row sums: {row_sums[0]}")  # Should be all 1.0

# Check column sums
col_sums = attention.sum(dim=-2)
print(f"Col sums: {col_sums[0]}")  # Will NOT be 1.0 (this is expected)
```

### Pitfall 4: Dimension Mismatches

**Symptom:** RuntimeError about matrix dimensions.

**Cause:**
```python
# WRONG: Transpose wrong dimensions
scores = Q @ K.transpose(0, 1)  # Transposes batch and seq_len!
```

**Fix:**
```python
# CORRECT: Transpose last two dimensions
scores = Q @ K.transpose(-2, -1)  # Always use -2, -1 for safety
```

**How to detect:**
```python
# Print shapes at each step
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"K.T shape: {K.transpose(-2, -1).shape}")
print(f"Q @ K.T shape: {(Q @ K.transpose(-2, -1)).shape}")
```

### Pitfall 5: Mask Broadcasting Issues

**Symptom:** RuntimeError about incompatible shapes.

**Cause:**
```python
# WRONG: Mask shape doesn't broadcast with scores
mask = torch.zeros(T)  # Shape (T,), but scores are (B, T, T)
scores = scores + mask  # Can't broadcast!
```

**Fix:**
```python
# CORRECT: Ensure mask broadcasts properly
mask = torch.zeros(T, T)  # (T, T) broadcasts to (B, T, T)
# or
mask = torch.zeros(1, T, T)  # (1, T, T) explicitly broadcasts
# or
mask = torch.zeros(B, T, T)  # Exact match
```

**Broadcasting rules:**
```
scores:     (B, T, T)
mask:       (   T, T)  ✓ Works
mask:       (B, 1, T)  ✓ Works (broadcasts middle dim)
mask:       (   1, T)  ✓ Works
mask:       (      T)  ✗ Fails (can't broadcast)
```

### Debugging Checklist

When attention isn't working:

1. **Check shapes:**
   ```python
   print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
   print(f"Scores: {scores.shape}")
   print(f"Attention: {attention.shape}")
   print(f"Output: {output.shape}")
   ```

2. **Check statistics:**
   ```python
   print(f"Scores: mean={scores.mean():.3f}, std={scores.std():.3f}")
   print(f"Attention: min={attention.min():.3f}, max={attention.max():.3f}")
   print(f"Attention row sums: {attention.sum(dim=-1)[0]}")
   ```

3. **Check for NaN/Inf:**
   ```python
   assert not torch.isnan(scores).any(), "NaN in scores!"
   assert not torch.isinf(scores).any(), "Inf in scores!"
   assert not torch.isnan(attention).any(), "NaN in attention!"
   ```

4. **Visualize attention:**
   ```python
   import matplotlib.pyplot as plt
   plt.imshow(attention[0].detach().cpu(), cmap='viridis')
   plt.colorbar()
   plt.xlabel('Key position')
   plt.ylabel('Query position')
   plt.title('Attention Weights')
   plt.show()
   ```

5. **Check gradients:**
   ```python
   loss = output.sum()
   loss.backward()
   print(f"Q grad: {Q.grad.abs().mean():.6f}")
   print(f"K grad: {K.grad.abs().mean():.6f}")
   print(f"V grad: {V.grad.abs().mean():.6f}")
   # Should be non-zero and finite
   ```

### Performance Tips

1. **Use fused operations when possible:**
   ```python
   # PyTorch 2.0+ has optimized scaled_dot_product_attention
   from torch.nn.functional import scaled_dot_product_attention as sdpa
   output = sdpa(Q, K, V, attn_mask=mask)  # Faster!
   ```

2. **Avoid materializing large intermediate tensors:**
   ```python
   # Less memory-efficient:
   scores = Q @ K.transpose(-2, -1)
   scores = scores / math.sqrt(d_k)

   # More memory-efficient:
   scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
   ```

3. **Use `torch.compile` (PyTorch 2.0+):**
   ```python
   attention_fn = torch.compile(scaled_dot_product_attention)
   output, weights = attention_fn(Q, K, V, mask)
   ```

4. **Profile your code:**
   ```python
   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       output, _ = scaled_dot_product_attention(Q, K, V)

   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

---

## Summary

### Key Takeaways

1. **Attention is a weighted aggregation mechanism** that allows each position to selectively focus on relevant parts of the input.

2. **The core equation is simple:**
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   ```
   But each component serves a crucial purpose.

3. **Query, Key, Value separation** enables:
   - Q: What information do I need?
   - K: What information do I offer?
   - V: What is the actual information?

4. **Scaling by √d_k is critical** to prevent softmax saturation and ensure stable gradients.

5. **Masking enables:**
   - Causal attention for autoregressive generation
   - Handling variable-length sequences with padding

6. **Computational complexity is O(n²)**, which limits sequence length and drives research into efficient variants.

7. **Implementation requires care** with:
   - Shape handling (batch, sequence, dimension)
   - Mask broadcasting
   - Numerical stability
   - Gradient flow

### The Big Picture

Attention is elegant in its simplicity yet powerful in its expressiveness. It solves the fundamental problem of **how to dynamically route information** in a neural network:

- **RNNs**: Fixed routing through hidden states (bottleneck)
- **CNNs**: Fixed receptive fields (limited context)
- **Attention**: Dynamic routing based on content (flexible, parallelizable)

This flexibility is why transformers have become the dominant architecture for:
- Natural language processing (GPT, BERT, T5)
- Computer vision (ViT, DETR)
- Speech processing (Whisper)
- Multi-modal models (CLIP, Flamingo)

### Next Steps

Now that you understand attention:

1. **Multi-Head Attention**: Running multiple attention operations in parallel
2. **Positional Encoding**: How transformers incorporate sequence order
3. **Feed-Forward Networks**: The other key component of transformer blocks
4. **Layer Normalization**: Stabilizing training
5. **The Full Transformer**: Putting it all together

### Further Reading

**Foundational Papers:**
- Vaswani et al. (2017): "Attention Is All You Need"
- Bahdanau et al. (2015): "Neural Machine Translation by Jointly Learning to Align and Translate"

**Tutorials:**
- The Illustrated Transformer (Jay Alammar)
- The Annotated Transformer (Harvard NLP)

**Advanced Topics:**
- Flash Attention (Dao et al., 2022)
- Linear Attention Transformers
- Sparse Attention Patterns

### Practice Exercises

1. **Implement from scratch**: Code the attention mechanism without looking at the reference implementation.

2. **Visualize attention patterns**: Generate text with a pretrained model and visualize what it attends to.

3. **Experiment with scaling**: Remove the √d_k scaling and observe training dynamics.

4. **Try different sequence lengths**: Measure memory usage for T = 128, 512, 1024, 2048.

5. **Implement sparse attention**: Create a local attention variant that only attends to k nearest neighbors.

---

## Appendix: Connection to Code

Our implementation at `/Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/attention.py` demonstrates all concepts covered:

**Key functions:**

1. `scaled_dot_product_attention()`: Core attention mechanism
   - Lines 19-84: Full implementation with detailed comments
   - Lines 56-63: Scaling factor with explanation
   - Lines 65-69: Masking logic
   - Lines 71-74: Softmax normalization

2. `create_causal_mask()`: Mask creation for autoregressive models
   - Lines 87-119: Generates lower-triangular mask
   - Example output in docstring

3. `Attention` class: Modular wrapper
   - Lines 122-161: PyTorch `nn.Module` interface
   - Includes dropout support

**Example usage:** Lines 164-216 show practical examples with different configurations.

**Testing:** See `/Users/shiongtan/projects/tiny-transformer-build/tests/test_attention.py` for comprehensive tests.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Implementation Reference:** `tiny_transformer/attention.py`
**Target Audience:** Intermediate developers learning transformers from scratch
