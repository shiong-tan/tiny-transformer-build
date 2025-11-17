# Embeddings & Positional Encoding: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [The Representation Problem](#the-representation-problem)
3. [Token Embeddings: From Discrete to Continuous](#token-embeddings-from-discrete-to-continuous)
4. [The Position Problem](#the-position-problem)
5. [Positional Encoding: Design Principles](#positional-encoding-design-principles)
6. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
7. [Learned Positional Embeddings](#learned-positional-embeddings)
8. [Combining Token and Positional Information](#combining-token-and-positional-information)
9. [Implementation Deep Dive](#implementation-deep-dive)
10. [Advanced Topics](#advanced-topics)
11. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
12. [Practical Considerations](#practical-considerations)
13. [Summary](#summary)

---

## Introduction

Embeddings and positional encodings form the input layer of transformer architectures. They solve two fundamental problems:

1. **Token Representation**: How do we represent discrete tokens (words, subwords, characters) as continuous vectors that neural networks can process?
2. **Position Information**: How do we inject sequence order into a position-invariant architecture like the transformer?

**What you'll learn:**
- Why we need embeddings and what they represent geometrically
- The mechanics of embedding lookup tables
- Why we scale embeddings by √d_model
- The position-invariance problem in transformers
- Mathematical foundations of sinusoidal positional encoding
- Why sinusoidal functions enable extrapolation
- Trade-offs between sinusoidal and learned positional embeddings
- How to combine token and positional information
- Implementation details and numerical stability considerations

**Prerequisites:**
- Completed Module 01 (Attention Mechanism)
- Completed Module 02 (Multi-Head Attention)
- Completed Module 03 (Transformer Block)
- Basic understanding of vector spaces and trigonometry
- Familiarity with neural network embedding layers

**Key Insight Preview:**

Unlike RNNs which inherently process sequences in order, transformers treat input as an **unordered set**. Without explicit positional information:
```
Attention("The cat sat") = Attention("sat The cat") = Attention("cat sat The")
```

This is both a feature (enables parallelization) and a bug (loses sequence order). Positional encoding is the elegant solution.

---

## The Representation Problem

### From Symbols to Vectors

Natural language consists of discrete symbols: words, characters, subwords. Neural networks, however, operate on continuous vectors. We need a bridge between these two worlds.

**The fundamental challenge:**
- **Input**: Discrete tokens from a vocabulary V = {w₁, w₂, ..., w_vocab_size}
- **Required output**: Continuous vectors in ℝ^d_model
- **Constraint**: Similar tokens should have similar representations

### Historical Context: One-Hot Encoding

Before embeddings, the naive approach was one-hot encoding:

```
Vocabulary: ["the", "cat", "sat", "mat"]

"cat" → [0, 1, 0, 0]  (4-dimensional)
"sat" → [0, 0, 1, 0]
```

**Problems with one-hot encoding:**

1. **Dimensionality**: Vector size = vocabulary size (often 10k-100k)
2. **Sparsity**: Only one non-zero element per vector
3. **No similarity**: All pairs of words are equidistant
   - distance("cat", "dog") = distance("cat", "computer") = √2
4. **No generalization**: Can't represent unseen words
5. **Computational cost**: Massive matrix multiplications

**Example:**
```python
vocab_size = 50000
batch_size = 32
seq_len = 512

# One-hot representation
one_hot_size = batch_size * seq_len * vocab_size
             = 32 * 512 * 50000
             = 819,200,000 values  # ~3.2 GB for single batch!

# First layer weights
W = (vocab_size, hidden_dim) = (50000, 512)
  = 25,600,000 parameters
```

This is computationally prohibitive.

### The Embedding Solution

Embeddings map discrete tokens to **dense, low-dimensional** continuous vectors:

```
Token ID → Embedding Vector
"cat" (ID: 42) → [0.21, -0.15, 0.87, ..., 0.34]  (512-dimensional)
```

**Key properties:**

1. **Dimensionality reduction**: vocab_size → d_model (e.g., 50000 → 512)
2. **Dense representation**: All dimensions contain meaningful information
3. **Learned similarity**: Similar words → similar vectors
4. **Efficient**: Lookup table operation, not matrix multiplication

**Geometric interpretation:**

Embeddings place each token in a d_model-dimensional space where geometric relationships reflect semantic relationships:

```
Geometric Space               Semantic Space
┌─────────────────┐          ┌──────────────┐
│  cat  •         │          │ Animals      │
│    dog •        │  ←→      │ close to     │
│                 │          │ each other   │
│        • car    │          │ Vehicles     │
│      • bus      │          │ separate     │
└─────────────────┘          └──────────────┘
```

Famous example (Word2Vec era):
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

---

## Token Embeddings: From Discrete to Continuous

### The Lookup Table Mechanism

At its core, an embedding layer is a **learnable lookup table**:

```
Embedding Matrix E ∈ ℝ^(vocab_size × d_model)

┌─────────────────────────────────┐
│ Token 0:  [0.12, -0.34, 0.56, ...]  ← Row 0
│ Token 1:  [0.89, 0.23, -0.45, ...]  ← Row 1
│ Token 2:  [-0.67, 0.91, 0.12, ...]  ← Row 2
│    ⋮           ⋮        ⋮
│ Token V:  [0.45, -0.78, 0.23, ...]  ← Row vocab_size-1
└─────────────────────────────────┘
     ↑
     └─ Each row is the embedding for one token
```

**Forward pass:**
```python
# Input: Token IDs
tokens = [42, 17, 9, 105]  # Shape: (seq_len,)

# Output: Embedding vectors
embeddings = E[tokens]      # Shape: (seq_len, d_model)
            = [E[42],       # Embedding for token 42
               E[17],       # Embedding for token 17
               E[9],        # Embedding for token 9
               E[105]]      # Embedding for token 105
```

**Batched operation:**
```python
# Input: Batch of sequences
tokens = [[42, 17, 9],      # Sequence 1
          [105, 3, 42],     # Sequence 2
          [9, 9, 17]]       # Sequence 3
# Shape: (batch_size=3, seq_len=3)

# Output: Batch of embedding sequences
embeddings = E[tokens]
# Shape: (batch_size=3, seq_len=3, d_model)
```

### Initialization Strategy

How we initialize embeddings matters for training dynamics:

**Random initialization (most common):**
```python
import torch.nn as nn

embedding = nn.Embedding(vocab_size, d_model)
# Default: Initialized from N(0, 1)

# Common practice: Scale initialization
embedding.weight.data.normal_(mean=0, std=0.02)
```

**Why small values?**
- Large initial embeddings → large gradients → training instability
- Small values allow gradients to shape the embedding space

**Alternative: Pre-trained embeddings**
```python
# Load pre-trained embeddings (e.g., Word2Vec, GloVe, FastText)
pretrained_weights = load_pretrained_embeddings()  # Shape: (vocab_size, d_model)
embedding = nn.Embedding.from_pretrained(pretrained_weights)
embedding.weight.requires_grad = True  # Fine-tune or freeze
```

### Embedding Scaling: The √d_model Factor

A crucial but often overlooked detail: **embeddings are typically scaled by √d_model**.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        # Key line: Scale by sqrt(d_model)
        return self.embedding(tokens) * math.sqrt(self.d_model)
```

**Why scale by √d_model?**

This comes from the original "Attention Is All You Need" paper. The reasoning:

1. **Variance preservation**:
   - Embeddings initialized with std ≈ 1
   - As d_model increases, the L2 norm of random vectors grows: E[||x||²] = d_model
   - Scaling by √d_model normalizes the magnitude

2. **Balance with positional encoding**:
   - Sinusoidal positional encodings have values in range [-1, 1]
   - Without scaling, embeddings (magnitude ~√d_model) would dominate
   - Scaling ensures both contribute meaningfully to the sum

**Example with numbers:**
```python
d_model = 512

# Unscaled embedding (initialized std=1)
embedding_unscaled = torch.randn(512)
norm_unscaled = embedding_unscaled.norm().item()
# norm_unscaled ≈ √512 ≈ 22.6

# Scaled embedding
embedding_scaled = embedding_unscaled * math.sqrt(512)
norm_scaled = embedding_scaled.norm().item()
# norm_scaled ≈ 22.6 * 22.6 ≈ 512

# Positional encoding (values in [-1, 1])
pos_encoding = torch.sin(torch.randn(512))
norm_pos = pos_encoding.norm().item()
# norm_pos ≈ 10-20 (much smaller than scaled embedding)
```

**When to skip scaling:**

Some modern implementations (e.g., GPT-2, GPT-3) **do not** use this scaling:
- Different initialization strategies
- LayerNorm before embedding addition
- Learned scaling factors

**Rule of thumb:** Follow the original paper's architecture unless you have a specific reason to deviate.

### Vocabulary Size Considerations

The vocabulary size directly impacts model size and performance:

**Trade-offs:**

| Aspect | Small Vocabulary (e.g., 5k) | Large Vocabulary (e.g., 50k) |
|--------|----------------------------|------------------------------|
| **Embedding parameters** | vocab_size × d_model = 2.5M | vocab_size × d_model = 25M |
| **Sequence length** | Longer (more subwords) | Shorter (fewer subwords) |
| **OOV handling** | More unknown tokens | Better coverage |
| **Training speed** | Faster (shorter sequences) | Slower (longer sequences) |
| **Memory** | Less embedding memory | More embedding memory |

**Modern practice (BPE/SentencePiece):**

Instead of word-level vocabularies, use **subword tokenization**:

```
Word-level (large vocab):
"unhappiness" → ["unhappiness"]  (might be OOV)

Subword-level (medium vocab):
"unhappiness" → ["un", "happiness"]

Character-level (tiny vocab):
"unhappiness" → ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]
```

**Optimal vocabulary size:**
- GPT-2: 50,257 tokens (BPE)
- BERT: 30,522 tokens (WordPiece)
- LLaMA: 32,000 tokens (SentencePiece)

Typical range: **30k-50k** balances coverage and efficiency.

### Shared vs. Separate Embeddings

In encoder-decoder models, you can share embeddings:

**Shared embeddings:**
```python
# Same embedding for encoder and decoder
shared_embedding = nn.Embedding(vocab_size, d_model)

encoder_input = shared_embedding(src_tokens)
decoder_input = shared_embedding(tgt_tokens)
```

**Benefits:**
- Fewer parameters
- Consistent representation across encoder/decoder
- Better for translation (source and target languages)

**Separate embeddings:**
```python
encoder_embedding = nn.Embedding(src_vocab_size, d_model)
decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
```

**Benefits:**
- Different vocabularies (e.g., multilingual)
- More flexibility
- Decoder can learn generation-specific representations

**For decoder-only models (GPT-style):** Only one embedding layer needed.

### Weight Tying with Output Layer

A common technique: **share embedding weights with output projection**:

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Output projection (logits over vocabulary)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: Share parameters
        self.output_projection.weight = self.embedding.weight
```

**Why this works:**

1. **Symmetry**: Embedding maps token → vector, output maps vector → token logits
2. **Parameter efficiency**: Reduces parameters by vocab_size × d_model
3. **Regularization**: Forces consistent representation between input and output
4. **Empirically better**: Shown to improve perplexity in language models

**Caveat:** Requires d_model to match output projection dimensions.

---

## The Position Problem

### Position-Invariance in Transformers

The attention mechanism has a beautiful property: it's **permutation-invariant**.

**Mathematical formulation:**

For any permutation π of positions:
```
Attention(X_π) = Attention(X)_π
```

**Concrete example:**

```python
# Original sequence
X = ["The", "cat", "sat"]

# Permuted sequence
X_perm = ["sat", "The", "cat"]

# Attention weights are the same (just permuted)
attn_weights = softmax(QK^T / √d_k)
attn_weights_perm = permute(attn_weights)

# Outputs are permuted versions
output = attn_weights @ V
output_perm = permute(output)
```

**Why is this a problem?**

Language is inherently **sequential**:
- "The dog bit the man" ≠ "The man bit the dog"
- "She didn't not come" ≠ "She didn't come not"
- Word order determines meaning in most languages

**Attention alone cannot distinguish:**
```
Input 1: ["I", "love", "Paris"]
Input 2: ["Paris", "love", "I"]
Input 3: ["love", "I", "Paris"]

Without positional info: All produce identical attention patterns!
```

### Why RNNs Don't Have This Problem

Recurrent networks process sequentially:

```
RNN:
h_0 = 0
h_1 = f(h_0, x_1)  ← Position 1
h_2 = f(h_1, x_2)  ← Position 2 knows it comes after 1
h_3 = f(h_2, x_3)  ← Position 3 knows it comes after 2
```

Position information is **implicit** in the recurrence.

**Transformer:**
```
All positions processed simultaneously:
h_1, h_2, h_3 = Attention([x_1, x_2, x_3])  ← No inherent order
```

Position information must be **explicit**.

### The Naive Solution: Position as a Feature

Why not just add position as an extra feature?

```python
# Bad idea: Position as a single number
x = [token_embedding, position]
# Shape: (d_model + 1,)

# Example
token_1 = [0.2, -0.5, 0.8, ..., 1]  # Position 1
token_2 = [0.2, -0.5, 0.8, ..., 2]  # Position 2
```

**Problems:**

1. **Scale mismatch**: Position grows unbounded (1, 2, 3, ..., 1000, ...), embedding values are bounded
2. **No similarity**: Position 100 and 101 are as different as 100 and 999
3. **Dimensionality**: Wastes only one dimension for positional information
4. **No structure**: Model must learn from scratch that adjacent positions are similar

We need a **structured, bounded, high-dimensional** positional representation.

---

## Positional Encoding: Design Principles

Before diving into specific methods, let's establish what makes a good positional encoding.

### Desiderata: What We Want

A good positional encoding should:

1. **Uniqueness**: Each position has a unique encoding
   ```
   PE(i) ≠ PE(j) for all i ≠ j
   ```

2. **Bounded values**: Encodings don't grow unbounded with position
   ```
   ||PE(i)|| ≤ C for all i (some constant C)
   ```

3. **Consistency**: Same position always gets same encoding (for deterministic methods)
   ```
   PE(i) is deterministic
   ```

4. **Relative position**: Model can learn to attend based on relative distances
   ```
   Relationship between PE(i) and PE(i+k) should be consistent
   ```

5. **Extrapolation**: Should generalize to longer sequences than seen during training
   ```
   PE(i) well-defined for i > max_training_length
   ```

6. **Computational efficiency**: Fast to compute
   ```
   O(1) or O(d_model) per position
   ```

### Two Main Approaches

**1. Fixed (Sinusoidal) Positional Encoding**
- Non-learnable, deterministic functions
- Original transformer approach
- Generalizes to unseen sequence lengths

**2. Learned Positional Embeddings**
- Learnable parameters (like token embeddings)
- Modern approach (GPT, BERT)
- More flexible but limited to max_seq_len

### Position Encoding vs. Positional Embedding: Terminology

**Encoding (Fixed):**
```python
# Computed from mathematical formula
PE(pos, i) = sin(pos / 10000^(2i/d_model))
# No learnable parameters
```

**Embedding (Learned):**
```python
# Learned lookup table
pos_embedding = nn.Embedding(max_seq_len, d_model)
# Learnable parameters: max_seq_len × d_model
```

We'll use "encoding" for sinusoidal and "embedding" for learned, though the terms are often used interchangeably.

---

## Sinusoidal Positional Encoding

The original transformer paper introduced sinusoidal positional encoding, one of the most elegant solutions in deep learning.

### Mathematical Formulation

For position `pos` and dimension index `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Breaking it down:**

- `pos`: Position in sequence (0, 1, 2, ..., seq_len-1)
- `i`: Dimension index (0, 1, 2, ..., d_model/2 - 1)
- `2i`: Even dimensions use sine
- `2i+1`: Odd dimensions use cosine
- `10000`: Wavelength scaling constant (explained below)

**Rewritten for clarity:**

Let's define the wavelength for dimension i:
```
λ_i = 10000^(2i/d_model)
```

Then:
```
PE(pos, 2i)   = sin(pos / λ_i)
PE(pos, 2i+1) = cos(pos / λ_i)
```

### Intuition: Multi-Scale Frequencies

The key insight: **each dimension uses a different frequency**.

**Low dimensions (i close to 0):**
```
i = 0:
λ_0 = 10000^(0/512) = 1
PE(pos, 0) = sin(pos / 1) = sin(pos)  ← High frequency, fast oscillation
```

**High dimensions (i close to d_model/2):**
```
i = 255 (for d_model=512):
λ_255 = 10000^(510/512) ≈ 9886
PE(pos, 510) = sin(pos / 9886)  ← Low frequency, slow oscillation
```

**Visualization for d_model=6:**

```
Position:    0      1      2      3      4      5      6      7
─────────────────────────────────────────────────────────────────
Dim 0 (sin): 0.00   0.84   0.91   0.14  -0.76  -0.96  -0.28   0.66  ← λ=1
Dim 1 (cos): 1.00   0.54  -0.42  -0.99  -0.65   0.28   0.96   0.75  ← λ=1

Dim 2 (sin): 0.00   0.10   0.20   0.30   0.39   0.48   0.56   0.64  ← λ=10
Dim 3 (cos): 1.00   0.99   0.98   0.95   0.92   0.88   0.83   0.77  ← λ=10

Dim 4 (sin): 0.00   0.01   0.02   0.03   0.04   0.05   0.06   0.07  ← λ=100
Dim 5 (cos): 1.00   1.00   1.00   1.00   0.99   0.99   0.99   0.99  ← λ=100
```

**Pattern:**
- **High-frequency dimensions** (low i): Change rapidly, encode fine-grained position
- **Low-frequency dimensions** (high i): Change slowly, encode coarse-grained position

This creates a **multi-resolution position signature**, similar to how Fourier series decompose signals into different frequencies.

### Why This Works: Unique Position Signatures

Each position gets a unique d_model-dimensional vector:

```
Position 0: [sin(0/1), cos(0/1), sin(0/10), cos(0/10), sin(0/100), cos(0/100)]
          = [0.00, 1.00, 0.00, 1.00, 0.00, 1.00]

Position 1: [sin(1/1), cos(1/1), sin(1/10), cos(1/10), sin(1/100), cos(1/100)]
          = [0.84, 0.54, 0.10, 0.99, 0.01, 1.00]

Position 2: [sin(2/1), cos(2/1), sin(2/10), cos(2/10), sin(2/100), cos(2/100)]
          = [0.91, -0.42, 0.20, 0.98, 0.02, 1.00]
```

**No two positions have the same encoding** (within practical sequence lengths).

### Why Sine and Cosine?

Why not other functions? Sine and cosine have special properties:

**1. Bounded values:**
```
-1 ≤ sin(x) ≤ 1
-1 ≤ cos(x) ≤ 1
```

Encodings don't grow with position (unlike position itself: 1, 2, 3, ..., 1000, ...).

**2. Periodic:**

Periodicity enables the model to learn relative positions through simple linear transformations.

**Key theorem:** For any fixed offset k, there exists a linear transformation that maps PE(pos) to PE(pos+k):

```
[sin(pos + k)]   [cos(k)  sin(k)] [sin(pos)]
[cos(pos + k)] = [-sin(k) cos(k)] [cos(pos)]
```

This is the **angle addition formula**:
```
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
```

**What this means:**

The model can learn to compute "position i + 3" from "position i" using a simple linear layer. This helps with learning relative positional relationships.

**3. Smooth:**

Sine and cosine are smooth, differentiable functions. Small changes in position lead to small changes in encoding:

```
PE(10) ≈ PE(11)  (nearby positions have similar encodings)
PE(10) ≠ PE(100) (distant positions have different encodings)
```

This smoothness helps the model interpolate between positions.

### The 10000 Constant: Where Does It Come From?

The wavelength scaling uses 10000 as a base. Why?

**Derivation:**

We want:
- Shortest wavelength (highest frequency): λ_min = 2π (completes one cycle per position)
- Longest wavelength (lowest frequency): λ_max = some large value (e.g., 10000 × 2π)

The geometric progression:
```
λ_i = λ_min × (λ_max / λ_min)^(2i/d_model)
    = 2π × (10000)^(2i/d_model)
```

Simplifying (absorbing 2π into the sine/cosine):
```
λ_i = 10000^(2i/d_model)
```

**Why 10000 specifically?**

It's somewhat arbitrary, but chosen to:
- Handle typical sequence lengths (up to ~10000 tokens)
- Create enough variation in high-dimensional spaces
- Balance between too much and too little variation

**What if we used different values?**

```
# Smaller base (e.g., 100):
λ_max = 100^(d_model/d_model) = 100
→ Repeats patterns more frequently
→ Limited to shorter sequences

# Larger base (e.g., 1000000):
λ_max = 1000000
→ Slower variation in high dimensions
→ May not capture fine-grained position info
```

In practice, 10000 works well for most applications. Some models adjust it:
- RoFormer uses different bases
- ALiBi removes positional encoding entirely (adds bias to attention)

### Implementation Details

**Efficient computation:**

Instead of computing sine and cosine separately for each position, we can vectorize:

```python
import torch
import math

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Compute sinusoidal positional encoding.

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)

    Returns:
        Tensor of shape (seq_len, d_model)
    """
    # Create position indices: [0, 1, 2, ..., seq_len-1]
    position = torch.arange(seq_len).unsqueeze(1)  # Shape: (seq_len, 1)

    # Create dimension indices: [0, 1, 2, ..., d_model/2 - 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )  # Shape: (d_model/2,)

    # Compute positional encoding
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

    return pe
```

**Breaking down the computation:**

**Step 1: Position matrix**
```python
position = torch.arange(seq_len).unsqueeze(1)
# Shape: (seq_len, 1)
# [[0], [1], [2], ..., [seq_len-1]]
```

**Step 2: Frequency divisors**
```python
# Original formula: 10000^(2i/d_model)
# Taking log: exp(log(10000^(2i/d_model))) = exp(2i * log(10000) / d_model)
# Rearranging: exp(2i * (-log(10000) / d_model))

div_term = torch.exp(
    torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
)
# Shape: (d_model/2,)
# [exp(0), exp(-2*log(10000)/d_model), exp(-4*log(10000)/d_model), ...]
# = [1, 1/10000^(2/d_model), 1/10000^(4/d_model), ...]
```

This is equivalent to `1 / λ_i` for each dimension.

**Step 3: Compute sin/cos**
```python
pe[:, 0::2] = torch.sin(position * div_term)
# Broadcasting: (seq_len, 1) * (d_model/2,) → (seq_len, d_model/2)
# Assigns to even columns: 0, 2, 4, ...

pe[:, 1::2] = torch.cos(position * div_term)
# Assigns to odd columns: 1, 3, 5, ...
```

**Result:**
```
pe[pos, 2i]   = sin(pos * div_term[i]) = sin(pos / λ_i)
pe[pos, 2i+1] = cos(pos * div_term[i]) = cos(pos / λ_i)
```

### Extrapolation to Longer Sequences

One of the key advantages of sinusoidal encoding: it **extrapolates to any sequence length**.

**Training:**
```python
# Train with sequences up to length 512
max_train_len = 512
pe_train = sinusoidal_positional_encoding(max_train_len, d_model)
```

**Inference:**
```python
# Inference with sequences up to length 2048 (4x longer!)
max_inference_len = 2048
pe_inference = sinusoidal_positional_encoding(max_inference_len, d_model)
# Works perfectly - no retraining needed
```

**Why this works:**

The sinusoidal functions are defined for all positions:
```
sin(1024 / 10000^(2i/d_model)) is well-defined
```

Even if the model never saw position 1024 during training, the encoding is computed from the same formula.

**Caveat:**

While the encoding is defined, the model may not **generalize well** to much longer sequences:
- Attention patterns may differ
- Long-range dependencies not seen during training
- Numerical precision issues for very large positions

**Modern improvements:**
- ALiBi (Attention with Linear Biases): Better extrapolation
- RoPE (Rotary Position Embedding): Relative positions in attention
- Learned embeddings with interpolation

### Periodicity and Uniqueness

Sinusoidal encodings are periodic. Does this cause collisions?

**Single dimension (wavelength λ):**
```
PE(pos, i) = sin(pos / λ)

Repeats every: pos = λ × 2π
```

**Example:**
```
For λ = 1 (dimension 0):
sin(0) = sin(2π) = sin(4π) = 0
```

**But in high dimensions:**

The combination of many wavelengths creates unique patterns up to very large positions.

**Uniqueness bound:**

For d_model = 512:
- Shortest wavelength: λ_0 = 1 → period ≈ 6.28
- Longest wavelength: λ_255 ≈ 10000 → period ≈ 62,832

The combined encoding is unique up to positions ~10000 before any significant collision.

**In practice:**
- Most sequences < 10000 tokens
- Rare collisions at extreme positions
- High-dimensional space makes exact collisions unlikely

### Visualization: Position Signatures

Let's visualize encodings for different positions:

```
d_model = 8 (simplified)

Position 0:  [0.00,  1.00,  0.00,  1.00,  0.00,  1.00,  0.00,  1.00]
Position 1:  [0.84,  0.54,  0.10,  1.00,  0.01,  1.00,  0.00,  1.00]
Position 2:  [0.91, -0.42,  0.20,  0.98,  0.02,  1.00,  0.00,  1.00]
Position 3:  [0.14, -0.99,  0.30,  0.95,  0.03,  1.00,  0.00,  1.00]
Position 4: [-0.76, -0.65,  0.39,  0.92,  0.04,  1.00,  0.00,  1.00]
Position 5: [-0.96,  0.28,  0.48,  0.88,  0.05,  0.99,  0.00,  1.00]
```

**Observations:**
- Dimensions 0-1: Rapid changes (high frequency)
- Dimensions 2-3: Moderate changes (medium frequency)
- Dimensions 4-7: Slow changes (low frequency)

**Geometric interpretation:**

Each position is a point in d_model-dimensional space. The sinusoidal encoding places these points in a structured pattern that:
- Nearby positions are nearby in space
- Distant positions are distant in space
- Linear transformations can capture relative offsets

### Comparison with Fourier Features

Sinusoidal positional encoding is essentially a **Fourier feature mapping**:

```
Input: pos (scalar)
Output: [sin(ω_0 pos), cos(ω_0 pos), sin(ω_1 pos), cos(ω_1 pos), ...]
```

Where ω_i = 1 / λ_i are the frequencies.

**Connections to signal processing:**
- Low frequencies encode global position (which part of sequence)
- High frequencies encode local position (exact position within part)
- Combined features span multiple scales

This multi-scale representation is why sinusoidal encoding works so well.

---

## Learned Positional Embeddings

While sinusoidal encoding is elegant, many modern transformers use **learned positional embeddings**.

### Motivation: Why Learn Positions?

**Advantages over sinusoidal:**

1. **Flexibility**: Model can learn task-specific positional patterns
2. **Simplicity**: No need to compute sine/cosine
3. **Empirically better**: Often performs slightly better on benchmarks
4. **Task adaptation**: Can capture domain-specific position patterns

**Example:**

In code:
```python
# Position 0 might learn "start of function" features
# Position 50 might learn "middle of block" features
```

In language:
```python
# Position 0 might learn "sentence start" features (capital letters, articles)
# Position -1 might learn "sentence end" features (punctuation)
```

### Implementation: Positional Embedding Lookup Table

Learned positional embeddings are implemented exactly like token embeddings:

```python
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # Learnable lookup table
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positional embeddings of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

        # Lookup embeddings
        return self.embedding(positions)  # Shape: (seq_len, d_model)
```

**Parameter count:**
```
Parameters = max_seq_len × d_model

Example (GPT-2 small):
max_seq_len = 1024
d_model = 768
Parameters = 1024 × 768 = 786,432
```

### Training Dynamics

Learned positional embeddings start random and evolve during training:

**Initialization:**
```python
# Random initialization
self.embedding.weight.data.normal_(mean=0, std=0.02)
```

**During training:**
- Gradients flow back from the loss
- Model learns which positional features are useful
- Embeddings adapt to the task

**What gets learned:**

For language modeling:
- Early positions: Sentence starters ("The", "A", capitals)
- Middle positions: General content
- Late positions: Sentence endings (periods, question marks)

For code:
- Early positions: Indentation level, function starts
- Middle positions: Logic flow
- Late positions: Return statements, closing brackets

### Absolute vs. Relative Position Information

Learned embeddings encode **absolute position**:

```python
# Position 5 always gets the same embedding (5th row of lookup table)
pos_5_embedding = embedding.weight[5]
```

**Limitations:**

The model doesn't inherently know:
- "Position 5 is 3 steps after position 2"
- "Position 100 and 101 are adjacent"

It must learn these relationships from data.

**Contrast with sinusoidal:**

Sinusoidal encodings have built-in relative position information (via the linear transformation property).

**Modern solutions:**
- Relative position embeddings (T5, Transformer-XL)
- Rotary Position Embedding (RoPE) - encodes relative positions directly in attention

### Extrapolation Limitations

**Critical limitation:** Learned embeddings cannot extrapolate beyond max_seq_len.

**Problem:**
```python
# Training: max_seq_len = 512
pos_embedding = nn.Embedding(512, d_model)

# Inference: sequence of length 1024
positions = torch.arange(1024)  # [0, 1, ..., 1023]
embeddings = pos_embedding(positions)  # ERROR! Index 512-1023 don't exist
```

**Solutions:**

**1. Truncation (most common):**
```python
# Only use first max_seq_len tokens
if seq_len > max_seq_len:
    x = x[:, :max_seq_len, :]
```

**2. Extrapolation via interpolation:**
```python
# Interpolate learned embeddings to longer sequences
def interpolate_position_embeddings(old_embeddings, new_max_len):
    """
    Interpolate embeddings from max_len to new_max_len.
    Used when fine-tuning on longer sequences.
    """
    old_max_len, d_model = old_embeddings.shape

    # Linear interpolation
    old_positions = torch.linspace(0, 1, old_max_len)
    new_positions = torch.linspace(0, 1, new_max_len)

    new_embeddings = torch.zeros(new_max_len, d_model)
    for dim in range(d_model):
        new_embeddings[:, dim] = torch.interp(
            new_positions, old_positions, old_embeddings[:, dim]
        )

    return new_embeddings
```

**3. Hybrid approach:**
```python
# Use sinusoidal for positions > max_seq_len
class HybridPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        self.learned = nn.Embedding(max_seq_len, d_model)
        self.sinusoidal = SinusoidalPositionalEncoding(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.learned.num_embeddings:
            return self.learned(torch.arange(seq_len))
        else:
            return self.sinusoidal(x)
```

### Comparison: Sinusoidal vs. Learned

| Aspect | Sinusoidal | Learned |
|--------|-----------|---------|
| **Parameters** | 0 (deterministic) | max_seq_len × d_model |
| **Extrapolation** | ✓ Works for any length | ✗ Limited to max_seq_len |
| **Flexibility** | Fixed pattern | Adapts to task |
| **Relative position** | Built-in (linear transform) | Must be learned |
| **Interpretability** | Clear mathematical meaning | Learned black box |
| **Performance** | Slightly worse (typically) | Slightly better (typically) |
| **Used in** | Original Transformer | GPT, BERT, most modern models |

**When to use which:**

**Sinusoidal:**
- Need to handle variable-length sequences
- Limited compute/memory
- Interpretability matters
- Mathematical elegance preferred

**Learned:**
- Fixed maximum length is acceptable
- Have sufficient data to learn patterns
- Want maximum performance
- Following modern best practices

**Current trend:** Most state-of-the-art models (GPT-3, GPT-4, LLaMA) use learned embeddings, but research into better positional encodings continues (RoPE, ALiBi, etc.).

---

## Combining Token and Positional Information

Now we have:
1. Token embeddings: `(batch_size, seq_len, d_model)`
2. Positional encodings/embeddings: `(seq_len, d_model)` or `(batch_size, seq_len, d_model)`

How do we combine them?

### Addition vs. Concatenation

Two obvious approaches:

**Approach 1: Addition (used in practice)**
```python
combined = token_embeddings + positional_encodings
# Shape: (batch_size, seq_len, d_model)
```

**Approach 2: Concatenation (NOT used)**
```python
combined = torch.cat([token_embeddings, positional_encodings], dim=-1)
# Shape: (batch_size, seq_len, 2 * d_model)
```

**Why addition?**

1. **Parameter efficiency**: No increase in dimensionality
   - Addition: d_model stays the same
   - Concatenation: d_model → 2 × d_model, doubles all subsequent layer sizes

2. **Empirically better**: Addition works well in practice
   - Transformer paper used addition
   - Extensive ablations show it's sufficient

3. **Interpretability**: Token and position live in the same semantic space
   - Addition: "Token at this position"
   - Concatenation: "Token" and "position" as separate features

**Mathematical perspective:**

Addition assumes:
```
embedding_space = token_space + position_space
```

The model learns to separate token and position information through attention and FFN layers.

**Proof it works:**

In attention, the model computes:
```
Q = (token_emb + pos_enc) W_Q
  = token_emb W_Q + pos_enc W_Q
```

The model can separately weight token and positional information via learned projections.

### Implementation Pattern

**Complete embedding layer:**

```python
class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        positional_type: str = "learned"  # or "sinusoidal"
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)  # Scaling factor

        # Positional embeddings
        if positional_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        elif positional_type == "learned":
            self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown positional type: {positional_type}")

        self.positional_type = positional_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        """
        Args:
            tokens: Token IDs of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = tokens.shape

        # Token embeddings with scaling
        token_emb = self.token_embedding(tokens) * self.scale
        # Shape: (batch_size, seq_len, d_model)

        # Positional embeddings
        if self.positional_type == "sinusoidal":
            pos_enc = self.positional_encoding(token_emb)
            # Shape: (seq_len, d_model)
            # Broadcasting: (batch_size, seq_len, d_model) + (seq_len, d_model)
        else:  # learned
            positions = torch.arange(seq_len, device=tokens.device)
            pos_enc = self.positional_embedding(positions)
            # Shape: (seq_len, d_model)

        # Combine via addition
        combined = token_emb + pos_enc
        # Shape: (batch_size, seq_len, d_model)

        # Apply dropout
        return self.dropout(combined)
```

**Shape flow:**
```
Input tokens:           (batch_size, seq_len)
                                ↓
Token embeddings:       (batch_size, seq_len, d_model)
                                ↓ [* √d_model]
Scaled embeddings:      (batch_size, seq_len, d_model)
                                ↓ [+]
Positional encodings:   (seq_len, d_model)  [broadcasted]
                                ↓
Combined:               (batch_size, seq_len, d_model)
                                ↓ [dropout]
Output:                 (batch_size, seq_len, d_model)
```

### Broadcasting Mechanics

Positional encodings are typically `(seq_len, d_model)` while token embeddings are `(batch_size, seq_len, d_model)`.

**How does addition work?**

PyTorch broadcasting rules:
```python
token_emb: (32, 512, 768)  # (batch_size, seq_len, d_model)
pos_enc:   (    512, 768)  # (seq_len, d_model)

# Broadcasting adds missing dimension at the beginning
pos_enc_broadcasted: (1, 512, 768)

# Then repeats to match batch size
combined = token_emb + pos_enc_broadcasted
# Shape: (32, 512, 768)
```

**Efficiency:**

Broadcasting doesn't copy data in memory. It's just a view, making it efficient.

**Alternative (explicit):**
```python
# Manual broadcasting (same result, less efficient)
pos_enc_expanded = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
combined = token_emb + pos_enc_expanded
```

### Dropout Considerations

**Why dropout after combining?**

```python
combined = token_emb + pos_enc
output = dropout(combined)
```

**Regularization:**
- Prevents overfitting to specific position patterns
- Forces model to be robust to missing positional information
- Typical dropout rate: 0.1

**Where NOT to apply dropout:**
```python
# Bad: Dropout before scaling
token_emb = dropout(self.token_embedding(tokens)) * self.scale  # ✗

# Bad: Separate dropout for token and position
token_emb = dropout(self.token_embedding(tokens) * self.scale)
pos_enc = dropout(self.positional_encoding(token_emb))
combined = token_emb + pos_enc  # ✗

# Good: Single dropout after combination
combined = token_emb + pos_enc
output = dropout(combined)  # ✓
```

**Why this matters:**

Applying dropout separately would break the relationship between token and position.

### Initialization and Scale Matching

For addition to work well, token and positional embeddings should have similar scales.

**Token embeddings (with scaling):**
```python
# After scaling by √d_model
token_emb = embedding(tokens) * math.sqrt(d_model)
# Norm ≈ √d_model × original_norm ≈ d_model (for std=1 initialization)
```

**Sinusoidal positional encoding:**
```python
# Values in [-1, 1]
pos_enc = sin(...)
# Norm ≈ √d_model (empirically)
```

**Learned positional embeddings:**
```python
# Initialized with std=0.02 (typical)
pos_embedding.weight.data.normal_(mean=0, std=0.02)
# Norm ≈ 0.02 × √d_model
```

**Problem:**

Scaled token embeddings (norm ~ d_model) dominate small positional embeddings (norm ~ 0.02 × √d_model).

**Solution:**

Either:
1. Don't scale token embeddings (GPT-2 approach)
2. Initialize positional embeddings with larger std
3. Use LayerNorm after combining (handles scale mismatch)

**Modern practice (GPT-2, GPT-3):**
```python
# No scaling on embeddings
token_emb = self.token_embedding(tokens)  # No * sqrt(d_model)
pos_emb = self.positional_embedding(positions)

# Embeddings are balanced
combined = token_emb + pos_emb
```

---

## Implementation Deep Dive

Let's implement everything from scratch with production-quality code.

### Sinusoidal Positional Encoding

**Full implementation:**

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Implements:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Embedding dimension (must be even)
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # Compute division term: 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x with positional encoding added, shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding (broadcasting across batch dimension)
        # x: (batch_size, seq_len, d_model)
        # self.pe: (1, max_len, d_model)
        # self.pe[:, :x.size(1)]: (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

**Key implementation details:**

1. **Pre-computation**: Encodings computed once in `__init__`, not every forward pass
2. **Buffer registration**: `register_buffer` ensures pe moves to GPU with model
3. **Flexible sequence length**: Slices pre-computed encodings to match input length
4. **Dropout**: Applied after adding positional encoding

**Alternative: On-demand computation**

For very long sequences (> max_len), compute on demand:

```python
def forward(self, x):
    batch_size, seq_len, d_model = x.shape

    if seq_len > self.pe.size(1):
        # Compute encoding for longer sequence
        pe = self._compute_pe(seq_len, d_model)
    else:
        pe = self.pe[:, :seq_len]

    return self.dropout(x + pe)

def _compute_pe(self, seq_len, d_model):
    """Compute positional encoding on-demand for long sequences."""
    # Same computation as __init__
    ...
```

### Learned Positional Embedding

**Full implementation:**

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (used in GPT, BERT, etc.).

    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Learnable positional embedding lookup table
        self.embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x with positional embedding added, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Create position indices
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)

        # Lookup embeddings
        pos_emb = self.embedding(positions)  # (seq_len, d_model)

        # Add to input (broadcasting across batch dimension)
        x = x + pos_emb.unsqueeze(0)  # (batch, seq, d_model) + (1, seq, d_model)

        return self.dropout(x)
```

**Alternative: Absolute position input**

Some implementations take absolute positions as input:

```python
def forward(self, x, positions=None):
    """
    Args:
        x: Input tensor (batch_size, seq_len, d_model)
        positions: Optional position indices (batch_size, seq_len)
                   If None, uses [0, 1, 2, ..., seq_len-1]
    """
    batch_size, seq_len, d_model = x.shape

    if positions is None:
        # Standard case: sequential positions
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

    # Lookup embeddings for each position
    pos_emb = self.embedding(positions)  # (batch_size, seq_len, d_model)

    return self.dropout(x + pos_emb)
```

This allows non-sequential positions (useful for some applications).

### Token Embedding with Scaling

**Full implementation:**

```python
class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional scaling by sqrt(d_model).

    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        padding_idx: Index of padding token (optional)
        scale_embeddings: Whether to scale by sqrt(d_model) (default: True)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = None,
        scale_embeddings: bool = True
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )

        self.d_model = d_model
        self.scale_embeddings = scale_embeddings
        self.scale_factor = math.sqrt(d_model) if scale_embeddings else 1.0

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Padding token should be zero
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)

    def forward(self, tokens):
        """
        Args:
            tokens: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Lookup embeddings
        emb = self.embedding(tokens)  # (batch_size, seq_len, d_model)

        # Scale if enabled
        if self.scale_embeddings:
            emb = emb * self.scale_factor

        return emb
```

**Padding considerations:**

When sequences have different lengths, we pad with a special token:

```python
# Example batch with different lengths
seq1 = [5, 12, 8, 3]       # Length 4
seq2 = [42, 7]             # Length 2

# Padded batch (padding_idx = 0)
batch = [
    [5, 12, 8, 3],         # Original
    [42, 7, 0, 0]          # Padded with 0s
]

# Embedding lookup
embeddings = token_embedding(batch)
# embeddings[:, 2:] for sequence 2 will be zeros (from padding)
```

The padding token embedding should be zero and not updated during training.

### Combined Embedding Layer

**Full implementation:**

```python
class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer for transformers.
    Combines token embeddings and positional encodings.

    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        positional_type: Type of positional encoding ("sinusoidal" or "learned")
        padding_idx: Padding token index (optional)
        scale_embeddings: Whether to scale embeddings by sqrt(d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        positional_type: str = "learned",
        padding_idx: int = None,
        scale_embeddings: bool = True
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
            scale_embeddings=scale_embeddings
        )

        # Positional encoding
        if positional_type == "sinusoidal":
            self.positional = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_seq_len,
                dropout=0.0  # Dropout applied once at the end
            )
        elif positional_type == "learned":
            self.positional = LearnedPositionalEmbedding(
                max_seq_len=max_seq_len,
                d_model=d_model,
                dropout=0.0  # Dropout applied once at the end
            )
        else:
            raise ValueError(
                f"Unknown positional_type: {positional_type}. "
                f"Choose 'sinusoidal' or 'learned'."
            )

        self.dropout = nn.Dropout(dropout)
        self.positional_type = positional_type

    def forward(self, tokens):
        """
        Args:
            tokens: Token IDs of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        # Token embeddings (with scaling if enabled)
        token_emb = self.token_embedding(tokens)
        # Shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        # Both implementations add positional info to token_emb
        combined = self.positional(token_emb)
        # Shape: (batch_size, seq_len, d_model)

        # Apply dropout once
        return self.dropout(combined)
```

**Usage example:**

```python
# Configuration
vocab_size = 50000
d_model = 512
max_seq_len = 1024
dropout = 0.1

# Create embedding layer (with learned positions)
embedding = TransformerEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    max_seq_len=max_seq_len,
    dropout=dropout,
    positional_type="learned",
    scale_embeddings=True
)

# Forward pass
tokens = torch.randint(0, vocab_size, (32, 128))  # (batch=32, seq_len=128)
embedded = embedding(tokens)
# Shape: (32, 128, 512)

# Can now pass to transformer blocks
# output = transformer_blocks(embedded)
```

---

## Advanced Topics

### Relative Positional Encodings

Absolute positional encodings (both sinusoidal and learned) encode "this is position 5".

**Limitation:**

The model must learn to compute relative distances:
- "Position 5 is 3 steps from position 2" (learned implicitly)

**Relative positional encodings** directly encode "this is 3 positions away":

**Approaches:**

**1. Relative Position Bias (T5)**

Add learned biases to attention scores based on relative distance:

```python
# In attention computation
scores = Q @ K.T / sqrt(d_k)

# Add relative position bias
relative_position = position_i - position_j
bias = learned_bias[relative_position]  # Lookup table
scores = scores + bias

# Then apply softmax
attention_weights = softmax(scores)
```

**2. Relative Position Embeddings (Transformer-XL)**

Modify the attention computation to use relative position embeddings:

```python
# Standard: Q @ K.T
# Relative: Q @ K.T + Q @ R^T
# where R is relative position embeddings
```

**3. Rotary Position Embedding (RoPE)**

Encodes positions as rotations in complex space. Used in many recent models (PaLM, LLaMA).

Key idea: Rotate queries and keys by an angle proportional to their position.

```python
# Simplified RoPE
def rotate(x, position):
    # Apply rotation matrix based on position
    # Relative position encoded in dot product
    ...
```

**Benefits:**
- Natural relative position encoding
- No maximum sequence length
- Efficient computation

### Absolute vs. Relative: Trade-offs

| Aspect | Absolute (Sinusoidal/Learned) | Relative (Bias/RoPE) |
|--------|-------------------------------|----------------------|
| **Position encoding** | "I am at position 5" | "I am 3 steps from you" |
| **Sequence length** | Depends on implementation | Often better extrapolation |
| **Parameters** | 0 (sin) or max_len×d (learned) | Varies by method |
| **Attention computation** | Standard | Modified |
| **Used in** | Original Transformer, BERT, GPT | T5, Transformer-XL, LLaMA |

### ALiBi: Attention with Linear Biases

ALiBi removes positional encoding entirely and adds biases to attention:

```python
# No positional encoding on inputs
embedded = token_embedding(tokens)  # No + pos_encoding

# Add linear bias to attention scores
attention_scores = Q @ K.T / sqrt(d_k)

# Add position-dependent bias
for head in heads:
    bias[i, j] = -m * abs(i - j)  # m is head-specific slope
    attention_scores[head] += bias

attention_weights = softmax(attention_scores)
```

**Properties:**
- No positional encoding parameters
- Excellent extrapolation to longer sequences
- Simpler architecture
- Used in BLOOM, MPT models

### 2D Positional Encodings (Vision Transformers)

For images, we need 2D positional encodings:

**Approach 1: Separate encodings per dimension**
```python
# Image: (H, W) patches
pos_h = sinusoidal_encoding(H, d_model // 2)  # Height positions
pos_w = sinusoidal_encoding(W, d_model // 2)  # Width positions

# Combine
pos_encoding[i, j] = concat(pos_h[i], pos_w[j])
```

**Approach 2: Learned 2D embeddings**
```python
pos_embedding = nn.Embedding(H * W, d_model)  # One embedding per patch
```

**Approach 3: Factorized learned embeddings**
```python
pos_h_emb = nn.Embedding(H, d_model)
pos_w_emb = nn.Embedding(W, d_model)

pos_encoding[i, j] = pos_h_emb[i] + pos_w_emb[j]
```

### Conditional Positional Encodings

Some applications encode multiple position types:

**Example: Hierarchical text (documents with paragraphs)**
```python
# Multiple position levels
paragraph_pos = positional_encoding(paragraph_id)
sentence_pos = positional_encoding(sentence_id)
word_pos = positional_encoding(word_id)

combined_pos = paragraph_pos + sentence_pos + word_pos
```

**Example: Time-series with multiple granularities**
```python
# Encode year, month, day, hour separately
year_pos = learned_embedding[year]
month_pos = learned_embedding[month]
day_pos = learned_embedding[day]

temporal_pos = year_pos + month_pos + day_pos
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Scale Embeddings

**Problem:**
```python
# Missing sqrt(d_model) scaling
embeddings = self.embedding(tokens)  # ✗
```

**Solution:**
```python
embeddings = self.embedding(tokens) * math.sqrt(self.d_model)  # ✓
```

**Why it matters:**

Without scaling, positional encodings (magnitude ~1) dominate small random embeddings (magnitude ~0.02).

**Caveat:** Modern models (GPT-2, GPT-3) often skip scaling and use LayerNorm instead.

### Pitfall 2: Applying Dropout Multiple Times

**Problem:**
```python
# Dropout applied twice
token_emb = self.dropout(self.token_embedding(tokens))
pos_enc = self.dropout(self.positional_encoding(tokens))
combined = token_emb + pos_enc  # ✗ Already dropped out
```

**Solution:**
```python
# Apply dropout once after combining
token_emb = self.token_embedding(tokens)
pos_enc = self.positional_encoding_no_dropout(tokens)
combined = self.dropout(token_emb + pos_enc)  # ✓
```

### Pitfall 3: Incorrect Positional Encoding Dimensionality

**Problem:**
```python
# Odd d_model with sinusoidal encoding
d_model = 513  # Odd number
pe[:, 0::2] = torch.sin(...)  # 257 dimensions
pe[:, 1::2] = torch.cos(...)  # 256 dimensions  # ✗ Mismatch!
```

**Solution:**
```python
# Ensure d_model is even
assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
```

Or use padding:
```python
if d_model % 2 != 0:
    # Compute for d_model - 1, then pad
    pe = sinusoidal_encoding(seq_len, d_model - 1)
    pe = F.pad(pe, (0, 1))  # Pad last dimension
```

### Pitfall 4: Sequence Length Exceeds max_seq_len

**Problem:**
```python
# Learned embeddings with max_seq_len = 512
pos_embedding = nn.Embedding(512, d_model)

# Inference with seq_len = 1024
positions = torch.arange(1024)  # ✗ IndexError: index out of range
```

**Solution:**

**Option 1: Truncate**
```python
seq_len = min(tokens.size(1), self.max_seq_len)
tokens = tokens[:, :seq_len]
```

**Option 2: Interpolate**
```python
if seq_len > max_seq_len:
    pos_emb = interpolate_embeddings(pos_emb, seq_len)
```

**Option 3: Use sinusoidal (extrapolates naturally)**
```python
# Sinusoidal works for any length
pos_enc = sinusoidal_encoding(seq_len, d_model)  # ✓
```

### Pitfall 5: Device Mismatch

**Problem:**
```python
# Model on GPU, but positional encoding on CPU
tokens = tokens.cuda()
embeddings = self.embedding(tokens)  # GPU

positions = torch.arange(seq_len)  # CPU by default
pos_emb = self.pos_embedding(positions)  # ✗ Device mismatch

combined = embeddings + pos_emb  # ✗ Error
```

**Solution:**
```python
# Ensure positions on same device as input
positions = torch.arange(seq_len, device=tokens.device)  # ✓
pos_emb = self.pos_embedding(positions)
```

Or use `register_buffer`:
```python
# In __init__
self.register_buffer('pe', pe)  # Automatically moves to correct device
```

### Pitfall 6: Not Freezing Padding Embeddings

**Problem:**
```python
# Padding token (ID=0) should always be zero
embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

# During training, embedding for token 0 changes
# Breaks padding mask logic
```

**Solution:**

PyTorch's `padding_idx` automatically handles this:
```python
embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
# Gradient for padding_idx automatically set to zero
```

Manual approach:
```python
# After each optimizer step
with torch.no_grad():
    self.embedding.weight[padding_idx] = 0
```

### Pitfall 7: Incorrect Broadcasting

**Problem:**
```python
# Positional encoding shape mismatch
token_emb = (batch, seq_len, d_model)
pos_enc = (batch, seq_len, d_model)  # Includes batch dimension

# Later code assumes pos_enc is shared across batch
pos_enc_broadcasted = pos_enc.unsqueeze(0)  # ✗ Now (1, batch, seq_len, d_model)
```

**Solution:**

Keep positional encoding without batch dimension:
```python
pos_enc = (seq_len, d_model)  # No batch dimension
# Broadcasting handles the rest automatically
```

---

## Practical Considerations

### Memory Efficiency

**Token embeddings:**
```
Memory = vocab_size × d_model × 4 bytes (float32)

Example (GPT-2 small):
50257 × 768 × 4 = 154,787,712 bytes ≈ 148 MB
```

**Positional embeddings (learned):**
```
Memory = max_seq_len × d_model × 4 bytes

Example (GPT-2 small):
1024 × 768 × 4 = 3,145,728 bytes ≈ 3 MB
```

**Optimization strategies:**

1. **Vocabulary pruning**: Remove rare tokens
2. **Shared embeddings**: Tie input and output embeddings
3. **Smaller d_model**: Use projection after embedding
4. **Quantization**: Use int8/int16 instead of float32

### Computational Efficiency

**Embedding lookup:**
- Time complexity: O(batch_size × seq_len)
- Very fast (just indexing)

**Sinusoidal encoding:**
- Time complexity: O(seq_len × d_model)
- Can pre-compute and cache

**Learned embedding:**
- Time complexity: O(seq_len)  (lookup)
- Slightly faster than sinusoidal

**Optimization:**

Cache positional encodings:
```python
# Pre-compute for common sequence lengths
self.position_cache = {}

def get_positional_encoding(self, seq_len):
    if seq_len not in self.position_cache:
        self.position_cache[seq_len] = self._compute_encoding(seq_len)
    return self.position_cache[seq_len][:seq_len]
```

### Numerical Stability

**Sinusoidal encoding:**

For very large positions (> 10000), be careful of numerical precision:

```python
# Potential issue: Large position × small div_term
position = 1_000_000
div_term = 1 / 10000^(510/512) ≈ 1/9886
angle = position * div_term ≈ 101.15

# Multiple rotations - precision loss
sin(101.15) vs sin(101.15 + 2π) - hard to distinguish
```

**Solution:**

Use modular arithmetic:
```python
angle = (position * div_term) % (2 * math.pi)
```

**Learned embeddings:**

Gradient clipping helps prevent exploding embeddings:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Hyperparameter Tuning

**Dropout rate:**
- Typical: 0.1
- Larger models: 0.0 - 0.2
- Overfitting: Increase dropout
- Underfitting: Decrease dropout

**Embedding initialization std:**
- Typical: 0.02
- Larger std → faster initial learning, more instability
- Smaller std → slower initial learning, more stable

**Scaling factor:**
- Standard: √d_model
- Modern: Often no scaling (scale=1.0)
- Experiment if using custom architectures

**Positional encoding type:**
- Default: Learned (used in most SOTA models)
- Need extrapolation: Sinusoidal or RoPE
- 2D inputs (images): 2D learned embeddings
- Long sequences: ALiBi or RoPE

### Debugging Tips

**Visualize embeddings:**
```python
import matplotlib.pyplot as plt

# Get embeddings for first 100 tokens
embeddings = model.embedding.weight[:100].detach().cpu()

# Project to 2D (e.g., PCA or t-SNE)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Token Embeddings (PCA)")
plt.show()
```

**Visualize positional encodings:**
```python
# Sinusoidal positional encoding
pe = sinusoidal_encoding(100, 128)  # 100 positions, 128 dimensions

# Heatmap
plt.figure(figsize=(12, 8))
plt.imshow(pe.T, aspect='auto', cmap='RdBu')
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.show()
```

**Check gradient flow:**
```python
# After backward pass
for name, param in model.named_parameters():
    if 'embedding' in name:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

**Verify shapes:**
```python
# Add assertions
def forward(self, tokens):
    assert tokens.dim() == 2, f"Expected 2D tokens, got {tokens.dim()}D"
    batch_size, seq_len = tokens.shape

    embeddings = self.embedding(tokens)
    assert embeddings.shape == (batch_size, seq_len, self.d_model)

    return embeddings
```

---

## Summary

### Key Takeaways

1. **Token Embeddings**
   - Map discrete tokens to continuous d_model-dimensional vectors
   - Implemented as learnable lookup table (vocab_size × d_model)
   - Often scaled by √d_model for balance with positional encodings
   - Initialization and vocabulary size significantly impact performance

2. **Positional Encoding**
   - Necessary because transformers are position-invariant
   - Injects sequence order information into the model
   - Two main approaches: sinusoidal (fixed) and learned

3. **Sinusoidal Positional Encoding**
   - Uses sine and cosine functions at different frequencies
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   - Extrapolates to unseen sequence lengths
   - No learnable parameters
   - Enables relative position computation via linear transformations

4. **Learned Positional Embeddings**
   - Trainable lookup table (max_seq_len × d_model)
   - More flexible, adapts to task
   - Limited to max_seq_len, cannot extrapolate
   - Used in most modern models (GPT, BERT, etc.)

5. **Combining Token and Position**
   - Use addition, not concatenation
   - Apply dropout after combination
   - Ensure scale matching between token and positional embeddings
   - Broadcasting handles batch dimension automatically

6. **Trade-offs**

| Aspect | Sinusoidal | Learned |
|--------|-----------|---------|
| Parameters | 0 | max_seq_len × d_model |
| Extrapolation | ✓ Yes | ✗ No |
| Flexibility | Fixed pattern | Task-adaptive |
| Performance | Good | Slightly better |
| Modern usage | Less common | Very common |

7. **Advanced Techniques**
   - Relative positional encodings (T5, Transformer-XL)
   - Rotary Position Embedding (RoPE) for better extrapolation
   - ALiBi for removing positional encoding entirely
   - 2D encodings for vision transformers

8. **Implementation Best Practices**
   - Pre-compute sinusoidal encodings in __init__
   - Use register_buffer for non-parameter tensors
   - Handle device placement correctly
   - Add shape assertions for debugging
   - Initialize embeddings with small std (0.02)
   - Ensure d_model is even for sinusoidal encoding
   - Set padding_idx for masked sequences

### Connection to Complete Transformer

Embeddings and positional encodings form the **input layer**:

```
Input Tokens (discrete)
        ↓
Token Embedding + Positional Encoding
        ↓
Combined Embeddings (continuous)
        ↓
Transformer Blocks (Module 01-03)
        ↓
Output Representations
        ↓
Task-Specific Head
        ↓
Predictions
```

Without proper embeddings and positional encodings:
- Model cannot process discrete tokens
- Model loses sequence order information
- Performance degrades significantly

### What's Next

In **Module 05: Full Model Assembly**, we'll combine all components:
- Embeddings (this module)
- Multi-head attention (Module 02)
- Feed-forward networks (Module 03)
- Layer normalization and residual connections (Module 03)

into a complete transformer language model capable of:
- Processing sequences of text
- Generating predictions
- Being trained end-to-end

### Further Reading

**Papers:**
- Vaswani et al. (2017): "Attention Is All You Need" - Original transformer with sinusoidal encoding
- Devlin et al. (2018): "BERT" - Learned positional embeddings
- Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al. (2021): "Train Short, Test Long: Attention with Linear Biases (ALiBi)"

**Resources:**
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Lilian Weng's blog: "Attention? Attention!"
- Hugging Face Transformers documentation

**Implementations:**
- PyTorch nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- Hugging Face: transformers/models/bert/modeling_bert.py (learned embeddings)
- Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html

---

### Quick Reference

**Sinusoidal encoding (one-liner):**
```python
pe[:, 0::2] = torch.sin(position * torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)))
pe[:, 1::2] = torch.cos(position * torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)))
```

**Learned embeddings (minimal):**
```python
pos_emb = nn.Embedding(max_seq_len, d_model)
pos = pos_emb(torch.arange(seq_len, device=x.device))
```

**Combined (production):**
```python
token_emb = self.token_embedding(tokens) * math.sqrt(self.d_model)
pos_emb = self.pos_embedding(token_emb)  # or self.pos_embedding(torch.arange(...))
output = self.dropout(token_emb + pos_emb)
```

You now have a complete understanding of embeddings and positional encodings. In the next module, we'll assemble the full transformer model!
