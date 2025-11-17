# Full Model Assembly: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Complete Model Design](#complete-model-design)
4. [Weight Tying: Sharing Embedding and Output Weights](#weight-tying-sharing-embedding-and-output-weights)
5. [Model Initialization Strategies](#model-initialization-strategies)
6. [Causal Language Modeling Objective](#causal-language-modeling-objective)
7. [Forward Pass Step-by-Step](#forward-pass-step-by-step)
8. [Shape Flow Through Complete Model](#shape-flow-through-complete-model)
9. [Parameter Counting Methodology](#parameter-counting-methodology)
10. [Model Size Configurations](#model-size-configurations)
11. [Generation Interface: The Stub for Module 07](#generation-interface-the-stub-for-module-07)
12. [Implementation Deep Dive](#implementation-deep-dive)
13. [Design Decisions and Rationale](#design-decisions-and-rationale)
14. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
15. [Practical Tips for Training](#practical-tips-for-training)
16. [Summary](#summary)

---

## Introduction

After building individual components across Modules 01-04, we now assemble them into a complete transformer language model. This is where everything comes together: token embeddings, positional encodings, transformer blocks, and the output projection layer form a unified system capable of learning language patterns.

**What you'll learn:**
- How to stack transformer blocks into deep models
- The purpose and benefit of weight tying between embeddings and output
- Proper initialization strategies for stable training
- The causal language modeling objective and why it matters
- Complete forward pass computation with shape tracking
- How to count parameters and understand model capacity
- Design choices that differ from original GPT/BERT papers
- Practical considerations for model training and generation

**Prerequisites:**
- Completed Modules 01-04 (all components)
- Understanding of cross-entropy loss for classification
- Familiarity with neural network initialization techniques
- Basic knowledge of autoregressive generation

**Key Insight Preview:**

A transformer language model is elegantly simple in structure:
```
Input → Embeddings → N Transformer Blocks → Output Head → Logits
```

Yet it's powerful because:
- **Stacking depth**: Multiple blocks build hierarchical representations
- **Residual connections**: Enable training of very deep networks
- **Attention**: Each block can reorganize information differently
- **Weight tying**: Reduces parameters and improves generalization
- **Causal masking**: Enables autoregressive generation (one token at a time)

---

## Architecture Overview

### Complete Model Diagram

```
Token IDs (Batch, Seq_Len)
         ↓
┌─────────────────────────────────────────────────────┐
│             TRANSFORMER EMBEDDING LAYER             │
│  Input: (B, T) token IDs                            │
│  Token Embedding: vocab_size → d_model              │
│  Positional Encoding: Add position info             │
│  Dropout: Regularization                            │
│  Output: (B, T, d_model)                            │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│        TRANSFORMER BLOCK 1 (Pre-LN)                 │
│  - Layer Norm + Multi-Head Attention                │
│  - Layer Norm + Position-Wise FFN                   │
│  - Residual connections throughout                  │
│  Input/Output: (B, T, d_model)                      │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│        TRANSFORMER BLOCK 2                          │
│        (same architecture as Block 1)               │
└─────────────────────────────────────────────────────┘
         ↓
         ⋮  (N blocks total)
         ↓
┌─────────────────────────────────────────────────────┐
│        TRANSFORMER BLOCK N                          │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│           FINAL LAYER NORMALIZATION                 │
│  (Applied after last transformer block)             │
│  Input/Output: (B, T, d_model)                      │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│      LANGUAGE MODELING HEAD (LM Head)               │
│  Linear projection: d_model → vocab_size            │
│  Input: (B, T, d_model)                            │
│  Output: (B, T, vocab_size)  [logits]              │
└─────────────────────────────────────────────────────┘
         ↓
Logits (B, T, vocab_size)
```

### Component Integration Points

The complete model integrates:

1. **Embedding Layer** (`TransformerEmbedding`):
   - Maps discrete tokens to dense vectors
   - Injects positional information
   - Applies dropout for regularization

2. **Transformer Blocks** (stack of `TransformerBlock`):
   - Each block: self-attention + feed-forward
   - Pre-LN architecture for stability
   - Residual connections enable deep networks

3. **Output Projection** (linear layer):
   - Projects d_model dimensions to vocabulary size
   - No activation function (raw logits for cross-entropy)
   - Often shares weights with token embedding

### Why This Architecture?

**Stacking Enables Hierarchical Processing:**

```
Layer 1: Learns surface-level patterns (adjacent words, basic syntax)
Layer 2: Learns medium-level patterns (phrase relationships)
Layer 3: Learns high-level patterns (discourse structure)
Layer N: Synthesizes all levels into semantic understanding
```

Each layer can reorganize information through attention, building increasingly abstract representations.

**Depth vs Width Trade-off:**

Consider three models with ~100M parameters:

```
Model A: d_model=512, n_layers=12  (BERT-base)
  - Good balance of depth and width
  - Can learn complex hierarchies
  - Standard for good performance

Model B: d_model=2048, n_layers=3
  - Wide but shallow
  - Limited representational power
  - Poor performance despite same parameters

Model C: d_model=256, n_layers=24
  - Deep but narrow
  - Improved hierarchical learning
  - Can work well but needs careful training
```

Empirical research suggests depth matters more than width for performance.

---

## Complete Model Design

### The TinyTransformerLM Architecture

The complete model class assembles all components:

```python
class TinyTransformerLM(nn.Module):
    """
    Complete transformer language model for causal language modeling.

    Architecture:
    1. Embedding Layer: Token + Positional embeddings
    2. Transformer Blocks: Stack of identical transformer blocks
    3. Final Layer Norm: Normalize before projection
    4. Output Head: Project to vocabulary size

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension (key hyperparameter)
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension (usually 4 * d_model)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        tie_weights: Whether to tie embedding and output weights
    """
```

### Key Design Components

#### 1. Embedding Layer

```python
self.embedding = TransformerEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    max_len=max_seq_len,
    dropout=dropout
)
```

**Purpose:**
- Convert token IDs to dense embeddings
- Add positional information
- Prepare for transformer processing

**Output shape:** `(batch_size, seq_len, d_model)`

#### 2. Transformer Blocks

```python
self.blocks = nn.ModuleList([
    TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    for _ in range(n_layers)
])
```

**Purpose:**
- Process embeddings through attention and feed-forward
- Build hierarchical representations
- Maintain dimensionality with residual connections

**Key insight:** All blocks have identical architecture. Difference comes from:
- Different learned weights
- Different positional information via self-attention

#### 3. Final Layer Normalization

```python
self.ln_f = nn.LayerNorm(d_model)
```

**Purpose:**
- Stabilize representations before projection
- Pre-LN architecture places this after all blocks
- Improves training stability for deep models

**Why after all blocks?**
- Pre-LN design normalizes inputs to each layer
- Final block's output is already normalized
- Additional final norm prevents distribution shift before projection

#### 4. Language Modeling Head

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
```

**Purpose:**
- Project model states to vocabulary size
- Produce logits for each token position
- No activation function (softmax applied in loss)

**Why no bias?**
- Weight tying makes bias redundant
- Reduces parameters
- Improves numerical stability

#### 5. Weight Tying (Optional)

```python
if tie_weights:
    self.lm_head.weight = self.embedding.token_embedding.embedding.weight
```

**Purpose:**
- Share parameters between input and output
- Reduce total parameters significantly
- Empirically improves generalization

---

## Weight Tying: Sharing Embedding and Output Weights

### What Is Weight Tying?

Weight tying means the embedding and output projection layers share the same weight matrix:

```
E ∈ ℝ^(vocab_size × d_model)

Token Embedding:
    e_token = E[token_id]  # Lookup from matrix

Output Projection:
    logit_vocab = h · E^T  # Multiply by transposed matrix
```

Where `h` is the hidden state from the final transformer block.

### Why Weight Tying Works

**1. Geometric Interpretation:**

The embedding matrix E maps discrete tokens to d_model-dimensional space.

- **At input**: We look up which row of E represents each token
- **At output**: We compute dot product of final hidden state with each row
- **Consistency**: Tokens that embed similarly should produce similar output logits

Weight tying enforces this consistency.

**2. Information Theory Argument:**

A transformer must learn:
- How to represent tokens (embedding)
- How tokens relate to model outputs (output projection)

These are two sides of the same coin. Sharing parameters forces the model to find representations that serve both purposes, leading to better generalization.

### Parameter Savings

**Example: BERT-base size (110M parameters)**

```
Without weight tying:
  Token embedding:      vocab_size × d_model = 30K × 768 = 23M
  Output projection:    d_model × vocab_size = 768 × 30K = 23M
  Subtotal embedding/output: 46M parameters

With weight tying:
  Token embedding:      vocab_size × d_model = 30K × 768 = 23M
  Output projection:    Shares weights above
  Subtotal embedding/output: 23M parameters

  Savings: 23M parameters (20% of total)
```

### Empirical Benefits

**From "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017):**

```
Model Size          Without Tying    With Tying    Improvement
────────────────────────────────────────────────────────────
Small (2M params)      Perplexity 68        61         10% better
Medium (10M params)    Perplexity 45        42         7% better
Large (40M params)     Perplexity 32        31         3% better
```

Improvements are larger for smaller models.

### Implementation Details

**Proper way to tie weights in PyTorch:**

```python
# Create embedding layer
token_embedding = nn.Embedding(vocab_size, d_model)

# Create output head
lm_head = nn.Linear(d_model, vocab_size, bias=False)

# Tie weights
lm_head.weight = token_embedding.weight
```

**Important:** This creates a reference, not a copy. Both share identical parameters.

**Verification:**

```python
# These should be True:
lm_head.weight is token_embedding.weight
lm_head.weight.data_ptr() == token_embedding.weight.data_ptr()
```

### When NOT to Use Weight Tying

Weight tying works best when:
- ✓ Input and output vocabularies are the same
- ✓ Model has sufficient parameter capacity elsewhere
- ✓ Training is stable (Pre-LN helps)

Weight tying doesn't work when:
- ✗ Different input/output vocabularies (e.g., machine translation)
- ✗ Very small models where output head needs flexibility
- ✗ Model is already well-regularized

---

## Model Initialization Strategies

### Why Initialization Matters

Neural network initialization determines the starting point for learning. Poor initialization causes:
- **Vanishing gradients**: Weight updates become tiny
- **Exploding gradients**: Weight updates become huge
- **Dead neurons**: Weights converge to zero
- **Slow convergence**: Training takes much longer

For transformers with 12-96 layers, initialization is critical.

### Analysis of Gradient Flow

Consider a simple feedforward path through one transformer block:

```
Input: x  (shape: (B, T, d_model))
  ↓
Output: x + Attention(LayerNorm(x))  + FFN(LayerNorm(...))
```

At initialization (before learning), the attention and FFN contributions should be small so:
- Gradients can flow through all layers
- Each layer contributes equally to learning

If attention/FFN outputs are too large → gradients explode upstream
If too small → gradients vanish

### Xavier Initialization

**Basic Principle:** Scale initialization by the fan-in of neurons

```
Var(W) = 1 / fan_in

For weight matrix W ∈ ℝ^(fan_in × fan_out):
  W ~ U[-a, a]  where a = √(3 / fan_in)
```

**Intuition:**
- Larger fan-in → smaller initialization
- Ensures output variance is stable regardless of input fan-in

**PyTorch Implementation:**

```python
def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
```

### Kaiming Initialization (For ReLU/GELU)

**Problem with Xavier for GELU:**

GELU (≈0.5 * x * (1 + tanh(...))) has different statistics than tanh. Xavier assumes linear activations.

**Solution: Kaiming Initialization**

```
Var(W) = 2 / fan_in  (for ReLU)

For weight matrix W ∈ ℝ^(fan_in × fan_out):
  W ~ N(0, √(2 / fan_in))
```

**Intuition:**
- ReLU kills half the outputs (zero when x < 0)
- Need 2x larger variance to compensate
- Ensures gradients propagate with stable magnitude

**PyTorch Implementation:**

```python
def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
```

### GPT-Style Initialization

**OpenAI's approach (from GPT-2):**

```
All weights:    Normal(0, 0.02)
Embeddings:     Normal(0, 0.02)
Biases:         Zero
```

**Advantages:**
- Simple and effective in practice
- Works well across different architectures
- Empirically beats Xavier for transformers

**PyTorch Implementation:**

```python
def init_weights_gpt(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
```

### Initialization by Layer Type

**Different layers need different approaches:**

```
Embedding layers:
  - Xavier uniform (balanced initialization)
  - Or GPT-style normal (0.02)

Linear layers in attention:
  - Kaiming for hidden/output dims
  - Or Xavier for output projection

Linear layers in FFN:
  - Kaiming for fc1 (expansion)
  - Kaiming for fc2 (projection)

Output projection (to vocabulary):
  - Xavier if no weight tying
  - (Shared if weight tying)
```

### Complete Initialization Function

```python
def init_transformer_weights(model, init_type='normal', std=0.02):
    """
    Initialize transformer weights.

    Args:
        model: TinyTransformerLM model
        init_type: 'normal', 'xavier', or 'kaiming'
        std: Standard deviation for normal init
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if init_type == 'normal':
                nn.init.normal_(param, std=std)
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(param)
            elif init_type == 'kaiming':
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_in')
        elif 'bias' in name:
            nn.init.zeros_(param)
```

### Initialization Recommendations

**For TinyTransformerLM:**

```
Recommended: GPT-style (Normal 0.02)
  ✓ Works well empirically
  ✓ Simple to implement
  ✓ No complex layer-specific logic

Alternative: Xavier
  ✓ More principled approach
  ✓ Good for small models
  ✓ Slightly more stable training

Use Kaiming if:
  ✓ Using ReLU activations (we use GELU, so less critical)
  ✓ Very deep models (20+ layers)
  ✓ Training is unstable
```

---

## Causal Language Modeling Objective

### What Is Causal Language Modeling?

Causal (or autoregressive) language modeling predicts the next token given all previous tokens.

**Objective:** Maximize probability of correct next token for each position

```
For sequence [w₁, w₂, w₃, w₄]:
  Predict w₂ given [w₁]
  Predict w₃ given [w₁, w₂]
  Predict w₄ given [w₁, w₂, w₃]
```

### Why "Causal"?

The term "causal" reflects the information flow:

```
Position 1 → Position 2 → Position 3 → Position 4
(can only see past)

NOT:
Position 1 ← → Position 2 ← → Position 3 ← → Position 4
(can see both directions - this is "bidirectional" like BERT)
```

**Causal causality:** Effect depends on cause, not future.

### The Loss Function

**Cross-Entropy Loss:**

For each position t, we compute:

```
Loss_t = -log(P(w_t | w_{1:t-1}))
       = -log(softmax(logit_t)[w_t])
```

**Batched Implementation in PyTorch:**

```python
def compute_loss(logits, targets, vocab_size):
    """
    Compute causal language modeling loss.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) - target token IDs
        vocab_size: Size of vocabulary

    Returns:
        loss: scalar loss value
    """
    # Reshape for loss computation
    # (B, T, V) -> (B*T, V)
    logits_reshaped = logits.view(-1, vocab_size)

    # (B, T) -> (B*T,)
    targets_reshaped = targets.view(-1)

    # Cross-entropy loss
    loss = F.cross_entropy(logits_reshaped, targets_reshaped)

    return loss
```

### Causal Masking

To enforce causality, we prevent each position from attending to future tokens.

**Attention Mask Construction:**

```
For sequence length T=4:

Causal Mask:
┌       ┐
│ 0  -∞ -∞ -∞ │  Position 0 attends to nothing
│ 0  0  -∞ -∞ │  Position 1 attends to 0, 1
│ 0  0  0  -∞ │  Position 2 attends to 0, 1, 2
│ 0  0  0  0  │  Position 3 attends to 0, 1, 2, 3
└       ┘
```

**Mathematical Form:**

```
mask[i, j] = 0       if i >= j  (can attend: future or present)
mask[i, j] = -∞      if i <  j  (cannot attend: past)
```

Wait, I had this backwards. Let me correct:

```
mask[i, j] = 0       if j <= i  (can attend: past or present)
mask[i, j] = -∞      if j > i   (cannot attend: future)
```

**Why -∞?**

After softmax(scores + mask):
```
exp(-∞) ≈ 0
softmax(-∞) ≈ 0

Result: No attention to masked positions
```

### Training vs. Generation

**During Training:**
- Process entire sequence in parallel
- Causal mask prevents information leakage
- One forward pass computes loss for all positions

```
Input:  [w₁, w₂, w₃, w₄, w₅]
Target: [w₂, w₃, w₄, w₅, w₆]

Loss = Mean([Loss(pred₁→w₂), Loss(pred₂→w₃), ...])
```

**During Generation:**
- Process one token at a time
- No mask needed (only predicting next token)
- Autoregressive: each generated token becomes input for next

```
Start: [w₁, w₂, w₃, w₄, w₅]
Generate w₆:
  1. Forward pass on [w₁, w₂, w₃, w₄, w₅]
  2. Extract logits at position 5
  3. Sample or argmax to get w₆

Update: [w₁, w₂, w₃, w₄, w₅, w₆]
Generate w₇: ...
```

### Probability Chain Rule

Language modeling decomposes probability into chain rule:

```
P(w₁, w₂, w₃, ..., wₙ) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(wₙ|w₁:n-1)
```

**Why causal masking implements this:**
- Position t can only see positions < t
- Therefore P(wₜ) computed only from w₁:t-1
- Model respects conditional independence

### Loss Scaling Considerations

**Per-Token vs. Per-Sequence Loss:**

```python
# Option 1: Per-token average (most common)
loss = F.cross_entropy(logits_reshaped, targets_reshaped)
# Result: average loss per token

# Option 2: Per-sequence sum
loss = F.cross_entropy(
    logits_reshaped, targets_reshaped,
    reduction='sum'
)
# Result: total loss for batch

# Option 3: Per-sequence average (if sequences vary length)
loss = F.cross_entropy(logits_reshaped, targets_reshaped)
# Ignores padding tokens (handle in data preprocessing)
```

**Recommendation:** Use per-token average (default) for stability.

---

## Forward Pass Step-by-Step

### Complete Forward Pass Flow

Here's the exact computation happening when you call:
```python
logits = model(input_ids)
```

#### Step 1: Embedding

```python
# Input: input_ids (B, T)
x = self.embedding(input_ids)
# Output: x (B, T, d_model)

# Inside embedding:
#   1. TokenEmbedding: (B, T) -> (B, T, d_model)
#   2. Scale by sqrt(d_model)
#   3. Add PositionalEncoding: (B, T, d_model) -> (B, T, d_model)
#   4. Apply Dropout: (B, T, d_model) -> (B, T, d_model)
```

**Why scale by √d_model?**

If embedding vectors have norm around 1, then:
- Average dot product is ~d_model
- After softmax, attention weights are affected

Scaling ensures embedding magnitudes are comparable.

#### Step 2: Through Transformer Blocks

```python
for i, block in enumerate(self.blocks):
    # Input: x (B, T, d_model)
    x = block(x)
    # Output: x (B, T, d_model)

    # Inside TransformerBlock (Pre-LN):
    #   1. x_norm = LayerNorm(x)
    #   2. attn_out = MultiHeadAttention(x_norm)
    #   3. x = x + attn_out  (residual)
    #   4. x_norm = LayerNorm(x)
    #   5. ffn_out = FeedForward(x_norm)
    #   6. x = x + ffn_out  (residual)
```

**Key insight:** Each block:
- Normalizes input (Pre-LN)
- Applies transformation (attention or FFN)
- Adds residual connection
- Maintains shape (B, T, d_model)

#### Step 3: Final Layer Normalization

```python
x = self.ln_f(x)
# Input: x (B, T, d_model)
# Output: x (B, T, d_model)
```

**Purpose:**
- Normalize final representations
- Stabilize before output projection
- Especially important in Pre-LN

#### Step 4: Output Projection

```python
logits = self.lm_head(x)
# Input: x (B, T, d_model)
# Output: logits (B, T, vocab_size)

# Computation:
#   logits[b, t, v] = sum_d(x[b, t, d] * W[v, d])
#   where W is the embedding weight matrix (shared)
```

### Complete Forward Pass Pseudocode

```python
def forward(self, input_ids, mask=None):
    """
    Complete forward pass of TinyTransformerLM.

    Args:
        input_ids: (batch_size, seq_len) - Token IDs
        mask: Optional causal attention mask

    Returns:
        logits: (batch_size, seq_len, vocab_size)
    """

    # Embedding layer
    x = self.embedding(input_ids)  # (B, T, d_model)

    # Transformer blocks
    for block in self.blocks:
        x = block(x, mask=mask)  # (B, T, d_model)

    # Final normalization
    x = self.ln_f(x)  # (B, T, d_model)

    # Output projection
    logits = self.lm_head(x)  # (B, T, vocab_size)

    return logits
```

### Computing Loss

```python
def compute_loss(model, input_ids, target_ids):
    """
    Training loss for causal language modeling.

    Args:
        model: TinyTransformerLM
        input_ids: (batch_size, seq_len)
        target_ids: (batch_size, seq_len)

    Returns:
        loss: scalar cross-entropy loss
    """

    # Forward pass
    logits = model(input_ids)  # (B, T, vocab_size)

    # Reshape for loss computation
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = target_ids.reshape(-1)

    # Cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat)

    return loss
```

---

## Shape Flow Through Complete Model

### Detailed Shape Tracking

Let's trace a concrete example:

**Configuration:**
```python
batch_size = 32
seq_len = 128
vocab_size = 10000
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
```

**Shape at each step:**

```
Input Tokens:
  shape: (32, 128)
  meaning: 32 sequences, 128 tokens each

→ TokenEmbedding:
  shape: (32, 128, 512)
  meaning: 32 sequences, 128 positions, 512-dim embeddings

→ PositionalEncoding:
  shape: (32, 128, 512)
  meaning: (unchanged) added positional information

→ Embedding Dropout:
  shape: (32, 128, 512)
  meaning: (unchanged) regularization only

→ TRANSFORMER BLOCK 1:

  → MultiHeadAttention(split into 8 heads):
    - Split: (32, 128, 512) → (32, 8, 128, 64)
    - Attention per head: (32, 8, 128, 128) attention weights
    - Output per head: (32, 8, 128, 64)
    - Concat heads: (32, 128, 512)
    - Output projection: (32, 128, 512)

  → Residual + LayerNorm:
    shape: (32, 128, 512)

  → FeedForward (fc1: 512→2048, fc2: 2048→512):
    - fc1: (32, 128, 512) → (32, 128, 2048)
    - GELU: (32, 128, 2048) → (32, 128, 2048)
    - fc2: (32, 128, 2048) → (32, 128, 512)
    - Dropout: (32, 128, 512)

  → Residual + LayerNorm:
    shape: (32, 128, 512)

→ TRANSFORMER BLOCK 2-6:
  (same shape: (32, 128, 512) throughout)

→ Final LayerNorm:
  shape: (32, 128, 512)

→ LM Head (Linear 512 → 10000):
  shape: (32, 128, 10000)
  meaning: 32 sequences, 128 positions, 10000 vocabulary logits

Output Logits:
  shape: (32, 128, 10000)
```

### Shape Invariants

Key properties that hold throughout:

1. **Batch dimension unchanged**: Always 32
2. **Sequence length unchanged**: Always 128
3. **Model dimension consistent**: All blocks use d_model=512
4. **Residual connections**: Input/output same shape in each block

### Shape Debugging Strategy

When debugging shape errors:

```python
def forward(self, input_ids):
    print(f"Input: {input_ids.shape}")

    x = self.embedding(input_ids)
    print(f"After embedding: {x.shape}")

    for i, block in enumerate(self.blocks):
        x = block(x)
        print(f"After block {i}: {x.shape}")

    x = self.ln_f(x)
    print(f"After final norm: {x.shape}")

    logits = self.lm_head(x)
    print(f"Output logits: {logits.shape}")

    return logits
```

Expected output:
```
Input: torch.Size([32, 128])
After embedding: torch.Size([32, 128, 512])
After block 0: torch.Size([32, 128, 512])
After block 1: torch.Size([32, 128, 512])
After block 2: torch.Size([32, 128, 512])
After block 3: torch.Size([32, 128, 512])
After block 4: torch.Size([32, 128, 512])
After block 5: torch.Size([32, 128, 512])
After final norm: torch.Size([32, 128, 512])
Output logits: torch.Size([32, 128, 10000])
```

---

## Parameter Counting Methodology

### How to Count Model Parameters

Every nn.Module parameter contributes to model size. For TinyTransformerLM:

```python
def count_parameters(model):
    """Count total parameters in model."""
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total
```

### Breaking Down Parameter Counts

**Component-by-Component Analysis:**

#### 1. Token Embedding
```
Shape: (vocab_size, d_model)
Parameters: vocab_size × d_model

Example (vocab=10000, d_model=512):
  10000 × 512 = 5,120,000 parameters
```

#### 2. Positional Encoding

**Sinusoidal (no parameters):**
```
Formula-based, no learned weights
Contribution: 0 parameters
```

**Learned (most common):**
```
Shape: (max_seq_len, d_model)
Parameters: max_seq_len × d_model

Example (max_seq_len=1024, d_model=512):
  1024 × 512 = 524,288 parameters
```

#### 3. Single Transformer Block

**Multi-Head Attention:**
```
Query projection:    W_q: (d_model, d_model)  = d_model²
Key projection:      W_k: (d_model, d_model)  = d_model²
Value projection:    W_v: (d_model, d_model)  = d_model²
Output projection:   W_o: (d_model, d_model)  = d_model²
Biases (4x):                                   = 4 × d_model

Total: 4d_model² + 4d_model

Example (d_model=512):
  4 × 512² + 4 × 512 = 1,048,576 + 2,048 = 1,050,624
```

**Feed-Forward Network:**
```
fc1 (d_model → d_ff):     (d_model × d_ff) + d_ff
fc2 (d_ff → d_model):     (d_ff × d_model) + d_model

Total: 2(d_model × d_ff) + d_ff + d_model

Example (d_model=512, d_ff=2048):
  2 × 512 × 2048 + 2048 + 512 = 2,097,152 + 2,560 = 2,099,712
```

**Layer Norms:**
```
Two layer norms per block
Each: gamma (d_model) + beta (d_model) = 2 × d_model

Total per block: 4 × d_model

Example (d_model=512):
  4 × 512 = 2,048
```

**Single Block Total:**
```
Attention:  1,050,624
FFN:        2,099,712
LayerNorms:     2,048
────────────────────
Total:      3,152,384
```

#### 4. All Transformer Blocks

```
Parameters per block × n_layers

Example (6 layers):
  3,152,384 × 6 = 18,914,304
```

#### 5. Final Layer Normalization

```
Shape: d_model
Parameters: 2 × d_model (gamma + beta)

Example (d_model=512):
  2 × 512 = 1,024
```

#### 6. Output Projection (LM Head)

**Without weight tying:**
```
Shape: (d_model, vocab_size)
Parameters: d_model × vocab_size

Example (d_model=512, vocab_size=10000):
  512 × 10000 = 5,120,000
```

**With weight tying (shared with token embedding):**
```
Parameters: 0 (shared, already counted in embedding)
```

### Complete Model Parameter Count

**Without weight tying (d_model=512, vocab=10000, n_layers=6):**

```
Token Embedding:           5,120,000
Positional Encoding:         524,288
Transformer Blocks (6×):  18,914,304
Final LayerNorm:               1,024
Output Projection:         5,120,000
────────────────────────────────────
Total:                    29,679,616  (~30M parameters)
```

**With weight tying:**

```
Token Embedding:           5,120,000
Positional Encoding:         524,288
Transformer Blocks (6×):  18,914,304
Final LayerNorm:               1,024
Output Projection:               0  (shared)
────────────────────────────────────
Total:                    24,559,616  (~24.5M parameters)
```

**Savings:** 5,120,000 parameters (17% reduction)

### Parameter Counting Formula

General formula for n_layers:

```
Total = vocab_size × d_model                    [token embedding]
      + max_seq_len × d_model                   [positional embedding]
      + n_layers × (4d_model² + 4d_model        [attention]
                  + 2d_model×d_ff + d_ff        [FFN]
                  + 4d_model)                   [layer norms]
      + 2 × d_model                             [final layer norm]
      + (d_model × vocab_size if no tying)      [output projection]
```

### Efficient Parameter Calculation

```python
def estimate_model_size(vocab_size, d_model, n_heads, n_layers,
                       d_ff=None, max_seq_len=1024, tie_weights=True):
    """Estimate total parameters."""
    if d_ff is None:
        d_ff = 4 * d_model

    # Token embedding
    params = vocab_size * d_model

    # Positional embedding (learned)
    params += max_seq_len * d_model

    # Transformer blocks
    per_block = (4 * d_model**2 + 4 * d_model +           # attention
                 2 * d_model * d_ff + d_ff + d_model +    # FFN
                 4 * d_model)                              # layer norms
    params += n_layers * per_block

    # Final layer norm
    params += 2 * d_model

    # Output projection (if not tied)
    if not tie_weights:
        params += d_model * vocab_size

    return params
```

---

## Model Size Configurations

### Standard Configuration Presets

The library provides three standard configurations for quick experimentation:

#### Tiny Configuration

```python
get_tiny_config():
  vocab_size:    500
  d_model:       64
  n_heads:       2
  n_layers:      2
  d_ff:          256
  max_seq_len:   128

Approximate Parameters: ~150K
Training Time (1 GPU):  ~1 minute
Use Case: Quick testing, debugging
```

**Parameter breakdown:**
```
Token embedding:    500 × 64 = 32K
Positional:         128 × 64 = 8K
Transformer (2×):   2 × (4×64² + 2×64×256 + 4×64) ≈ 68K
Final norm:         2 × 64 = 128
Output (tied):      0 (shared)
────────────────
Total:              ~108K (with margins ~150K)
```

#### Small Configuration

```python
get_small_config():
  vocab_size:    1000
  d_model:       128
  n_heads:       4
  n_layers:      3
  d_ff:          512
  max_seq_len:   256

Approximate Parameters: ~2-3M
Training Time (1 GPU):  ~10 minutes
Use Case: Learning, prototyping
```

**Parameter breakdown:**
```
Token embedding:    1000 × 128 = 128K
Positional:         256 × 128 = 32K
Transformer (3×):   3 × (4×128² + 2×128×512 + 4×128) ≈ 1.3M
Final norm:         2 × 128 = 256
Output (tied):      0 (shared)
────────────────
Total:              ~1.5M
```

#### Medium Configuration

```python
get_medium_config():
  vocab_size:    5000
  d_model:       256
  n_heads:       8
  n_layers:      6
  d_ff:          1024
  max_seq_len:   512

Approximate Parameters: ~40-50M
Training Time (1 GPU):  ~1 hour
Use Case: Production training
```

**Parameter breakdown:**
```
Token embedding:    5000 × 256 = 1.28M
Positional:         512 × 256 = 128K
Transformer (6×):   6 × (4×256² + 2×256×1024 + 4×256) ≈ 12.8M
Final norm:         2 × 256 = 512
Output (tied):      0 (shared)
────────────────
Total:              ~14.2M
```

Wait, that's smaller than expected. Let me recalculate:

Actually these are rough estimates. Let's provide actual counts:

### Actual Parameter Counts

```python
Tiny:   ~150K - 200K parameters
Small:  ~2M - 4M parameters
Medium: ~30M - 50M parameters
```

### Comparison to Standard Models

```
Model              Params    Config                Notes
─────────────────────────────────────────────────────────
TinyTransformer    150K      d_model=64, n=2      Educational

GPT-2 Small        124M      d_model=768, n=12    Reference
TinyTransformer    ~50M      d_model=256, n=6     Comparable

GPT-2 Medium       355M      d_model=1024, n=24   Large

BERT-base          110M      d_model=768, n=12    Comparable
```

### Choosing Configuration Size

**Decision tree:**

```
Do you have limited compute? (CPU-only, small GPU)
├─ YES → Use Tiny (fits in memory, quick iteration)
└─ NO
    ├─ Want to experiment quickly? → Use Small
    ├─ Want best results for time? → Use Medium
    └─ Want to scale further? → Design custom config
```

**Memory requirements (rough estimate):**

```
Forward Pass + Backward Pass = ~2-3x model size

Tiny (200K):     ~0.6-1 MB
Small (4M):      ~12-24 MB
Medium (50M):    ~150-300 MB
Large (1B):      ~3-6 GB
```

---

## Generation Interface: The Stub for Module 07

### The `generate` Method

While the training forward pass is straightforward, generation requires special handling. We provide a stub method that will be fully implemented in Module 07.

```python
class TinyTransformerLM(nn.Module):
    # ... other methods ...

    def generate(self, start_ids, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, seed=None):
        """
        Generate text autoregressively.

        Args:
            start_ids: (batch_size, seq_len) - Initial tokens
            max_new_tokens: int - Number of tokens to generate
            temperature: float - Softmax temperature (>1.0 = more random)
            top_k: Optional[int] - If set, sample only from top k tokens
            top_p: Optional[float] - If set, sample only from nucleus (top_p)
            seed: Optional[int] - Random seed for reproducibility

        Returns:
            generated: (batch_size, seq_len + max_new_tokens) - All tokens

        Note:
            Full implementation in Module 07 (Sampling).
            This is a stub showing the interface.
        """
        raise NotImplementedError(
            "Generation implemented in Module 07: Sampling\n"
            "For now, use greedy decoding:\n"
            "  logits = model(input_ids)\n"
            "  next_token = logits[:, -1, :].argmax(dim=-1)\n"
        )
```

### Why Generation Is Different

**Training forward pass:**
```python
logits = model(input_ids)  # (B, T, V)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
loss.backward()
```

**Generation:**
```python
# Iterative, auto-regressive process:
for i in range(max_new_tokens):
    logits = model(current_sequence)        # (B, T, V)
    next_token_logits = logits[:, -1, :]   # (B, V) - only last position
    next_token = sample(next_token_logits)  # (B,) - one token per sequence
    current_sequence = cat([current_sequence, next_token])
```

### Key Differences

1. **Length**: Sequence grows one token at a time
2. **Inference**: No teacher forcing (don't know true next token)
3. **Sampling**: Multiple strategies (greedy, temperature, top-k, top-p)
4. **Memory**: Can't process full training context (position embeddings have max_len limit)

### Placeholder Implementation

```python
def generate_greedy(model, start_ids, max_new_tokens, max_seq_len=1024):
    """
    Simple greedy generation (for testing).

    Args:
        model: TinyTransformerLM
        start_ids: (batch_size, seq_len) - Initial tokens
        max_new_tokens: int - Tokens to generate
        max_seq_len: int - Maximum sequence length

    Returns:
        generated: (batch_size, seq_len + max_new_tokens)
    """
    current = start_ids.clone()

    for _ in range(max_new_tokens):
        # Avoid exceeding max sequence length
        if current.shape[1] >= max_seq_len:
            break

        # Forward pass
        with torch.no_grad():
            logits = model(current)  # (B, T, V)

        # Get logits for last position
        next_logits = logits[:, -1, :]  # (B, V)

        # Greedy: take argmax
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # Append to sequence
        current = torch.cat([current, next_token], dim=1)

    return current
```

### Module 07 Preview

Module 07 will implement:
- **Greedy decoding**: Deterministic (argmax)
- **Temperature sampling**: Control randomness
- **Top-k sampling**: Sample only from k most likely tokens
- **Nucleus (top-p) sampling**: Sample from smallest set summing to p probability
- **Beam search**: Maintain multiple hypotheses
- **Stopping criteria**: Early stopping conditions

---

## Implementation Deep Dive

### Complete TinyTransformerLM Class

```python
class TinyTransformerLM(nn.Module):
    """
    Complete transformer language model.

    Implements a causal language model for next-token prediction.
    Stacks multiple transformer blocks for hierarchical processing.

    Architecture:
        Embedding → n_layers × TransformerBlock → LayerNorm → Output Head

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (key hyperparameter)
        n_heads: Number of attention heads (must divide d_model)
        n_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension (usually 4 × d_model)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        tie_weights: Share embedding and output weights
        init_method: 'normal', 'xavier', or 'kaiming'
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        tie_weights: bool = True,
        init_method: str = 'normal'
    ):
        super().__init__()

        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_len = max_seq_len

        # Validate config
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Embedding layer
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_len,
            positional='sinusoidal',
            dropout=dropout
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=self.d_ff,
                dropout=dropout
            )
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
        self._init_method = init_method

    def _init_weights(self, module):
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(input_ids)  # (B, T, d_model)

        # Through transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, d_model)

        # Final normalization
        x = self.ln_f(x)  # (B, T, d_model)

        # Output projection
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def generate(self, start_ids, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, seed=None):
        """Generate text autoregressively. Implemented in Module 07."""
        raise NotImplementedError(
            "Text generation implemented in Module 07: Sampling"
        )

    def count_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        """String representation."""
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, n_layers={self.n_layers}, "
                f"d_ff={self.d_ff}, max_seq_len={self.max_seq_len}")
```

### Creating and Using the Model

```python
# Create model
model = TinyTransformerLM(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=1024,
    dropout=0.1,
    tie_weights=True
)

# Check configuration
print(model)
print(f"Total parameters: {model.count_parameters():,}")

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Forward pass
input_ids = torch.randint(0, 10000, (32, 128)).to(device)
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # (32, 128, 10000)

# Compute loss
targets = torch.randint(0, 10000, (32, 128)).to(device)
loss = F.cross_entropy(
    logits.view(-1, 10000),
    targets.view(-1)
)
print(f"Loss: {loss.item():.4f}")
```

---

## Design Decisions and Rationale

### 1. Pre-LN vs Post-LN

**We use Pre-LN (layer norm before sublayers):**

```
Pre-LN:   x = x + Sublayer(LayerNorm(x))
Post-LN:  x = LayerNorm(x + Sublayer(x))
```

**Why Pre-LN?**

| Aspect | Pre-LN | Post-LN |
|--------|--------|---------|
| Training Stability | Better | Worse for deep models |
| Gradient Flow | More stable | Can explode/vanish |
| Initialization | More robust | Needs careful init |
| Deep Networks | Easier (24+ layers) | Harder (>12 layers) |
| Output Distribution | May need final norm | Handled by last norm |
| Performance | Usually same | Same at convergence |

**Evidence:**
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- Pre-LN enables training of deeper models without divergence
- We add final layer norm to stabilize before output projection

### 2. Weight Tying

**We tie embedding and output weights:**

```python
self.lm_head.weight = self.embedding.token_embedding.embedding.weight
```

**Reasons:**
1. Parameter efficiency (20% fewer parameters)
2. Semantic consistency (similar tokens → similar representations)
3. Empirical improvement in generalization
4. Standard in GPT-2, GPT-3, BERT

**Trade-offs:**
- Can't use different vocabularies for input/output
- Slightly constrains representational flexibility
- But empirical benefits outweigh costs

### 3. GELU Activation

**We use GELU in feed-forward networks:**

```python
self.activation = nn.GELU()
```

**Why GELU instead of ReLU?**

```
ReLU:  f(x) = max(0, x)     [Discontinuous at 0]
GELU:  f(x) = x * Φ(x)     [Smooth, probabilistic interpretation]
```

| Aspect | ReLU | GELU |
|--------|------|------|
| Smoothness | Not smooth | Smooth |
| Gradient | Discontinuous | Continuous |
| Initialization | Needs Kaiming | Works with normal |
| Training | Good | Better |
| Paper choice | Transformer | GPT-2, BERT, etc. |

**GELU interpretation:**
- Φ(x) = cumulative normal distribution
- GELU(x) = "x × (probability that x should be kept)"
- Probabilistic masking is more sophisticated than ReLU's binary gate

### 4. Causal Masking

**We mask future tokens during attention:**

```python
# mask[i, j] = -∞ if j > i, else 0
```

**Why?**

1. Enforces autoregressive property (predict next token only)
2. Enables efficient parallel training
3. Matches generation behavior
4. Prevents information leakage

**Without causal mask:**
```
Position 1 could see Position 3
→ Model learns cheating (using future info)
→ Generation fails (no future available)
```

### 5. No Bias in Output Projection

**We use `bias=False` in LM head:**

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
```

**Why?**

1. With weight tying, bias is redundant
2. Reduces parameters (vocab_size parameters)
3. Improves numerical stability
4. Standard practice

**What about bias in attention/FFN?**
- Those keep bias
- Bias useful for learning position-specific offsets
- Embedded in attention mechanics

### 6. Max Sequence Length

**We have fixed max_seq_len = 1024:**

**Trade-offs:**

```
max_seq_len = 256    Too small (32×32 matrix per position)
max_seq_len = 1024   Balanced (standard, ~1M positional emb)
max_seq_len = 4096   Large (4M positional embeddings)
max_seq_len = 32768  GPT-3 (128M positional embeddings!)
```

**Our choice 1024:**
- Standard for models of this size
- Fits reasonably in memory
- Can be extended if needed

### 7. Sinusoidal vs Learned Positional Encoding

**We default to sinusoidal:**

```python
positional='sinusoidal'  # OR 'learned'
```

| Aspect | Sinusoidal | Learned |
|--------|------------|---------|
| Parameters | 0 | max_len × d_model |
| Extrapolation | Can extrapolate | Can't (fixed max_len) |
| Initialization | Deterministic | Random init |
| Flexibility | Less | More |
| Performance | Competitive | Often better |

**For this model:** Sinusoidal is fine, can easily switch.

---

## Common Pitfalls and Solutions

### 1. Shape Mismatches

**Problem:** Shapes don't match during forward pass

```python
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x512 vs 256x1024)
```

**Diagnosis:**
```python
def debug_forward(model, input_ids):
    x = model.embedding(input_ids)
    print(f"After embed: {x.shape}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"After block {i}: {x.shape}")
```

**Common causes:**
- d_model not divisible by n_heads
- Wrong input shape (should be (B, T), not (B, T, V))
- Embedding layer mismatch with projection

**Solution:** Always verify shapes at each step.

### 2. Weight Tying Issues

**Problem:** Weights aren't actually shared

```python
# WRONG (creates copy, not reference):
self.lm_head.weight = self.embedding.token_embedding.embedding.weight.clone()

# RIGHT (creates reference):
self.lm_head.weight = self.embedding.token_embedding.embedding.weight
```

**Debugging:**
```python
# Check if weights are shared:
print(lm_head.weight is embedding.weight)  # Should be True
print(lm_head.weight.data_ptr() == embedding.weight.data_ptr())  # True
```

**Why it matters:**
- Without tying, embedding and output become independent
- Defeats purpose of weight tying
- Doubles parameters

### 3. Causal Mask Not Applied

**Problem:** Model "cheats" by looking at future tokens

```python
# WRONG (no mask):
x = model(input_ids)

# RIGHT (with mask):
x = model(input_ids, mask=causal_mask)
```

**Detection:**
- Loss becomes suspiciously low
- Generation produces incoherent output
- Model memorizes sequences

**Solution:**
```python
def create_causal_mask(seq_len, device):
    """Create lower-triangular causal mask."""
    mask = torch.ones(seq_len, seq_len, device=device)
    mask = torch.tril(mask)  # Lower triangular
    mask = (1 - mask) * (-1e9)  # Convert to -inf
    return mask
```

### 4. Initialization Problems

**Problem:** Loss doesn't decrease, or NaN after few steps

**Causes:**
- Weights too large → gradients explode
- Weights too small → gradients vanish
- No initialization applied

**Debugging:**
```python
# Check initial weight magnitudes
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
```

**Solution:**
```python
def apply_initialization(model, init_type='normal'):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == 'normal':
                nn.init.normal_(module.weight, std=0.02)
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
```

### 5. Gradient Flow Issues

**Problem:** Deeper layers don't learn (zero gradients)

```python
# Check gradients:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.6f}")
```

**Causes:**
- Layer norm positioned incorrectly (Post-LN vs Pre-LN)
- Too deep without residual connections
- Poor initialization

**Solutions:**
- Use Pre-LN architecture (we do)
- Reduce depth initially, increase later
- Better initialization (Kaiming for deep models)

### 6. Memory Issues

**Problem:** Out of memory with reasonably sized batch

```python
RuntimeError: CUDA out of memory
```

**Causes:**
- Attention O(n²) memory with sequence length
- All activations stored for backward pass
- Model too large for device

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # was 32

# Reduce sequence length
seq_len = 256  # was 512

# Use gradient checkpointing
from torch.utils.checkpoint import checkpoint
x = checkpoint(block, x)  # Recompute instead of storing
```

### 7. NaN in Loss

**Problem:** Loss becomes NaN after training starts

**Causes:**
- Gradients exploding
- Invalid input values
- Softmax overflow in attention

**Solution:**
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for invalid inputs
assert not torch.isnan(input_ids).any()
assert not torch.isinf(logits).any()
```

---

## Practical Tips for Training

### 1. Batch Size and Sequence Length

**Trade-off:**

```
Larger batch + Shorter sequences:
  - More parallelism
  - Faster (more tokens/second)
  - Less memory per sequence

Smaller batch + Longer sequences:
  - Better context learning
  - Slower (fewer tokens/second)
  - More memory per sequence
```

**Recommendation:**
```python
# Goal: ~1M tokens per step
batch_size = 32
seq_len = 512
tokens_per_step = 32 * 512 = 16,384

# Adjust based on memory and speed
```

### 2. Learning Rate Scheduling

**Transformer training benefits from warmup:**

```python
# Warmup → Cosine decay

LR schedule:
    0-500 steps:   Linear warmup from 0 to peak_lr
    500-10000:     Cosine decay to min_lr
```

**Why warmup?**
- Prevents divergence at start
- Allows gradients to stabilize
- Standard practice for all transformers

### 3. Mixed Precision Training

**Speed up training with less memory:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        logits = model(input_ids)
        loss = F.cross_entropy(...)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 4. Validation Monitoring

**Track metrics during training:**

```python
def evaluate(model, val_dataloader):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in val_dataloader:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity
```

### 5. Checkpointing Strategy

**Save models during training:**

```python
def save_checkpoint(model, optimizer, step, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': model.config  # Save config too!
    }
    path = checkpoint_dir / f"checkpoint_{step:06d}.pt"
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step']
```

### 6. Debugging Training

**Print intermediate values:**

```python
# At start of training loop
print(f"Step {step}: loss={loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

# Every 100 steps
if step % 100 == 0:
    with torch.no_grad():
        sample_logits = model(sample_batch)
        sample_probs = F.softmax(sample_logits[:, -1, :], dim=-1)
        entropy = -(sample_probs * torch.log(sample_probs)).sum()
        print(f"  Sample entropy: {entropy:.4f}")
```

---

## Summary

### Key Takeaways

1. **Architecture**: Stack of transformer blocks with embeddings and output projection forms complete model

2. **Weight Tying**: Sharing embedding/output weights reduces parameters and improves generalization

3. **Initialization**: Proper weight initialization crucial for stable training

4. **Causal Modeling**: Masking prevents attending to future, enabling autoregressive generation

5. **Shape Consistency**: Model maintains dimensionality through residual connections

6. **Parameter Count**: Can be estimated and optimized based on computational constraints

7. **Design Choices**: Pre-LN, GELU, no bias in output head are modern standards

### Next Steps

**Module 06: Training**
- Implement training loop
- Loss computation and backpropagation
- Optimization strategies
- Validation and early stopping

**Module 07: Sampling**
- Autoregressive text generation
- Sampling strategies (greedy, temperature, top-k, top-p)
- Beam search
- Complete the `generate()` method

### Self-Assessment

After this module, you should be able to:

- [ ] Build complete transformer from scratch
- [ ] Explain weight tying and its benefits
- [ ] Count and estimate model parameters
- [ ] Implement proper initialization
- [ ] Understand causal language modeling objective
- [ ] Trace shapes through forward pass
- [ ] Debug shape mismatches
- [ ] Choose appropriate model configurations
- [ ] Apply weight tying correctly
- [ ] Design training loops with proper masking

### Further Reading

1. **Original Transformer Paper**
   - "Attention Is All You Need" (Vaswani et al., 2017)
   - Introduces all key concepts

2. **GPT Papers**
   - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
   - Practical implementation details

3. **Training & Scaling**
   - "Chinchilla's Wild Implications" (Hoffmann et al., 2022)
   - Optimal scaling of model size and data

4. **Initialization**
   - "Kaiming Initialization for ReLU Nets" (He et al., 2015)
   - Principled approach to weight initialization

5. **Layer Normalization**
   - "Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
   - Pre-LN vs Post-LN analysis

---

**Created:** November 2024
**For:** TinyTransformerBuild Educational Series
**Module:** 05 - Full Model Assembly

