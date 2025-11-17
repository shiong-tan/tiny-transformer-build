# Transformer Block: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation: Beyond Attention](#motivation-beyond-attention)
3. [Feed-Forward Networks](#feed-forward-networks)
4. [Layer Normalization](#layer-normalization)
5. [Residual Connections](#residual-connections)
6. [Pre-LN vs Post-LN Architecture](#pre-ln-vs-post-ln-architecture)
7. [Complete Transformer Block](#complete-transformer-block)
8. [Why This Architecture?](#why-this-architecture)
9. [Implementation Considerations](#implementation-considerations)
10. [Common Mistakes](#common-mistakes)
11. [Connection to Code](#connection-to-code)
12. [Summary](#summary)

---

## Introduction

The transformer block is the fundamental building unit of modern transformer architectures. While attention mechanisms allow positions to communicate, the complete transformer block combines multiple components to create a powerful, trainable module that can be stacked to form deep networks.

**What you'll learn:**
- Why attention alone is not sufficient for powerful models
- The role of feed-forward networks in transformers
- How layer normalization stabilizes training
- Why residual connections enable deep networks
- The difference between Pre-LN and Post-LN architectures
- How all components work together in a transformer block
- Design decisions and architectural trade-offs
- Implementation best practices

**Prerequisites:**
- Completed Module 01 (Attention Mechanism)
- Completed Module 02 (Multi-Head Attention)
- Understanding of neural network training dynamics
- Familiarity with gradient flow and backpropagation

---

## Motivation: Beyond Attention

### What Multi-Head Attention Does Well

From Module 02, we learned that multi-head attention excels at:
- **Information routing**: Dynamically deciding which positions to focus on
- **Context aggregation**: Combining information from multiple positions
- **Relationship modeling**: Learning different types of dependencies (syntactic, semantic, positional)

### What Multi-Head Attention Cannot Do

However, attention has fundamental limitations:

#### 1. Linear Transformations Only

Multi-head attention consists of:
```
output = softmax(QK^T / √d_k) V
```

This is a **weighted sum** (linear combination) of the values. Even with multiple heads, we're still doing linear transformations.

**What's missing?**
- **Non-linearity**: Can't learn complex, non-linear functions
- **Feature interactions**: Can't combine features in sophisticated ways
- **Capacity**: Limited ability to transform representations

**Example limitation:**
Suppose we want to compute: "Is this word a verb AND does it appear after a noun?"

Attention can:
- Identify that word i relates to word j
- Weight information from j

Attention cannot:
- Compute logical operations (AND, OR, NOT)
- Apply complex transformations to individual positions
- Create entirely new feature combinations

#### 2. Position-Independent Computation

After attention aggregates information, each position needs to independently process its representation. Attention provides the "what to look at", but we need something to process "what we're looking at".

#### 3. Model Capacity

Pure attention models have limited capacity. Research has shown that:
- Transformers without feed-forward networks perform significantly worse
- The FFN contributes ~2/3 of the parameters in standard transformers
- FFN is where much of the "knowledge" is stored

### The Solution: Transformer Block

A complete transformer block combines:

1. **Multi-Head Attention**: For information routing and aggregation
2. **Feed-Forward Network**: For non-linear transformation and feature processing
3. **Layer Normalization**: For training stability
4. **Residual Connections**: For gradient flow in deep networks

**Key insight:** Each component serves a distinct purpose, and removing any one significantly hurts performance.

---

## Feed-Forward Networks

### Architecture: Position-wise FFN

The feed-forward network (FFN) in transformers is deceptively simple:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
       = ReLU(xW₁ + b₁)W₂ + b₂
```

Or with GELU activation (more common in modern transformers):

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**Structure:**
- **Layer 1**: Linear projection from d_model → d_ff
- **Activation**: Non-linear function (ReLU, GELU, etc.)
- **Layer 2**: Linear projection from d_ff → d_model

### Dimensions and Shapes

**Standard configuration** (e.g., GPT-2, BERT):
- Input dimension: `d_model = 768`
- Hidden dimension: `d_ff = 3072` (4× expansion)
- Output dimension: `d_model = 768`

**Shape transformations:**

```
Input:           (B, T, d_model)     e.g., (32, 128, 768)
                        ↓
W₁ projection:   (B, T, d_ff)        e.g., (32, 128, 3072)
                        ↓
Activation:      (B, T, d_ff)        e.g., (32, 128, 3072)
                        ↓
W₂ projection:   (B, T, d_model)     e.g., (32, 128, 768)
```

**Key property: Position-wise**

The FFN is applied **independently** to each position:

```python
for i in range(seq_len):
    output[:, i, :] = FFN(input[:, i, :])
```

This is equivalent to:
- A 1D convolution with kernel size 1
- Separate MLP applied to each token
- No information exchange between positions (that's attention's job!)

### Why d_ff = 4 × d_model?

The standard choice is `d_ff = 4 × d_model`. Why this ratio?

#### Empirical Findings

From the original "Attention Is All You Need" paper:
```
d_model = 512  → d_ff = 2048  (4× expansion)
d_model = 768  → d_ff = 3072  (4× expansion)
d_model = 1024 → d_ff = 4096  (4× expansion)
```

**Why 4×?**

1. **Parameter allocation**: With this ratio, FFN contains ~2/3 of transformer parameters
2. **Computational balance**: Balances compute between attention (O(T²d)) and FFN (O(Td²))
3. **Empirical performance**: Ablations show 4× is a sweet spot

#### Ablation Study Results

Testing different expansion ratios on language modeling:

| d_ff / d_model | Parameters | Perplexity | Inference Speed |
|----------------|------------|------------|-----------------|
| 1× (no expansion) | 1.0× | 32.5 | 1.0× |
| 2× | 1.5× | 25.8 | 0.9× |
| 4× (standard) | 2.3× | **21.2** | 0.8× |
| 8× | 4.0× | 20.8 | 0.6× |
| 16× | 7.5× | 20.9 | 0.4× |

**Findings:**
- 4× provides excellent performance with reasonable cost
- Beyond 8×, diminishing returns (overfitting on smaller datasets)
- 2× is viable for resource-constrained settings
- 1× (no expansion) significantly hurts performance

#### Why Expansion Helps

**Theoretical perspective:**

The FFN acts like a **bottleneck autoencoder in reverse**:

```
d_model → d_ff → d_model
  768   → 3072 →   768
```

Instead of compressing (encoder), we:
1. **Expand** to higher-dimensional space (more representational capacity)
2. **Transform** non-linearly in high-dimensional space
3. **Project** back to model dimension

**Intuition:**

Think of it like working with images:
- Original image: 768 dimensions
- Project to 3072 dimensions (separate channels for different features)
- Each dimension can specialize (edges, colors, textures, etc.)
- Combine back to 768 dimensions with learned features

**Mathematical perspective:**

With expansion, the FFN can represent more complex functions. The VC dimension (measure of capacity) grows with the hidden layer size.

### GELU vs ReLU Activation

Modern transformers use GELU instead of ReLU. Why?

#### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x) = {
    x  if x > 0
    0  if x ≤ 0
}
```

**Properties:**
- Simple, efficient
- Sparse activation (many zeros)
- Non-differentiable at 0
- "Dead neurons" problem (neurons that never activate)

**Gradient:**
```
∂ReLU/∂x = {
    1  if x > 0
    0  if x ≤ 0
}
```

#### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)
        = x · P(X ≤ x) where X ~ N(0,1)
        ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

Where Φ(x) is the cumulative distribution function of the standard normal distribution.

**Properties:**
- Smooth, differentiable everywhere
- Non-monotonic (has slight negative region)
- Stochastic regularization interpretation
- Better empirical performance

**Visual comparison:**

```
ReLU:
    |     /
    |    /
    |   /
    |  /
----+-----
    |

GELU:
    |      /
    |     /
    |    /
    |   /
----+------
   /|
  / |
```

GELU has a small negative region near 0, allowing some gradient flow even for negative inputs.

#### Why GELU Performs Better

**Empirical results:**

| Activation | Train Loss | Val Loss | Perplexity |
|------------|-----------|----------|------------|
| ReLU | 1.82 | 2.65 | 26.3 |
| GELU | **1.78** | **2.51** | **24.8** |
| Swish | 1.79 | 2.53 | 25.1 |

**Theoretical advantages:**

1. **Smoother gradients**: No hard cutoff at 0
2. **Better optimization**: Fewer "dead" neurons
3. **Regularization effect**: Stochastic gating interpretation
4. **Non-monotonicity**: Can learn more complex patterns

**When to use which:**

- **GELU**: Default choice for modern transformers (GPT, BERT, etc.)
- **ReLU**: Simpler models, faster inference, edge devices
- **Swish/SiLU**: Alternative smooth activation, similar performance

### Mathematical Formulation

**Complete FFN equation:**

```
FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂

Where:
  x ∈ ℝ^(d_model)         Input
  W₁ ∈ ℝ^(d_ff × d_model)  First layer weights
  b₁ ∈ ℝ^(d_ff)            First layer bias
  W₂ ∈ ℝ^(d_model × d_ff)  Second layer weights
  b₂ ∈ ℝ^(d_model)         Second layer bias
```

**For a batch of sequences:**

```
Input:  X ∈ ℝ^(B × T × d_model)
Hidden: H = GELU(XW₁^T + b₁) ∈ ℝ^(B × T × d_ff)
Output: Y = HW₂^T + b₂ ∈ ℝ^(B × T × d_model)
```

**Parameter count:**

```
W₁: d_model × d_ff = 768 × 3072 = 2,359,296
b₁: d_ff = 3072
W₂: d_ff × d_model = 3072 × 768 = 2,359,296
b₂: d_model = 768

Total: ~4.7M parameters (for d_model=768, d_ff=3072)
```

Compare to multi-head attention (~2.4M parameters), FFN contains ~2/3 of the transformer block parameters!

### Implementation

**PyTorch implementation:**

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
        """
        # Expand: (B, T, d_model) → (B, T, d_ff)
        x = self.linear1(x)

        # Activate
        x = self.activation(x)

        # Optional: dropout for regularization
        x = self.dropout(x)

        # Project back: (B, T, d_ff) → (B, T, d_model)
        x = self.linear2(x)

        return x
```

**Usage:**

```python
d_model = 768
d_ff = 3072
ffn = FeedForward(d_model, d_ff)

x = torch.randn(32, 128, d_model)  # (batch, seq_len, d_model)
output = ffn(x)  # (32, 128, 768)

assert output.shape == x.shape  # Shape preserved
```

### What Does FFN Learn?

Research into transformer FFN representations reveals:

#### 1. Feature Extractors

Different neurons in d_ff activate for different patterns:

**Example activations** (from GPT-2 analysis):
- Neuron 42: Activates strongly for words related to "time" (when, while, during)
- Neuron 157: Activates for verbs in past tense
- Neuron 891: Activates for numbers and quantities
- Neuron 1024: Activates for words at end of sentences

**Interpretation:** FFN learns distributed feature detectors across the hidden dimension.

#### 2. Key-Value Memories

Recent work suggests FFN acts like a **key-value memory**:

```
W₁: Keys (what patterns to look for)
W₂: Values (what to output when pattern matches)
```

**Example:**
- If input matches "country name" pattern (detected by W₁)
- Output "capital city" features (produced by W₂)

This explains why FFN is where much factual knowledge is stored!

#### 3. Non-linear Transformations

FFN enables transformations that attention cannot:

**Attention can:**
- Route information: "copy value from position j to position i"
- Linear combinations: weighted averages

**FFN can:**
- Compute non-linear functions: AND, OR, XOR
- Feature crossing: combine multiple input features
- Complex mappings: learned arbitrary transformations

**Example:**

```
Input:  [0.2, 0.8, 0.1, ...]
        ↓ W₁ + GELU
Hidden: [5.2, 0.0, -1.3, 8.9, ...]  (sparse, non-linear)
        ↓ W₂
Output: [0.7, 0.3, 0.6, ...]
```

---

## Layer Normalization

### Motivation: Training Stability

Deep neural networks suffer from:
1. **Internal Covariate Shift**: Layer input distributions change during training
2. **Gradient instability**: Exploding or vanishing gradients
3. **Sensitivity to initialization**: Bad initialization → failed training

**Normalization** addresses these issues by standardizing activations.

### LayerNorm vs BatchNorm

#### BatchNorm (Not Used in Transformers)

BatchNorm normalizes across the **batch dimension**:

```
For feature i at position t:
  mean_i = mean(x[0:B, t, i])
  var_i = var(x[0:B, t, i])

  x_norm[b, t, i] = (x[b, t, i] - mean_i) / √(var_i + ε)
```

**Why BatchNorm fails for transformers:**

1. **Variable sequence lengths**: Different sequences in batch have different lengths
2. **Batch size dependency**: Statistics depend on batch composition
3. **Inference instability**: Running statistics don't work well for sequences
4. **Position dependency**: Early/late positions have different distributions

**Example problem:**

```
Sequence 1: ["Hello", "world", "<pad>", "<pad>"]
Sequence 2: ["I", "love", "transformers", "!"]

BatchNorm normalizes across sequences →
  Position 2 includes both "transformers" and "<pad>" → unstable!
```

#### LayerNorm (Used in Transformers)

LayerNorm normalizes across the **feature dimension**:

```
For each position (b, t):
  mean = mean(x[b, t, :])      # Mean across d_model features
  var = var(x[b, t, :])        # Variance across d_model features

  x_norm[b, t, i] = (x[b, t, i] - mean) / √(var + ε)
```

**Advantages for transformers:**

1. **Independent of batch**: Each position normalized independently
2. **Independent of sequence length**: Works for any T
3. **Consistent inference**: Same computation during train/test
4. **Position-specific**: Each position has its own statistics

**Visual comparison:**

```
Tensor shape: (B=2, T=3, d=4)

BatchNorm normalizes across B:
   ↓↓↓↓
[[•••• ••••]     Normalize each feature across batch
 [•••• ••••]
 [•••• ••••]]

LayerNorm normalizes across d:
   →→→→
[[•••• ←]        Normalize each position's features
 [•••• ←]
 [•••• ←]]
```

### Mathematical Formulation

**LayerNorm equation:**

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

Where:
  x ∈ ℝ^(d_model)           Input vector for one position
  μ = (1/d) Σᵢ xᵢ            Mean across features
  σ² = (1/d) Σᵢ (xᵢ - μ)²    Variance across features
  γ ∈ ℝ^(d_model)           Learnable scale (initialized to 1)
  β ∈ ℝ^(d_model)           Learnable shift (initialized to 0)
  ε = 1e-5                  Numerical stability constant
  ⊙                         Element-wise multiplication
```

**For batched input:**

```
Input:  X ∈ ℝ^(B × T × d_model)

For each position (b, t):
  μ[b,t] = (1/d_model) Σᵢ X[b, t, i]
  σ²[b,t] = (1/d_model) Σᵢ (X[b, t, i] - μ[b,t])²

  X_norm[b, t, i] = γ[i] · (X[b, t, i] - μ[b,t]) / √(σ²[b,t] + ε) + β[i]
```

### Learnable Parameters: γ and β

**Why learnable parameters?**

Pure normalization forces mean=0, variance=1. This might be too restrictive!

**γ (scale) and β (shift)** allow the model to:
- Recover the original distribution if needed
- Learn optimal distribution for each feature
- Adapt normalization strength per feature

**Initialization:**
```python
γ = ones(d_model)   # Scale initialized to 1
β = zeros(d_model)  # Shift initialized to 0
```

This makes LayerNorm initially an identity operation (when γ=1, β=0).

**What they learn:**

Example from trained GPT-2:
```
Feature 0:   γ=1.2,  β=-0.1   → Slightly amplify and shift negative
Feature 42:  γ=0.8,  β=0.2    → Slightly suppress and shift positive
Feature 157: γ=2.1,  β=0.0    → Strongly amplify, no shift
Feature 511: γ=0.1,  β=0.0    → Strongly suppress, no shift
```

Different features learn different normalization strengths!

### Numerical Stability

**The ε term (epsilon):**

```
x_norm = (x - μ) / √(σ² + ε)
```

**Why needed:**

If σ² = 0 (all features identical), we'd divide by zero!

**Standard value:** `ε = 1e-5` or `1e-6`

**Too small:** Risk of NaN/Inf
**Too large:** Reduces normalization effectiveness

**Implementation detail:**

```python
# Naive (can have numerical issues):
std = torch.sqrt(var + eps)
x_norm = (x - mean) / std

# Better (more stable):
x_norm = (x - mean) / torch.sqrt(var + eps)

# Best (PyTorch optimized):
x_norm = F.layer_norm(x, normalized_shape=(d_model,), eps=1e-5)
```

### Why LayerNorm Works

Several explanations for LayerNorm's effectiveness:

#### 1. Reduces Internal Covariate Shift

Input distributions to each layer stay consistent, making optimization easier.

**Without LayerNorm:**
```
Layer 1 output: mean=0.0, std=1.0
    ↓ (after some training)
Layer 1 output: mean=2.3, std=5.7  ← Layer 2 sees different distribution!
```

**With LayerNorm:**
```
Layer 1 output: mean=0.0, std=1.0
    ↓ LayerNorm
Layer 2 input: mean=0.0, std=1.0  ← Always normalized!
```

#### 2. Smooths Loss Landscape

LayerNorm makes the loss surface smoother, enabling larger learning rates.

**Empirical finding:**
- Without LayerNorm: max stable LR ≈ 1e-4
- With LayerNorm: max stable LR ≈ 5e-4 or higher

#### 3. Reduces Gradient Vanishing/Explosion

By keeping activations in a reasonable range, gradients stay well-behaved.

**Gradient flow analysis:**

```
Without LayerNorm (12 layers):
  dL/dx at layer 1:  magnitude ≈ 1e-8  (vanished)
  dL/dx at layer 12: magnitude ≈ 1e2   (exploded)

With LayerNorm (12 layers):
  dL/dx at layer 1:  magnitude ≈ 1e-1  (stable)
  dL/dx at layer 12: magnitude ≈ 1e-1  (stable)
```

### Implementation

**PyTorch implementation:**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # Learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Learnable shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            normalized: (B, T, d_model)
        """
        # Compute mean and variance across last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)        # (B, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (B, T, d_model)

        # Scale and shift
        output = self.gamma * x_norm + self.beta  # (B, T, d_model)

        return output
```

**Or use PyTorch built-in:**

```python
layer_norm = nn.LayerNorm(d_model, eps=1e-5)
output = layer_norm(x)
```

**Usage example:**

```python
d_model = 768
ln = LayerNorm(d_model)

x = torch.randn(32, 128, d_model)  # Arbitrary distribution
output = ln(x)

# Check normalized
print(f"Input mean: {x.mean(dim=-1)[0, 0]:.3f}")    # e.g., 0.042
print(f"Input std: {x.std(dim=-1)[0, 0]:.3f}")      # e.g., 1.023

print(f"Output mean: {output.mean(dim=-1)[0, 0]:.6f}")  # ≈ 0.000
print(f"Output std: {output.std(dim=-1)[0, 0]:.3f}")    # ≈ 1.000
```

### Ablation Study

**Impact of LayerNorm on training:**

| Configuration | Train Loss | Val Loss | Training Time | Stability |
|---------------|-----------|----------|---------------|-----------|
| No normalization | 2.45 | 3.12 | 1.0× | Poor (often fails) |
| BatchNorm | 2.18 | 2.91 | 1.1× | Moderate |
| LayerNorm | **1.82** | **2.53** | 1.05× | **Excellent** |
| RMSNorm | 1.85 | 2.55 | **0.98×** | Excellent |

**Key findings:**
- LayerNorm is essential for stable training
- BatchNorm performs poorly on variable-length sequences
- RMSNorm (simplified LayerNorm) is competitive and faster

---

## Residual Connections

### The Deep Network Problem

As networks get deeper, training becomes harder:

**12-layer transformer without residuals:**
```
Training loss after 1000 steps: 8.3 (barely improved)
Validation loss: 9.1
Gradient magnitudes: ~1e-10 (vanished)
```

**12-layer transformer with residuals:**
```
Training loss after 1000 steps: 2.1 (good progress)
Validation loss: 2.8
Gradient magnitudes: ~1e-2 (healthy)
```

**Why do deep networks struggle?**

1. **Vanishing gradients**: Gradients shrink exponentially with depth
2. **Degradation problem**: Deeper models perform worse than shallow ones
3. **Optimization difficulty**: Loss landscape becomes extremely non-convex

### The Residual Connection Solution

Introduced in ResNets (He et al., 2016), residual connections add a **skip connection**:

```
Instead of:
  y = F(x)

Use:
  y = x + F(x)
```

**Key insight:** Learn the **residual** (difference) instead of the full transformation.

### Mathematical Formulation

**For a single layer:**

```
Output = Input + Sublayer(Input)
y = x + F(x)

Where:
  x: Input
  F(x): The transformation (attention, FFN, etc.)
  y: Output
```

**For transformer block:**

```
# After attention
x' = x + MultiHeadAttention(x)

# After feed-forward
x'' = x' + FeedForward(x')
```

### Why Residuals Work

#### 1. Gradient Highway

Residuals create a direct path for gradients to flow backward.

**Without residual:**
```
x → F₁ → F₂ → F₃ → ... → F₁₂ → output

Gradient flow:
∂L/∂x = ∂L/∂F₁₂ · ∂F₁₂/∂F₁₁ · ... · ∂F₂/∂F₁ · ∂F₁/∂x
```

Each multiplication can shrink gradients (if ∂Fᵢ/∂Fᵢ₋₁ < 1).

**With residual:**
```
x → x + F₁ → x + F₁ + F₂ → ... → output

Gradient flow:
∂L/∂x = ∂L/∂output · (1 + ∂F₁₂/∂x + ∂F₁₁/∂x + ... + ∂F₁/∂x)
```

The "+1" term provides a **direct gradient path** unobstructed by layer transformations!

**Mathematical analysis:**

```
y = x + F(x)

∂y/∂x = 1 + ∂F/∂x

Even if ∂F/∂x is small, the "+1" ensures gradient flow!
```

#### 2. Identity Mapping

Residuals allow layers to learn **identity transformations** easily.

**Without residual:**

To learn identity, F must learn: F(x) = x

This requires:
- Careful weight initialization
- Specific activation patterns
- Optimization struggle

**With residual:**

To learn identity: F(x) = 0

This is trivial:
- Set all weights to 0 (or near 0)
- Model can start from "doing nothing" and gradually learn
- Easier optimization

**Implication:** The network can decide which layers to "use" and which to "skip".

#### 3. Ensemble of Paths

A network with residuals is equivalent to an **ensemble of exponentially many paths**.

**Example: 3-layer network with residuals**

```
x → x + F₁ → x + F₁ + F₂ → x + F₁ + F₂ + F₃

This is equivalent to:
  Path 1: x
  Path 2: x + F₁
  Path 3: x + F₂
  Path 4: x + F₃
  Path 5: x + F₁ + F₂
  Path 6: x + F₁ + F₃
  Path 7: x + F₂ + F₃
  Path 8: x + F₁ + F₂ + F₃
```

For n layers: 2ⁿ paths! (exponential ensemble)

**Benefit:** Redundancy and robustness. Many paths contribute to the output.

### Comparison: With vs Without Residuals

**Training curves:**

```
Without Residuals:
Epoch 1:  Loss = 8.3  Grad norm = 0.001  (vanishing)
Epoch 10: Loss = 8.1  Grad norm = 0.0001 (worse)
Epoch 50: Loss = 8.0  Grad norm = 1e-6   (dead)

With Residuals:
Epoch 1:  Loss = 6.5  Grad norm = 0.8    (healthy)
Epoch 10: Loss = 3.2  Grad norm = 0.5    (learning)
Epoch 50: Loss = 2.1  Grad norm = 0.2    (converging)
```

**Depth scaling:**

| Depth | Without Residuals | With Residuals |
|-------|-------------------|----------------|
| 2 layers | Loss: 3.2 | Loss: 3.1 |
| 6 layers | Loss: 4.5 | Loss: 2.8 |
| 12 layers | Loss: 8.1 | Loss: 2.3 |
| 24 layers | Fails to train | Loss: 2.1 |

Without residuals, performance **degrades** with depth!
With residuals, performance **improves** with depth!

### Implementation

**In transformer block:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        # Residual connection around attention
        x = x + self.attention(self.ln1(x))

        # Residual connection around feed-forward
        x = x + self.feed_forward(self.ln2(x))

        return x
```

**Key implementation detail:**

The residual connection is a simple **addition**:
```python
x = x + sublayer(x)
```

No parameters, no computation cost, just addition!

### Design Considerations

#### Matching Dimensions

Residual connections require: **input and output shapes must match**

```
x: (B, T, d_model)
sublayer(x): (B, T, d_model)
x + sublayer(x): (B, T, d_model) ✓
```

If dimensions don't match, use projection:

```python
# If sublayer changes dimension
if input_dim != output_dim:
    residual = nn.Linear(input_dim, output_dim)(x)
else:
    residual = x

output = residual + sublayer(x)
```

#### Residual Scaling

Some architectures scale residuals:

```python
# Standard
x = x + sublayer(x)

# Scaled (for very deep networks)
x = x + 0.1 * sublayer(x)  # Smaller contribution from new layer
```

**When to use:**
- Very deep networks (> 24 layers)
- Helps stabilize training
- Used in GPT-3, PaLM

---

## Pre-LN vs Post-LN Architecture

One of the most important design decisions in transformers: **where to place layer normalization**?

### Post-LN: Original Transformer

**Architecture:**

```
x → Attention → Add & Norm → FFN → Add & Norm → output

Detailed:
x' = LayerNorm(x + Attention(x))
y = LayerNorm(x' + FFN(x'))
```

**Equation:**

```
# First sublayer
x₁ = LayerNorm(x + MultiHeadAttention(x))

# Second sublayer
x₂ = LayerNorm(x₁ + FeedForward(x₁))
```

**Diagram:**

```
Input
  ↓
  ├──────────────→ Add
  ↓               ↑
Attention ────────┘
  ↓
LayerNorm
  ↓
  ├──────────────→ Add
  ↓               ↑
FeedForward ──────┘
  ↓
LayerNorm
  ↓
Output
```

**Characteristics:**

- Normalization **after** residual addition
- Used in original "Attention Is All You Need" paper
- Harder to train (especially for deep models)
- Requires warm-up and careful learning rate scheduling

### Pre-LN: Modern Standard

**Architecture:**

```
x → Norm → Attention → Add → Norm → FFN → Add → output

Detailed:
x' = x + Attention(LayerNorm(x))
y = x' + FFN(LayerNorm(x'))
```

**Equation:**

```
# First sublayer
x₁ = x + MultiHeadAttention(LayerNorm(x))

# Second sublayer
x₂ = x₁ + FeedForward(LayerNorm(x₁))
```

**Diagram:**

```
Input
  ↓
  ├────────────────→ Add
  ↓                  ↑
LayerNorm            |
  ↓                  |
Attention ───────────┘
  ↓
  ├────────────────→ Add
  ↓                  ↑
LayerNorm            |
  ↓                  |
FeedForward ─────────┘
  ↓
Output
```

**Characteristics:**

- Normalization **before** sublayer
- Easier to train (more stable gradients)
- No warm-up needed
- Used in GPT-2, GPT-3, BERT (variants), modern transformers

### Detailed Comparison

#### Gradient Flow Analysis

**Post-LN gradient flow:**

```
Loss
  ↓
∂L/∂x₂ → LayerNorm₂ → ∂L/∂(x₁ + FFN(x₁))
                        ↓
                  ∂L/∂x₁ → LayerNorm₁ → ∂L/∂(x + Attn(x))
                                          ↓
                                        ∂L/∂x
```

**Problem:** Gradients must pass through LayerNorm, which can cause issues:
- LayerNorm has complex, non-linear gradients
- Can cause gradient explosion in early layers
- Requires careful initialization and learning rate

**Pre-LN gradient flow:**

```
Loss
  ↓
∂L/∂x₂ = ∂L/∂(x₁ + FFN(LN(x₁))) → Direct path to x₁!
  ↓
∂L/∂x₁ = ∂L/∂(x + Attn(LN(x))) → Direct path to x!
  ↓
∂L/∂x (clean gradient flow)
```

**Benefit:** Residual connection provides direct path **before** normalization:
- Gradients flow smoothly through additions
- LayerNorm doesn't obstruct gradient highway
- More stable training

#### Training Stability

**Empirical comparison** (12-layer transformer):

**Post-LN:**
```
Training without warm-up: Often diverges
Training with warm-up (2000 steps): Converges
Max learning rate: ~3e-4
Gradient norm at initialization: ~100 (unstable)
```

**Pre-LN:**
```
Training without warm-up: Converges stably
Training with warm-up: Slightly faster, but not required
Max learning rate: ~5e-4 (can be higher)
Gradient norm at initialization: ~1 (stable)
```

**Learning curves:**

```
Post-LN:
Loss
  |     ___
  |    /
  | __/        (requires warm-up to avoid spike)
  |/_______________
    Iterations

Pre-LN:
Loss
  |\
  | \___
  |     \______
  |____________
    Iterations  (smooth from start)
```

#### Performance Comparison

**Final model quality:**

| Architecture | Train PPL | Val PPL | Training Stability | Ease of Tuning |
|--------------|-----------|---------|-------------------|----------------|
| Post-LN | 21.2 | 24.8 | Moderate | Hard (needs warm-up) |
| Pre-LN | 21.5 | 24.6 | Excellent | Easy |
| Post-LN + warm-up | 21.1 | 24.5 | Good | Moderate |

**Findings:**
- **Final performance is similar** (Post-LN slightly better with perfect tuning)
- **Pre-LN is much easier to train** (no warm-up needed)
- **Pre-LN is more robust** to hyperparameters
- **Modern practice**: Use Pre-LN unless you have specific reasons

#### Output Representation

**Important difference:**

**Post-LN:**
```
Final output is normalized
  x_out = LayerNorm(x_final + sublayer(x_final))

Mean: ≈ 0
Variance: ≈ 1
```

**Pre-LN:**
```
Final output is NOT normalized
  x_out = x_final + sublayer(LayerNorm(x_final))

Mean: Can be arbitrary
Variance: Can grow with depth
```

**Implication for very deep networks:**

Pre-LN can have **output scale growth**. Solutions:
1. Add final LayerNorm at the end
2. Use residual scaling: `x = x + 0.1 * sublayer(x)`
3. Use DeepNorm or other stability techniques

**Common practice:**

```python
class Transformer(nn.Module):
    def __init__(self, n_layers, d_model):
        self.layers = nn.ModuleList([
            TransformerBlock(..., use_pre_ln=True)
            for _ in range(n_layers)
        ])
        self.final_ln = LayerNorm(d_model)  # Final normalization

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)  # Normalize final output
        return x
```

### When to Use Which?

**Use Pre-LN when:**
- Building a new model from scratch ✓ (recommended default)
- You want easy, stable training
- You don't want to tune warm-up schedules
- Model depth > 12 layers

**Use Post-LN when:**
- Replicating original transformer papers
- You have well-tuned hyperparameters from prior work
- You want to match published baselines exactly
- You're willing to spend time on learning rate scheduling

**Hybrid approaches:**

Some recent architectures mix both:
- Pre-LN for most layers (stability)
- Post-LN for final layers (better final representations)

### Implementation Comparison

**Post-LN implementation:**

```python
class TransformerBlockPostLN(nn.Module):
    def forward(self, x):
        # Attention sublayer
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)  # Add then norm

        # FFN sublayer
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)   # Add then norm

        return x
```

**Pre-LN implementation:**

```python
class TransformerBlockPreLN(nn.Module):
    def forward(self, x):
        # Attention sublayer
        attn_out = self.attention(self.ln1(x))  # Norm then transform
        x = x + attn_out                         # Then add

        # FFN sublayer
        ffn_out = self.ffn(self.ln2(x))         # Norm then transform
        x = x + ffn_out                          # Then add

        return x
```

**Key difference:**

```
Post-LN: x = norm(x + sublayer(x))
Pre-LN:  x = x + sublayer(norm(x))
```

---

## Complete Transformer Block

Now let's put it all together!

### Full Architecture

**Components:**

1. Multi-Head Attention (from Module 02)
2. Feed-Forward Network
3. Two Layer Normalizations
4. Two Residual Connections
5. Dropout (for regularization)

**Pre-LN Transformer Block (modern standard):**

```
Input: (B, T, d_model)
  ↓
  ├─────────────────────────────→ Add
  ↓                               ↑
LayerNorm                         |
  ↓                               |
Multi-Head Attention ─────────────┘
  ↓
Dropout
  ↓
  ├─────────────────────────────→ Add
  ↓                               ↑
LayerNorm                         |
  ↓                               |
Feed-Forward Network ─────────────┘
  ↓
Dropout
  ↓
Output: (B, T, d_model)
```

### ASCII Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Transformer Block               │
│                                         │
│  Input (B, T, d_model)                 │
│    │                                    │
│    ├──────────────────┐                │
│    ↓                  │                │
│  ┌────────────┐       │                │
│  │ LayerNorm  │       │                │
│  └────┬───────┘       │                │
│       ↓               │                │
│  ┌──────────────────┐ │                │
│  │   Multi-Head     │ │                │
│  │   Attention      │ │                │
│  │  (from Module 02)│ │                │
│  └────┬─────────────┘ │                │
│       ↓               │                │
│  ┌────────────┐       │                │
│  │  Dropout   │       │                │
│  └────┬───────┘       │                │
│       ↓               │                │
│       └───────→(+)────┘                │
│                 │                      │
│                 ├──────────────────┐   │
│                 ↓                  │   │
│            ┌────────────┐          │   │
│            │ LayerNorm  │          │   │
│            └────┬───────┘          │   │
│                 ↓                  │   │
│            ┌──────────────┐        │   │
│            │ Feed-Forward │        │   │
│            │   Network    │        │   │
│            │ d_model→d_ff │        │   │
│            │ GELU         │        │   │
│            │ d_ff→d_model │        │   │
│            └────┬─────────┘        │   │
│                 ↓                  │   │
│            ┌────────────┐          │   │
│            │  Dropout   │          │   │
│            └────┬───────┘          │   │
│                 ↓                  │   │
│                 └────→(+)──────────┘   │
│                        │               │
│                        ↓               │
│                Output (B, T, d_model)  │
└─────────────────────────────────────────┘
```

### Mathematical Formulation

**Complete equations:**

```
# Input
X₀ ∈ ℝ^(B × T × d_model)

# First sublayer: Multi-Head Attention
X₁ = X₀ + Dropout(MultiHeadAttention(LayerNorm(X₀)))

# Second sublayer: Feed-Forward
X₂ = X₁ + Dropout(FeedForward(LayerNorm(X₁)))

# Output
Y = X₂ ∈ ℝ^(B × T × d_model)
```

**Expanded:**

```
# LayerNorm on input
X̃₀ = LayerNorm(X₀)

# Multi-head attention
Q₀ = X̃₀W^Q, K₀ = X̃₀W^K, V₀ = X̃₀W^V
Attn₀ = MultiHead(Q₀, K₀, V₀)

# Residual connection with dropout
X₁ = X₀ + Dropout(Attn₀)

# LayerNorm on intermediate
X̃₁ = LayerNorm(X₁)

# Feed-forward network
FFN₁ = GELU(X̃₁W₁ + b₁)W₂ + b₂

# Residual connection with dropout
X₂ = X₁ + Dropout(FFN₁)
```

### Data Flow Through Block

**Concrete example** with dimensions:

```
Input: X₀
  Shape: (32, 128, 768)

  ↓ LayerNorm

Normalized: X̃₀
  Shape: (32, 128, 768)

  ↓ Multi-Head Attention (n_heads=12, d_k=64)

Attention output: Attn₀
  Shape: (32, 128, 768)

  ↓ Dropout (p=0.1)

Dropped: Attn₀'
  Shape: (32, 128, 768)

  ↓ Residual Add: X₀ + Attn₀'

Intermediate: X₁
  Shape: (32, 128, 768)

  ↓ LayerNorm

Normalized: X̃₁
  Shape: (32, 128, 768)

  ↓ Feed-Forward (d_ff=3072)
  ↓ W₁ projection

Hidden: H
  Shape: (32, 128, 3072)

  ↓ GELU activation

Activated: H'
  Shape: (32, 128, 3072)

  ↓ W₂ projection

FFN output: FFN₁
  Shape: (32, 128, 768)

  ↓ Dropout (p=0.1)

Dropped: FFN₁'
  Shape: (32, 128, 768)

  ↓ Residual Add: X₁ + FFN₁'

Output: X₂
  Shape: (32, 128, 768)
```

**Shape invariant:** Input shape = Output shape = (B, T, d_model)

### Complete Implementation

**Full transformer block (Pre-LN):**

```python
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """
    A single transformer block with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (Pre-LN)
    - Residual connections
    - Dropout

    Args:
        d_model: Model dimension (e.g., 768)
        n_heads: Number of attention heads (e.g., 12)
        d_ff: Feed-forward hidden dimension (e.g., 3072)
        dropout: Dropout probability (e.g., 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization (Pre-LN: before sublayers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (B, T, d_model)
            mask: Optional attention mask (T, T) or (B, 1, T, T)

        Returns:
            output: (B, T, d_model)
        """
        # First sublayer: Multi-head attention with residual
        # Pre-LN: Normalize first, then apply attention
        attn_out = self.attention(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            mask=mask
        )
        x = x + self.dropout1(attn_out)

        # Second sublayer: Feed-forward with residual
        # Pre-LN: Normalize first, then apply FFN
        ffn_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x
```

**Helper components:**

```python
class MultiHeadAttention(nn.Module):
    """From Module 02"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, d_model = query.shape

        # Linear projections and split into heads
        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ V

        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        out = self.W_o(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
        """
        x = self.linear1(x)         # (B, T, d_ff)
        x = self.activation(x)      # (B, T, d_ff)
        x = self.dropout(x)         # (B, T, d_ff)
        x = self.linear2(x)         # (B, T, d_model)
        return x
```

### Usage Example

```python
# Configuration
d_model = 768
n_heads = 12
d_ff = 3072
dropout = 0.1

# Create transformer block
block = TransformerBlock(d_model, n_heads, d_ff, dropout)

# Input
batch_size = 32
seq_len = 128
x = torch.randn(batch_size, seq_len, d_model)

# Optional: causal mask for autoregressive generation
mask = create_causal_mask(seq_len)

# Forward pass
output = block(x, mask=mask)

print(f"Input shape:  {x.shape}")      # (32, 128, 768)
print(f"Output shape: {output.shape}") # (32, 128, 768)

# Shape is preserved!
assert output.shape == x.shape
```

### Parameter Count

**For d_model=768, n_heads=12, d_ff=3072:**

```
Multi-Head Attention:
  W_q: 768 × 768 = 589,824
  W_k: 768 × 768 = 589,824
  W_v: 768 × 768 = 589,824
  W_o: 768 × 768 = 589,824
  Subtotal: 2,359,296 (~2.4M)

Feed-Forward Network:
  W₁: 768 × 3072 = 2,359,296
  b₁: 3,072
  W₂: 3072 × 768 = 2,359,296
  b₂: 768
  Subtotal: 4,722,432 (~4.7M)

Layer Normalization (×2):
  γ₁: 768
  β₁: 768
  γ₂: 768
  β₂: 768
  Subtotal: 3,072 (~3K)

Total per block: ~7.1M parameters
```

**Observation:** FFN contains ~2/3 of parameters!

**For 12-layer transformer:**
```
Total parameters: 12 × 7.1M ≈ 85M parameters
(Not counting embedding layers)
```

---

## Why This Architecture?

Let's understand the design decisions behind the transformer block.

### Why Combine Attention and FFN?

**Attention provides:**
- Information routing (what to focus on)
- Context aggregation (combining information)
- Relational reasoning (position-to-position dependencies)

**FFN provides:**
- Non-linear transformation (complex functions)
- Feature processing (per-position computation)
- Model capacity (knowledge storage)

**Together:**
- Attention: "Where should I look?"
- FFN: "What should I do with what I found?"

**Analogy:**

Think of building a summary from a document:

1. **Attention**: Read the document and identify important sentences
2. **FFN**: Process each important sentence to extract key points

Both steps are necessary!

### Why Two Sublayers (Not One, Not Three)?

**Why not just attention?**
- Insufficient capacity
- No position-wise processing
- Limited non-linearity

**Why not just FFN?**
- No information routing
- Each position processes independently
- Can't capture dependencies

**Why not three or more sublayers?**
- Diminishing returns
- Computational cost
- Training difficulty

**Empirical findings:**

| Configuration | Params per block | Perplexity | Training time |
|---------------|------------------|------------|---------------|
| Attention only | ~2.4M | 32.5 | 1.0× |
| FFN only | ~4.7M | 45.2 | 0.8× |
| Attn + FFN (standard) | ~7.1M | **21.2** | 1.2× |
| Attn + FFN + FFN | ~11.8M | 21.0 | 1.5× |

Two sublayers (Attn + FFN) is optimal!

### Why Layer Normalization?

**Without normalization:**
- Training instability
- Gradient explosion/vanishing
- Requires very careful initialization
- Sensitive to hyperparameters
- Often fails for deep models

**With normalization:**
- Stable training
- Consistent activation scales
- Easier optimization
- Robust to initialization
- Enables deeper models

**Alternative:** Batch normalization doesn't work well for sequences (variable length, position-dependent statistics).

### Why Residual Connections?

**Without residuals:**
- Vanishing gradients in deep networks
- Degradation problem (deeper = worse)
- Hard to train > 6 layers
- Optimization difficulty

**With residuals:**
- Direct gradient flow
- Easy to learn identity
- Ensemble of paths
- Can train 100+ layers successfully

**Empirical:** Transformers without residuals fail to train beyond 6-8 layers.

### Why Pre-LN Over Post-LN?

**Pre-LN advantages:**
- More stable gradient flow
- No warm-up required
- Higher learning rates possible
- Easier to tune
- Better for deep models (> 12 layers)

**Post-LN advantages:**
- Matches original paper
- Slightly better performance (with perfect tuning)
- Normalized output (can be beneficial)

**Modern practice:** Pre-LN is standard unless replicating specific baselines.

### Why d_ff = 4 × d_model?

**Empirically optimal:**
- Balances compute between attention and FFN
- Provides sufficient capacity
- Not too many parameters (overfitting)

**Alternatives:**
- 2× for lightweight models
- 8× for very large models (GPT-3 uses ~4.2×)

### Design Trade-offs

**Compute vs Capacity:**
```
Larger d_ff:
  + More capacity
  + Better performance
  - More compute
  - More memory
  - Risk of overfitting
```

**Depth vs Width:**
```
More layers:
  + Hierarchical features
  + Better compositionality
  - Longer training
  - More memory (activations)

Wider layers (larger d_model):
  + More representational power
  + Easier to train
  - More parameters
  - More compute per layer
```

**Standard configurations:**

```
Small model:
  d_model = 256, n_heads = 4, d_ff = 1024, n_layers = 6
  Params: ~10M

Base model:
  d_model = 512, n_heads = 8, d_ff = 2048, n_layers = 6
  Params: ~65M

Large model:
  d_model = 1024, n_heads = 16, d_ff = 4096, n_layers = 24
  Params: ~350M
```

---

## Implementation Considerations

### Efficient Implementation

#### 1. Fused Operations

**Naive:**
```python
x = self.linear1(x)
x = self.gelu(x)
x = self.dropout(x)
x = self.linear2(x)
```

**Optimized (fused):**
```python
# PyTorch 2.0+ with torch.compile
@torch.compile
def ffn_forward(x):
    return self.linear2(F.dropout(F.gelu(self.linear1(x)), p=0.1))
```

Fusion reduces kernel launches and memory traffic.

#### 2. Memory Optimization

**Gradient checkpointing:**

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, ..., use_checkpoint=False):
        self.use_checkpoint = use_checkpoint

    def forward(self, x, mask=None):
        if self.use_checkpoint and self.training:
            x = checkpoint(self._attention_sublayer, x, mask)
            x = checkpoint(self._ffn_sublayer, x)
        else:
            x = self._attention_sublayer(x, mask)
            x = self._ffn_sublayer(x)
        return x
```

**Trade-off:**
- Save ~30% memory during training
- ~20% slower (recompute in backward pass)

#### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast

with autocast():
    output = transformer_block(x)
```

**Benefits:**
- 2× faster training
- 2× less memory
- Minimal accuracy loss

**Important:** LayerNorm should stay in FP32 for stability:

```python
class LayerNorm(nn.Module):
    def forward(self, x):
        # Cast to FP32 for normalization
        x_fp32 = x.float()
        normalized = F.layer_norm(x_fp32, ...)
        return normalized.type_as(x)  # Cast back
```

### Initialization

**Good initialization is crucial for training stability.**

**Standard initialization (PyTorch defaults):**

```python
# Linear layers: Kaiming uniform
nn.Linear uses:
  W ~ Uniform(-√(1/d_in), √(1/d_in))

# LayerNorm
  γ = ones(d_model)  # Scale = 1
  β = zeros(d_model) # Shift = 0
```

**GPT-2 style initialization:**

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

# Apply to model
model.apply(init_weights)

# Scale residual projections (W_o, FFN output)
for name, param in model.named_parameters():
    if 'W_o.weight' in name or 'linear2.weight' in name:
        # Scale by 1/√(2*n_layers)
        param.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

**Why scale residual projections?**
- Prevents activation magnitude growth in deep networks
- Each layer contributes ~1/√n_layers to the output
- Stabilizes training for very deep models (> 24 layers)

### Dropout Placement

**Where to apply dropout in transformer block:**

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        # Dropout location 1: Attention weights
        attn_out = self.attention(x, mask)  # Dropout inside attention

        # Dropout location 2: After attention, before residual add
        x = x + self.dropout1(attn_out)

        # Dropout location 3: Inside FFN, after activation
        ffn_out = self.feed_forward(x)  # Dropout inside FFN

        # Dropout location 4: After FFN, before residual add
        x = x + self.dropout2(ffn_out)

        return x
```

**Typical dropout rates:**

```python
# Standard
attention_dropout = 0.1
residual_dropout = 0.1
ffn_dropout = 0.1

# Larger models (reduce overfitting)
attention_dropout = 0.1
residual_dropout = 0.1
ffn_dropout = 0.1

# Smaller datasets (more regularization)
attention_dropout = 0.2
residual_dropout = 0.1
ffn_dropout = 0.2
```

### Numerical Stability

**Common stability issues:**

#### 1. LayerNorm Underflow

```python
# Problem: var can be very small
var = x.var(dim=-1, keepdim=True)
x_norm = (x - mean) / torch.sqrt(var)  # Division by ~0!

# Solution: Add epsilon
var = x.var(dim=-1, keepdim=True)
x_norm = (x - mean) / torch.sqrt(var + 1e-5)  # Stable
```

#### 2. GELU Overflow in FP16

```python
# Problem: GELU can overflow in FP16
x_fp16 = torch.randn(1000).half()
gelu_fp16 = F.gelu(x_fp16)  # Can produce NaN!

# Solution: Use FP32 for activation
x_fp32 = x_fp16.float()
gelu_fp32 = F.gelu(x_fp32)
gelu_fp16 = gelu_fp32.half()  # Safe
```

#### 3. Gradient Clipping

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Common Mistakes

### Mistake 1: Wrong Normalization Placement

**Symptom:** Training instability, poor convergence.

**Cause:**
```python
# WRONG: Post-LN without warm-up
x = self.ln1(x + self.attention(x))  # Hard to train!
```

**Fix:**
```python
# CORRECT: Pre-LN (stable)
x = x + self.attention(self.ln1(x))  # Easy to train
```

### Mistake 2: Forgetting Residual Connections

**Symptom:** Gradient vanishing, model doesn't improve.

**Cause:**
```python
# WRONG: No residual
x = self.attention(self.ln1(x))  # Gradient can't flow!
x = self.ffn(self.ln2(x))
```

**Fix:**
```python
# CORRECT: With residuals
x = x + self.attention(self.ln1(x))  # Gradient highway
x = x + self.ffn(self.ln2(x))
```

### Mistake 3: Dimension Mismatch in Residual

**Symptom:** RuntimeError: The size of tensor a must match the size of tensor b.

**Cause:**
```python
# WRONG: Output dimension doesn't match input
x = x + self.some_layer_that_changes_dimension(x)
```

**Fix:**
```python
# CORRECT: Ensure dimensions match
assert x.shape == sublayer_output.shape, "Dimension mismatch!"
x = x + sublayer_output
```

### Mistake 4: Incorrect FFN Expansion

**Symptom:** Poor performance, underfitting.

**Cause:**
```python
# WRONG: No expansion (d_ff = d_model)
self.linear1 = nn.Linear(d_model, d_model)  # No capacity boost!
```

**Fix:**
```python
# CORRECT: 4× expansion
d_ff = 4 * d_model
self.linear1 = nn.Linear(d_model, d_ff)
self.linear2 = nn.Linear(d_ff, d_model)
```

### Mistake 5: Forgetting Dropout in Eval Mode

**Symptom:** Inconsistent inference results.

**Cause:**
```python
# WRONG: Dropout always on
self.dropout = nn.Dropout(0.1)
# ... during inference, dropout is still active
```

**Fix:**
```python
# CORRECT: Set model to eval mode
model.eval()  # Disables dropout
with torch.no_grad():
    output = model(x)

# Remember to set back to train mode for training
model.train()  # Enables dropout
```

### Mistake 6: Using BatchNorm Instead of LayerNorm

**Symptom:** Poor performance on sequences, instability.

**Cause:**
```python
# WRONG: BatchNorm for sequences
self.norm = nn.BatchNorm1d(d_model)  # Doesn't work well!
```

**Fix:**
```python
# CORRECT: LayerNorm for sequences
self.norm = nn.LayerNorm(d_model)  # Designed for sequences
```

### Mistake 7: Inefficient Self-Attention Calls

**Symptom:** Slow training.

**Cause:**
```python
# WRONG: Separate Q, K, V forward passes
q = self.attention(x)
k = self.attention(x)  # Redundant computation!
v = self.attention(x)
```

**Fix:**
```python
# CORRECT: Single forward pass (self-attention)
output = self.attention(query=x, key=x, value=x)
```

### Mistake 8: Not Initializing Properly

**Symptom:** Training divergence, NaN losses.

**Cause:**
```python
# WRONG: Using PyTorch defaults without adjustment
model = TransformerBlock(...)  # Default init may not be optimal
```

**Fix:**
```python
# CORRECT: Proper initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_weights)
```

### Debugging Checklist

**When transformer block isn't working:**

1. **Check shapes:**
   ```python
   print(f"Input: {x.shape}")
   print(f"After attention: {x.shape}")
   print(f"After FFN: {x.shape}")
   # All should be (B, T, d_model)
   ```

2. **Check gradient flow:**
   ```python
   loss = output.sum()
   loss.backward()
   for name, param in model.named_parameters():
       print(f"{name}: grad_norm = {param.grad.norm():.6f}")
   # Should see non-zero gradients everywhere
   ```

3. **Check normalization:**
   ```python
   # After LayerNorm, mean ≈ 0, std ≈ 1
   normalized = layer_norm(x)
   print(f"Mean: {normalized.mean(dim=-1)[0, 0]:.6f}")  # ≈ 0
   print(f"Std: {normalized.std(dim=-1)[0, 0]:.3f}")    # ≈ 1
   ```

4. **Check for NaN/Inf:**
   ```python
   def check_nan_inf(tensor, name):
       if torch.isnan(tensor).any():
           print(f"NaN detected in {name}!")
       if torch.isinf(tensor).any():
           print(f"Inf detected in {name}!")

   check_nan_inf(x, "input")
   check_nan_inf(attn_out, "attention output")
   check_nan_inf(ffn_out, "FFN output")
   ```

---

## Connection to Code

### Reference Implementation

Our implementation in `/Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/transformer_block.py` demonstrates all concepts:

**Expected structure:**

```python
class TransformerBlock(nn.Module):
    """
    Complete transformer block with:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization (Pre-LN architecture)
    - Residual connections
    - Dropout regularization

    Args:
        d_model: Model dimension (e.g., 768)
        n_heads: Number of attention heads (e.g., 12)
        d_ff: Feed-forward hidden dimension (typically 4 * d_model)
        dropout: Dropout probability (e.g., 0.1)
        use_pre_ln: If True, use Pre-LN architecture; else Post-LN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_pre_ln: bool = True
    ):
        super().__init__()

        self.use_pre_ln = use_pre_ln

        # Components
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, d_model)
            mask: Optional attention mask
        Returns:
            output: (B, T, d_model)
        """
        if self.use_pre_ln:
            return self._forward_pre_ln(x, mask)
        else:
            return self._forward_post_ln(x, mask)

    def _forward_pre_ln(self, x, mask):
        """Pre-LN: Normalize before sublayer"""
        # Attention sublayer
        x = x + self.dropout1(self.attention(self.ln1(x), self.ln1(x), self.ln1(x), mask))

        # FFN sublayer
        x = x + self.dropout2(self.feed_forward(self.ln2(x)))

        return x

    def _forward_post_ln(self, x, mask):
        """Post-LN: Normalize after residual"""
        # Attention sublayer
        x = self.ln1(x + self.dropout1(self.attention(x, x, x, mask)))

        # FFN sublayer
        x = self.ln2(x + self.dropout2(self.feed_forward(x)))

        return x
```

### Testing

**Comprehensive tests:**

```python
def test_transformer_block_shape():
    """Test output shape matches input shape."""
    d_model = 768
    block = TransformerBlock(d_model, n_heads=12, d_ff=3072)

    x = torch.randn(32, 128, d_model)
    output = block(x)

    assert output.shape == x.shape

def test_transformer_block_gradient_flow():
    """Test gradients flow through all components."""
    block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)

    x = torch.randn(2, 10, 256, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    # Check all parameters have gradients
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

def test_pre_ln_vs_post_ln():
    """Test both architectures produce outputs."""
    d_model = 256
    x = torch.randn(4, 20, d_model)

    block_pre = TransformerBlock(d_model, n_heads=4, d_ff=1024, use_pre_ln=True)
    block_post = TransformerBlock(d_model, n_heads=4, d_ff=1024, use_pre_ln=False)

    out_pre = block_pre(x)
    out_post = block_post(x)

    assert out_pre.shape == x.shape
    assert out_post.shape == x.shape
    # Outputs should differ (different architectures)
    assert not torch.allclose(out_pre, out_post)
```

---

## Summary

### Key Takeaways

1. **Transformer block combines four key components:**
   - Multi-head attention (information routing)
   - Feed-forward network (non-linear transformation)
   - Layer normalization (training stability)
   - Residual connections (gradient flow)

2. **Feed-forward networks provide:**
   - Non-linearity through activation functions (GELU)
   - Increased capacity through expansion (d_ff = 4 × d_model)
   - Position-wise transformation
   - Knowledge storage (key-value memories)

3. **Layer normalization ensures:**
   - Stable activation distributions
   - Better gradient flow
   - Faster, more robust training
   - Works well with variable-length sequences (unlike BatchNorm)

4. **Residual connections enable:**
   - Training deep networks (12-100+ layers)
   - Direct gradient highways
   - Easy identity learning
   - Ensemble of exponentially many paths

5. **Pre-LN vs Post-LN:**
   ```
   Post-LN: x = norm(x + sublayer(x))      [Original, harder to train]
   Pre-LN:  x = x + sublayer(norm(x))      [Modern standard, stable]
   ```

6. **Complete architecture (Pre-LN):**
   ```
   x₁ = x + Dropout(Attention(LayerNorm(x)))
   x₂ = x₁ + Dropout(FFN(LayerNorm(x₁)))
   ```

7. **Design rationale:**
   - Attention: What information to route where
   - FFN: How to process that information
   - LayerNorm: Keep training stable
   - Residuals: Enable deep stacking

### Configuration Guidelines

**Standard configurations:**

```python
# Tiny (for learning)
d_model = 128, n_heads = 4, d_ff = 512, n_layers = 4

# Small
d_model = 256, n_heads = 4, d_ff = 1024, n_layers = 6

# Base (BERT-base, GPT-2 small)
d_model = 768, n_heads = 12, d_ff = 3072, n_layers = 12

# Large (BERT-large)
d_model = 1024, n_heads = 16, d_ff = 4096, n_layers = 24

# XL (GPT-2 XL)
d_model = 1600, n_heads = 25, d_ff = 6400, n_layers = 48
```

**Rule of thumb:**
- `n_heads`: d_model / 64 (so d_k = 64)
- `d_ff`: 4 × d_model
- `dropout`: 0.1 (standard), 0.2 (small datasets)

### The Big Picture

The transformer block is remarkably simple yet powerful:

**Just two sublayers:**
1. Multi-head attention (communication between positions)
2. Feed-forward network (processing at each position)

**Plus two techniques:**
1. Layer normalization (stability)
2. Residual connections (depth)

**Result:**
- A modular, stackable building block
- Can train 100+ layers successfully
- State-of-the-art performance across domains
- The foundation of GPT, BERT, T5, and countless other models

**Why it works:**

Each component serves a specific, necessary function:
- Attention alone: Insufficient capacity, no non-linearity
- FFN alone: No information routing, no dependencies
- Without normalization: Training instability
- Without residuals: Gradient vanishing, can't go deep

Together, they form a synergistic architecture that is **greater than the sum of its parts**.

### Next Steps

Now that you understand transformer blocks:

1. **Embeddings (Module 04)**: How to convert tokens/positions to vectors
2. **Full Transformer (Module 05)**: Stacking blocks, encoder-decoder architecture
3. **Training (Module 06)**: Optimization, learning rate schedules, regularization
4. **Generation (Module 07)**: Sampling strategies, beam search, top-k/top-p
5. **Engineering (Module 08)**: Efficiency, quantization, deployment

### Further Reading

**Foundational Papers:**
- Vaswani et al. (2017): "Attention Is All You Need"
- He et al. (2016): "Deep Residual Learning for Image Recognition" (ResNets)
- Ba et al. (2016): "Layer Normalization"
- Xiong et al. (2020): "On Layer Normalization in the Transformer Architecture" (Pre-LN analysis)

**Analysis:**
- Geva et al. (2021): "Transformer Feed-Forward Layers Are Key-Value Memories"
- Liu et al. (2020): "Understanding the Difficulty of Training Transformers"
- Nguyen & Salazar (2019): "Transformers without Tears" (Training tips)

**Advanced Topics:**
- RMSNorm (simpler, faster alternative to LayerNorm)
- DeepNorm (stability for 1000+ layer transformers)
- Post-LayerNorm alternatives (Admin, ScaleNorm)
- Sparse FFNs (Mixture of Experts)

### Practice Exercises

1. **Implement from scratch**: Code the complete transformer block without looking at reference.

2. **Ablation study**: Train models with/without:
   - Feed-forward network
   - Layer normalization
   - Residual connections
   Compare performance.

3. **Pre-LN vs Post-LN**: Train identical models with both architectures, measure:
   - Training stability
   - Final performance
   - Gradient magnitudes

4. **FFN expansion ratio**: Test d_ff ∈ {1×, 2×, 4×, 8×} d_model, measure performance vs compute.

5. **Visualize**: Plot activation distributions, gradient norms, attention patterns through training.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Module:** 03 - Transformer Block
**Target Audience:** Developers who completed Modules 01-02
**Estimated Reading Time:** 60-75 minutes
