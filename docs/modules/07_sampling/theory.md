# Sampling & Generation: Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [The Generation Problem](#the-generation-problem)
3. [Autoregressive Language Generation](#autoregressive-language-generation)
4. [The Logits-to-Probabilities Pipeline](#the-logits-to-probabilities-pipeline)
5. [Greedy Decoding](#greedy-decoding)
6. [Temperature Sampling](#temperature-sampling)
7. [Top-K Sampling](#top-k-sampling)
8. [Nucleus (Top-P) Sampling](#nucleus-top-p-sampling)
9. [Combined Sampling Strategies](#combined-sampling-strategies)
10. [The Generation Loop](#the-generation-loop)
11. [KV Cache Optimization](#kv-cache-optimization)
12. [Trade-offs and Comparisons](#trade-offs-and-comparisons)
13. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
14. [Summary](#summary)

---

## Introduction

Once you've trained a transformer language model on text, the next natural question is: **How do we generate new text?**

Training teaches the model to predict the next token given a sequence. Generation leverages this by repeatedly asking "what's the next most likely token?" and building a sequence one token at a time. But this simple idea contains surprising complexity.

**What you'll learn:**
- The autoregressive generation paradigm: predicting one token at a time
- Why the naive approach (always picking the most likely token) produces repetitive, boring text
- How temperature controls the randomness of predictions
- Top-K sampling: excluding unlikely tokens to improve coherence
- Nucleus (top-P) sampling: the state-of-the-art approach used in modern LLMs
- Building a complete generation loop with proper tensor handling
- KV cache optimization: achieving 10-100× speedups in generation
- Trade-offs between speed, quality, and diversity
- Debugging common issues in text generation

**Key Insight Preview:**

The fundamental generation equation is deceptively simple:

```
logits = model(input_tokens)
probabilities = softmax(logits)
next_token = sample(probabilities)
```

Yet the choices you make in each step dramatically affect output quality:

```
Always pick most likely?  → Repetitive, deterministic
Always sample fairly?     → Incoherent, random
Apply temperature?        → Control randomness
Filter tokens?            → Improve coherence
Combine strategies?       → Production quality
```

**Prerequisites:**
- Completed Modules 01-06 (attention through training)
- Understanding of probability distributions and softmax
- Familiarity with PyTorch sampling operations (torch.multinomial)
- A trained transformer model available for inference

---

## The Generation Problem

### From Training to Inference

During training, the model learns a supervised task:

```
Input:  "The cat sat on"
Target: "the"
Loss:   Cross-entropy(logits, target)
```

The model receives the full input at once and produces logits for all positions in parallel (Figure 1).

During generation, the task is completely different:

```
1. Start with: "The cat"
2. Ask model: "What comes next?"
3. Model predicts probability distribution over tokens
4. We choose a token
5. Append to sequence: "The cat sat"
6. Repeat
```

We must generate **sequentially**, one token at a time. Why?

**Fundamental constraint:** At generation time, we don't know what future tokens will be. We can only predict the next one, commit to a choice, and let that inform future predictions.

### The Decoding Objective

**During training:** Minimize cross-entropy loss on ground truth targets.

**During generation:** Maximize quality metrics:
- **Coherence**: Do tokens logically follow?
- **Diversity**: Is output varied or repetitive?
- **Correctness**: Are facts accurate? (requires external knowledge)
- **Fluency**: Does it read naturally?
- **Determinism**: Is behavior predictable?

These objectives often conflict. For example:

```
High coherence + low diversity = Boring, repetitive text
Low coherence + high diversity = Incoherent gibberish
Balance needed!
```

### Problem Formulation

Given:
- A trained language model: `logits = f(tokens)`
- A starting sequence: `start_tokens = [w₁, w₂, ..., wₜ]`
- Generation parameters: `temperature, top_k, top_p, max_length`

**Find:** Sequence `[w₁, w₂, ..., wₜ, wₜ₊₁, ..., wₙ]`

**Constraints:**
- Each token is sampled from a probability distribution
- The distribution is computed from model logits
- Sampling strategies determine which tokens are likely to be chosen

---

## Autoregressive Language Generation

### The Autoregressive Property

A model is **autoregressive** if its predictions depend on previous predictions:

```
P(w₁, w₂, w₃, w₄ | context) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × P(w₄|w₁,w₂,w₃)
```

Chain rule of probability! The transformer naturally learns this decomposition during training.

**During generation, this becomes:**

```
Step 1: tokens = [w₁, w₂]
        logits = model(tokens)  # Predict position 2
        p(w₃ | w₁, w₂) = softmax(logits[-1])
        sample w₃

Step 2: tokens = [w₁, w₂, w₃]
        logits = model(tokens)  # Predict position 3
        p(w₄ | w₁, w₂, w₃) = softmax(logits[-1])
        sample w₄

Step 3: Continue...
```

### Sequence Building Example

Let's trace through a concrete example with small vocabulary:

```
Vocabulary: ["<START>", "the", "cat", "sat", "on", "mat", "<END>"]
IDs:         [0,        1,     2,     3,    4,    5,    6]

Starting tokens: [0, 1]  ("<START> the")

Step 1: Forward pass
  Input: [0, 1]                    Shape: (1, 2)
  Output logits: [-, -, L₂]        Shape: (1, 2, 7)
                                   Only L₂ is used!
  logits_at_position_2: [0.2, 0.8, 2.1, 0.3, 0.5, 0.1, 0.2]
  probabilities: softmax(logits) ≈ [0.02, 0.04, 0.80, 0.03, 0.05, 0.02, 0.04]

  Sample from this distribution → Pick token 2 ("cat") with probability 0.80
  New sequence: [0, 1, 2]

Step 2: Forward pass
  Input: [0, 1, 2]                 Shape: (1, 3)
  Output logits: [-, -, -, L₃]     Shape: (1, 3, 7)
                                   Only L₃ is used!
  logits_at_position_3: [0.1, 0.2, 0.3, 1.5, 0.8, 0.4, 0.6]
  probabilities: softmax(logits) ≈ [0.01, 0.02, 0.03, 0.60, 0.15, 0.06, 0.13]

  Sample from this distribution → Pick token 3 ("sat") with probability 0.60
  New sequence: [0, 1, 2, 3]

Step 3: Continue...
```

### Why Autoregressive?

This sequential dependency is both powerful and limiting:

**Advantages:**
- Natural fit for language (next token depends on context)
- Theoretically grounded in probability
- Efficient during generation (process one token at a time)

**Disadvantages:**
- Can't parallelize generation (unlike training where we compute all positions at once)
- Errors compound: a bad token choice affects all future predictions
- Can't generate multiple tokens in parallel easily

### Computational Cost

Let's compare training vs. generation cost for sequence of length T:

**Training:** Process entire sequence in one forward pass
```
Input: (batch, T, d_model)
Output: (batch, T, vocab_size)
Cost: O(T) forward pass
```

**Generation without optimization:** Recompute from scratch each step
```
Step 1: Forward with (1, 1) input  → Cost O(1)
Step 2: Forward with (1, 2) input  → Cost O(4)
Step 3: Forward with (1, 3) input  → Cost O(9)
        ...
Step T: Forward with (1, T) input  → Cost O(T²)

Total: O(1 + 4 + 9 + ... + T²) = O(T³)
```

This is why KV caching (Section 11) is crucial for practical generation.

---

## The Logits-to-Probabilities Pipeline

### From Raw Outputs to Probability Distributions

The transformer model's final output is **logits**: raw, unnormalized scores for each vocabulary token.

```python
logits = model(tokens)
# Shape: (batch_size, vocab_size) for single position
# Example: [0.2, 1.5, -0.3, 2.1, 0.5, ..., -1.2]
```

Logits have some properties:
- Can be any real number (negative, positive, large, small)
- Not constrained to sum to 1
- Not interpretable as probabilities directly

**We must convert them to probabilities.**

### The Softmax Function

The softmax function maps logits to a valid probability distribution:

```
softmax(logits)ᵢ = exp(logitsᵢ) / Σⱼ exp(logitsⱼ)
```

**Key properties:**
- All outputs in [0, 1]
- Sum to exactly 1.0
- Preserves ordering (highest logit → highest probability)
- Differentiable everywhere

**Example:**

```python
import torch.nn.functional as F

logits = torch.tensor([0.2, 1.5, -0.3, 2.1, 0.5])
probs = F.softmax(logits, dim=-1)

# Result: tensor([0.0521, 0.1424, 0.0314, 0.7069, 0.0672])
# Sum: 1.0000
```

Notice how the highest logit (2.1) gets the highest probability (0.71).

### Why Softmax?

**Alternative 1: Simple normalization**
```python
probs = logits / logits.sum()  # WRONG!
```
Problem: Negative logits become negative "probabilities".

**Alternative 2: ReLU normalization**
```python
probs = F.relu(logits) / F.relu(logits).sum()  # Wrong!
```
Problem: Doesn't preserve uncertainty; negative logits become zero.

**Why softmax is optimal:**
1. Mathematically grounded in maximum entropy (information theory)
2. Gives more weight to larger logits, less to smaller (exponential relationship)
3. Never produces zeros or negative values
4. Numerically stable (with implementation tricks)

### Numerical Stability: The Log-Sum-Exp Trick

In practice, computing softmax directly can cause numerical overflow:

```python
# Direct approach (UNSTABLE)
exp_logits = torch.exp(logits)  # Can overflow if logits are large!
probs = exp_logits / exp_logits.sum()

# Stable approach (what PyTorch uses)
max_logit = logits.max()
exp_shifted = torch.exp(logits - max_logit)  # Prevents overflow
probs = exp_shifted / exp_shifted.sum()

# Result is identical but numerically stable
```

**Why this works:** Subtracting the maximum logit ensures all exponents are ≤ 0, preventing overflow.

```
e^2000 = overflow!
e^0 = 1.0 (safe)
e^-1000 = 0 (underflow, but acceptable)
```

PyTorch's `F.softmax` does this automatically.

### Probability Distribution Properties

The output of softmax is a valid probability distribution P(token | history):

```python
# For single position
probs.shape = (vocab_size,)  # e.g., (10000,)
probs.sum() == 1.0
all(probs >= 0)
```

**Interpretation:** For each token in vocabulary, we have a probability it's the next token.

```
Token "the"   : probability 0.42
Token "cat"   : probability 0.15
Token "dog"   : probability 0.08
Token "runs"  : probability 0.07
... (9996 more tokens)
```

This distribution encodes the model's uncertainty about what comes next.

---

## Greedy Decoding

### The Simplest Strategy

Greedy decoding is straightforward: **always pick the token with the highest probability**.

```python
def greedy_sample(logits):
    """
    Greedy decoding: pick token with highest logit.

    Args:
        logits: (vocab_size,) tensor of logits

    Returns:
        token_id: int, index of token with highest logit
    """
    return torch.argmax(logits, dim=-1)
```

**Equivalent approaches:**

```python
# Method 1: Argmax logits
token = torch.argmax(logits, dim=-1)

# Method 2: Argmax probabilities (same result)
probs = F.softmax(logits, dim=-1)
token = torch.argmax(probs, dim=-1)

# Method 3: Top-1 sampling with temperature→0
token = torch.multinomial(F.softmax(logits / 1e-6, dim=-1), num_samples=1)
```

### Complete Greedy Generation

```python
@torch.no_grad()
def greedy_generate(model, start_tokens, max_new_tokens):
    """
    Generate text using greedy decoding.

    Args:
        model: Transformer language model
        start_tokens: (batch_size, seq_len) initial tokens
        max_new_tokens: int, number of tokens to generate

    Returns:
        generated: (batch_size, seq_len + max_new_tokens) token IDs
    """
    model.eval()
    tokens = start_tokens.clone()

    for _ in range(max_new_tokens):
        # Forward pass: compute logits for all positions
        logits = model(tokens)  # (batch_size, seq_len, vocab_size)

        # Only use logits at the last position
        logits_last = logits[:, -1, :]  # (batch_size, vocab_size)

        # Greedy: pick highest logit
        next_token = torch.argmax(logits_last, dim=-1, keepdim=True)  # (batch_size, 1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens
```

### Example: Generating with Greedy Decoding

```
Model vocabulary: ["<START>", "the", "cat", "sat", "on", "mat", "and", "dog", "<END>"]
Indices:          [0,        1,     2,     3,    4,    5,     6,    7,     8]

Start: tokens = [0, 1]  ("<START> the")

Iteration 1:
  Forward: model([0, 1])
  Logits at last position: [0.1, 0.2, 1.8, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2]
  argmax = 2 ("cat")
  tokens = [0, 1, 2]

Iteration 2:
  Forward: model([0, 1, 2])
  Logits at last position: [0.2, 0.3, 0.5, 2.1, 0.8, 0.3, 0.2, 0.1, 0.2]
  argmax = 3 ("sat")
  tokens = [0, 1, 2, 3]

Iteration 3:
  Forward: model([0, 1, 2, 3])
  Logits at last position: [0.1, 0.2, 0.3, 0.2, 1.5, 0.8, 0.3, 0.2, 0.1]
  argmax = 4 ("on")
  tokens = [0, 1, 2, 3, 4]

Iteration 4:
  Forward: model([0, 1, 2, 3, 4])
  Logits at last position: [0.2, 0.3, 0.2, 0.1, 0.3, 2.2, 0.2, 0.1, 0.1]
  argmax = 5 ("mat")
  tokens = [0, 1, 2, 3, 4, 5]

Final output: "the cat sat on mat"
```

### Advantages and Disadvantages

**Advantages:**
- ✓ Deterministic: Same input always produces same output
- ✓ Fast: Just one argmax operation
- ✓ Simple: Easy to implement and understand
- ✓ Useful for debugging: Reproducible behavior

**Disadvantages:**
- ✗ Repetitive: Mode-seeking (always picks the peak)
- ✗ Unnatural: Real language has randomness
- ✗ Boring: Same token patterns repeat
- ✗ Poor quality: Often produces degraded text

### Why Greedy Generation Fails

**Problem: Exposure bias**

During training, the model sees actual tokens at each position:
```
Input:  [w₁, w₂, w₃, ...]
Actual sequence that occurred in data
```

During greedy generation, the model sees its own predicted tokens:
```
Step 1: [w₁, w₂, ŵ₃]     ← ŵ₃ is model's prediction, not ground truth
Step 2: [w₁, w₂, ŵ₃, ŵ₄]  ← Compounding errors
```

Once the model makes a slightly wrong prediction, it's stuck: it must condition on that mistake for future predictions.

**Example of error compounding:**

```
Correct continuation: "The cat sat on the mat"
Greedy chooses:     "The cat"

Step 3: Model predicts "sat" (correct)
Step 4: Model predicts "on"  (correct)
Step 5: Model predicts "the" (correct)
Step 6: Model predicts "dog" (ERROR - should be "mat")
Step 7: Model predicts "runs" (ERROR - now stuck with dog)

Output: "The cat sat on the dog runs..."
```

The single error at step 6 cascades because the model never saw "the dog" in training context.

**Solution: Add randomness!**

Instead of always picking the most likely token, sample from the distribution. This allows the model to:
1. Make different predictions on repeated runs
2. Explore different continuations
3. Avoid getting stuck in repetitive loops

---

## Temperature Sampling

### The Temperature Parameter

Temperature is a hyperparameter that controls how **uniform** or **peaked** the probability distribution is.

```
τ (tau) ranges from 0 to ∞
- τ → 0:   Distribution becomes sharper (approaches greedy)
- τ = 1:   Standard softmax (no change)
- τ → ∞:   Distribution becomes flatter (approaches uniform random)
```

### Mathematical Formulation

Temperature is applied by scaling logits before softmax:

```
probs = softmax(logits / τ)
```

**Dividing by temperature:**
- Small τ (e.g., 0.7): Divide by small number → Logits get larger → Softmax becomes peakier
- Large τ (e.g., 1.5): Divide by large number → Logits get smaller → Softmax becomes flatter

### Concrete Example

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.2])

# τ = 0.5 (cooler, more deterministic)
probs_cool = F.softmax(logits / 0.5, dim=-1)
# tensor([0.0021, 0.0144, 0.0005, 0.9827, 0.0003])  # Very peaked!

# τ = 1.0 (no change)
probs_normal = F.softmax(logits / 1.0, dim=-1)
# tensor([0.0900, 0.2447, 0.0662, 0.5761, 0.0230])  # Moderate

# τ = 2.0 (warmer, more random)
probs_warm = F.softmax(logits / 2.0, dim=-1)
# tensor([0.1538, 0.2097, 0.1406, 0.3662, 0.1297])  # Flatter
```

Visualized:

```
Logits: [1.0, 2.0, 0.5, 3.0, 0.2]

τ=0.5:  ┃                    ██████████
        ┃                    ██████████
        ┃                    ██████████
        ┃
        ┃
        └─ Very sharp (deterministic)

τ=1.0:  ┃                ██████
        ┃            ██████████
        ┃        ██████
        ┃        ██████
        ┃
        └─ Moderate (balanced)

τ=2.0:  ┃        ███████
        ┃        ███████
        ┃        ███████
        ┃        ███████
        ┃        ███████
        └─ Flat (more random)
```

### Implementation

```python
def temperature_sample(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample from distribution with temperature scaling.

    Args:
        logits: (vocab_size,) or (batch_size, vocab_size) tensor
        temperature: float in (0, inf]
                    - < 1.0: More deterministic
                    - = 1.0: Standard sampling
                    - > 1.0: More random
        top_k: Optional, apply top-k filtering
        top_p: Optional, apply top-p filtering

    Returns:
        token: sampled token id
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Optional: apply top-k filtering
    if top_k is not None:
        scaled_logits = top_k_filter(scaled_logits, k=top_k)

    # Optional: apply top-p filtering
    if top_p is not None:
        scaled_logits = top_p_filter(scaled_logits, p=top_p)

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from distribution
    # torch.multinomial: samples from categorical distribution
    sampled_idx = torch.multinomial(probs, num_samples=1)

    return sampled_idx
```

### Temperature Effects on Generation Quality

**Low temperature (τ = 0.5):**
```
Input:   "The quick brown fox"
Output1: "jumps over the lazy dog. The quick brown fox..."  (repetition)
Output2: "jumps over the lazy dog. The quick brown fox..."  (identical)
Output3: "jumps over the lazy dog. The quick brown fox..."  (identical)

Behavior: Highly deterministic, repetitive
```

**Medium temperature (τ = 1.0):**
```
Input:   "The quick brown fox"
Output1: "jumps over the lazy dog and runs away"
Output2: "jumps over a fence and escapes"
Output3: "runs through the forest quickly"

Behavior: Varied and coherent
```

**High temperature (τ = 1.5):**
```
Input:   "The quick brown fox"
Output1: "purple elephant sings wednesday"
Output2: "hamburger quantum thinks basketball"
Output3: "crystalline sideways magnificently"

Behavior: Very random, often incoherent
```

### Semantic Interpretation

Think of temperature as **model confidence**:

- **Low τ**: Model is confident → Pick most likely tokens
- **High τ**: Model is uncertain → Sample more broadly

In practice:

```
τ = 0.7   → Creative but sensible (dialogue, storytelling)
τ = 1.0   → Balanced (general text generation)
τ = 1.3   → More random (brainstorming, creative writing)
```

### Why Not Higher Temperature?

At very high temperatures, the distribution becomes nearly uniform:

```
τ = 10.0: All tokens equally likely
         Probability ≈ 1 / vocab_size for each token
         Result: Random gibberish
```

This is useless for practical generation.

---

## Top-K Sampling

### The Problem with Naive Sampling

Temperature sampling alone has a subtle problem:

```
High temperature makes distribution flatter, but...
All tokens get some probability, including terrible ones!

Example vocabulary with model's probabilities:
- "the" (high probability): 0.35
- "cat" (medium probability): 0.20
- "runs" (medium probability): 0.18
- ... 9990 more tokens with tiny probabilities (0.0001 each)

Even with high temperature, the model might pick token 9998
just because we're sampling from the entire distribution.
```

The problem: The **tail** of the probability distribution contains tokens that should never be selected, even with randomness.

### The Solution: Truncate the Distribution

Top-K sampling solves this elegantly: **Only consider the K most likely tokens, ignore the rest.**

```
Step 1: Sort tokens by logits
Step 2: Keep only top K tokens
Step 3: Set all other logits to -∞
Step 4: Apply softmax and sample
```

### Mathematical Formulation

```
1. Compute logits from model
2. Get top-K values: logits_top_k = topk(logits, k=K)
3. Create filtered logits:
   filtered_logits[i] = logits[i] if logits[i] in top_k
                      = -∞ otherwise
4. Renormalized: probs = softmax(filtered_logits)
5. Sample: token ~ Categorical(probs)
```

**Key insight:** When we apply softmax to filtered logits, the -∞ values become 0 probabilities, and the remaining probabilities renormalize to sum to 1.

```python
# Softmax with -inf
logits = [-inf, 0.5, 1.0, -inf, 2.0]
exp(logits) = [0, 1.65, 2.72, 0, 7.39]
softmax(logits) = [0, 0.15, 0.25, 0, 0.60]  # Sums to 1!
```

### Implementation

```python
def top_k_sample(logits, k=50, temperature=1.0):
    """
    Sample from top-k tokens.

    Args:
        logits: (vocab_size,) tensor
        k: int, number of top tokens to keep
        temperature: float, temperature scaling

    Returns:
        token_id: sampled token
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(scaled_logits, k=k)

    # Create filtered logits tensor
    filtered_logits = torch.full_like(scaled_logits, float('-inf'))
    filtered_logits[top_k_indices] = top_k_values

    # Compute probabilities (softmax handles -inf correctly)
    probs = F.softmax(filtered_logits, dim=-1)

    # Sample from filtered distribution
    token_id = torch.multinomial(probs, num_samples=1)

    return token_id
```

### Concrete Example

```
Vocabulary size: 10000
Logits (unsorted): [0.2, 1.5, -0.3, 2.1, 0.5, ..., -1.2]

Step 1: Sort by magnitude
Sorted logits: [2.1, 1.5, 0.5, 0.2, -0.3, -0.7, ..., -1.2]
Sorted indices: [3, 1, 4, 0, 2, 7, ..., 9999]

Step 2: Keep only top-50
Top-50 logits: [2.1, 1.5, 0.5, 0.2, -0.3, ..., last_small_value]
Top-50 indices: [3, 1, 4, 0, 2, ..., some_id]

Step 3: Set rest to -inf
filtered_logits[3] = 2.1
filtered_logits[1] = 1.5
filtered_logits[4] = 0.5
filtered_logits[0] = 0.2
filtered_logits[2] = -0.3
filtered_logits[7] = -0.7
...
filtered_logits[9950..9999] = -inf  (9950 tokens get zero probability)

Step 4: Softmax
probs[3] ≈ 0.60  (highest)
probs[1] ≈ 0.20
probs[4] ≈ 0.10
... (47 more non-zero probabilities)
probs[9950..9999] = 0.0  (impossible to sample)

Step 5: Sample
Might pick token 3 (60% chance) or 1 (20% chance) or others (20% combined)
Never picks tokens 9950-9999!
```

### Why This Works

**Intuition:** The top-K tokens represent the model's "intended" outputs. Everything else is noise.

**Benefits:**
1. ✓ Prevents garbage tokens
2. ✓ Still allows randomness (sampling from top-K)
3. ✓ Much better quality than temperature alone
4. ✓ Fast (one topk operation)

**Typical values:**
```
K = 50     → Recommended for general text
K = 100    → More diversity
K = 10     → Very restrictive, deterministic
K = 1      → Equivalent to greedy (top-1)
```

---

## Nucleus (Top-P) Sampling

### The Limitation of Top-K

Top-K has a subtle problem:

```
Scenario A: Model is very certain
Logits:     [10.0, 9.0, 1.0, 0.5, 0.4, ...]
Top-50:     [10.0, 9.0, 1.0, 0.5, 0.4, ..., 0.01, 0.001]
            = ~50 tokens with reasonable probabilities

Scenario B: Model is uncertain
Logits:     [1.1, 1.0, 0.9, 0.8, 0.7, ...]
Top-50:     [1.1, 1.0, 0.9, 0.8, 0.7, ..., 0.01, 0.001]
            = 50 tokens, but many of them equally likely!

Problem: Same K for different levels of uncertainty?
```

When the model is **very certain**, maybe we only need top-5.
When the model is **uncertain**, maybe we need top-100.

**Solution:** Instead of fixed K, use dynamic threshold based on **cumulative probability**.

### The Core Idea

**Nucleus (Top-P) Sampling:** Keep the smallest set of tokens whose cumulative probability ≥ P.

```
Sort tokens by probability (descending)
Compute cumulative probabilities
Keep tokens until cumulative probability reaches P (e.g., 0.9)
Discard the rest
```

### Mathematical Formulation

```
1. Compute probabilities: probs = softmax(logits / temperature)
2. Sort by probability (descending)
3. Compute cumulative sum: cumsum_probs = cumsum(sorted_probs)
4. Create mask: keep if cumsum_probs <= P, else discard
5. Renormalize remaining probabilities
6. Sample
```

### Concrete Example

```
Vocabulary: [token_A, token_B, token_C, ... token_Z]
Logits:     [2.0,     1.5,     1.0,     ... -5.0]
After softmax:
Probabilities: [0.41, 0.27, 0.15, 0.08, 0.05, 0.02, 0.01, 0.001, ...]
(sorted descending)

Cumulative:    [0.41, 0.68, 0.83, 0.91, 0.96, 0.98, 0.99, 0.991, ...]

With P = 0.9:
Keep A (cumsum=0.41 ≤ 0.9) ✓
Keep B (cumsum=0.68 ≤ 0.9) ✓
Keep C (cumsum=0.83 ≤ 0.9) ✓
Keep D (cumsum=0.91 > 0.9)  ✗ STOP!

Selected tokens: {A, B, C}
Renormalized probabilities:
A: 0.41 / 0.83 ≈ 0.49
B: 0.27 / 0.83 ≈ 0.33
C: 0.15 / 0.83 ≈ 0.18

Sample from: {A (49%), B (33%), C (18%)}
```

### Implementation

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Sample from nucleus (top-p).

    Args:
        logits: (vocab_size,) tensor
        p: float in [0, 1], cumulative probability threshold
        temperature: float, temperature scaling

    Returns:
        token_id: sampled token
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Compute probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask: keep tokens until cumsum exceeds p
    # Important: ensure at least one token is always kept
    sorted_mask = cumsum_probs <= p
    sorted_mask[0] = True  # Keep highest probability token

    # Set probabilities of discarded tokens to 0
    sorted_probs[~sorted_mask] = 0.0

    # Renormalize (renormalization needed after masking)
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample from sorted distribution
    sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    # Map back to original indices
    token_id = sorted_indices[sorted_idx]

    return token_id
```

### Dynamic K Behavior

Top-P adapts K automatically:

```
Scenario A: Model very certain
Probs: [0.80, 0.15, 0.03, 0.01, 0.01]
Cumsum: [0.80, 0.95, ...]
With P=0.9: Keep first 2 tokens (cumsum=0.95)
Effective K = 2

Scenario B: Model uncertain
Probs: [0.20, 0.19, 0.18, 0.17, 0.16, ...]
Cumsum: [0.20, 0.39, 0.57, 0.74, 0.90]
With P=0.9: Keep first 5 tokens (cumsum=0.90)
Effective K = 5
```

Same hyperparameter (P=0.9), but different behavior based on model's confidence!

### Why Top-P is Superior

**Intuition:** The nucleus contains the tokens the model thinks are plausible. Everything else is outliers.

**Why it works better than top-K:**

1. ✓ Adapts to model's uncertainty
2. ✓ Respects semantic quality (tokens with low probability are excluded)
3. ✓ Robust across different model architectures
4. ✓ Used in state-of-the-art models (GPT-3, GPT-4, etc.)

**Disadvantages:**
- Slightly slower than top-K (requires sorting and cumsum)
- Still slow compared to greedy

**Typical values:**
```
P = 0.9   → Standard, most papers
P = 0.95  → More variety
P = 0.75  → More conservative
P = 1.0   → No filtering (equivalent to temperature alone)
```

---

## Combined Sampling Strategies

### Using Top-K AND Top-P Together

Interestingly, top-K and top-P can be applied **together** for even better results:

```python
def combined_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    Apply top-k filtering first, then top-p filtering.

    This combines the advantages of both strategies:
    - Top-K: Hard constraint, prevents garbage tokens
    - Top-P: Soft constraint, adapts to uncertainty
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # First: Apply top-k filtering
    top_k_values, top_k_indices = torch.topk(scaled_logits, k=top_k)
    filtered = torch.full_like(scaled_logits, float('-inf'))
    filtered[top_k_indices] = top_k_values

    # Convert to probabilities
    probs = F.softmax(filtered, dim=-1)

    # Second: Apply top-p filtering on the filtered distribution
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumsum_probs <= top_p
    sorted_mask[0] = True
    sorted_probs[~sorted_mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample
    sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    token_id = sorted_indices[sorted_idx]

    return token_id
```

**Why use both?**

```
Top-K alone: Might include 50 very different tokens
             If none are good, we're forced to pick from bad options

Top-P alone: Might give high probability to outlier tokens
             Requires probability to accumulate, but doesn't cap
             the number of tokens (could be 100+ if uncertain)

Combined: Top-K ensures max 50 tokens considered
          Top-P ensures we don't use all 50 (only nucleus)
          Best of both worlds!
```

### Practical Recommendation

For production use:

```python
# Recommended settings (from GPT-3 paper):
temperature = 0.7 - 1.0     # Moderate randomness
top_k = 50                   # Hard limit
top_p = 0.9                  # Soft nucleus

# For creative generation:
temperature = 1.2 - 1.5
top_k = 100
top_p = 0.95

# For faithful generation (fact-checking, code):
temperature = 0.5 - 0.7
top_k = 40
top_p = 0.9
```

---

## The Generation Loop

### Complete Generation Implementation

```python
@torch.no_grad()
def generate(
    model: nn.Module,
    start_tokens: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Generate text using specified sampling strategy.

    Args:
        model: Trained transformer language model
        start_tokens: (batch_size, seq_len) initial tokens
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (default 1.0 = no change)
        top_k: Keep only top-k tokens (None = no filtering)
        top_p: Keep tokens with cumsum prob <= p (None = no filtering)
        eos_token_id: Stop generation if this token is sampled
        pad_token_id: Pad token (usually not sampled)
        verbose: Print generation progress

    Returns:
        generated: (batch_size, seq_len + max_new_tokens) token IDs

    Shape examples:
        Input:  (batch_size=2, seq_len=4)
        Output: (batch_size=2, seq_len=4 + max_new_tokens)
    """
    model.eval()
    batch_size = start_tokens.shape[0]
    tokens = start_tokens.clone()

    # Track which sequences have ended (for batched generation)
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)

    for step in range(max_new_tokens):
        if verbose and step % 10 == 0:
            print(f"  Generating token {step}/{max_new_tokens}")

        # Forward pass: compute logits
        logits = model(tokens)  # (batch_size, seq_len, vocab_size)

        # Extract logits for last position only
        logits_last = logits[:, -1, :]  # (batch_size, vocab_size)

        # Sample next token for each item in batch
        next_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=tokens.device)

        for i in range(batch_size):
            if is_finished[i]:
                # Already finished, pad with pad_token
                next_tokens[i] = pad_token_id if pad_token_id is not None else 0
                continue

            # Get logits for this batch item
            logits_i = logits_last[i]  # (vocab_size,)

            # Apply temperature
            if temperature != 1.0:
                logits_i = logits_i / temperature

            # Apply top-k filtering
            if top_k is not None:
                values, indices = torch.topk(logits_i, k=top_k)
                filtered = torch.full_like(logits_i, float('-inf'))
                filtered[indices] = values
                logits_i = filtered

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create mask
                mask = cumsum_probs <= top_p
                mask[0] = True  # Always keep at least highest prob

                # Apply mask
                sorted_logits[~mask] = float('-inf')
                logits_i = sorted_logits

            # Convert to probabilities and sample
            probs_i = F.softmax(logits_i, dim=-1)

            # torch.multinomial requires probabilities, not logits
            next_token = torch.multinomial(probs_i, num_samples=1)

            next_tokens[i] = next_token

            # Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                is_finished[i] = True

        # Append to sequence
        tokens = torch.cat([tokens, next_tokens], dim=1)

    return tokens
```

### Batched Generation Example

```python
# Setup
model = TinyTransformerLM(vocab_size=10000, d_model=512, ...)
model.eval()

# Generate batch
batch_size = 4
seq_len = 32

# Start tokens for batch of 4 sequences
start_tokens = torch.tensor([
    [0, 1, 2, 3],        # Sequence 1
    [0, 4, 5, 6],        # Sequence 2
    [0, 7, 8, 9],        # Sequence 3
    [0, 10, 11, 12]      # Sequence 4
])  # Shape: (4, 4)

# Generate
generated = generate(
    model=model,
    start_tokens=start_tokens,
    max_new_tokens=128,
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    eos_token_id=2,  # End-of-sequence token
)  # Shape: (4, 4 + 128) = (4, 132)

# Shapes at each step
for step in range(5):
    print(f"Step {step}: tokens shape = {start_tokens[:, :4+step].shape}")
# Output:
# Step 0: tokens shape = torch.Size([4, 4])
# Step 1: tokens shape = torch.Size([4, 5])
# Step 2: tokens shape = torch.Size([4, 6])
# Step 3: tokens shape = torch.Size([4, 7])
# Step 4: tokens shape = torch.Size([4, 8])
```

### High-Level Generation Flow

```
Input: Start tokens [B, T]
  ↓
┌─────────────────────────────┐
│ For step = 1 to max_tokens: │
├─────────────────────────────┤
│ 1. Forward pass             │
│    logits = model(tokens)   │
│    [B, T, V]                │
│                             │
│ 2. Extract last position    │
│    logits = logits[:, -1]   │
│    [B, V]                   │
│                             │
│ 3. Sampling (per item)      │
│    for each in batch:       │
│      - Apply temperature    │
│      - Apply top-k/top-p    │
│      - Sample from dist.    │
│                             │
│ 4. Check for stop condition │
│    if token == EOS:         │
│      mark sequence as done  │
│                             │
│ 5. Append to sequence       │
│    tokens = [B, T+step]     │
│                             │
│ 6. (Optional) Handle padding│
│    for finished sequences   │
└─────────────────────────────┘
  ↓
Output: Generated tokens [B, T + max_tokens]
```

---

## KV Cache Optimization

### The Inefficiency Without Caching

Currently, our generation loop does this at each step:

```python
for step in range(max_new_tokens):
    logits = model(tokens)  # tokens shape: [B, T+step]
    next_token = sample(logits[:, -1, :])
    tokens = cat([tokens, next_token], dim=1)
```

**The problem:** At step 10, `tokens` has shape `[B, 20]`. The model must process all 20 positions through attention, even though we only care about position 20's output!

**Computational cost:**

```
Step 1: Process [B, 1+1] tokens through attention
        Attention computation: O((1+1)²) = O(1)

Step 2: Process [B, 1+2] tokens through attention
        Attention computation: O((1+2)²) = O(4)

Step 3: Process [B, 1+3] tokens through attention
        Attention computation: O((1+3)²) = O(9)

...

Step T: Process [B, 1+T] tokens through attention
        Attention computation: O((1+T)²) = O(T²)

Total: O(1 + 4 + 9 + ... + T²) = O(T³)
```

For T=512, this is **250,000** attention operations for one generation!

### The Key-Value Cache Solution

**Key insight:** Attention computation has this structure:

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

For position t:
- Q_t (query) is the new token embedding
- K_{1..t} (keys) are all past token embeddings
- V_{1..t} (values) are all past token embeddings

At step t+1:
- We need K_{1..t+1} and V_{1..t+1}
- K_{1..t} and V_{1..t} are identical to previous step!

**Solution:** Cache the K and V values, reuse them each step.

### Conceptual Example

```
Step 1: Generate position 2
  Input: [token_0, token_1]
  Need K,V from: [token_0, token_1]
  Compute and cache them
  Output: logits for position 2

Step 2: Generate position 3
  Input: [token_0, token_1, token_2]
  Need K,V from: [token_0, token_1, token_2]
  Reuse cached K,V from [token_0, token_1]
  Only compute K,V for [token_2]
  Output: logits for position 3

Step 3: Generate position 4
  Input: [token_0, token_1, token_2, token_3]
  Need K,V from: [token_0, token_1, token_2, token_3]
  Reuse cached K,V from [token_0, token_1, token_2]
  Only compute K,V for [token_3]
  Output: logits for position 4
```

### Data Structure for Cache

The cache stores K and V for each layer:

```python
cache = [
    {
        'k': torch.tensor([B, T_so_far, n_heads, d_k]),  # Keys for all layers
        'v': torch.tensor([B, T_so_far, n_heads, d_v]),  # Values for all layers
    }
    for _ in range(n_layers)
]
```

Or as nested dict:

```python
cache = {
    0: {'k': [...], 'v': [...]},  # Layer 0
    1: {'k': [...], 'v': [...]},  # Layer 1
    ...
    n_layers-1: {'k': [...], 'v': [...]},  # Last layer
}
```

### Modified Attention with Cache

Without caching:
```python
def attention(Q, K, V, mask=None):
    scores = Q @ K.T / sqrt(d_k)
    if mask: scores = scores + mask
    weights = softmax(scores)
    return weights @ V
```

With caching:
```python
def attention_cached(Q, K_new, V_new, cache, mask=None):
    # Concatenate with cached values
    K = cat([cache['k'], K_new], dim=time_axis)  # Reuse past
    V = cat([cache['v'], V_new], dim=time_axis)  # Reuse past

    # Compute attention (same as before)
    scores = Q @ K.T / sqrt(d_k)
    if mask: scores = scores + mask
    weights = softmax(scores)
    output = weights @ V

    # Update cache with new K, V
    cache['k'] = K
    cache['v'] = V

    return output, cache
```

### Speedup Comparison

**Without KV cache (current):**
```
Generation of 512 tokens:
Cost = O(512³) = ~134 million operations
Time ≈ 100 seconds (assuming 1M ops/ms)
```

**With KV cache:**
```
Generation of 512 tokens:
Cost = O(512) forward passes × O(512) each = O(512²) ≈ 260K operations
Time ≈ 0.26 seconds

Speedup: 100 / 0.26 ≈ 385×
```

This isn't exact, but shows the dramatic improvement. In practice, typical speedups are **10-100×**.

### Implementation Outline

```python
@torch.no_grad()
def generate_with_cache(
    model,
    start_tokens,
    max_new_tokens,
    cache=None,
    **sampling_kwargs
):
    """
    Generate with KV cache for efficiency.

    The model must support cache in its forward pass.
    """
    tokens = start_tokens.clone()

    for _ in range(max_new_tokens):
        # Forward pass WITH cache
        logits, cache = model(
            tokens[:, -1:],  # ONLY new token!
            cache=cache
        )

        # Sample next token
        next_token = sample(logits, **sampling_kwargs)

        # Append
        tokens = cat([tokens, next_token], dim=1)

    return tokens, cache
```

**Key difference:** Input is `tokens[:, -1:]` (only last token) instead of `tokens` (all tokens).

### Cache Memory Trade-off

**Memory usage with cache:**

```python
# Per layer, per batch item, per head
K_cache = [B, T, n_heads, d_k]
V_cache = [B, T, n_heads, d_v]

# With T=512, B=1, n_heads=8, d_k=64:
cache_size = 2 × 1 × 512 × 8 × 64 = 524,288 floats
            ≈ 2 MB per layer
            ≈ 12 MB for 6 layers

# Acceptable for single request, but with large batches:
# B=32 → 384 MB per request
# Can add up if serving many users!
```

**Trade-off:**
- ✓ Massive speedup (10-100×)
- ✗ Requires extra memory (linear in sequence length)

In practice, virtually all production systems use KV cache.

---

## Trade-offs and Comparisons

### Sampling Strategies Head-to-Head

| Aspect | Greedy | Temperature | Top-K | Top-P | Top-K+Top-P |
|--------|--------|-------------|-------|-------|------------|
| **Deterministic** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Speed** | Fastest | Fast | Fast | Medium | Medium |
| **Quality** | Poor | Good | Very Good | Excellent | Excellent |
| **Repetition** | High | Low | Very Low | Very Low | Very Low |
| **Controllability** | None | Temperature | K value | P value | Both |
| **Reproducible** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Use Case** | Debug | Creative | General | Production | Production |

### Quality Comparison

Let's compare outputs from same start prompt:

**Prompt:** "The scientist discovered a new"

**Greedy (deterministic):**
```
The scientist discovered a new way to make money.
The scientist discovered a new way to make money.
The scientist discovered a new way to make money.
(repetitive, boring)
```

**Temperature (τ=0.8):**
```
The scientist discovered a new compound that can heal infections.
The scientist discovered a new species of frog in the Amazon.
The scientist discovered a new method for time travel.
(varied, reasonable)
```

**Top-K (K=50):**
```
The scientist discovered a new element in the periodic table.
The scientist discovered a new theorem in mathematics.
The scientist discovered a new treatment for cancer.
(coherent, high quality)
```

**Top-P (P=0.9):**
```
The scientist discovered a new form of renewable energy.
The scientist discovered a new fossil in the ancient ruins.
The scientist discovered a new virus in the deep ocean.
(excellent quality, natural variation)
```

### When to Use Each Strategy

**Greedy Decoding:**
- Debugging generation code
- Need determinism for testing
- Want baseline quality
- Not recommended for production

**Temperature Sampling:**
- Standalone, without other filters: rarely recommended
- Combined with top-K or top-P: good approach

**Top-K Sampling:**
- General-purpose text generation
- When you want to limit vocabulary without probability threshold
- Slightly easier to tune than top-P

**Top-P (Nucleus) Sampling:**
- Production systems (OpenAI uses this)
- When you want to adapt to model's uncertainty
- Most modern LLMs use this
- Best overall quality

**Combined Top-K + Top-P:**
- Maximum quality and safety
- Production-grade generation
- Recommended by OpenAI and others
- Slight performance overhead but worth it

### Hyperparameter Tuning Guidelines

| Use Case | Temperature | Top-K | Top-P |
|----------|-------------|-------|-------|
| Creative writing | 1.2-1.5 | 50-100 | 0.9-0.95 |
| General text | 0.8-1.0 | 40-50 | 0.9 |
| Faithful generation | 0.5-0.7 | 30-40 | 0.85-0.9 |
| Question answering | 0.7 | 40 | 0.9 |
| Code generation | 0.5 | 50 | 0.95 |
| Dialogue | 0.9 | 40 | 0.9 |

### Speed vs. Quality Trade-off

```
                Speed
                  ↑
         Greedy   │
                  │     Temperature
                  │     + Top-K/P
                  │
    With KVCache  │
         10-100×  │        Top-P
         faster   │
                  ├─────────────────→ Quality
                  │
              Temperature alone
              (moderate speed/quality)

                  With KV Cache:
                  All strategies become
                  nearly real-time
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Numerical Instability in Softmax

**Problem:**
```python
def bad_softmax(logits):
    return torch.exp(logits) / torch.exp(logits).sum()
```

Logits can be large (e.g., 1000), and exp(1000) overflows!

**Solution:**
```python
def good_softmax(logits):
    max_logit = logits.max()
    exp_shifted = torch.exp(logits - max_logit)
    return exp_shifted / exp_shifted.sum()
```

PyTorch's `F.softmax` does this automatically.

### Pitfall 2: Forgetting to Set Model to Eval Mode

**Problem:**
```python
def generate(model, tokens):
    # Missing: model.eval()
    logits = model(tokens)
    ...
```

Model may use dropout, batch norm, etc., which behave differently in training vs. eval!

**Solution:**
```python
@torch.no_grad()
def generate(model, tokens):
    model.eval()  # Set to evaluation mode
    logits = model(tokens)
    ...
```

### Pitfall 3: Running out of Memory with Long Sequences

**Problem:**
```python
def naive_generate(model, tokens, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(tokens)  # tokens gets longer each iteration
        next_token = sample(logits[:, -1, :])
        tokens = cat([tokens, next_token], dim=1)  # Keep growing!
```

After 512 new tokens, `tokens` shape becomes huge.

**Solution:** Use KV cache (Section 11) to avoid reprocessing.

### Pitfall 4: Sampling from Logits Instead of Probabilities

**Problem:**
```python
def bad_sample(logits):
    # torch.multinomial requires probabilities, not logits!
    return torch.multinomial(logits, num_samples=1)
```

Result: Incorrect probability distribution! Negative logits become nonsensical.

**Solution:**
```python
def good_sample(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Pitfall 5: Incorrect Top-K Implementation

**Problem:**
```python
def bad_top_k(logits, k=50):
    top_k_logits, _ = torch.topk(logits, k=k)
    # Missing: Need to zero out other logits!
    probs = F.softmax(top_k_logits, dim=-1)  # Wrong size!
    return torch.multinomial(probs, num_samples=1)
```

The returned probabilities have size K, not vocab_size!

**Solution:**
```python
def good_top_k(logits, k=50):
    top_k_vals, top_k_indices = torch.topk(logits, k=k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered[top_k_indices] = top_k_vals
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Pitfall 6: Top-P Mask Off-by-One Error

**Problem:**
```python
def bad_top_p(logits, p=0.9):
    sorted_probs, _ = torch.sort(logits, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > p  # Should be <=!
    # Now we KEEP tokens with cumsum > p, which is backwards!
```

**Solution:**
```python
def good_top_p(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cumsum <= p
    mask = cumsum <= p
    mask[0] = True  # Always keep top token

    sorted_logits[~mask] = float('-inf')
    ...
```

### Pitfall 7: Forgetting @torch.no_grad() Decorator

**Problem:**
```python
def generate(model, tokens, max_new_tokens):
    # Missing @torch.no_grad()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        ...
```

This computes gradients during generation (wasteful!). Might run out of memory.

**Solution:**
```python
@torch.no_grad()
def generate(model, tokens, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(tokens)
        ...
```

The `@torch.no_grad()` decorator disables gradient computation, saving memory and time.

### Pitfall 8: Batch Processing Indexing Errors

**Problem:**
```python
def generate_batch(model, start_tokens):  # (B, T)
    tokens = start_tokens
    for _ in range(max_new_tokens):
        logits = model(tokens)  # (B, T, V)
        next_token = argmax(logits[:, -1, :])  # (B,)
        tokens = cat([tokens, next_token], dim=1)  # Wrong!
```

`next_token` has shape (B,), but we need (B, 1) for concatenation.

**Solution:**
```python
def generate_batch(model, start_tokens):  # (B, T)
    tokens = start_tokens
    for _ in range(max_new_tokens):
        logits = model(tokens)  # (B, T, V)
        next_token = argmax(logits[:, -1, :], keepdim=True)  # (B, 1)
        tokens = cat([tokens, next_token], dim=1)  # Correct!
```

---

## Summary

### Key Takeaways

1. **Autoregressive Generation:** Transformers generate one token at a time, using previous tokens to predict the next one. Simple loop, but contains many implementation details.

2. **Logits to Probabilities:** Use softmax to convert model logits to valid probability distributions. Use numerical stability tricks (log-sum-exp).

3. **Greedy Decoding:** Always pick most likely token. Fast and deterministic, but produces boring, repetitive text. Not recommended for production.

4. **Temperature Sampling:** Scale logits before softmax to control randomness. Low temperature = deterministic, high = random. Useful but needs other filters for quality.

5. **Top-K Sampling:** Only keep top-K most likely tokens, ignore the rest. Prevents garbage tokens, still allows randomness. Good general-purpose approach.

6. **Top-P (Nucleus) Sampling:** Keep smallest set of tokens with cumulative probability ≥ P. Adapts to model's uncertainty. Used in GPT-3, GPT-4, and other state-of-the-art models.

7. **Combined Strategies:** Using top-K + top-P + temperature together provides best quality. Recommended for production.

8. **KV Cache Optimization:** Cache key and value tensors from previous steps to avoid recomputing attention. Provides 10-100× speedup. Essential for practical generation.

9. **Trade-offs:** Speed vs. quality, determinism vs. variety, memory vs. time. Choose strategy based on use case.

10. **Debugging:** Use greedy decoding for determinism and debugging. Check tensor shapes carefully. Remember @torch.no_grad() and model.eval().

### Conceptual Hierarchy

```
┌────────────────────────────────┐
│ Autoregressive Generation      │ (Core concept: sequential prediction)
└─────────────────┬──────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
    ┌───▼────────┐   ┌──────▼────────┐
    │ Sampling   │   │ Optimization  │
    │ Strategies │   │ (KV Cache)    │
    └───┬────────┘   └───────────────┘
        │
    ┌───┴──────────────────────────┬────────┐
    │                              │        │
┌───▼───────┐  ┌──────────────┐  ┌┴────┐  ┌┴──────┐
│  Greedy   │  │ Temperature  │  │Top-K│  │Top-P  │
│ (Baseline)│  │(Randomness)  │  │     │  │(State-│
└───────────┘  └──────────────┘  │     │  │of-Art)│
                                  └─────┘  └───────┘
```

### Next Steps

- Implement all sampling strategies
- Build complete generation loop
- Profile and optimize with KV cache
- Generate text from your trained model
- Experiment with different hyperparameters
- Move to Module 08: Engineering Practices

---
