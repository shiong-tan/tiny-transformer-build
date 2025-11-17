# Module 06: Training Theory and Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading for Language Modeling](#data-loading-for-language-modeling)
3. [Cross-Entropy Loss Function](#cross-entropy-loss-function)
4. [The AdamW Optimizer](#the-adamw-optimizer)
5. [Learning Rate Schedules](#learning-rate-schedules)
6. [Gradient Clipping and Stability](#gradient-clipping-and-stability)
7. [Training Checkpointing](#training-checkpointing)
8. [The Training Loop](#the-training-loop)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Debugging Non-Decreasing Loss](#debugging-non-decreasing-loss)
11. [Complete Training Example](#complete-training-example)
12. [Summary](#summary)

---

## Introduction

Training a transformer language model requires coordinating several moving pieces: loading data efficiently, computing the right loss, optimizing with a sophisticated algorithm, scheduling learning rates, and monitoring progress. This module ties together everything from Modules 01-05 into a complete training pipeline.

**What you'll learn:**
- How to prepare text data for language modeling (batching, tokenization)
- Why cross-entropy loss is the right objective
- How AdamW optimizer works and why it's better than standard Adam
- Learning rate scheduling with warmup and cosine decay
- Gradient clipping to prevent training instability
- Checkpointing for saving and resuming training
- Monitoring metrics to diagnose training problems
- Debugging when loss doesn't decrease as expected

**Key Insight Preview:**

Training a transformer follows this simple loop:

```
1. Load batch of tokens
2. Forward pass: tokens → logits
3. Compute loss: cross-entropy(logits, targets)
4. Backward pass: compute gradients
5. Clip gradients (prevent explosion)
6. Optimizer step: update weights
7. Schedule: adjust learning rate
8. Repeat
```

Yet this simple loop contains many subtleties: how to structure data, why certain optimizers work better, what learning rate to use when, and how to detect when something goes wrong.

**Prerequisites:**
- Completed Modules 01-05 (attention, multi-head, transformer blocks, embeddings, full model)
- Understanding of gradient descent and backpropagation
- Familiarity with PyTorch training loops
- Basic probability (softmax, cross-entropy)

---

## Data Loading for Language Modeling

### The Language Modeling Objective

Language modeling predicts the next token given previous tokens. For a sequence of tokens:

```
[w₁, w₂, w₃, w₄, w₅]
```

We want to:
- At position 0: predict w₁ given nothing (or start token)
- At position 1: predict w₂ given [w₁]
- At position 2: predict w₃ given [w₁, w₂]
- At position 3: predict w₄ given [w₁, w₂, w₃]
- At position 4: predict w₅ given [w₁, w₂, w₃, w₄]

**Key insight:** The target at position t is the token at position t+1.

### Input-Target Alignment

For a sequence of N tokens, we create:

```python
# Sequence: [w₀, w₁, w₂, w₃, w₄]
input_ids  = [w₀, w₁, w₂, w₃, w₄]
target_ids = [w₁, w₂, w₃, w₄, w₅]
```

Wait, that's wrong. Let me reconsider. In a typical language modeling batch:

```python
# We read tokens: [w₀, w₁, w₂, w₃, w₄, w₅, ...]
# Sequence for training: [w₀, w₁, w₂, w₃, w₄]
input_ids  = [w₀, w₁, w₂, w₃, w₄]      # What we feed to model
target_ids = [w₁, w₂, w₃, w₄, w₅]      # What model should predict
```

The targets are just the inputs shifted by one position.

### Sliding Window Batching

**Method: Create overlapping windows from continuous text**

```
Raw text tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
seq_len = 4

Window 1: input [0, 1, 2, 3]       target [1, 2, 3, 4]
Window 2: input [4, 5, 6, 7]       target [5, 6, 7, 8]
Window 3: input [8, 9, 10, 11]     target [9, 10, 11, ?]
```

Each position in the token stream is used exactly once as an input, and once as a target (except the last position).

### Batch Construction

**Format:**

```python
# All data: (total_tokens,)
all_tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, ...])

# Create batch of windows
batch_size = 32
seq_len = 128

# Shape: (batch_size, seq_len)
batch_inputs = torch.stack([
    all_tokens[i : i + seq_len]
    for i in range(0, len(all_tokens) - seq_len, seq_len)
])

# Targets are shifted by 1
batch_targets = torch.stack([
    all_tokens[i + 1 : i + seq_len + 1]
    for i in range(0, len(all_tokens) - seq_len, seq_len)
])
```

**Shape example:**

```
all_tokens:     (N,)           e.g., (1,000,000,)
batch_inputs:   (B, T)         e.g., (32, 128)
batch_targets:  (B, T)         e.g., (32, 128)
```

### Dataset Implementation

```python
class TextDataset:
    """Load text and create language modeling batches."""

    def __init__(self, text_path: str, tokenizer, seq_len: int):
        """
        Args:
            text_path: Path to text file
            tokenizer: Function to convert text to token IDs
            seq_len: Sequence length for windows
        """
        # Load text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize
        self.tokens = tokenizer(text)  # (N,)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

        self.seq_len = seq_len
        self.n_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        """Number of complete windows."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one training example.

        Args:
            idx: Window index

        Returns:
            input_ids: (seq_len,)
            target_ids: (seq_len,)
        """
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.tokens[start : end]
        target_ids = self.tokens[start + 1 : end + 1]

        return input_ids, target_ids
```

### Shape Flow Through Data Loading

```
Text file: "The cat sat on the mat"
    ↓ Read
Raw text: "The cat sat on the mat"
    ↓ Tokenize (word-level for simplicity)
Token IDs: [1, 2, 3, 4, 1, 5]
    Meaning: [The=1, cat=2, sat=3, on=4, the=1, mat=5]
    ↓ Create windows (seq_len=3)
Window 1 input:  [1, 2, 3]   target: [2, 3, 4]
Window 2 input:  [4, 1, 5]   target: [1, 5, ?]
    ↓ Batch (batch_size=1 for clarity)
batch_input_ids:  (1, 3) = [[1, 2, 3]]
batch_target_ids: (1, 3) = [[2, 3, 4]]
    ↓ Model forward
logits: (1, 3, vocab_size)
    ↓ Compute loss
loss: scalar
```

### Important: Don't Leak Future Information

**Wrong approach:**

```python
# If we give model all tokens at once without causal masking:
# Position 0 can see positions 0, 1, 2, ...
# → Model learns to copy from future (cheating!)
```

**Right approach:**

```python
# Causal mask ensures position i can only see positions ≤ i
# Model genuinely predicts next token from previous only
```

The transformer model applies causal masking internally (done in Module 05 transformer blocks).

---

## Cross-Entropy Loss Function

### The Classification Problem

Language modeling is multi-class classification at each position:

```
Input at position i: hidden state h_i (shape: d_model)
Task: Classify which of vocab_size tokens comes next
Output: Probability distribution over vocab_size tokens
```

### Cross-Entropy Formula

For a single position, the loss is:

```
Loss = -log(P(correct_token))

Where:
  P(correct_token) = softmax(logits)[correct_token_id]
  logits = h_i @ W_output^T + b  (shape: vocab_size)
```

**Example:**

```
logits = [2.1, -1.3, 0.5, 3.2]  (scores for 4 vocab items)
correct_token_id = 3

softmax(logits) = [0.05, 0.01, 0.04, 0.90]
P(correct) = 0.90
Loss = -log(0.90) = 0.105
```

Lower loss means higher probability on correct token (good!).

### Batched Cross-Entropy

During training, we process many positions in parallel:

```python
# Model output
logits: (B, T, vocab_size)  # e.g., (32, 128, 10000)

# Target tokens
targets: (B, T)              # e.g., (32, 128)

# Cross-entropy expects:
# - Input: (N, C) where C=num_classes, N=total samples
# - Target: (N,) with class indices

# Reshape
logits_flat = logits.view(-1, logits.size(-1))    # (B*T, vocab_size)
targets_flat = targets.view(-1)                    # (B*T,)

# Compute loss (averaged over all positions)
loss = F.cross_entropy(logits_flat, targets_flat)
```

**Shape transformation:**

```
logits:  (32, 128, 10000)
  ↓ Reshape to (B*T, V)
logits_flat: (4096, 10000)   # 32*128 = 4096

targets: (32, 128)
  ↓ Reshape to (B*T,)
targets_flat: (4096,)

cross_entropy: scalar (average loss per position)
```

### Why Cross-Entropy for Language Modeling

**Advantages:**

1. **Proper probability**: Softmax converts logits to probabilities summing to 1
2. **Log penalty**: Log function penalizes low-confidence correct predictions heavily
3. **Numerical stability**: PyTorch's `F.cross_entropy` uses log-sum-exp trick internally
4. **Gradient flow**: Logarithm creates well-behaved gradients

**Alternative (not recommended):**

```python
# KL divergence - creates one-hot target
# Similar but more complex
target_probs = torch.zeros(batch_size, vocab_size)
target_probs[torch.arange(batch_size), targets] = 1.0
loss = F.kl_div(logits, target_probs)
```

Cross-entropy is simpler and more efficient.

### Per-Token vs Aggregate Loss

**PyTorch options:**

```python
# Option 1: Average over all positions (most common)
loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
# Result: Single scalar, averaged across all B*T positions

# Option 2: Sum over all positions
loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
# Result: Single scalar, summed across all B*T positions
# Useful if batches have variable-length sequences

# Option 3: No reduction (raw losses)
loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
# Result: (B*T,) - one loss per position
```

**Recommendation:** Use `reduction='mean'` for stable gradient magnitudes regardless of batch size.

### Perplexity Metric

A complementary metric to loss:

```
Perplexity = exp(Loss)
```

**Intuition:** Average "branching factor" - how confused the model is.

```
Loss = 0.0 → Perplexity = 1.0    (perfect predictions)
Loss = 2.3 → Perplexity ≈ 10.0   (average: 10 equally likely options)
Loss = 4.6 → Perplexity ≈ 100.0  (very confused)
```

Lower perplexity is always better.

---

## The AdamW Optimizer

### Standard Adam (Why It's Not Ideal)

Adam (Adaptive Moment Estimation) computes per-parameter learning rates:

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        [first moment: momentum]
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²      [second moment: adaptive LR]
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

**Problem with Adam + Weight Decay:**

Standard Adam applies weight decay incorrectly:

```python
# This is what Adam does:
loss = loss + (weight_decay / 2) * (params ** 2)

# But weight decay should be:
params = params * (1 - weight_decay * lr)
```

The first method couples weight decay with gradients, making it less effective with adaptive learning rates.

### AdamW: Decoupled Weight Decay

AdamW separates weight decay from gradient-based updates:

```python
# Update from gradients (like Adam)
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ = θ - α * m_t / (√v_t + ε)

# Then apply weight decay separately
θ = θ * (1 - λ * α)  # λ is weight_decay coefficient
```

This makes weight decay work consistently regardless of gradient statistics.

**Empirical benefit:** AdamW trains transformers better than Adam.

### AdamW Hyperparameters

**Typical settings for transformers:**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,              # Learning rate
    betas=(0.9, 0.95),    # Momentum and adaptive LR decay
    eps=1e-8,             # Numerical stability in denominator
    weight_decay=0.1      # L2 regularization strength
)
```

**Parameter meanings:**

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| `lr` | 3e-4 | Peak learning rate after warmup |
| `betas[0]` | 0.9 | Exponential decay for momentum |
| `betas[1]` | 0.95 | Exponential decay for second moment |
| `eps` | 1e-8 | Prevents division by zero |
| `weight_decay` | 0.1 | L2 regularization strength |

### Understanding Momentum (β₁)

Momentum accumulates gradients over time:

```
β₁ = 0.9 (standard):   Keep 90% of previous momentum + 10% new gradient
m_t = 0.9 * m_{t-1} + 0.1 * g_t
```

**Effect:**

```
Step 1: m = 0.1 * g₁
Step 2: m = 0.9 * (0.1 * g₁) + 0.1 * g₂ = 0.09 * g₁ + 0.1 * g₂
Step 3: m = 0.081 * g₁ + 0.09 * g₂ + 0.1 * g₃
...
```

Momentum creates exponentially-weighted average of recent gradients.

**Benefits:**
- Smooths noisy gradient signal
- Accelerates learning in consistent directions
- Helps escape local minima

### Understanding Adaptive Learning Rate (β₂)

Second moment adapts learning rate per parameter:

```
v_t = 0.999 * v_{t-1} + 0.001 * g_t²
θ = θ - α * m_t / √v_t
```

**Effect:**

```
Large gradients → large v_t → smaller effective update
Small gradients → small v_t → larger effective update
```

Each parameter gets its own learning rate based on gradient magnitude.

**Benefits:**
- Parameters with sparse gradients get boosted
- Parameters with dense gradients are damped
- Generally reduces need for learning rate tuning

### Implementation Pattern

```python
def training_step(model, optimizer, batch):
    """One optimizer step."""
    input_ids, target_ids = batch

    # Forward pass
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    # Backward pass
    optimizer.zero_grad()        # Clear old gradients
    loss.backward()              # Compute new gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip

    # Optimizer step
    optimizer.step()             # Update parameters
    scheduler.step()             # Update learning rate

    return loss.item()
```

---

## Learning Rate Schedules

### Why Learning Rate Matters

The learning rate controls step size in parameter updates:

```
θ_new = θ_old - lr * gradient
```

**Too high:** Overshoots optima, diverges, training unstable
**Too low:** Converges slowly, gets stuck in poor minima
**Just right:** Smooth convergence to good minima

For transformers, a fixed learning rate doesn't work well. We need a schedule.

### Three-Phase Learning Rate Schedule

**Phase 1: Warmup (0 → peak_lr)**
- Gradually increase learning rate from 0
- Prevents early divergence
- Allows gradients to stabilize
- Typically 1000-10000 steps depending on model size

**Phase 2: Cosine Decay (peak_lr → min_lr)**
- Smoothly decrease learning rate
- Follows cosine function from π to π/2
- Empirically works better than exponential decay
- Rest of training

**Phase 3: Final Region (min_lr)**
- Optional: very small learning rate at end
- Helps convergence to local minima
- Often peak_lr / 10 to peak_lr / 100

### Warmup: Linear Warmup

```
LR schedule:
    Step 0:           lr = 0
    Step warmup/2:    lr = peak_lr / 2
    Step warmup:      lr = peak_lr
    Step warmup+1:    begin cosine decay
```

**Warmup formula:**

```
If step <= warmup_steps:
    lr = (step / warmup_steps) * peak_lr
Else:
    lr = peak_lr * cosine_decay(step)
```

**Code:**

```python
def linear_warmup(step, warmup_steps, peak_lr):
    """Linear warmup from 0 to peak_lr."""
    if step <= warmup_steps:
        return (step / warmup_steps) * peak_lr
    return peak_lr
```

**Why linear?**
- Simple and effective
- Gradual enough to avoid divergence
- Allows gradient signal to stabilize

### Cosine Annealing: Smooth Decay

After warmup, learning rate follows cosine curve:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

Where:
- `t` is current step (after warmup)
- `T` is total steps (after warmup)
- `lr_max` is peak learning rate (after warmup)
- `lr_min` is minimum learning rate (can be 0 or small value)

**Shape:**

```
Learning Rate

peak_lr ┤ ╱───╲
        │╱       ╲___
min_lr  ├────────────╲___
        └─────────────────► Steps
        0    warmup     total
```

**Why cosine?**
- Smooth (no abrupt changes)
- Theoretically motivated (relates to curriculum learning)
- Works empirically better than exponential decay
- No plateau at end (always decreasing)

**Code:**

```python
import math

def cosine_annealing(step, warmup_steps, total_steps, peak_lr, min_lr=0):
    """Cosine annealing with warmup."""
    if step <= warmup_steps:
        # Linear warmup
        return (step / warmup_steps) * peak_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        # progress goes from 0 to 1
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cos_decay
```

### PyTorch Scheduler Integration

PyTorch provides scheduler classes:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Create scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=500,        # Initial period
    T_mult=2,       # Multiply period after restart
    eta_min=1e-5,   # Minimum learning rate
    last_epoch=-1
)

# Training loop
for step, batch in enumerate(dataloader):
    loss = forward_backward(model, batch)
    optimizer.step()
    scheduler.step()
```

Or use Hugging Face's implementation:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=50000
)

for step, batch in enumerate(dataloader):
    loss = forward_backward(model, batch)
    optimizer.step()
    scheduler.step()
```

### Choosing Warmup and Total Steps

**Warmup steps:**

```
Rule of thumb: 5-10% of total training steps
OR: 1000-5000 steps (whichever is larger)

Example:
    Total steps: 50,000
    Warmup: ~5,000 steps (10%)
```

**Total steps:**

```
total_steps = num_epochs * steps_per_epoch
            = num_epochs * (dataset_size / batch_size)

Example:
    Dataset: 1M tokens
    Batch size: 32
    Seq length: 128
    Steps per epoch: (1M / (32 * 128)) ≈ 244 steps
    Epochs: 10
    Total steps: 2,440 steps
```

### Common Learning Rate Values

```
Model Size          Typical Peak LR
─────────────────────────────────
Tiny (< 1M params)  1e-3  to 5e-3
Small (1-10M)       5e-4  to 1e-3
Medium (10-100M)    3e-4  to 5e-4
Large (> 100M)      1e-4  to 3e-4
```

Start with 3e-4 and adjust based on:
- Loss not decreasing → increase LR
- Training diverges (NaN) → decrease LR
- Slow convergence → increase LR slightly

---

## Gradient Clipping and Stability

### The Exploding Gradient Problem

During backpropagation through deep networks, gradients can grow exponentially:

```
Loss → Block N → ... → Block 1 → Input

If each block multiplies gradients by > 1.0:
gradient_final = gradient_initial × (scale₁ × scale₂ × ... × scaleN)
```

For a 12-layer transformer, small multipliers compound:

```
scales: [1.1, 1.05, 1.08, ...] (12 times)
total: 1.1^12 ≈ 3.1x amplification
```

Sometimes worse:

```
scales: [2.0, 2.0, 2.0] (3 times)
total: 2^3 = 8x amplification
```

When gradient norms become huge (e.g., > 100), parameter updates become massive, causing divergence (NaN loss).

### Gradient Clipping by Norm

**Solution: Cap the gradient norm**

```python
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

**What it does:**

```
1. Compute total gradient norm: ‖∇‖ = √(Σ(grad_i²))
2. If ‖∇‖ > max_norm:
   scale all gradients by max_norm / ‖∇‖
3. Else:
   leave gradients unchanged
```

**Example:**

```
Original gradients: [5.0, 3.0, 4.0]
Norm: √(25 + 9 + 16) = √50 ≈ 7.07
max_norm = 1.0

Scale factor: 1.0 / 7.07 ≈ 0.14
Clipped: [0.7, 0.42, 0.56]
New norm: √(0.49 + 0.18 + 0.31) ≈ 1.0
```

The clipped gradients have magnitude at most 1.0 while preserving direction.

### Gradient Clipping by Value

Alternative (not recommended for transformers):

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

Clips each gradient element individually to [-clip_value, clip_value].

**Why norm clipping is better:**
- Norm clipping preserves gradient direction
- Value clipping distorts gradient direction
- Norm clipping is standard for transformers

### Detecting Exploding Gradients

**Monitor during training:**

```python
def log_gradient_stats(model, step):
    """Log gradient statistics for debugging."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2) ** 2
    total_norm = total_norm ** 0.5

    print(f"Step {step}: gradient_norm = {total_norm:.4f}")

    # Flag if too large
    if total_norm > 100:
        print("  WARNING: Very large gradient norm!")
```

**What to look for:**

```
Normal training:        gradient_norm ≈ 0.1 to 1.0
Warning:               gradient_norm ≈ 10-100
Critical:              gradient_norm > 1000 (definitely clipping)
```

### Why Clipping Matters

Without clipping in deep transformers:

```
Step 1: loss = 4.0 (normal)
Step 2: loss = 3.9 (improving)
Step 3: loss = NaN (diverged!)
```

With clipping:

```
Step 1: loss = 4.0
Step 2: loss = 3.9
Step 3: loss = 3.8
...
Step 1000: loss = 0.5 (converged!)
```

Clipping prevents these catastrophic divergences.

### Implementation Pattern

```python
for step, batch in enumerate(dataloader):
    input_ids, target_ids = batch

    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients (CRITICAL)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step
    optimizer.step()
    scheduler.step()

    # Monitor
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

**Always clip! It's standard practice.**

---

## Training Checkpointing

### Why Checkpointing?

Training large models takes hours or days. We need to:
1. Save progress periodically
2. Resume from checkpoints if interrupted
3. Load best model after training
4. Experiment with different configurations

### What to Save

A checkpoint should contain:

```python
checkpoint = {
    'step': current_step,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'config': model_config,  # Can recreate model
    'loss': current_loss,
    'epoch': current_epoch,
}
```

Each component is restorable.

### Saving Checkpoints

```python
def save_checkpoint(model, optimizer, scheduler, config, step, loss, path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'epoch': step // steps_per_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
```

**Best practices:**

```python
# Save every N steps
if step % save_interval == 0:
    save_checkpoint(..., f"checkpoint_{step:06d}.pt")

# Save only best model
if loss < best_loss:
    best_loss = loss
    save_checkpoint(..., "checkpoint_best.pt")

# Keep only last K checkpoints (save disk space)
old_path = f"checkpoint_{step - K*save_interval:06d}.pt"
if Path(old_path).exists():
    Path(old_path).unlink()
```

### Loading Checkpoints

```python
def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint and resume."""
    checkpoint = torch.load(path)

    # Restore model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get metadata
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Loaded checkpoint from step {step}, epoch {epoch}")
    print(f"Last loss: {loss:.4f}")

    return step, epoch, loss

# In training loop
if resume_from_checkpoint:
    step, epoch, _ = load_checkpoint(
        model, optimizer, scheduler,
        "checkpoint_best.pt"
    )
    start_step = step
    start_epoch = epoch
else:
    start_step = 0
    start_epoch = 0
```

### Checkpoint Organization

**Recommended structure:**

```
checkpoints/
├── checkpoint_000000.pt  (step 0)
├── checkpoint_001000.pt  (step 1000)
├── checkpoint_002000.pt  (step 2000)
├── checkpoint_best.pt    (lowest validation loss)
└── checkpoint_latest.pt  (most recent)
```

### Disk Space Considerations

A checkpoint typically contains:

```
Model state:       model_size (e.g., 100MB for 50M params)
Optimizer state:   2-3x model_size (momentum + second moment)
Scheduler state:   ~1KB
Config:            ~1KB
────────────────
Total:             ~3-4x model_size
```

For a 50M parameter model:
```
Model:             50MB
Checkpoint:        200MB

10 recent checkpoints: 2GB
```

Keep only recent checkpoints unless storing important snapshots.

---

## The Training Loop

### Basic Training Loop Structure

```python
def train_one_epoch(model, train_loader, optimizer, scheduler, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0

    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update
        optimizer.step()
        scheduler.step()

        # Track
        total_loss += loss.item()
        num_steps += 1

        # Log
        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_steps
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Batch {batch_idx}: "
                  f"loss={avg_loss:.4f}, lr={lr:.6f}")

    return total_loss / num_steps
```

### Multi-Epoch Training

```python
def train(model, train_loader, val_loader, config):
    """Train for multiple epochs."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = config.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, config)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_best.pt')
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    return model
```

### Evaluation During Training

```python
def evaluate(model, val_loader, config):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_steps = 0

    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            total_loss += loss.item()
            num_steps += 1

    avg_loss = total_loss / num_steps
    return avg_loss
```

### Gradient Accumulation

For very large effective batch sizes without GPU memory:

```python
def train_with_accumulation(model, train_loader, optimizer, scheduler,
                            accumulation_steps, config):
    """Train with gradient accumulation."""
    model.train()

    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        # Backward (accumulate gradients)
        loss.backward()

        # Step every N batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

This allows effective batch_size = batch_size * accumulation_steps with less memory.

---

## Monitoring and Metrics

### Essential Metrics to Track

**1. Training Loss**

```python
# Per step
print(f"Step {step}: loss = {loss.item():.4f}")

# Smoothed (exponential moving average)
ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
```

Should decrease smoothly. If it increases or stays flat, something's wrong.

**2. Validation Loss**

```python
with torch.no_grad():
    val_loss = evaluate(model, val_loader)
print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
```

Should eventually plateau or increase (overfitting indicator).

**3. Perplexity**

```python
perplexity = math.exp(loss.item())
print(f"Perplexity: {perplexity:.1f}")
```

More interpretable than loss. Lower is better.

**4. Learning Rate**

```python
lr = optimizer.param_groups[0]['lr']
print(f"Learning rate: {lr:.6f}")
```

Should follow your schedule (warmup then decay).

**5. Gradient Norm**

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2) ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

Healthy range: 0.1 to 1.0. Very large (> 10) suggests clipping is active.

### Visualization Tools

**Tensorboard:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/')

writer.add_scalar('train/loss', loss.item(), step)
writer.add_scalar('train/perplexity', math.exp(loss.item()), step)
writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
writer.add_scalar('val/loss', val_loss, epoch)
```

Launch:
```bash
tensorboard --logdir=logs/
```

**Weights & Biases (wandb):**

```python
import wandb

wandb.init(project="transformer-training")

wandb.log({
    'train_loss': loss.item(),
    'val_loss': val_loss,
    'learning_rate': optimizer.param_groups[0]['lr'],
    'step': step
})
```

**Simple CSV logging:**

```python
import csv

with open('training_log.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([step, loss.item(), val_loss, lr])
```

### Metrics Summary

```python
class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []

    def record_train_loss(self, loss):
        self.losses.append(loss)

    def record_val_loss(self, loss):
        self.val_losses.append(loss)

    def record_lr(self, lr):
        self.learning_rates.append(lr)

    def get_summary(self):
        return {
            'mean_train_loss': np.mean(self.losses[-100:]),  # Last 100
            'mean_val_loss': np.mean(self.val_losses),
            'min_val_loss': np.min(self.val_losses),
            'current_lr': self.learning_rates[-1]
        }
```

---

## Debugging Non-Decreasing Loss

### Loss Doesn't Decrease at All

**Symptoms:**
- Loss stays constant or oscillates
- Model doesn't seem to be learning

**Diagnosis checklist:**

1. **Is the optimizer working?**
   ```python
   # Check parameter changes
   params_before = [p.clone() for p in model.parameters()]
   optimizer.step()
   params_after = [p for p in model.parameters()]

   # Some parameters should have changed
   for before, after in zip(params_before, params_after):
       if (before != after).any():
           print("✓ Optimizer updated parameters")
       else:
           print("✗ Optimizer did NOT update parameters")
   ```

2. **Is the learning rate zero?**
   ```python
   lr = optimizer.param_groups[0]['lr']
   if lr == 0:
       print("✗ Learning rate is zero!")
   elif lr < 1e-6:
       print("⚠ Learning rate is very small")
   ```

3. **Are gradients flowing?**
   ```python
   for name, param in model.named_parameters():
       if param.grad is None:
           print(f"✗ No gradient for {name}")
       elif param.grad.abs().max() < 1e-8:
           print(f"⚠ Tiny gradient for {name}")
   ```

4. **Is the data loading correctly?**
   ```python
   batch = next(iter(train_loader))
   input_ids, target_ids = batch
   print(f"Input shape: {input_ids.shape}")
   print(f"Input range: [{input_ids.min()}, {input_ids.max()}]")
   print(f"Input dtype: {input_ids.dtype}")
   # Should match model expectations
   ```

5. **Is the loss computed correctly?**
   ```python
   logits = model(input_ids)
   print(f"Logits shape: {logits.shape}")
   print(f"Logits range: [{logits.min():.1f}, {logits.max():.1f}]")

   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
   print(f"Loss: {loss.item():.4f}")
   # Loss should be finite and reasonable
   ```

### Loss Becomes NaN

**Symptoms:**
- Loss starts fine, then suddenly becomes NaN
- Training crashes

**Causes:**

1. **Exploding gradients**
   ```python
   # Solution: increase gradient clipping or reduce learning rate
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

2. **Invalid data**
   ```python
   assert not torch.isnan(input_ids).any()
   assert not torch.isinf(input_ids).any()
   assert input_ids.min() >= 0
   assert input_ids.max() < vocab_size
   ```

3. **Learning rate too high**
   ```python
   # Reduce peak learning rate
   peak_lr = 1e-4  # was 3e-4
   ```

4. **Bad initialization**
   ```python
   # Check initial weight magnitudes
   for name, param in model.named_parameters():
       if 'weight' in name:
           mean = param.abs().mean()
           std = param.abs().std()
           print(f"{name}: mean={mean:.4f}, std={std:.4f}")
   ```

### Loss Oscillates or Becomes Noisy

**Symptoms:**
- Loss decreases then increases unpredictably
- High variance in loss values

**Causes:**

1. **Learning rate too high**
   ```python
   # Reduce learning rate
   lr = 1e-4  # was 3e-4
   ```

2. **Batch size too small**
   ```python
   # Increase batch size (if memory allows)
   batch_size = 64  # was 32
   ```

3. **Noisy data or labels**
   ```python
   # Check data quality
   for i in range(10):
       sample = dataset[i]
       print(f"Sample {i}: {sample}")
   ```

### Loss Decreases Then Stops (Plateau)

**Symptoms:**
- Loss decreases initially, then flattens
- Model stops learning

**Possible causes:**

1. **Too long training without learning rate decay**
   ```python
   # Verify scheduler is active
   print(f"Scheduler state: {scheduler.state_dict()}")
   ```

2. **Model capacity too small**
   ```python
   # Increase model size
   d_model = 512  # was 256
   n_layers = 8   # was 4
   ```

3. **Learning rate schedule not working**
   ```python
   # Check learning rates over time
   for step in [0, 1000, 5000, 10000]:
       lr = cosine_schedule(step, warmup_steps=1000, total_steps=50000)
       print(f"Step {step}: lr = {lr:.6f}")
   ```

### Loss Diverges (Goes to Infinity)

**Symptoms:**
- Loss grows rapidly
- Training fails catastrophically

**Root causes:**

1. **Gradient clipping not applied or too high**
   ```python
   # MUST clip gradients
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

   # Or reduce max_norm if still diverging
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
   ```

2. **Learning rate way too high**
   ```python
   # Reduce significantly
   lr = 1e-5  # Start very small, increase if stable
   ```

3. **Initialization explodes**
   ```python
   def init_weights(model):
       for module in model.modules():
           if isinstance(module, nn.Linear):
               nn.init.normal_(module.weight, std=0.01)  # Smaller std
               if module.bias is not None:
                   nn.init.zeros_(module.bias)
   ```

### Systematic Debugging Process

When loss doesn't behave as expected:

```python
def debug_training(model, single_batch):
    """Systematic debugging of training."""
    input_ids, target_ids = single_batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    print("=== Data Check ===")
    print(f"Input IDs: {input_ids.shape}, range [{input_ids.min()}, {input_ids.max()}]")
    print(f"Target IDs: {target_ids.shape}, range [{target_ids.min()}, {target_ids.max()}]")

    print("\n=== Forward Pass ===")
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Logits: {logits.shape}, range [{logits.min():.1f}, {logits.max():.1f}]")

    print("\n=== Loss Computation ===")
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    print(f"Loss: {loss.item():.4f}, is finite: {math.isfinite(loss.item())}")

    print("\n=== Backward Pass ===")
    model.zero_grad()
    loss.backward()

    print("\n=== Gradient Check ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"{name}: mean={grad_mean:.6f}, max={grad_max:.6f}")
        else:
            print(f"{name}: NO GRADIENT")

    print("\n=== Optimizer Check ===")
    optimizer.step()
    print("✓ Optimizer step completed")

    print("\n=== Parameter Update Check ===")
    # (compare parameters before and after)
```

---

## Complete Training Example

### Full Training Script

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
import math
from pathlib import Path

class TextDataset(Dataset):
    """Simple language modeling dataset."""

    def __init__(self, text_path, tokenizer, seq_len):
        with open(text_path, 'r') as f:
            text = f.read()

        self.tokens = torch.tensor(tokenizer(text), dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = len(self.tokens) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        input_ids = self.tokens[idx : idx + self.seq_len]
        target_ids = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return input_ids, target_ids


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for step, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        # Update
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % config['log_interval'] == 0:
            avg_loss = total_loss / (step + 1)
            perplexity = math.exp(avg_loss)
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step + 1}: loss={avg_loss:.4f}, "
                  f"ppl={perplexity:.1f}, lr={lr:.6f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(model, train_loader, val_loader, config):
    """Complete training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    total_steps = config['epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device)
        val_perplexity = math.exp(val_loss)
        print(f"Val loss: {val_loss:.4f}, Perplexity: {val_perplexity:.1f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'model_best.pt')
            print(f"✓ Saved best model")

        # Regular checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f'model_epoch_{epoch + 1}.pt'
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


# Configuration
config = {
    'batch_size': 32,
    'seq_len': 128,
    'epochs': 10,
    'learning_rate': 3e-4,
    'weight_decay': 0.1,
    'warmup_steps': 1000,
    'grad_clip': 1.0,
    'log_interval': 100,
    'save_interval': 2,
}

# Simple character-level tokenizer
def char_tokenizer(text):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    return [char_to_idx[c] for c in text]

# Load data
train_dataset = TextDataset('data/train.txt', char_tokenizer, config['seq_len'])
val_dataset = TextDataset('data/val.txt', char_tokenizer, config['seq_len'])

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

# Create model (assume TinyTransformerLM is imported)
model = TinyTransformerLM(
    vocab_size=50,  # Adjust based on tokenizer
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1
)

# Train
trained_model = train(model, train_loader, val_loader, config)
```

---

## Summary

### Key Takeaways

1. **Data Loading**: Language modeling requires shifting targets by one position and batching into windows

2. **Cross-Entropy Loss**: Appropriate for multi-class classification at each position

3. **AdamW Optimizer**: Better than standard Adam due to decoupled weight decay

4. **Learning Rate Schedule**: Warmup prevents early divergence; cosine decay enables convergence

5. **Gradient Clipping**: Essential for preventing exploding gradients in deep networks

6. **Checkpointing**: Save and resume training checkpoints for long experiments

7. **Training Loop**: Coordinates forward/backward passes, optimization, and monitoring

8. **Monitoring Metrics**: Loss, perplexity, learning rate, and gradient norm reveal training health

9. **Debugging**: Systematic approach to diagnose when loss doesn't behave as expected

### Checklist for Training Your Model

- [ ] Implement TextDataset for language modeling
- [ ] Verify data loading produces correct shapes (B, T) for inputs and targets
- [ ] Create optimizer with AdamW and appropriate hyperparameters
- [ ] Implement learning rate schedule with warmup + cosine decay
- [ ] Add gradient clipping to prevent divergence
- [ ] Implement checkpointing for save/resume
- [ ] Monitor loss, perplexity, learning rate, gradient norm
- [ ] Run on single batch to verify forward/backward pass
- [ ] Train on small dataset and verify loss decreases
- [ ] Investigate if loss doesn't decrease using debugging checklist

### Common Training Commands

```python
# Inspect batch
batch = next(iter(train_loader))
input_ids, target_ids = batch
print(f"Input: {input_ids.shape}")  # Should be (B, T)

# Test forward pass
with torch.no_grad():
    logits = model(input_ids)
print(f"Logits: {logits.shape}")  # Should be (B, T, vocab_size)

# Compute loss
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
print(f"Loss: {loss.item():.4f}")

# Single training step
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

### Next Steps

**Module 07: Sampling & Generation**
- Implement autoregressive text generation
- Explore sampling strategies (greedy, temperature, top-k, nucleus)
- Complete the `generate()` method stub from Module 05

---

**Created:** November 2024
**For:** TinyTransformerBuild Educational Series
**Module:** 06 - Training
**Target Audience:** Developers learning to train transformers from scratch
