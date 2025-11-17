# Module 07: Sampling & Generation

Implementing various sampling strategies to generate text from your trained transformer - from greedy decoding to nucleus sampling.

## What You'll Learn

1. **Greedy Sampling** - Always pick most likely token
2. **Temperature Sampling** - Controlling randomness
3. **Top-K Sampling** - Sample from K most likely tokens
4. **Top-P (Nucleus) Sampling** - Sample from cumulative probability mass
5. **Generation Loop** - Autoregressive text generation
6. **KV Caching** - Optimizing inference speed

## Prerequisites

- ✓ Completed Module 06 (Training)
- ✓ Trained model available
- ✓ Understanding of probability distributions

## Generation Process

```
Start tokens: "The cat"
    ↓
┌──────────────────────────────────┐
│ Model Forward                    │
│  Input: [5, 12]                  │
│  Output: logits for position 2   │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ Apply Sampling Strategy          │
│  logits → probabilities → sample │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ Sample Next Token: "sat"         │
│  token_id = 8                    │
└──────────────────────────────────┘
    ↓
Append to sequence: [5, 12, 8]
    ↓
Repeat until max_length or <EOS>
```

## Sampling Strategies

### 1. Greedy Decoding
**Always pick the most probable token**

```python
def greedy_sample(logits):
    # logits: (vocab_size,)
    return torch.argmax(logits, dim=-1)
```

**Pros**: Deterministic, fast
**Cons**: Repetitive, boring text

### 2. Temperature Sampling
**Control randomness with temperature τ**

```python
def temperature_sample(logits, temperature=1.0):
    # Higher temperature = more random
    # Lower temperature = more deterministic
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

- **τ → 0**: Approaches greedy (peaked distribution)
- **τ = 1**: Standard softmax
- **τ > 1**: Flatter distribution (more random)

### 3. Top-K Sampling
**Sample from K most likely tokens**

```python
def top_k_sample(logits, k=50):
    # Keep only top-k logits, set rest to -inf
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[sampled_idx]
```

**Typical K**: 40-50
**Effect**: Cuts off unlikely tokens, more coherent than pure temperature

### 4. Top-P (Nucleus) Sampling
**Sample from smallest set of tokens with cumulative probability ≥ p**

```python
def top_p_sample(logits, p=0.9):
    # Sort by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Remove tokens with cumulative probability > p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[0] = False  # Keep at least one

    # Set removed tokens to -inf
    filtered_logits = sorted_logits.clone()
    filtered_logits[sorted_indices_to_remove] = float('-inf')

    # Sample from filtered distribution
    final_probs = F.softmax(filtered_logits, dim=-1)
    sampled_idx = torch.multinomial(final_probs, num_samples=1)
    return sorted_indices[sampled_idx]
```

**Typical P**: 0.9-0.95
**Best practice**: Often used in modern LLMs (GPT-3, etc.)

## Complete Generation Loop

```python
@torch.no_grad()
def generate(
    model,
    start_tokens,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    """
    Generate text autoregressively.

    Args:
        model: Trained transformer model
        start_tokens: (batch_size, seq_len) initial tokens
        max_new_tokens: How many tokens to generate
        temperature: Sampling temperature
        top_k: K for top-k sampling (None = disabled)
        top_p: P for nucleus sampling (None = disabled)

    Returns:
        generated: (batch_size, seq_len + max_new_tokens)
    """
    model.eval()
    tokens = start_tokens

    for _ in range(max_new_tokens):
        # Get logits for last position
        logits = model(tokens)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size) - last position

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            logits = top_k_filter(logits, k=top_k)

        # Apply top-p filtering
        if top_p is not None:
            logits = top_p_filter(logits, p=top_p)

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)  # (B, T+1)

    return tokens
```

## KV Caching Optimization

**Problem**: Recomputing attention for all previous tokens is wasteful

**Solution**: Cache key and value tensors from previous steps

```python
# Without caching: O(T²) per token
for t in range(max_tokens):
    logits = model(tokens[:, :t+1])  # Recompute all positions!

# With caching: O(T) per token
cache = None
for t in range(max_tokens):
    logits, cache = model(tokens[:, t:t+1], cache=cache)  # Only new token!
```

**Speedup**: 10-100× faster generation for long sequences

## What You'll Implement

```python
# 1. Sampling functions
def greedy_sample(logits): ...
def temperature_sample(logits, temperature): ...
def top_k_sample(logits, k): ...
def top_p_sample(logits, p): ...

# 2. Complete generation
class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)

        # Generate
        generated_tokens = generate(
            self.model,
            tokens,
            max_tokens,
            temperature,
            top_k,
            top_p
        )

        # Decode to text
        return self.tokenizer.decode(generated_tokens)

# 3. KV Cache (Advanced)
class CachedTransformer(nn.Module):
    def forward(self, tokens, cache=None):
        # Use cached K, V for previous positions
        # Only compute attention for new position
        pass
```

## Sampling Comparison

| Strategy | Deterministic | Quality | Speed | Use Case |
|----------|--------------|---------|-------|----------|
| Greedy | ✓ | Poor (repetitive) | Fast | Debugging |
| Temperature (0.7) | ✗ | Good | Fast | Creative text |
| Top-K (50) | ✗ | Very Good | Fast | General purpose |
| Top-P (0.9) | ✗ | Best | Fast | Production LLMs |
| Top-K + Top-P | ✗ | Best | Fast | **Recommended** |

## Module Contents

- `theory.md` - Sampling strategies deep dive
- `../../tiny_transformer/sampling/` - All sampling implementations
- `../../examples/generate.py` - Interactive generation script
- `../../tests/test_sampling.py` - Sampling tests

## Success Criteria

- [ ] Implement all sampling strategies (greedy, temperature, top-k, top-p)
- [ ] Implement complete generation loop
- [ ] Generate coherent text from trained model
- [ ] Understand trade-offs between strategies
- [ ] Compare different temperature values
- [ ] (Bonus) Implement KV caching for faster generation

## Next Module

**Module 08: Engineering Practices** - Production-ready code with logging, checkpointing, experiment tracking, and more.
