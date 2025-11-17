# Shape Debugging Primer

**Master the #1 skill for transformer development**

## Why Shape Debugging Matters

> "90% of bugs in transformer implementations are shape errors."
> — Every transformer developer ever

When you're building transformers, you'll spend more time debugging tensor shapes than any other issue. This primer gives you the superpowers to:

1. **Read shape errors instantly** and know exactly what's wrong
2. **Prevent shape errors** with defensive coding
3. **Debug complex multi-dimensional operations** with confidence
4. **Use our shape checking utilities** effectively

## The Transformer Shape Landscape

### Core Dimensions

Every tensor in a transformer has a specific shape with meaningful names:

```python
# Input to attention mechanism
query: Tensor      # Shape: (batch_size, seq_len_q, d_model)
key: Tensor        # Shape: (batch_size, seq_len_k, d_model)
value: Tensor      # Shape: (batch_size, seq_len_v, d_model)

# Attention computation
scores: Tensor     # Shape: (batch_size, seq_len_q, seq_len_k)
attention: Tensor  # Shape: (batch_size, seq_len_q, seq_len_k)  # After softmax
output: Tensor     # Shape: (batch_size, seq_len_q, d_model)
```

### Common Dimension Names

| Name | Meaning | Typical Values |
|------|---------|----------------|
| `batch_size` or `B` | Number of examples in batch | 1, 8, 16, 32, 64 |
| `seq_len` or `L` | Sequence length (# tokens) | 1-512 (tiny), 512-4096 (real) |
| `d_model` or `D` | Model dimension (embedding size) | 64, 128, 256, 512, 768, 1024 |
| `d_k` | Key/Query dimension per head | 32, 64, 128 |
| `d_v` | Value dimension per head | 32, 64, 128 |
| `n_heads` or `H` | Number of attention heads | 4, 8, 12, 16 |
| `d_ff` | Feed-forward hidden dimension | 2048, 4096 (usually 4× d_model) |
| `vocab_size` or `V` | Vocabulary size | 256 (char), 50k (subword) |

## Reading Shape Errors

### Example 1: Dimension Mismatch

```python
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x512 and 256x64)
```

**Translation:**
- Trying to do: `(32, 512) @ (256, 64)` ← Invalid!
- Matrix multiplication requires: `(m, n) @ (n, p) = (m, p)`
- Inner dimensions must match: 512 ≠ 256 ❌

**Common cause:** Wrong projection matrix or transposed dimensions

**Fix:**
```python
# Wrong
output = x @ W_q  # x:(32,512), W_q:(256,64) ❌

# Right
output = x @ W_q  # x:(32,512), W_q:(512,64) ✓
```

### Example 2: Broadcasting Error

```python
RuntimeError: The size of tensor a (128) must match the size of tensor b (64)
at non-singleton dimension 2
```

**Translation:**
- Tensor a: `(?, ?, 128)`
- Tensor b: `(?, ?, 64)`
- Trying to add/multiply these → dimensions don't broadcast

**Common cause:** Mismatched d_model, d_k, or d_v

**Fix:**
```python
# Wrong: d_k mismatch
q = x @ W_q  # (B, L, 128)
k = x @ W_k  # (B, L, 64)   ← Different d_k!
scores = q @ k.transpose(-2, -1)  # ❌

# Right: consistent dimensions
q = x @ W_q  # (B, L, 128)
k = x @ W_k  # (B, L, 128)  ← Match!
scores = q @ k.transpose(-2, -1)  # ✓
```

### Example 3: Unexpected Batch Dimension

```python
RuntimeError: Expected 3-dimensional input for 3-dimensional weight [256, 512, 1],
but got 2-dimensional input of size [512, 256]
```

**Translation:**
- Expected: `(batch_size, seq_len, d_model)`
- Got: `(seq_len, d_model)` ← Missing batch dimension!

**Common cause:** Forgot to add batch dimension or used wrong indexing

**Fix:**
```python
# Wrong: single example without batch dimension
x = torch.randn(512, 256)  # ❌
output = model(x)

# Right: add batch dimension
x = torch.randn(1, 512, 256)  # (batch=1, seq=512, dim=256) ✓
output = model(x)

# Or use unsqueeze
x = torch.randn(512, 256)
x = x.unsqueeze(0)  # Add batch dimension
output = model(x)
```

## Our Shape Checking Utilities

### 1. `check_shape()` - Your Best Friend

```python
from tiny_transformer.utils import check_shape

def attention(q, k, v):
    batch_size, seq_len, d_model = q.shape

    # Check shapes defensively
    check_shape(q, (batch_size, seq_len, d_model), "query")
    check_shape(k, (batch_size, seq_len, d_model), "key")
    check_shape(v, (batch_size, seq_len, d_model), "value")

    # ... rest of computation
```

**Benefits:**
- Clear error messages with named tensors
- Catches errors early before they propagate
- Self-documenting code (shows expected shapes)

### 2. Named Dimensions with Wildcards

```python
# Use None or -1 for "any value" dimensions
check_shape(x, (None, 512, 256), "input")
# Accepts: (1, 512, 256), (32, 512, 256), etc.

# Use variables for dynamic checking
batch_size = x.size(0)
check_shape(output, (batch_size, 512, 256), "output")
# Ensures batch_size matches input
```

### 3. `assert_batch_consistency()` - Verify Matching Batches

```python
from tiny_transformer.utils import assert_batch_consistency

def forward(q, k, v, mask=None):
    # Ensure all inputs have same batch size
    tensors = [q, k, v]
    if mask is not None:
        tensors.append(mask)

    assert_batch_consistency(tensors, ["query", "key", "value", "mask"])
```

### 4. `print_shape_info()` - Quick Debugging

```python
from tiny_transformer.utils import print_shape_info

# During debugging, inspect shapes
print_shape_info(q, "query after projection")
print_shape_info(scores, "attention scores")
print_shape_info(output, "final output")

# Output:
# query after projection: shape=(32, 128, 512), dtype=float32, device=cpu
# attention scores: shape=(32, 128, 128), dtype=float32, device=cpu
# final output: shape=(32, 128, 512), dtype=float32, device=cpu
```

### 5. `ShapeTracer()` - Trace Full Forward Pass

```python
from tiny_transformer.utils import ShapeTracer

with ShapeTracer(enabled=True) as tracer:
    output = model(input_ids)

# Prints shape of every intermediate tensor!
```

## Common Shape Patterns in Transformers

### Pattern 1: Projection (Linear Layer)

```python
# Input:  (batch, seq_len, d_in)
# Weight: (d_in, d_out)
# Output: (batch, seq_len, d_out)

x = torch.randn(32, 128, 512)  # (B, L, D_in)
W = torch.randn(512, 256)       # (D_in, D_out)
output = x @ W                  # (B, L, D_out) = (32, 128, 256)
```

**Mental model:** Batch and sequence dimensions are preserved, only last dimension changes.

### Pattern 2: Attention Scores (Q @ K^T)

```python
# Query: (batch, seq_len_q, d_k)
# Key:   (batch, seq_len_k, d_k)
# Scores: (batch, seq_len_q, seq_len_k)

q = torch.randn(32, 128, 64)  # (B, L_q, d_k)
k = torch.randn(32, 128, 64)  # (B, L_k, d_k)

# Transpose k: (B, L_k, d_k) → (B, d_k, L_k)
k_t = k.transpose(-2, -1)

# Multiply: (B, L_q, d_k) @ (B, d_k, L_k) = (B, L_q, L_k)
scores = q @ k_t  # (32, 128, 128)
```

**Mental model:** Each query (row) attends to all keys (columns), creating a 2D attention map per batch.

### Pattern 3: Attention Application (Attn @ V)

```python
# Attention: (batch, seq_len_q, seq_len_v)
# Value:     (batch, seq_len_v, d_v)
# Output:    (batch, seq_len_q, d_v)

attention = torch.randn(32, 128, 128)  # (B, L_q, L_v) - weights
v = torch.randn(32, 128, 64)           # (B, L_v, d_v) - values

# Multiply: (B, L_q, L_v) @ (B, L_v, d_v) = (B, L_q, d_v)
output = attention @ v  # (32, 128, 64)
```

**Mental model:** Weighted average of values, one output vector per query position.

### Pattern 4: Multi-Head Reshape

```python
# Input:  (batch, seq_len, d_model)
# Output: (batch, n_heads, seq_len, d_k)

batch_size = 32
seq_len = 128
d_model = 512
n_heads = 8
d_k = d_model // n_heads  # 64

x = torch.randn(batch_size, seq_len, d_model)

# Reshape for multi-head
x = x.view(batch_size, seq_len, n_heads, d_k)  # (B, L, H, d_k)
x = x.transpose(1, 2)                           # (B, H, L, d_k)
```

**Mental model:** Split d_model into n_heads chunks, move heads to separate dimension.

## Debugging Strategies

### Strategy 1: Add Shape Checks Liberally

```python
def complex_operation(x):
    # Check input
    check_shape(x, (None, None, 512), "input")

    # Step 1
    x = self.layer1(x)
    check_shape(x, (None, None, 256), "after layer1")  # ← Catches errors early

    # Step 2
    x = self.layer2(x)
    check_shape(x, (None, None, 128), "after layer2")

    # Step 3
    x = self.layer3(x)
    check_shape(x, (None, None, 64), "after layer3")

    return x
```

**Benefit:** Pinpoint exact operation that caused shape error.

### Strategy 2: Use Inline Shape Comments

```python
def attention(q, k, v):
    # q, k, v: (B, L, D)
    scores = q @ k.transpose(-2, -1)  # (B, L, D) @ (B, D, L) → (B, L, L)
    scores = scores / math.sqrt(self.d_k)  # (B, L, L)
    attention = torch.softmax(scores, dim=-1)  # (B, L, L)
    output = attention @ v  # (B, L, L) @ (B, L, D) → (B, L, D)
    return output  # (B, L, D)
```

**Benefit:** Document expected shapes for every operation.

### Strategy 3: Print Intermediate Shapes

```python
def forward(self, x):
    print(f"Input: {x.shape}")

    x = self.layer1(x)
    print(f"After layer1: {x.shape}")

    x = self.layer2(x)
    print(f"After layer2: {x.shape}")

    return x
```

**When to use:** Quick debugging, remove after fixing.

### Strategy 4: Use Assertions

```python
def attention(q, k, v):
    assert q.dim() == 3, f"Expected 3D query, got {q.dim()}D"
    assert k.dim() == 3, f"Expected 3D key, got {k.dim()}D"
    assert v.dim() == 3, f"Expected 3D value, got {v.dim()}D"
    assert q.size(-1) == k.size(-1), "d_model mismatch between q and k"
```

**Benefit:** Fast, built-in, no dependencies.

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting Batch Dimension

```python
# ❌ Wrong: Creating test data without batch
x = torch.randn(128, 512)  # (seq_len, d_model)

# ✓ Right: Always include batch dimension
x = torch.randn(1, 128, 512)  # (batch=1, seq_len, d_model)
```

### Pitfall 2: Transposing the Wrong Dimensions

```python
# ❌ Wrong: Transpose all dimensions
k_t = k.transpose(0, 1)  # Swaps batch and seq_len!

# ✓ Right: Only transpose last two dimensions
k_t = k.transpose(-2, -1)  # Swaps seq_len and d_k
# Or equivalently:
k_t = k.transpose(1, 2)
```

### Pitfall 3: Inconsistent d_k and d_v

```python
# ❌ Wrong: Different dimensions per component
self.W_q = nn.Linear(d_model, 64)  # d_k = 64
self.W_k = nn.Linear(d_model, 128)  # d_k = 128 ← Mismatch!

# ✓ Right: Consistent dimensions
self.W_q = nn.Linear(d_model, d_k)
self.W_k = nn.Linear(d_model, d_k)  # Same d_k
self.W_v = nn.Linear(d_model, d_v)  # Can differ from d_k
```

### Pitfall 4: Broadcasting Surprises

```python
# Shapes
a = torch.randn(32, 1, 512)
b = torch.randn(1, 128, 512)

# This works (broadcasting) but might not be what you want!
c = a + b  # (32, 128, 512) ← Broadcasted!

# Be explicit
assert a.size(1) == b.size(1), "Sequence lengths must match"
```

### Pitfall 5: In-Place Operations

```python
# ❌ Wrong: In-place op changes shape unexpectedly
x.transpose_(-2, -1)  # Modifies x in-place, can break autograd

# ✓ Right: Create new tensor
x = x.transpose(-2, -1)
```

## Quick Reference Card

### Shape Check Template

```python
from tiny_transformer.utils import check_shape

def my_function(x):
    B, L, D = x.shape  # Destructure for named dimensions
    check_shape(x, (B, L, D), "input")

    # ... computation ...

    check_shape(output, (B, L, D), "output")
    return output
```

### Common Operations

| Operation | Input Shape | Output Shape |
|-----------|-------------|--------------|
| Linear(D_in, D_out) | (B, L, D_in) | (B, L, D_out) |
| Q @ K^T | (B, L, d_k), (B, L, d_k) | (B, L, L) |
| Attn @ V | (B, L, L), (B, L, d_v) | (B, L, d_v) |
| Softmax(x, dim=-1) | (B, L, L) | (B, L, L) |
| LayerNorm(x) | (B, L, D) | (B, L, D) |
| x.transpose(-2,-1) | (B, L, D) | (B, D, L) |
| x.view(B, L, H, d_k) | (B, L, H*d_k) | (B, L, H, d_k) |

## Practice Exercise

Debug this broken code:

```python
def broken_attention(q, k, v):
    # q, k, v: (32, 128, 512)
    scores = q @ k  # ← BUG 1
    scores = scores / math.sqrt(512)
    attention = torch.softmax(scores, dim=1)  # ← BUG 2
    output = attention @ v.transpose(0, 1)  # ← BUG 3
    return output
```

**Solutions:**
1. Missing transpose: should be `q @ k.transpose(-2, -1)`
2. Wrong dim: should be `dim=-1` (softmax over keys, not sequence)
3. Wrong transpose: should be `attention @ v` (no transpose)

## Next Steps

Now that you understand shape debugging:

1. ✓ Install our shape checking utilities: `from tiny_transformer.utils import *`
2. ✓ Add shape checks to your code proactively
3. ✓ Read inline shape comments in our codebase
4. ✓ Use `ShapeTracer` when confused
5. ✓ Move on to Module 01 with confidence!

**Remember:** Shape errors are normal. The pros just debug them faster!

---

**Pro tip:** Keep this doc open in a tab while coding. You'll reference it constantly!
