# 📊 Learning Path & Architecture Diagram

## 🗺️ Module Progression Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module 00: Setup & Orientation                │
│  ✅ Environment verification                                    │
│  ✅ Shape debugging primer                                      │
│  ✅ PyTorch refresher                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Module 01: Attention Fundamentals (60% DONE)           │
│  ✅ Scaled dot-product attention                               │
│  ✅ Causal masking                                             │
│  ✅ Shape: (B, T, d_k) → (B, T, d_v)                          │
│  📝 TODO: theory, exercises, notebook                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Module 02: Multi-Head Attention                     │
│  Split d_model into n_heads                                     │
│  Parallel attention (different perspectives)                    │
│  Concatenate & project back                                     │
│  Shape: (B, T, d_model) → (B, T, d_model)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             Module 03: Transformer Blocks                        │
│  Multi-head attention + FFN                                      │
│  Residual connections                                            │
│  Layer normalization (Pre-LN)                                    │
│  Shape: (B, T, d_model) → (B, T, d_model)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                          │
        ▼                                          ▼
┌──────────────────┐                    ┌──────────────────┐
│  Module 04:      │                    │  Module 05:      │
│  Embeddings      │ ─────combines────▶ │  Tiny LM         │
│                  │                    │                  │
│  Token embed     │                    │  Stack N blocks  │
│  + Positional    │                    │  + Output head   │
└──────────────────┘                    └────────┬─────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │  Untrained Model       │
                                    │  Can forward pass      │
                                    │  Random outputs        │
                                    └────────┬───────────────┘
                                             │
        ┌────────────────────────────────────┴──────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                    ┌──────────────────┐
│  Module 06:      │                    │  Module 07:      │
│  Training        │                    │  Sampling        │
│                  │                    │                  │
│  Data loader     │                    │  Greedy          │
│  Cross-entropy   │                    │  Temperature     │
│  Optimization    │                    │  Top-k / Top-p   │
└────────┬─────────┘                    └────────┬─────────┘
         │                                        │
         └───────────────┬────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Trained Model         │
            │  Generates text        │
            │  Coherent samples      │
            └────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Module 08: Engineering        │
        │  Add logging, checkpointing    │
        │  Experiment tracking           │
        │  Configuration management      │
        └────────┬───────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Module 09: CAPSTONE PROJECT                         │
│                                                                  │
│  Session 1 (90 min): Build complete model from components       │
│  Session 2 (90 min): Train on Shakespeare dataset               │
│  Session 3 (90 min): Sample, refine, create gallery             │
│                                                                  │
│  DELIVERABLE: GitHub repo + Colab notebook + trained model      │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Transformer Architecture (What You'll Build)

```
Input: "The cat sat"
   │
   ▼
┌──────────────────────────────────────┐
│  Token Embedding                     │
│  "The" → [0.1, -0.3, 0.8, ...]      │
│  "cat" → [0.2, 0.1, -0.4, ...]      │
│  "sat" → [-0.1, 0.5, 0.2, ...]      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  + Positional Embedding              │
│  Position 0 + Position 1 + Position 2│
└────────────┬─────────────────────────┘
             │
             ▼ (B, T, d_model)
    ┌────────────────────┐
    │ Transformer Block 1 │
    │ ┌────────────────┐ │
    │ │ Multi-Head     │ │ ──┐
    │ │ Attention      │ │   │ Residual
    │ └────────────────┘ │   │ Connection
    │         +  ←────────────┘
    │         ▼          │
    │ ┌────────────────┐ │
    │ │ Layer Norm     │ │
    │ └────────────────┘ │
    │         │          │
    │         ▼          │
    │ ┌────────────────┐ │
    │ │ Feed Forward   │ │ ──┐
    │ │ Network (FFN)  │ │   │ Residual
    │ └────────────────┘ │   │ Connection
    │         +  ←────────────┘
    │         ▼          │
    │ ┌────────────────┐ │
    │ │ Layer Norm     │ │
    │ └────────────────┘ │
    └─────────┬──────────┘
              │
              ▼ (B, T, d_model)
    ┌────────────────────┐
    │ Transformer Block 2 │
    │  (same structure)   │
    └─────────┬──────────┘
              │
              ⋮  (more blocks)
              │
              ▼ (B, T, d_model)
    ┌────────────────────┐
    │  Output Projection  │
    │  Linear: d_model    │
    │          → vocab    │
    └─────────┬──────────┘
              │
              ▼ (B, T, vocab_size)
    ┌────────────────────┐
    │  Logits             │
    │  [0.1, 0.8, ...]   │ → "on"  (highest)
    │  [0.3, 0.2, ...]   │ → "the" (second)
    │  ...                │
    └────────────────────┘
              │
              ▼ (apply softmax + sample)
         Output: "on"
```

## 🔍 Attention Mechanism (Core Operation)

```
Query, Key, Value all from same input (self-attention)

Input: (B, T, d_model)
   │
   ├──── Linear_Q ──→ Query (B, T, d_model)
   ├──── Linear_K ──→ Key   (B, T, d_model)
   └──── Linear_V ──→ Value (B, T, d_model)

Then split into heads:
Query → (B, n_heads, T, d_k)  where d_k = d_model / n_heads
Key   → (B, n_heads, T, d_k)
Value → (B, n_heads, T, d_k)

For each head:

    Query @ Key^T / √d_k
         │
         ▼
    Attention Scores (B, T, T)
    ┌───────────────────┐
    │ 1.2  -∞   -∞   -∞ │  Position 0 can only see itself
    │ 0.8  1.5  -∞   -∞ │  Position 1 can see 0 and 1
    │ 0.3  0.9  2.1  -∞ │  Position 2 can see 0, 1, 2
    │ 0.1  0.4  0.7  1.8│  Position 3 can see all
    └───────────────────┘
         │
         ▼ softmax
    Attention Weights (B, T, T)
    ┌───────────────────┐
    │ 1.0  0.0  0.0  0.0│  Normalized probabilities
    │ 0.3  0.7  0.0  0.0│  (each row sums to 1.0)
    │ 0.1  0.2  0.7  0.0│
    │ 0.1  0.1  0.2  0.6│
    └───────────────────┘
         │
         ▼ @ Value
    Output (B, n_heads, T, d_k)
         │
         ▼ concatenate heads
    Output (B, T, d_model)
```

## 📈 Training Loop

```
while step < max_steps:
    
    1. Sample Batch
       ┌──────────────┐
       │ Input:  [1,2]│  (token IDs)
       │ Target: [2,3]│  (shifted by 1)
       └──────────────┘
    
    2. Forward Pass
       input → embeddings → transformer → logits
       Shape: (B, T) → (B, T, d_model) → (B, T, vocab_size)
    
    3. Compute Loss
       Cross-entropy between logits and targets
       Reshape: (B, T, vocab_size) → (B*T, vocab_size)
               (B, T) → (B*T,)
    
    4. Backward Pass
       loss.backward() → gradients for all parameters
    
    5. Optimizer Step
       optimizer.step() → update weights
       optimizer.zero_grad() → clear gradients
    
    6. Log & Checkpoint
       if step % log_interval == 0:
           log metrics
       if step % checkpoint_interval == 0:
           save model
```

## 🎲 Sampling (Text Generation)

```
Input: "The cat"

while len(generated) < max_tokens:
    
    1. Encode current text
       "The cat" → [34, 89] (token IDs)
    
    2. Forward pass
       [34, 89] → logits for next token
       Shape: (1, 2, vocab_size) → take logits[-1]
       → (vocab_size,) = [0.1, 0.8, 0.3, ...]
    
    3. Apply temperature
       logits / temperature
       
       Temperature = 1.0: balanced
       Temperature = 0.5: more peaked (deterministic)
       Temperature = 2.0: more flat (random)
    
    4. Sample strategy
       
       Greedy: argmax(logits) → highest probability
       
       Temperature: softmax + random sample
       
       Top-k: keep only top k, renormalize, sample
       
       Top-p: keep cumulative prob p, sample
    
    5. Append to sequence
       [34, 89] → [34, 89, 102]
       "The cat sat"
    
    6. Repeat until done
```

## 🔧 Engineering Workflow

```
Day N workflow:

1. Read module theory    (30 min)
   ├─ Conceptual understanding
   └─ Mathematical foundations

2. Study reference code  (45 min)
   ├─ Shape annotations
   ├─ Implementation details
   └─ Run examples

3. Complete exercises    (60 min)
   ├─ Implement functions
   ├─ Debug shapes
   └─ Verify correctness

4. Run tests            (15 min)
   ├─ pytest XX_module/tests/
   └─ All tests pass ✅

5. Interactive notebook  (30 min)
   ├─ Experiment with parameters
   ├─ Visualize attention
   └─ Build intuition

Total: 3 hours per module
```

## 🎯 Success Metrics by Module

```
Module 01 ✓:
├─ Understand Q, K, V
├─ Implement attention
├─ Visualize patterns
└─ All tests pass

Module 02 ✓:
├─ Split into heads
├─ Parallel attention
├─ Concatenate correctly
└─ Shape preserved

Module 03 ✓:
├─ Build complete block
├─ Add residuals
├─ Layer norm
└─ Gradient flows

Module 04 ✓:
├─ Token embeddings
├─ Positional encodings
└─ Combined correctly

Module 05 ✓:
├─ Stack blocks
├─ Forward pass works
├─ Can generate (random)
└─ Shape: (B, T) → (B, T, vocab_size)

Module 06 ✓:
├─ Training loop
├─ Loss decreases
├─ Checkpoints work
└─ Logs saved

Module 07 ✓:
├─ Multiple strategies
├─ Coherent text
├─ Temperature effects
└─ Quality samples

Module 08 ✓:
├─ Config management
├─ Reproducible runs
├─ Clean logs
└─ Easy to resume

Module 09 ✓:
├─ End-to-end training
├─ Sample gallery
├─ Colab notebook
└─ Documentation
```

---

**Total Journey**: From attention basics → production-ready transformer LM

**Time Investment**: 8 days × 3-4 hours = 24-32 hours

**Outcome**: Deep understanding + working implementation + best practices
