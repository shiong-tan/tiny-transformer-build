# Tiny Transformer - Datasets

## Overview

This directory contains datasets for training and evaluating the Tiny Transformer language model.

## Quick Start

```bash
# Download all datasets
bash download_shakespeare.sh

# Verify downloads
ls -lh *.txt
```

## Datasets

### 1. Tiny Shakespeare (`tiny_shakespeare.txt`)

**Description**: A compilation of Shakespeare's works, widely used for character-level language modeling.

**Statistics**:
- **Size**: ~1.1 MB (1,115,394 characters)
- **Lines**: ~40,000
- **Unique characters**: ~65 (a-z, A-Z, punctuation, newlines)
- **Vocabulary**: Printable ASCII characters

**Source**: [Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)

**Sample**:
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

**Character Distribution**:
- Letters: ~63% (lowercase ~45%, uppercase ~18%)
- Whitespace: ~25% (spaces, newlines)
- Punctuation: ~12% (.,!?:;-')

**Use Cases**:
- Character-level language modeling
- Text generation in Shakespearean style
- Benchmark for small transformer models
- Educational demonstrations

**Training Recommendations**:
- **Sequence length**: 256-512 characters
- **Batch size**: 32-64 (character-level allows larger batches)
- **Model size**: d_model=384, n_heads=6, n_layers=6 works well
- **Training steps**: 10,000-20,000 for good results
- **Learning rate**: 5e-4 (higher for character-level)

### 2. Simple Sequences (`simple_sequences.txt`)

**Description**: Minimal test dataset for quick validation and debugging.

**Statistics**:
- **Size**: ~150 characters
- **Lines**: 6
- **Content**: Alphabet, numbers, simple phrases

**Use Cases**:
- Quick smoke tests
- Debugging tokenization
- Validating training pipeline
- Fast iteration during development

**Sample**:
```
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
Hello world!
```

## Data Preprocessing

### Character-Level Tokenization

The Tiny Transformer uses character-level tokenization for simplicity:

```python
from tiny_transformer.training import CharTokenizer

# Create tokenizer
tokenizer = CharTokenizer()

# Fit on text
with open('data/tiny_shakespeare.txt', 'r') as f:
    text = f.read()
tokenizer.fit(text)

# Encode/decode
tokens = tokenizer.encode("Hello")
decoded = tokenizer.decode(tokens)
```

**Vocabulary Construction**:
- Each unique character → unique integer ID
- Typical Shakespeare vocab size: ~65 characters
- No special tokens needed (no [PAD], [UNK], etc.)

### Data Splitting

**Default split** (when no separate validation file):
```python
# 90% train, 10% validation
val_size = len(tokens) // 10
train_tokens = tokens[:-val_size]
val_tokens = tokens[-val_size:]
```

**Custom split**:
```bash
# Create separate validation file
head -n 36000 tiny_shakespeare.txt > shakespeare_train.txt
tail -n 4000 tiny_shakespeare.txt > shakespeare_val.txt
```

### Sliding Window Dataset

The `TextDataset` uses overlapping windows for efficient training:

```python
from tiny_transformer.training import TextDataset

dataset = TextDataset(
    tokens=train_tokens,
    seq_len=256,     # Context window
    stride=128       # 50% overlap for more training data
)
```

**Example with stride=128**:
- Sequence 1: tokens[0:256]
- Sequence 2: tokens[128:384]  (overlaps by 128)
- Sequence 3: tokens[256:512]  (overlaps by 128)

## Training Configuration

### Shakespeare-Optimized Config

See `configs/shakespeare.yaml`:

```yaml
model:
  d_model: 384
  n_heads: 6
  n_layers: 6
  max_seq_len: 256

training:
  learning_rate: 5.0e-4
  batch_size: 64
  max_steps: 20000

data:
  train_file: "data/tiny_shakespeare.txt"
  seq_len: 256
  stride: 128
```

### Quick Test Config

See `configs/tiny.yaml` for fast validation:

```yaml
model:
  d_model: 128
  n_layers: 2

training:
  batch_size: 16
  max_steps: 1000
```

## Expected Results

### Shakespeare Model Performance

**After 10,000 steps** (~20 minutes on M1 Mac):
- **Training loss**: ~1.2-1.4
- **Validation loss**: ~1.4-1.6
- **Perplexity**: ~4.0-5.0
- **Sample quality**: Coherent character names, basic grammar

**After 20,000 steps** (~40 minutes):
- **Training loss**: ~1.0-1.2
- **Validation loss**: ~1.2-1.4
- **Perplexity**: ~3.3-4.0
- **Sample quality**: Better structure, dialogue format, occasional coherent phrases

**Sample Generation** (after 20k steps):
```
ROMEO:
What say you to my lady's love?

JULIET:
I know not, sir, but I do love thee well.
```

### Loss Curves

Expected training progression:
- **Steps 0-1000**: Rapid drop from ~4.0 to ~2.0
- **Steps 1000-5000**: Steady decrease to ~1.5
- **Steps 5000-20000**: Gradual improvement to ~1.2

## Troubleshooting

### Issue: "Unknown characters in vocabulary"

**Cause**: Prompt contains characters not in training data.

**Solution**:
```python
# Check vocabulary
tokenizer.vocab  # See all valid characters

# Filter to valid characters
valid_chars = set(tokenizer.vocab.keys())
prompt = ''.join(c for c in prompt if c in valid_chars)
```

### Issue: Poor generation quality

**Possible causes**:
1. **Undertraining**: Train for more steps (20k recommended)
2. **Model too small**: Increase d_model, n_layers
3. **Learning rate too high**: Reduce to 3e-4
4. **Bad sampling**: Try temperature=0.8, top_p=0.95

### Issue: Training too slow

**Solutions**:
1. **Reduce batch size**: 32 instead of 64
2. **Shorter sequences**: seq_len=128 instead of 256
3. **Smaller model**: Use tiny.yaml config
4. **Use GPU/MPS**: Ensure PyTorch detects accelerator

## Additional Datasets (Optional)

### Other Character-Level Datasets

**Linux Kernel Code**:
```bash
curl -o linux_source.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux_input.txt
```

**Wikipedia Articles**:
```bash
# Download from Hugging Face datasets
# Requires: pip install datasets
```

## Data Licenses

- **Tiny Shakespeare**: Public domain (Shakespeare's works)
- **Simple Sequences**: Created for this course, public domain

## References

1. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy
2. [char-rnn Repository](https://github.com/karpathy/char-rnn) - Character-level language modeling
3. [nanoGPT](https://github.com/karpathy/nanoGPT) - Modern character-level transformers

---

**Next Steps**: After downloading data, see `docs/modules/09_capstone/walkthrough.md` for complete training guide.
