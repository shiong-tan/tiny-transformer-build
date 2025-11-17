# Module 09: Capstone Project

The final challenge - train a complete transformer language model on Shakespeare's works and generate original text in Shakespeare's style!

## Project Goals

1. **Train from scratch** on Shakespeare dataset
2. **Generate coherent text** that resembles Shakespeare
3. **Apply everything learned** from Modules 01-08
4. **Create a shareable demo** of your working transformer
5. **Understand the full pipeline** end-to-end

## Prerequisites

- ✓ Completed Modules 01-08
- ✓ All components implemented and tested
- ✓ Understanding of complete training pipeline

## The Challenge

**Build and train a character-level transformer to generate Shakespeare-style text.**

### Dataset: Shakespeare's Complete Works
- **Source**: Tiny Shakespeare dataset (~1MB)
- **Content**: All of Shakespeare's plays
- **Vocabulary**: ~65 unique characters (a-z, A-Z, punctuation, space)
- **Size**: ~1 million characters

### Target Output Example
```
ROMEO:
Thou art thyself, though not a Montague.
What's Montague? It is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
```

## Project Phases

### Phase 1: Data Preparation (30 min)
```python
# 1. Download dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. Build character tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

# 3. Create train/val split
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]
```

### Phase 2: Model Configuration (15 min)
```yaml
# shakespeare_config.yaml
model:
  vocab_size: 65  # Character vocabulary
  d_model: 256    # Small for fast training
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  max_len: 256
  dropout: 0.2

training:
  batch_size: 64
  seq_len: 128
  epochs: 20
  learning_rate: 3e-4
  warmup_steps: 500
  weight_decay: 0.1
  grad_clip: 1.0

generation:
  temperature: 0.8
  top_k: 40
  top_p: 0.9
```

### Phase 3: Training (2-4 hours)
```python
# Complete training script
python tools/train.py \\
    --config configs/shakespeare.yaml \\
    --data data/shakespeare.txt \\
    --experiment-name shakespeare-v1

# Expected results:
# Epoch 1: train_loss=2.5, val_loss=2.3
# Epoch 5: train_loss=1.5, val_loss=1.6
# Epoch 10: train_loss=1.2, val_loss=1.4
# Epoch 20: train_loss=1.0, val_loss=1.3

# Target: val_loss < 1.5 for decent generation
```

### Phase 4: Generation (30 min)
```python
# Generate Shakespeare-style text
python tools/generate.py \\
    --checkpoint checkpoints/shakespeare_best.pt \\
    --prompt "ROMEO:" \\
    --max-tokens 200 \\
    --temperature 0.8 \\
    --top-p 0.9

# Try different prompts:
# - "ROMEO:"
# - "JULIET:"
# - "To be or not to be,"
# - "Shall I compare thee"
```

### Phase 5: Analysis & Demo (1 hour)
```python
# 1. Analyze learned patterns
# - Plot attention heads
# - Visualize embeddings
# - Check perplexity

# 2. Create sample gallery
generate_samples(
    prompts=[
        "ROMEO:",
        "JULIET:",
        "KING:",
        "FOOL:"
    ],
    temperatures=[0.5, 0.8, 1.0, 1.2],
    save_to="samples/"
)

# 3. Interactive demo
python tools/interactive.py \\
    --checkpoint checkpoints/shakespeare_best.pt
# > Enter prompt: ROMEO:
# > [Generated text appears...]
# > Continue? (y/n):
```

## Complete Project Structure

```
tiny-transformer-build/
├── data/
│   └── shakespeare.txt
├── configs/
│   └── shakespeare.yaml
├── checkpoints/
│   ├── shakespeare_step_1000.pt
│   ├── shakespeare_step_2000.pt
│   └── shakespeare_best.pt
├── experiments/
│   └── shakespeare-v1/
│       ├── config.yaml
│       ├── logs.json
│       └── samples/
├── tools/
│   ├── train.py
│   ├── generate.py
│   └── interactive.py
└── notebooks/
    └── shakespeare_analysis.ipynb
```

## Evaluation Criteria

### Quantitative Metrics
- **Validation Loss**: < 1.5 (good), < 1.3 (excellent)
- **Perplexity**: exp(val_loss) - lower is better
- **Training Time**: 2-4 hours on CPU, < 1 hour on GPU
- **Generation Speed**: > 100 tokens/sec

### Qualitative Assessment
- **Coherence**: Sentences make grammatical sense
- **Style**: Resembles Shakespeare (archaic language, iambic pentameter hints)
- **Character**: Text fits character (ROMEO vs KING)
- **Creativity**: Not just memorization - novel combinations

## Sample Outputs to Expect

**Early Training (Epoch 1, Loss ~2.0)**:
```
ROMEO:
Whath is the mone boud the sert of the warld
And the sert whan I was the have the sime
```
*(Gibberish, but character-level patterns emerging)*

**Mid Training (Epoch 10, Loss ~1.4)**:
```
ROMEO:
What is the matter with the world and the sun,
And the world that I have seen the world,
And I will not be the same to the world.
```
*(Coherent sentences, repetitive)*

**Late Training (Epoch 20, Loss ~1.2)**:
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief.
```
*(Recognizable Shakespeare style, creative!)*

## Extensions & Challenges

### Easy
- [ ] Train with different model sizes (tiny, small, medium)
- [ ] Experiment with sampling temperatures
- [ ] Generate dialogue between multiple characters
- [ ] Create a web interface for generation

### Medium
- [ ] Add prompt conditioning (genre, mood)
- [ ] Implement beam search for generation
- [ ] Fine-tune on specific plays
- [ ] Add rhyme scheme detection

### Hard
- [ ] Train on word-level tokenization (subword)
- [ ] Implement sparse attention for longer context
- [ ] Multi-GPU training
- [ ] Deploy as REST API

## Deliverables

By the end of this module, you should have:

1. **Trained Model**
   - `shakespeare_best.pt` (< 50MB)
   - Training logs and metrics
   - Configuration file

2. **Sample Gallery**
   - 20+ generated samples
   - Different prompts and temperatures
   - Quality comparison

3. **Interactive Demo**
   - CLI or web interface
   - Real-time generation
   - Parameter controls

4. **Analysis Notebook**
   - Training curves
   - Attention visualizations
   - Embedding analysis
   - Error analysis

5. **Documentation**
   - README with usage instructions
   - Model card with hyperparameters
   - Sample outputs
   - Lessons learned

## Submission (Optional)

Share your capstone project:
- **GitHub**: Push code and samples
- **Blog Post**: Write about your experience
- **Demo Video**: Show generation in action
- **Colab Notebook**: Interactive demo for others

## Success Criteria

- [ ] Model trains successfully to val_loss < 1.5
- [ ] Generated text is coherent and Shakespeare-like
- [ ] Can generate 200+ tokens without degradation
- [ ] Different prompts produce different styles
- [ ] Interactive demo works smoothly
- [ ] Full documentation and samples
- [ ] Understanding of complete transformer pipeline

## Celebration Time!

**Congratulations!** You've built a complete transformer language model from scratch! You now understand:

- ✓ Attention mechanisms (single and multi-head)
- ✓ Transformer architecture
- ✓ Embeddings and positional encoding
- ✓ Training pipelines
- ✓ Text generation strategies
- ✓ Production engineering practices

**You're now equipped to:**
- Build custom transformer models
- Fine-tune existing models (GPT, BERT)
- Understand research papers
- Contribute to ML projects
- Continue learning advanced topics

---

**Thank you for completing Tiny Transformer!** 🎉

Share what you built: #TinyTransformer
