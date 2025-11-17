# Tiny Transformer - Learning Journey Guide

**Welcome!** This guide will take you from zero to building a complete transformer language model from scratch.

---

## 🚀 Quick Navigation

**Choose your path:**

- [**Absolute Beginner**](#path-1-absolute-beginner-30-hours) - Start from scratch, learn everything (30 hours)
- [**Experienced ML Engineer**](#path-2-experienced-ml-engineer-10-15-hours) - Skip basics, focus on transformers (10-15 hours)
- [**Just Show Me Code**](#path-3-just-show-me-code-2-hours) - Run the demo, explore later (2 hours)

**Repository status:** ✅ All critical bugs fixed, fully functional!

---

## 📚 What You'll Build

By the end of this journey:

1. **Complete Transformer Language Model** - Built from scratch, no black boxes
2. **Shakespeare Text Generator** - Trained on Shakespeare's complete works
3. **Production-Ready Codebase** - Proper engineering, testing, documentation
4. **Deep Understanding** - Know exactly how transformers work

**Final Demo:**
```bash
$ python tools/interactive.py --checkpoint checkpoints/shakespeare_best.pt
> ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon...
```

---

## 🗺️ The Complete Learning Path

```
Module 00: Setup & Orientation (1-2 hours)
    ↓
Module 01: Attention Mechanism (2-3 hours)
    ↓
Module 02: Multi-Head Attention (2-3 hours)
    ↓
Module 03: Transformer Block (2-3 hours)
    ↓
Module 04: Embeddings (2-3 hours)
    ↓
Module 05: Full Model (2-3 hours)
    ↓
Module 06: Training Pipeline (3-4 hours)
    ↓
Module 07: Sampling Strategies (2-3 hours)
    ↓
Module 08: Engineering Best Practices (2-3 hours)
    ↓
Module 09: CAPSTONE PROJECT (4-6 hours)
    ↓
    🎉 You built a transformer!
```

**Total time:** ~25-35 hours for complete mastery

---

## 🎯 Path 1: Absolute Beginner (30 hours)

**Perfect if:** You're new to deep learning or PyTorch

### Step-by-Step Journey

#### **Module 00: Setup & Orientation** (1-2 hours)
📍 Location: `docs/modules/00_setup/`

**What you'll do:**
1. Set up your Python environment
2. Learn shape debugging (THE critical skill)
3. Review PyTorch fundamentals

**Commands:**
```bash
# Verify environment
python tools/verify_environment.py

# Run setup notebook
jupyter notebook docs/modules/00_setup/setup_walkthrough.ipynb

# Read shape debugging primer
open docs/modules/00_setup/shape_debugging_primer.md
```

**Success criteria:** ✅ `make verify` passes all checks

---

#### **Module 01: Attention Mechanism** (2-3 hours)
📍 Location: `docs/modules/01_attention/`

**What you'll learn:**
- Scaled dot-product attention
- Query, Key, Value matrices
- Attention scores and weights
- Causal masking for autoregressive models

**Learning flow:**
```bash
# 1. Read theory
open docs/modules/01_attention/theory.md

# 2. Explore the implementation
python tiny_transformer/attention.py

# 3. Interactive notebook
jupyter notebook docs/modules/01_attention/notebook.ipynb

# 4. Do exercises
python docs/modules/01_attention/exercises/exercises.py

# 5. Run tests
pytest tests/test_attention.py -v
```

**Key file:** `tiny_transformer/attention.py:19-84`

---

#### **Module 02: Multi-Head Attention** (2-3 hours)
📍 Location: `docs/modules/02_multi_head/`

**What you'll learn:**
- Why multiple attention heads?
- Parallel attention computation
- Head splitting and concatenation
- Linear projections

**Learning flow:**
```bash
# 1. Read theory
open docs/modules/02_multi_head/theory.md

# 2. Examine implementation
open tiny_transformer/multi_head.py

# 3. Run tests
pytest tests/test_multi_head.py -v
```

**Key insight:** Multiple heads let the model attend to different aspects simultaneously (syntax, semantics, position, etc.)

---

#### **Module 03: Transformer Block** (2-3 hours)
📍 Location: `docs/modules/03_transformer_block/`

**What you'll learn:**
- Complete transformer layer
- Pre-LN vs Post-LN architecture
- Residual connections
- Layer normalization
- Feed-forward networks

**Learning flow:**
```bash
# 1. Read theory
open docs/modules/03_transformer_block/theory.md

# 2. Interactive notebook
jupyter notebook docs/modules/03_transformer_block/notebook.ipynb

# 3. Exercises
python docs/modules/03_transformer_block/exercises/exercises.py

# 4. Tests
pytest tests/test_transformer_block.py -v
```

**Key file:** `tiny_transformer/transformer_block.py`

---

#### **Module 04: Embeddings** (2-3 hours)
📍 Location: `docs/modules/04_embeddings/`

**What you'll learn:**
- Token embeddings
- Sinusoidal positional encoding
- Learned positional embeddings
- Why position matters

**Learning flow:**
```bash
# 1. Theory
open docs/modules/04_embeddings/theory.md

# 2. Notebook
jupyter notebook docs/modules/04_embeddings/notebook.ipynb

# 3. Exercises
python docs/modules/04_embeddings/exercises/exercises.py

# 4. Tests
pytest tests/test_embeddings.py -v
```

**Key insight:** Position encoding formula from "Attention is All You Need" paper

---

#### **Module 05: Full Model** (2-3 hours)
📍 Location: `docs/modules/05_full_model/`

**What you'll learn:**
- Stack multiple transformer blocks
- Output projection layer
- Weight tying (embedding = projection)
- Complete forward pass

**Learning flow:**
```bash
# 1. Theory
open docs/modules/05_full_model/theory.md

# 2. Notebook - build complete model!
jupyter notebook docs/modules/05_full_model/notebook.ipynb

# 3. Exercises
python docs/modules/05_full_model/exercises/exercises.py

# 4. Test your complete model
pytest tests/test_model.py -v
```

**Milestone:** 🎉 You have a working transformer model!

---

#### **Module 06: Training Pipeline** (3-4 hours)
📍 Location: `docs/modules/06_training/`

**What you'll learn:**
- Data loading and batching
- Character tokenization
- AdamW optimizer
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- Checkpointing

**Learning flow:**
```bash
# 1. Theory
open docs/modules/06_training/theory.md

# 2. Notebook - train a tiny model!
jupyter notebook docs/modules/06_training/notebook.ipynb

# 3. Exercises
python docs/modules/06_training/exercises/exercises.py

# 4. Tests
pytest tests/test_training.py -v
```

**Key files:**
- `tiny_transformer/training/trainer.py`
- `tiny_transformer/training/scheduler.py`
- `tiny_transformer/training/dataset.py`

---

#### **Module 07: Sampling Strategies** (2-3 hours)
📍 Location: `docs/modules/07_sampling/`

**What you'll learn:**
- Greedy sampling
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Combined strategies

**Learning flow:**
```bash
# 1. Theory
open docs/modules/07_sampling/theory.md

# 2. Notebook - generate text!
jupyter notebook docs/modules/07_sampling/notebook.ipynb

# 3. Exercises
python docs/modules/07_sampling/exercises/exercises.py

# 4. Tests
pytest tests/test_sampling.py -v
```

**Fun part:** Experiment with different sampling strategies and see how text quality changes!

---

#### **Module 08: Engineering Best Practices** (2-3 hours)
📍 Location: `docs/modules/08_engineering/`

**What you'll learn:**
- Atomic checkpoint saves
- Input validation
- Error handling
- Testing strategies
- Code organization
- Documentation

**Learning flow:**
```bash
# 1. Read theory
open docs/modules/08_engineering/theory.md

# 2. Review fixed code
open review_logs/CRITICAL_FIXES_APPLIED.md
```

**Key insight:** All 8 critical bugs have been fixed! The repo is production-ready.

---

#### **Module 09: Capstone Project** (4-6 hours)
📍 Location: `docs/modules/09_capstone/`

**🎯 THE BIG MOMENT:** Train your transformer on Shakespeare!

**Phase 1: Quick Start (30 minutes)**
```bash
# Download Shakespeare dataset
bash data/download_shakespeare.sh

# Quick training run (2000 steps, ~10 minutes)
python examples/shakespeare_demo.py --train --steps 2000

# Generate text!
python examples/shakespeare_demo.py --generate \
    --checkpoint checkpoints/demo/demo_model.pt \
    --prompt "ROMEO:"
```

**Phase 2: Full Training (2-4 hours)**
```bash
# Train production model (10k-20k steps)
python tools/shakespeare_train.py \
    --max-steps 20000 \
    --batch-size 64 \
    --d-model 384

# Monitor training
tail -f experiments/shakespeare/logs.txt
```

**Phase 3: Generation & Analysis (1-2 hours)**
```bash
# Generate with different settings
python tools/shakespeare_generate.py \
    --checkpoint checkpoints/shakespeare/best.pt \
    --prompt "JULIET:" \
    --temperature 0.8 \
    --max-tokens 200

# Interactive mode
python tools/interactive.py \
    --checkpoint checkpoints/shakespeare/best.pt

# Compare sampling strategies
python tools/shakespeare_generate.py \
    --checkpoint checkpoints/shakespeare/best.pt \
    --compare

# Batch generation
python tools/shakespeare_generate.py \
    --checkpoint checkpoints/shakespeare/best.pt \
    --batch \
    --output samples/gallery.txt
```

**Phase 4: Analysis Notebook**
```bash
jupyter notebook docs/modules/09_capstone/notebook.ipynb
```

**Success criteria:**
- ✅ Validation loss < 1.5
- ✅ Generated text is coherent
- ✅ Shakespeare-like style
- ✅ Different prompts → different outputs

**Deliverables:**
- Trained model checkpoint
- Sample gallery (20+ generations)
- Analysis notebook
- Documentation

**🎉 Congratulations! You've built a transformer from scratch!**

---

## ⚡ Path 2: Experienced ML Engineer (10-15 hours)

**Perfect if:** You know PyTorch and neural networks, want to focus on transformers

### Accelerated Track

**Skip these:**
- Module 00 (just run `make verify`)
- PyTorch refresher notebook
- Basic exercises

**Focus on:**

#### Quick Start (2 hours)
```bash
# 1. Environment
python tools/verify_environment.py

# 2. Skim the implementations
open tiny_transformer/attention.py
open tiny_transformer/multi_head.py
open tiny_transformer/model.py

# 3. Run all tests
pytest tests/ -v

# 4. Quick demo
python examples/shakespeare_demo.py --train --steps 2000
python examples/shakespeare_demo.py --generate \
    --checkpoint checkpoints/demo/demo_model.pt
```

#### Deep Dive (8-10 hours)

**Focus modules:**
- **Module 01**: Attention mechanism (theory + implementation)
- **Module 03**: Transformer block architecture
- **Module 06**: Training pipeline details
- **Module 07**: Sampling strategies

**For each module:**
1. Read theory.md (15 min)
2. Review implementation code (20 min)
3. Run notebook cells (30 min)
4. Do advanced exercises (30 min)

#### Capstone (3-4 hours)
```bash
# Full training run
python tools/shakespeare_train.py --max-steps 20000

# Experimentation
# - Try different hyperparameters
# - Implement beam search
# - Add special tokens
# - Multi-GPU training
```

**Extensions for experienced engineers:**
- Implement sparse attention
- Add KV caching for faster generation
- Profile and optimize bottlenecks
- Add mixed precision training
- Deploy as REST API

---

## 🔥 Path 3: Just Show Me Code (2 hours)

**Perfect if:** You want to see it work first, understand later

### Super Quick Start

```bash
# 1. Clone and setup (5 min)
git clone <repo-url>
cd tiny-transformer-build
pip install -e .

# 2. Verify installation (2 min)
python tools/verify_environment.py

# 3. Download data (1 min)
bash data/download_shakespeare.sh

# 4. Train a tiny model (10 min)
python examples/shakespeare_demo.py --train --steps 2000

# 5. Generate text! (1 min)
python examples/shakespeare_demo.py --generate \
    --checkpoint checkpoints/demo/demo_model.pt \
    --prompt "ROMEO:"

# 6. Interactive mode (5 min - play around!)
python examples/shakespeare_demo.py --interactive \
    --checkpoint checkpoints/demo/demo_model.pt
```

**Expected output:**
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief...
```

**Now explore:**
```bash
# Try different prompts
--prompt "JULIET:"
--prompt "HAMLET:"
--prompt "To be or not to be,"

# Adjust temperature
--temperature 0.5  # More conservative
--temperature 1.2  # More creative

# Longer output
--max-tokens 500
```

**Next step:** Once you see it work, go back and learn how! Start with Module 01.

---

## 📂 Repository Structure

```
tiny-transformer-build/
│
├── tiny_transformer/              # Core library (your implementations)
│   ├── attention.py               # Scaled dot-product attention
│   ├── multi_head.py              # Multi-head attention
│   ├── feedforward.py             # Position-wise FFN
│   ├── transformer_block.py       # Complete transformer layer
│   ├── embeddings.py              # Token + positional embeddings
│   ├── model.py                   # Full TinyTransformerLM
│   ├── training/                  # Training pipeline
│   │   ├── trainer.py             # Training loop
│   │   ├── dataset.py             # Data loading
│   │   ├── scheduler.py           # LR scheduling
│   │   └── utils.py               # Training utilities
│   ├── sampling/                  # Text generation
│   │   ├── strategies.py          # Sampling algorithms
│   │   └── generator.py           # TextGenerator class
│   └── utils/                     # Utilities
│       ├── checkpoint.py          # Checkpoint management
│       └── shape_check.py         # Shape debugging tools
│
├── docs/modules/                  # Learning materials
│   ├── 00_setup/                  # Setup & orientation
│   ├── 01_attention/              # Attention mechanism
│   ├── 02_multi_head/             # Multi-head attention
│   ├── 03_transformer_block/      # Transformer block
│   ├── 04_embeddings/             # Embeddings
│   ├── 05_full_model/             # Complete model
│   ├── 06_training/               # Training pipeline
│   ├── 07_sampling/               # Sampling strategies
│   ├── 08_engineering/            # Best practices
│   └── 09_capstone/               # Final project
│
├── tests/                         # Comprehensive test suite
│   ├── test_attention.py
│   ├── test_multi_head.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_sampling.py
│
├── tools/                         # Standalone scripts
│   ├── train.py                   # Generic training
│   ├── shakespeare_train.py       # Shakespeare-specific training
│   ├── generate.py                # Text generation
│   ├── shakespeare_generate.py    # Shakespeare generation
│   ├── interactive.py             # Interactive mode
│   └── verify_environment.py      # Environment validation
│
├── examples/                      # Quick examples
│   └── shakespeare_demo.py        # All-in-one demo
│
├── data/                          # Datasets (gitignored)
│   └── download_shakespeare.sh    # Download script
│
├── checkpoints/                   # Model checkpoints (gitignored)
├── experiments/                   # Training logs (gitignored)
│
├── review_logs/                   # Code review documentation
│   ├── CRITICAL_FIXES_APPLIED.md  # All bugs fixed!
│   └── COMPREHENSIVE_REVIEW_FINDINGS.md
│
└── QUICKSTART.md                  # This file!
```

---

## 🛠️ Development Workflow

### Standard Module Workflow

For each module:

1. **Read theory** (`theory.md`) - Understand concepts (15-30 min)
2. **Study implementation** - Read the actual code (20-40 min)
3. **Run notebook** - Interactive exploration (30-60 min)
4. **Do exercises** - Hands-on practice (30-60 min)
5. **Run tests** - Verify correctness (5-10 min)

### Example: Module 01 (Attention)

```bash
# Terminal 1: Read theory
open docs/modules/01_attention/theory.md

# Terminal 2: Explore implementation
python -i tiny_transformer/attention.py
>>> # Try the example at the bottom
>>> # Experiment with different inputs

# Terminal 3: Notebook
jupyter notebook docs/modules/01_attention/notebook.ipynb

# Terminal 4: Exercises
cd docs/modules/01_attention/exercises
python exercises.py
# Read the instructions, implement the TODOs

# Terminal 5: Tests
pytest tests/test_attention.py -v
```

### Testing Your Work

```bash
# Test single module
pytest tests/test_attention.py -v

# Test everything
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=tiny_transformer

# Run specific test
pytest tests/test_model.py::test_forward_pass -v
```

---

## 📊 Progress Tracking

### Module Checklist

Track your progress through the learning journey:

- [ ] **Module 00: Setup** - Environment ready
- [ ] **Module 01: Attention** - Understand attention mechanism
- [ ] **Module 02: Multi-Head** - Parallel attention heads
- [ ] **Module 03: Transformer Block** - Complete layer
- [ ] **Module 04: Embeddings** - Token + position encoding
- [ ] **Module 05: Full Model** - Stack blocks into model
- [ ] **Module 06: Training** - Train your model
- [ ] **Module 07: Sampling** - Generate text
- [ ] **Module 08: Engineering** - Production practices
- [ ] **Module 09: Capstone** - Complete Shakespeare project

### Skills Mastery

By module completion, you should be able to:

**After Module 01:**
- [ ] Explain attention mechanism
- [ ] Implement scaled dot-product attention
- [ ] Use causal masking

**After Module 03:**
- [ ] Build complete transformer block
- [ ] Understand residual connections
- [ ] Apply layer normalization

**After Module 05:**
- [ ] Build end-to-end transformer model
- [ ] Understand weight tying
- [ ] Debug shape mismatches

**After Module 07:**
- [ ] Generate coherent text
- [ ] Control generation with temperature
- [ ] Apply top-k and top-p sampling

**After Module 09:**
- [ ] Train production models
- [ ] Generate Shakespeare-style text
- [ ] Analyze and improve models

---

## 🚨 Common Issues & Solutions

### Installation Issues

**Problem:** `pip install -e .` fails
```bash
# Solution: Use Python 3.11+
python --version  # Should be 3.11 or higher
python3.11 -m pip install -e .
```

**Problem:** PyTorch not found
```bash
# Solution: Install PyTorch separately
pip install torch torchvision torchaudio
```

**Problem:** Jupyter kernel not found
```bash
# Solution: Install kernel
python -m ipykernel install --user --name=tiny-transformer
```

### Training Issues

**Problem:** Out of memory
```bash
# Solution: Reduce batch size
python tools/shakespeare_train.py --batch-size 32  # Instead of 64
```

**Problem:** Training too slow on CPU
```bash
# Solution: Use smaller model or GPU
python examples/shakespeare_demo.py --train --steps 1000  # Fewer steps
```

**Problem:** Generated text is gibberish
```bash
# Solution: Train longer or check checkpoint loading
# Make sure validation loss < 1.5
# Verify tokenizer vocabulary is loaded correctly
```

### Shape Errors

**Problem:** "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
```python
# Solution: Use shape debugging tools
from tiny_transformer.utils import check_shape

query = ...
check_shape(query, (batch_size, seq_len, d_model))
```

**Problem:** Attention mask shape mismatch
```python
# Solution: Check mask dimensions
# Mask should be (seq_len, seq_len) or (batch, seq_len, seq_len)
mask = create_causal_mask(seq_len)
print(f"Mask shape: {mask.shape}")
```

---

## 🎓 Learning Resources

### Within This Repository

**Essential reads:**
1. `docs/modules/00_setup/shape_debugging_primer.md` - Master shape debugging
2. `docs/modules/01_attention/theory.md` - Deep dive into attention
3. `review_logs/CRITICAL_FIXES_APPLIED.md` - See what was fixed and why

**Interactive notebooks:**
- Setup walkthrough
- Attention visualization
- Training experiments
- Sampling comparison

### External Resources

**Papers:**
- ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Decoder-only architecture

**Videos:**
- Andrej Karpathy - "Let's build GPT"
- 3Blue1Brown - "Attention in transformers, visually explained"

**Blogs:**
- Jay Alammar - "The Illustrated Transformer"
- Lilian Weng - "Attention? Attention!"

---

## 💡 Tips for Success

### Learning Tips

1. **Shapes First:** Always think in terms of tensor shapes
   - (batch, seq_len, d_model) is your mantra
   - Use `check_shape()` liberally

2. **Build Incrementally:** Don't rush
   - Verify each component works before moving on
   - Run tests after each change

3. **Visualize:** Use the provided visualization tools
   - Plot attention patterns
   - Visualize embeddings
   - Watch loss curves

4. **Experiment:** Break things and fix them
   - Try different hyperparameters
   - Implement variations
   - Compare results

5. **Document:** Keep notes on what you learn
   - What worked / didn't work
   - Interesting patterns
   - Questions for later

### Debugging Strategy

When something breaks:

1. **Check shapes first** - 90% of bugs are shape mismatches
2. **Print intermediate values** - See what's actually happening
3. **Use smaller inputs** - Debug with batch_size=1, seq_len=2
4. **Read error messages** - They tell you exactly what's wrong
5. **Check the tests** - See how it's supposed to work

### Time Management

**If you have:**

- **2 hours:** Path 3 (demo only)
- **1 weekend:** Modules 00-05 + quick demo
- **2 weekends:** Complete all modules
- **1 month (1hr/day):** Full mastery with extensions

**Suggested schedule:**
- Week 1: Modules 00-03 (setup + attention)
- Week 2: Modules 04-06 (model + training)
- Week 3: Modules 07-08 (generation + engineering)
- Week 4: Module 09 (capstone project)

---

## 🎯 Next Steps After Completion

### Immediate Next Steps

1. **Share your work:**
   - Push to GitHub
   - Write a blog post
   - Share on Twitter/LinkedIn with #TinyTransformer

2. **Extend the project:**
   - Train on different datasets (code, poetry, lyrics)
   - Implement beam search
   - Add special tokens (BOS, EOS, PAD)
   - Create web interface

3. **Dive deeper:**
   - Read transformer research papers
   - Explore GPT, BERT, T5 architectures
   - Study attention variants (sparse, linear, etc.)

### Career Applications

**You can now:**
- ✅ Understand transformer research papers
- ✅ Fine-tune pre-trained models (GPT, BERT)
- ✅ Build custom transformers for specific tasks
- ✅ Contribute to ML projects
- ✅ Interview confidently for ML roles

**Projects to showcase:**
- This repository (complete transformer implementation)
- Fine-tuned models on custom datasets
- Performance optimizations
- Novel architectures or attention variants

### Advanced Topics

**Next learning goals:**
- Vision Transformers (ViT)
- Encoder-decoder architectures (T5)
- Efficient transformers (Linformer, Performer)
- Multi-modal models (CLIP, Flamingo)
- Reinforcement learning from human feedback (RLHF)

---

## 🙋 Getting Help

### Self-Help Resources

1. **Check the code comments** - Every function is documented
2. **Read error messages** - They're usually correct
3. **Review the tests** - See working examples
4. **Search the theory docs** - Explanations for everything

### Community

- **GitHub Issues:** Report bugs or ask questions
- **Discussions:** Share what you built
- **Pull Requests:** Contribute improvements

---

## 📝 Conclusion

You're now ready to start your transformer learning journey!

**Choose your path:**
- [Absolute Beginner](#path-1-absolute-beginner-30-hours) - Start with Module 00
- [Experienced Engineer](#path-2-experienced-ml-engineer-10-15-hours) - Jump to Module 01
- [Just Show Me Code](#path-3-just-show-me-code-2-hours) - Run the demo now!

**First command to run:**
```bash
# Verify your environment
python tools/verify_environment.py

# If all checks pass:
# Path 1 → jupyter notebook docs/modules/00_setup/setup_walkthrough.ipynb
# Path 2 → open docs/modules/01_attention/theory.md
# Path 3 → python examples/shakespeare_demo.py --train --steps 2000
```

**Remember:**
- All critical bugs are fixed ✅
- The repository is fully functional ✅
- Tests pass ✅
- You can do this! ✅

---

**Let's build transformers!** 🚀

Good luck on your journey, and remember: every expert was once a beginner.

*Questions? Check the module READMEs or open a GitHub issue.*
