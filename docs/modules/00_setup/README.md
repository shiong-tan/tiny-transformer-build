# Module 00: Setup & Orientation

Welcome to **Tiny Transformer**! This is your starting point for building a complete transformer language model from scratch.

## What You'll Learn

This module prepares you for the journey ahead by:

1. **Setting up your environment** - Getting Python, PyTorch, and all dependencies working
2. **Understanding shape debugging** - The #1 skill for deep learning development
3. **Refreshing PyTorch fundamentals** - Tensors, autograd, and nn.Module basics
4. **Orienting to the codebase** - Understanding our project structure and philosophy

## Why Start Here?

Building transformers requires precision. A single shape mismatch can cause cryptic errors hours into training. This module gives you the **debugging superpowers** you'll need for the rest of the course.

## Learning Path

### 1. Environment Setup (15 minutes)
**File**: `setup_walkthrough.ipynb`

Follow the interactive notebook to:
- Verify Python 3.11+ installation
- Install PyTorch (with MPS support on Mac)
- Test your hardware (GPU/MPS/CPU)
- Install course dependencies
- Run validation suite

**Goal**: Get a passing ✓ from `make verify`

### 2. Shape Debugging Primer (30 minutes)
**File**: `shape_debugging_primer.md`

Master the essential skill of tensor shape debugging:
- Reading shape errors (BatchSize × SeqLen × d_model)
- Using our `check_shape()` utility
- Named dimensions for clarity
- Common shape pitfalls
- Debugging strategies

**Goal**: Never fear a shape error again

### 3. PyTorch Refresher (45 minutes)
**File**: `pytorch_refresher.ipynb`

Quick review of PyTorch essentials:
- Tensor operations and broadcasting
- Autograd and backpropagation
- Building with nn.Module
- Forward/backward passes
- Parameter management

**Goal**: Comfortable with PyTorch before we build transformers

## Project Structure Overview

```
tiny-transformer-build/
├── tiny_transformer/          # Core implementation
│   ├── attention.py           # ✓ Scaled dot-product attention
│   ├── multi_head.py          # TODO: Multi-head attention
│   ├── transformer_block.py   # TODO: Full transformer block
│   ├── model.py               # TODO: Complete language model
│   └── utils/                 # Shape checking & visualization
├── tests/                     # Comprehensive test suite
├── docs/modules/              # This learning path
│   ├── 00_setup/              # You are here
│   ├── 01_attention/          # Single-head attention
│   ├── 02_multi_head/         # Multi-head attention
│   └── ...                    # More modules
└── tools/                     # Validation & environment tools
```

## Our Learning Philosophy

### 1. **Build Before You Use**
You'll implement every component from scratch. No black boxes.

### 2. **Shapes First**
Every operation has explicit shape annotations. You'll always know what's flowing through your model.

### 3. **Test Everything**
Each component has comprehensive tests. You'll verify correctness at every step.

### 4. **Visualize to Understand**
We provide visualization tools to see attention patterns, embeddings, and gradients.

### 5. **Learn by Debugging**
You'll encounter intentional challenges that build real debugging skills.

## Prerequisites

**Required:**
- Python programming (intermediate level)
- Basic linear algebra (matrix multiplication, vectors)
- High school calculus (derivatives concept)

**Helpful but not required:**
- Previous PyTorch experience (we'll review)
- Neural network basics (we'll build from ground up)
- Understanding of language models (we'll explain everything)

## Time Commitment

- **Module 00 (Setup)**: 1-2 hours
- **Modules 01-05 (Core Architecture)**: 12-15 hours
- **Modules 06-07 (Training & Generation)**: 8-10 hours
- **Modules 08-09 (Engineering & Capstone)**: 6-8 hours

**Total**: ~30 hours for complete mastery

You can go faster if you skip optional exercises, or slower if you want deep understanding.

## What You'll Build

By the end of this course, you'll have:

1. **A working transformer language model** that generates text
2. **Complete test suite** with 100+ tests
3. **Comprehensive understanding** of attention mechanisms
4. **Production-ready code** with proper engineering practices
5. **A trained model** generating Shakespeare-style text
6. **Portfolio project** demonstrating transformer expertise

## Getting Help

### During Setup
- Check `tools/verify_environment.py` for diagnostic info
- Run `make verify` to test your installation
- Review `CONTRIBUTING.md` for development setup

### During Learning
- Read inline code comments (we explain every operation)
- Run test suites (`pytest tests/test_*.py`)
- Use visualization tools (`tiny_transformer.utils`)
- Check shape annotations in function signatures

### Debugging
- Use `check_shape()` liberally (imported from utils)
- Enable `ShapeTracer()` context manager
- Print intermediate tensors with `print_shape_info()`
- Read the shape debugging primer (next file!)

## Ready to Begin?

1. **First**: Complete the setup walkthrough notebook
2. **Then**: Read the shape debugging primer
3. **Finally**: Do the PyTorch refresher if needed

After Module 00, you'll jump into Module 01 where you'll deeply understand the attention mechanism you've already started using!

## Success Criteria

You're ready for Module 01 when:
- [ ] `make verify` passes all checks
- [ ] You can run and understand the attention.py examples
- [ ] You're comfortable with tensor shapes (batch × seq × dim)
- [ ] You understand basic PyTorch operations
- [ ] You've run at least one test file successfully

## Next Module

**Module 01: Attention Mechanism** - Deep dive into scaled dot-product attention, the core building block of transformers.

---

**Let's build transformers!** Start with `setup_walkthrough.ipynb` →
