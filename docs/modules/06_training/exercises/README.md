# Module 06: Training - Exercises

Welcome to the training exercises! This module teaches you how to train transformer language models from scratch.

## Overview

These exercises cover the complete training pipeline:
- Data loading and preprocessing
- Learning rate scheduling
- Training loops and optimization
- Gradient clipping and numerical stability
- Checkpointing and model persistence
- Evaluation metrics (loss, perplexity)
- Early stopping and regularization

## Exercise List

### Easy (1-3)
1. **TextDataset** - Create sliding window datasets for language modeling
2. **CharTokenizer** - Implement character-level tokenization
3. **Scheduler Configuration** - Set up learning rate schedules (warmup + decay)

### Medium (4-8)
4. **Data Loaders** - Create train/val DataLoaders with proper batching
5. **Basic Training Loop** - Implement forward/backward/update cycle
6. **Gradient Clipping** - Add gradient clipping to prevent explosion
7. **Checkpointing** - Save and load model checkpoints
8. **Perplexity** - Compute perplexity for language model evaluation

### Hard (9-11)
9. **Custom Warmup Scheduler** - Implement warmup + cosine decay from scratch
10. **Custom Trainer** - Build a complete trainer with logging and validation
11. **Early Stopping** - Implement patience-based early stopping

### Very Hard (12)
12. **Shakespeare Training** - End-to-end training pipeline on real text data

## Prerequisites

Before starting these exercises, you should have:
- Completed Modules 01-05 (attention through full model assembly)
- Understanding of PyTorch DataLoader and Dataset
- Familiarity with Adam/AdamW optimizer
- Knowledge of cross-entropy loss for classification

## Getting Started

1. **Read the theory first**: Check `../theory.md` for concepts
2. **Work sequentially**: Exercises build on each other
3. **Read docstrings carefully**: Each exercise has detailed instructions
4. **Implement TODOs**: Fill in the marked sections
5. **Test your code**: Uncomment assertions and run tests

## File Structure

```
06_training/exercises/
├── README.md          # This file
├── __init__.py        # Module initialization
├── exercises.py       # All exercises (1,560 lines)
└── solutions.py       # Reference solutions (create if needed)
```

## Running the Exercises

### Individual Exercise

```python
from docs.modules.06_training.exercises import exercises

# Exercise 1: TextDataset
tokens = list(range(1000))
dataset = exercises.Exercise01_TextDataset(tokens, seq_len=128, stride=128)
print(f"Dataset size: {len(dataset)}")

# Exercise 2: CharTokenizer
tokenizer = exercises.Exercise02_CharTokenizer()
tokenizer.fit("hello world")
tokens = tokenizer.encode("hello")
print(f"Tokens: {tokens}")
```

### Run All Tests

```python
from docs.modules.06_training.exercises import exercises

# Run comprehensive test suite
exercises.run_all_tests()
```

### Direct Execution

```bash
# Run the exercises file directly to see instructions
python docs/modules/06_training/exercises/exercises.py
```

## Learning Objectives

By completing these exercises, you will:

1. **Data Management**
   - Create sliding window datasets for sequence modeling
   - Implement character-level tokenization
   - Split data into train/validation sets
   - Configure DataLoaders for efficient batching

2. **Training Infrastructure**
   - Build training loops from scratch
   - Implement learning rate schedules with warmup
   - Add gradient clipping for stability
   - Create checkpointing systems

3. **Evaluation & Monitoring**
   - Compute cross-entropy loss
   - Calculate perplexity for language models
   - Track training metrics
   - Implement early stopping

4. **Integration**
   - Combine all components into working trainer
   - Train on real text data (Shakespeare)
   - Generate text from trained models
   - Debug and improve training

## Tips for Success

### Understanding Shapes
Track tensor shapes at each step:
```python
print(f"Input shape: {input_ids.shape}")   # (batch, seq_len)
print(f"Logits shape: {logits.shape}")     # (batch, seq_len, vocab_size)
print(f"Loss shape: {loss.shape}")         # scalar
```

### Debugging Training
Monitor these metrics:
- Loss (should decrease over time)
- Gradient norms (should be reasonable, not NaN)
- Learning rate (should follow schedule)
- Perplexity (should decrease)

### Common Pitfalls
1. **Wrong loss shape**: Use `view(-1, vocab_size)` to flatten
2. **Missing zero_grad**: Always call `optimizer.zero_grad()` before backward
3. **Eval mode**: Use `model.eval()` and `torch.no_grad()` for validation
4. **Sequence alignment**: Target is input shifted by 1 position

## Exercise Details

### Exercise 01: TextDataset
**Difficulty**: Easy
**Time**: 15-20 minutes
**Key Concepts**: Sliding windows, input-target alignment

Learn to create datasets for causal language modeling where each input predicts the next token.

### Exercise 02: CharTokenizer
**Difficulty**: Easy
**Time**: 15-20 minutes
**Key Concepts**: Vocabulary building, encoding/decoding

Build a simple character-level tokenizer with encode and decode methods.

### Exercise 03: Scheduler Configuration
**Difficulty**: Easy
**Time**: 20-25 minutes
**Key Concepts**: Warmup, cosine/linear decay

Configure learning rate schedulers and visualize the LR curve.

### Exercise 04: Data Loaders
**Difficulty**: Medium
**Time**: 25-30 minutes
**Key Concepts**: Train/val split, DataLoader configuration

Create train and validation DataLoaders from raw text with proper batching.

### Exercise 05: Basic Training Loop
**Difficulty**: Medium
**Time**: 30-40 minutes
**Key Concepts**: Forward/backward/update cycle

Implement the core training loop: forward pass, loss computation, backward pass, optimizer step.

### Exercise 06: Gradient Clipping
**Difficulty**: Medium
**Time**: 20-25 minutes
**Key Concepts**: Gradient norms, numerical stability

Add gradient clipping to prevent exploding gradients in transformer training.

### Exercise 07: Checkpointing
**Difficulty**: Medium
**Time**: 25-30 minutes
**Key Concepts**: Model persistence, resumable training

Implement checkpointing to save model state during training.

### Exercise 08: Perplexity
**Difficulty**: Medium
**Time**: 20-25 minutes
**Key Concepts**: Language model metrics, evaluation

Compute perplexity (exp(loss)) as a quality metric for language models.

### Exercise 09: Custom Warmup Scheduler
**Difficulty**: Hard
**Time**: 40-50 minutes
**Key Concepts**: Learning rate schedules, cosine annealing

Implement a warmup + cosine decay scheduler from scratch using PyTorch's LRScheduler API.

### Exercise 10: Custom Trainer
**Difficulty**: Hard
**Time**: 60-75 minutes
**Key Concepts**: Training infrastructure, logging, validation

Build a complete Trainer class that integrates all training components with proper logging.

### Exercise 11: Early Stopping
**Difficulty**: Hard
**Time**: 30-40 minutes
**Key Concepts**: Overfitting prevention, patience

Implement early stopping with patience to stop training when validation loss plateaus.

### Exercise 12: Shakespeare Training
**Difficulty**: Very Hard
**Time**: 90-120 minutes
**Key Concepts**: End-to-end pipeline, text generation

Train a transformer on Shakespeare text and generate new samples. This integrates everything!

## Reference Implementations

Check these files for reference:
- `tiny_transformer/training/dataset.py` - TextDataset and CharTokenizer
- `tiny_transformer/training/scheduler.py` - Learning rate schedulers
- `tiny_transformer/training/trainer.py` - Complete Trainer class
- `tiny_transformer/training/utils.py` - Training utilities

## Testing Your Solutions

Each exercise includes assertions you can uncomment to test:

```python
# Uncomment to test:
# assert logits.shape == (batch_size, seq_len, vocab_size)
# assert len(losses) == num_steps
# assert losses[0] > losses[-1]  # Loss should decrease
```

Run all tests at once:
```python
from docs.modules.06_training.exercises import exercises
exercises.run_all_tests()
```

## Expected Outcomes

After completing these exercises:
- You can train transformers from scratch
- You understand the complete training pipeline
- You can debug training issues
- You're ready for Module 07 (Sampling) and beyond

## Time Estimate

- **Easy exercises (1-3)**: 1 hour
- **Medium exercises (4-8)**: 2-2.5 hours
- **Hard exercises (9-11)**: 2-2.5 hours
- **Very hard (12)**: 1.5-2 hours
- **Total**: 5-6 hours

## Next Steps

After completing these exercises:
1. Review `../theory.md` to deepen understanding
2. Experiment with different hyperparameters
3. Try training on different datasets
4. Move on to Module 07: Sampling and Generation

## Questions?

If you get stuck:
1. Re-read the exercise docstring
2. Check the theory document
3. Look at reference implementations
4. Review earlier modules
5. Use print statements to debug

Good luck! 🚀
