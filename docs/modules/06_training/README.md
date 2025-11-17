# Module 06: Training

Implementing the complete training pipeline - data loading, loss computation, optimization, and monitoring to train your transformer language model.

## What You'll Learn

1. **Data Loading** - Creating batches for language modeling
2. **Loss Functions** - Cross-entropy for next-token prediction
3. **Optimization** - AdamW with learning rate scheduling
4. **Training Loop** - Putting it all together
5. **Checkpointing** - Saving and resuming training
6. **Monitoring** - Tracking metrics and debugging

## Prerequisites

- ✓ Completed Module 05 (Full Model)
- ✓ Understanding of gradient descent
- ✓ Familiarity with PyTorch training loops

## Training Pipeline

```
┌─────────────────────────────────┐
│ Dataset (Text File)             │
│  "The cat sat on the mat..."    │
└─────────────────────────────────┘
    ↓ Tokenization
┌─────────────────────────────────┐
│ Token IDs                       │
│  [5, 12, 8, 42, 17, ...]        │
└─────────────────────────────────┘
    ↓ Batching
┌─────────────────────────────────┐
│ Batches (batch_size, seq_len)   │
│  [[5, 12, 8, ...],              │
│   [42, 17, 9, ...], ...]        │
└─────────────────────────────────┘
    ↓ Model Forward
┌─────────────────────────────────┐
│ Logits (B, T, vocab_size)       │
└─────────────────────────────────┘
    ↓ Compute Loss
┌─────────────────────────────────┐
│ Cross-Entropy Loss              │
│  Compare logits vs targets      │
└─────────────────────────────────┘
    ↓ Backward
┌─────────────────────────────────┐
│ Gradients                       │
└─────────────────────────────────┘
    ↓ Optimizer Step
┌─────────────────────────────────┐
│ Update Weights                  │
└─────────────────────────────────┘
    ↓ Repeat
```

## Key Concepts

### 1. Language Modeling Data
- **Input**: Tokens [0, 1, 2, ..., T-1]
- **Target**: Tokens [1, 2, 3, ..., T] (shifted by 1)
- **Prediction**: At position i, predict token i+1

### 2. Cross-Entropy Loss
```python
# Flatten for loss computation
logits = model(tokens)  # (B, T, vocab_size)
logits = logits.view(-1, vocab_size)  # (B*T, vocab_size)
targets = targets.view(-1)  # (B*T,)

loss = F.cross_entropy(logits, targets)
```

### 3. AdamW Optimizer
- **Adam with weight decay decoupling**
- **Better than Adam** for transformers
- **Hyperparameters**: lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1

### 4. Learning Rate Schedule
- **Warmup**: Linear increase from 0 to max_lr
- **Cosine decay**: Smooth decrease to min_lr
- **Critical for training**: Prevents early instability

### 5. Gradient Clipping
- **Prevent exploding gradients**: Clip norm to max value
- **Typical value**: 1.0
- **Essential for stability**

## What You'll Implement

```python
# 1. Data Loader
class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_len):
        # Load and tokenize text
        # Create sliding window batches

    def __getitem__(self, idx):
        # Return (input_tokens, target_tokens)
        pass

# 2. Training Configuration
@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    save_interval: int = 5000

# 3. Trainer Class
class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: TrainingConfig
    ):
        self.model = model
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps
        )

    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            tokens, targets = batch

            # Forward
            logits = self.model(tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Update
            self.optimizer.step()
            self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                tokens, targets = batch
                logits = self.model(tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
        }, path)
```

## Training Best Practices

### 1. Monitoring
- **Loss curves**: Should decrease smoothly
- **Perplexity**: exp(loss) - lower is better
- **Learning rate**: Track schedule
- **Gradient norms**: Watch for explosion/vanishing

### 2. Debugging
- **Overfit single batch**: Verify model can learn
- **Check gradients**: All parameters should have gradients
- **Validate shapes**: Shape errors are common
- **Loss not decreasing**: Check LR, initialization, data

### 3. Optimization
- **Mixed precision**: FP16 for faster training
- **Gradient accumulation**: Effective larger batch sizes
- **Data loading**: Use multiple workers
- **Distributed training**: Multi-GPU (advanced)

## Module Contents

- `theory.md` - Training deep dive and best practices
- `../../tiny_transformer/training/dataset.py` - Data loading
- `../../tiny_transformer/training/trainer.py` - Training loop
- `../../tiny_transformer/training/scheduler.py` - LR schedules
- `../../examples/train.py` - Complete training script

## Success Criteria

- [ ] Implement text dataset for language modeling
- [ ] Implement training loop with proper loss computation
- [ ] Add AdamW optimizer with cosine schedule
- [ ] Implement gradient clipping
- [ ] Add checkpointing for save/resume
- [ ] Monitor training metrics
- [ ] Successfully train on small dataset
- [ ] Observe decreasing loss

## Next Module

**Module 07: Sampling & Generation** - Implementing different sampling strategies to generate text from your trained model.
