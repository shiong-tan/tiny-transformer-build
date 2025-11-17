"""
Module 06: Training - Exercises

This module contains hands-on exercises for learning how to train transformer
language models. You'll practice:

1. Creating datasets for language modeling
2. Character-level tokenization
3. Learning rate scheduling (warmup + decay)
4. Complete training loops
5. Gradient clipping and optimization
6. Checkpointing and model saving
7. Perplexity computation
8. Custom trainers with logging
9. Early stopping strategies
10. End-to-end training pipelines

Each exercise includes:
- Clear docstrings with learning objectives
- Type hints and shape annotations
- TODO sections for implementation
- Test assertions for self-assessment
- Progressive difficulty (Easy → Medium → Hard → Very Hard)

Prerequisites:
- Completed Modules 01-05 (attention through full model)
- Understanding of cross-entropy loss
- Basic knowledge of PyTorch DataLoader
- Familiarity with Adam optimizer

Reference implementations:
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/training/dataset.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/training/scheduler.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/training/trainer.py
- /Users/shiongtan/projects/tiny-transformer-build/tiny_transformer/training/utils.py

Theory reference:
- /Users/shiongtan/projects/tiny-transformer-build/docs/modules/06_training/theory.md

Time estimate: 5-6 hours for all exercises

Good luck! Let's train some transformers!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math
from typing import Optional, Tuple, Dict, List, Callable
from pathlib import Path
import time


# ==============================================================================
# EXERCISE 1: Create TextDataset from Tokens
# Difficulty: Easy
# ==============================================================================

class Exercise01_TextDataset(Dataset):
    """
    EXERCISE 01: Implement a sliding window dataset for language modeling.

    Learning Objectives:
    - Understand the input-target relationship in causal LM
    - Learn the sliding window pattern for creating sequences
    - Practice computing dataset sizes from strides

    The dataset takes a list of tokens and creates (input, target) pairs where:
    - Input: [t0, t1, t2, ..., t(n-1)]
    - Target: [t1, t2, t3, ..., tn]

    This is the standard setup for next-token prediction.

    Args:
        tokens: List or tensor of token IDs
        seq_len: Sequence length for each example
        stride: Step size between windows (default: seq_len for no overlap)

    Shape:
        Input: (seq_len,) - Token IDs
        Target: (seq_len,) - Next token IDs (shifted by 1)

    Example:
        >>> tokens = list(range(100))
        >>> dataset = Exercise01_TextDataset(tokens, seq_len=10, stride=10)
        >>> len(dataset)  # How many windows fit?
        9
        >>> input_ids, target_ids = dataset[0]
        >>> input_ids.shape, target_ids.shape
        (torch.Size([10]), torch.Size([10]))
        >>> # Verify: target[i] == input[i+1]
        >>> torch.equal(target_ids[:-1], input_ids[1:])
        True

    Theory reference: theory.md, Section "Data Loading and Preprocessing"

    Self-Assessment Questions:
    1. Why do we need seq_len + 1 tokens for each example?
    2. How does stride affect dataset size and token coverage?
    3. What's the difference between stride=seq_len and stride=1?
    """

    def __init__(
        self,
        tokens: List[int],
        seq_len: int,
        stride: Optional[int] = None
    ):
        # TODO: Implement initialization

        # Step 1: Convert tokens to tensor if needed
        # self.tokens = torch.tensor(tokens, dtype=torch.long) if isinstance(tokens, list) else tokens

        # Step 2: Store sequence length and stride
        # self.seq_len = seq_len
        # self.stride = stride if stride is not None else seq_len

        # Step 3: Calculate number of valid windows
        # We need seq_len + 1 tokens for each example (seq_len input + 1 target)
        # total_tokens = len(self.tokens)
        # window_size = seq_len + 1

        # Step 4: Validate we have enough tokens
        # if total_tokens < window_size:
        #     raise ValueError(f"Not enough tokens...")

        # Step 5: Compute number of windows
        # Number of valid starting positions
        # self.num_windows = (total_tokens - window_size) // self.stride + 1

        pass  # Remove this and add your implementation

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        # TODO: Return the number of windows
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair for language modeling.

        Args:
            idx: Index of example

        Returns:
            Tuple of (input_ids, target_ids)
        """
        # TODO: Implement this method

        # Step 1: Calculate starting position
        # start_idx = idx * self.stride

        # Step 2: Extract window of size seq_len + 1
        # window = self.tokens[start_idx : start_idx + self.seq_len + 1]

        # Step 3: Split into input (first seq_len) and target (last seq_len)
        # input_ids = window[:-1]   # [t0, t1, ..., t(n-1)]
        # target_ids = window[1:]   # [t1, t2, ..., tn]

        # Step 4: Return tuple
        # return input_ids, target_ids

        pass

        # Uncomment to test:
        # assert input_ids.shape[0] == self.seq_len
        # assert target_ids.shape[0] == self.seq_len


# ==============================================================================
# EXERCISE 2: Character-Level Tokenizer
# Difficulty: Easy
# ==============================================================================

class Exercise02_CharTokenizer:
    """
    EXERCISE 02: Implement a simple character-level tokenizer.

    Learning Objectives:
    - Understand vocabulary building from text
    - Learn encode/decode for tokenization
    - Practice bidirectional mappings (char ↔ idx)

    The tokenizer:
    1. Builds vocab from unique characters in training text
    2. Encodes text to token IDs
    3. Decodes token IDs back to text

    Example:
        >>> tokenizer = Exercise02_CharTokenizer()
        >>> tokenizer.fit("hello world")
        >>> tokenizer.vocab_size
        8
        >>> tokens = tokenizer.encode("hello")
        >>> tokenizer.decode(tokens)
        'hello'

    Theory reference: theory.md, Section "Data Loading and Preprocessing"

    Self-Assessment Questions:
    1. Why do we sort characters when building vocab?
    2. What happens if we try to encode a character not in vocab?
    3. How would you add special tokens (PAD, UNK, BOS, EOS)?
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize tokenizer with optional pre-built vocabulary.

        Args:
            vocab: Optional character to index mapping
        """
        # TODO: Implement initialization

        # Step 1: Initialize vocab and inverse vocab
        # self.vocab = vocab if vocab is not None else {}
        # self.inverse_vocab = {idx: char for char, idx in self.vocab.items()} if vocab else {}

        pass

    def fit(self, text: str):
        """
        Build vocabulary from text.

        Args:
            text: Training text to build vocabulary from
        """
        # TODO: Implement this method

        # Step 1: Get unique characters (sorted for consistency)
        # unique_chars = sorted(set(text))

        # Step 2: Create char → idx mapping
        # self.vocab = {char: idx for idx, char in enumerate(unique_chars)}

        # Step 3: Create idx → char mapping (for decoding)
        # self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

        pass

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # TODO: Implement this method

        # Convert each character to its token ID
        # return [self.vocab[char] for char in text]

        pass

    def decode(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        # TODO: Implement this method

        # Convert each token ID back to character and join
        # return ''.join([self.inverse_vocab[token] for token in tokens])

        pass

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        # TODO: Return the size of vocab
        pass


# ==============================================================================
# EXERCISE 3: Learning Rate Scheduler Configuration
# Difficulty: Easy
# ==============================================================================

def exercise_03_create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    peak_lr: float = 1e-3,
    min_lr: float = 1e-5,
    scheduler_type: str = "cosine"
) -> Tuple[LRScheduler, List[float]]:
    """
    EXERCISE 03: Create and visualize a learning rate scheduler.

    Learning Objectives:
    - Understand warmup + decay schedules
    - Learn to configure schedulers
    - Visualize LR curves over training

    You'll create either:
    - Cosine schedule: Linear warmup → Cosine decay
    - Linear schedule: Linear warmup → Linear decay

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate (after warmup)
        min_lr: Minimum learning rate (end of training)
        scheduler_type: "cosine" or "linear"

    Returns:
        Tuple of (scheduler, lr_history)
        - scheduler: Configured LRScheduler
        - lr_history: List of LR values for each step

    Example:
        >>> model = nn.Linear(10, 10)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler, lrs = exercise_03_create_scheduler(
        ...     optimizer, warmup_steps=100, total_steps=1000
        ... )
        >>> len(lrs)
        1001
        >>> lrs[0] < lrs[100]  # Warmup increases LR
        True
        >>> lrs[100] > lrs[-1]  # Decay decreases LR
        True

    Theory reference: theory.md, Section "Learning Rate Schedules"

    Self-Assessment Questions:
    1. Why do we need warmup at the start of training?
    2. What's the difference between cosine and linear decay?
    3. At what step does warmup end and decay begin?
    """
    # TODO: Implement this exercise

    # Step 1: Import the scheduler classes
    # from tiny_transformer.training.scheduler import (
    #     WarmupCosineScheduler,
    #     WarmupLinearScheduler
    # )

    # Step 2: Create the appropriate scheduler
    # if scheduler_type == "cosine":
    #     scheduler = WarmupCosineScheduler(
    #         optimizer,
    #         warmup_steps=warmup_steps,
    #         total_steps=total_steps,
    #         peak_lr=peak_lr,
    #         min_lr=min_lr
    #     )
    # elif scheduler_type == "linear":
    #     scheduler = WarmupLinearScheduler(...)
    # else:
    #     raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Step 3: Collect learning rates for all steps
    # lr_history = []
    # for step in range(total_steps + 1):
    #     lr_history.append(scheduler.get_lr()[0])
    #     scheduler.step()

    # Step 4: Reset scheduler to initial state
    # scheduler = WarmupCosineScheduler(...) or WarmupLinearScheduler(...)

    # return scheduler, lr_history
    pass


# ==============================================================================
# EXERCISE 4: Create Data Loaders with Sliding Windows
# Difficulty: Medium
# ==============================================================================

def exercise_04_create_data_loaders(
    text: str,
    seq_len: int = 128,
    batch_size: int = 32,
    train_split: float = 0.9,
    stride: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    EXERCISE 04: Create train/val data loaders from text.

    Learning Objectives:
    - Practice end-to-end dataset creation
    - Learn train/val splitting for language modeling
    - Configure DataLoader for efficient batching

    You'll:
    1. Tokenize text with CharTokenizer
    2. Split into train/val tokens
    3. Create TextDataset for each split
    4. Wrap in DataLoaders with proper settings

    Args:
        text: Input text to tokenize
        seq_len: Sequence length for each example
        batch_size: Batch size
        train_split: Fraction of data for training (0-1)
        stride: Stride for sliding window (default: seq_len)

    Returns:
        Tuple of (train_loader, val_loader, vocab_size)

    Example:
        >>> text = "hello world! " * 100  # Sample text
        >>> train_loader, val_loader, vocab_size = exercise_04_create_data_loaders(
        ...     text, seq_len=32, batch_size=8
        ... )
        >>> # Check batch shapes
        >>> inputs, targets = next(iter(train_loader))
        >>> inputs.shape, targets.shape
        (torch.Size([8, 32]), torch.Size([8, 32]))

    Theory reference: theory.md, Section "Data Loading and Preprocessing"

    Self-Assessment Questions:
    1. Why do we split tokens (not text) for train/val?
    2. Should we shuffle the validation DataLoader?
    3. What does pin_memory do in DataLoader?
    """
    # TODO: Implement this exercise

    # Step 1: Create and fit tokenizer
    # tokenizer = Exercise02_CharTokenizer()
    # tokenizer.fit(text)
    # vocab_size = tokenizer.vocab_size

    # Step 2: Encode entire text
    # tokens = tokenizer.encode(text)

    # Step 3: Split into train/val
    # split_idx = int(len(tokens) * train_split)
    # train_tokens = tokens[:split_idx]
    # val_tokens = tokens[split_idx:]

    # Step 4: Create datasets
    # train_dataset = Exercise01_TextDataset(train_tokens, seq_len, stride)
    # val_dataset = Exercise01_TextDataset(val_tokens, seq_len, seq_len)  # No overlap for val

    # Step 5: Create data loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,      # Shuffle training data
    #     pin_memory=True,   # Faster GPU transfer
    #     num_workers=0      # Single-threaded for simplicity
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,     # Don't shuffle validation
    #     pin_memory=True,
    #     num_workers=0
    # )

    # return train_loader, val_loader, vocab_size
    pass


# ==============================================================================
# EXERCISE 5: Basic Training Loop
# Difficulty: Medium
# ==============================================================================

def exercise_05_basic_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    learning_rate: float = 1e-3
) -> List[float]:
    """
    EXERCISE 05: Implement a basic training loop.

    Learning Objectives:
    - Understand the training loop structure
    - Practice forward/backward/update cycle
    - Learn loss computation for language modeling

    Training loop structure:
    1. Set model to train mode
    2. For each batch:
       a. Forward pass
       b. Compute loss
       c. Backward pass
       d. Optimizer step
       e. Track metrics

    Args:
        model: TinyTransformerLM or similar
        train_loader: Training data loader
        num_steps: Number of training steps
        learning_rate: Learning rate

    Returns:
        List of loss values (one per step)

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        >>> # Create dummy data loader
        >>> train_loader = ...
        >>> losses = exercise_05_basic_training_loop(model, train_loader, num_steps=50)
        >>> len(losses)
        50
        >>> losses[0] > losses[-1]  # Loss should decrease
        True

    Theory reference: theory.md, Section "Training Loop"

    Self-Assessment Questions:
    1. Why do we use cross_entropy instead of MSE?
    2. What does optimizer.zero_grad() do?
    3. Why do we call loss.backward() before optimizer.step()?
    """
    # TODO: Implement this exercise

    # Step 1: Set model to training mode
    # model.train()

    # Step 2: Create optimizer
    # optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Step 3: Track losses
    # losses = []

    # Step 4: Training loop
    # step = 0
    # while step < num_steps:
    #     for input_ids, target_ids in train_loader:
    #         # Step 4a: Forward pass
    #         logits, _ = model(input_ids)
    #
    #         # Step 4b: Compute loss
    #         # Reshape: (batch, seq_len, vocab) → (batch*seq_len, vocab)
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             target_ids.view(-1)
    #         )
    #
    #         # Step 4c: Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Step 4d: Update weights
    #         optimizer.step()
    #
    #         # Step 4e: Track loss
    #         losses.append(loss.item())
    #         step += 1
    #
    #         if step >= num_steps:
    #             break

    # return losses
    pass


# ==============================================================================
# EXERCISE 6: Add Gradient Clipping
# Difficulty: Medium
# ==============================================================================

def exercise_06_training_with_grad_clip(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0
) -> Tuple[List[float], List[float]]:
    """
    EXERCISE 06: Add gradient clipping to prevent exploding gradients.

    Learning Objectives:
    - Understand gradient clipping
    - Learn to compute and monitor gradient norms
    - Practice numerical stability in training

    Gradient clipping:
    - Prevents gradient explosion
    - Stabilizes training
    - Especially important for RNNs and transformers

    Args:
        model: PyTorch model
        train_loader: Training data loader
        num_steps: Number of training steps
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm (clip threshold)

    Returns:
        Tuple of (losses, grad_norms)
        - losses: Loss value per step
        - grad_norms: Gradient norm per step (before clipping)

    Example:
        >>> losses, grad_norms = exercise_06_training_with_grad_clip(
        ...     model, train_loader, num_steps=50, max_grad_norm=1.0
        ... )
        >>> max(grad_norms) <= 1.0  # Gradients are clipped
        True

    Theory reference: theory.md, Section "Gradient Clipping"

    Self-Assessment Questions:
    1. Why do gradients explode in transformers?
    2. What's the difference between clipping by norm vs value?
    3. How do you choose the clipping threshold?
    """
    # TODO: Implement this exercise

    # Step 1: Import gradient clipping utility
    # from tiny_transformer.training.utils import clip_gradient_norm

    # Step 2: Set up model and optimizer
    # model.train()
    # optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Step 3: Track metrics
    # losses = []
    # grad_norms = []

    # Step 4: Training loop
    # step = 0
    # while step < num_steps:
    #     for input_ids, target_ids in train_loader:
    #         # Forward pass
    #         logits, _ = model(input_ids)
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             target_ids.view(-1)
    #         )
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Gradient clipping (returns norm before clipping)
    #         grad_norm = clip_gradient_norm(model, max_grad_norm)
    #
    #         # Optimizer step
    #         optimizer.step()
    #
    #         # Track metrics
    #         losses.append(loss.item())
    #         grad_norms.append(grad_norm)
    #         step += 1
    #
    #         if step >= num_steps:
    #             break

    # return losses, grad_norms
    pass


# ==============================================================================
# EXERCISE 7: Implement Checkpointing
# Difficulty: Medium
# ==============================================================================

def exercise_07_training_with_checkpoints(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    save_interval: int = 25,
    checkpoint_dir: str = "checkpoints"
) -> List[str]:
    """
    EXERCISE 07: Add checkpointing to save model during training.

    Learning Objectives:
    - Learn to save/load model checkpoints
    - Understand what to include in checkpoints
    - Practice resumable training

    Checkpoints should include:
    - Model state dict (weights)
    - Optimizer state dict (momentum, etc.)
    - Training step/epoch
    - Best validation loss
    - Configuration/hyperparameters

    Args:
        model: PyTorch model
        train_loader: Training data loader
        num_steps: Number of training steps
        save_interval: Save checkpoint every N steps
        checkpoint_dir: Directory to save checkpoints

    Returns:
        List of checkpoint file paths created

    Example:
        >>> checkpoint_paths = exercise_07_training_with_checkpoints(
        ...     model, train_loader, num_steps=100, save_interval=25
        ... )
        >>> len(checkpoint_paths)  # 100 / 25 = 4 checkpoints
        4

    Theory reference: theory.md, Section "Checkpointing"

    Self-Assessment Questions:
    1. Why save optimizer state in checkpoints?
    2. How do you resume training from a checkpoint?
    3. What's the difference between state_dict and the full model?
    """
    # TODO: Implement this exercise

    # Step 1: Create checkpoint directory
    # checkpoint_path = Path(checkpoint_dir)
    # checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Setup training
    # model.train()
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    # checkpoint_files = []

    # Step 3: Training loop with checkpointing
    # step = 0
    # while step < num_steps:
    #     for input_ids, target_ids in train_loader:
    #         # Forward + backward + update (same as before)
    #         logits, _ = model(input_ids)
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             target_ids.view(-1)
    #         )
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         step += 1
    #
    #         # Save checkpoint at intervals
    #         if step % save_interval == 0:
    #             checkpoint = {
    #                 'step': step,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': loss.item(),
    #             }
    #
    #             filename = checkpoint_path / f"checkpoint_step_{step}.pt"
    #             torch.save(checkpoint, filename)
    #             checkpoint_files.append(str(filename))
    #             print(f"Saved checkpoint: {filename}")
    #
    #         if step >= num_steps:
    #             break

    # return checkpoint_files
    pass


# ==============================================================================
# EXERCISE 8: Calculate Perplexity
# Difficulty: Medium
# ==============================================================================

def exercise_08_compute_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    EXERCISE 08: Compute perplexity on a dataset.

    Learning Objectives:
    - Understand perplexity as a language modeling metric
    - Learn evaluation without gradient computation
    - Practice setting model to eval mode

    Perplexity = exp(average_loss)
    - Lower is better
    - Measures "surprise" of the model
    - Common metric for language models

    Args:
        model: Trained language model
        data_loader: Data loader (train or val)
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, perplexity)

    Example:
        >>> loss, ppl = exercise_08_compute_perplexity(model, val_loader)
        >>> print(f"Validation Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    Theory reference: theory.md, Section "Loss Functions"

    Self-Assessment Questions:
    1. Why is perplexity exp(loss)?
    2. What does a perplexity of 10 mean?
    3. Why do we use @torch.no_grad() for evaluation?
    """
    # TODO: Implement this exercise

    # Step 1: Import perplexity utility
    # from tiny_transformer.training.utils import compute_perplexity

    # Step 2: Set model to eval mode
    # model.eval()
    # model = model.to(device)

    # Step 3: Evaluate without gradients
    # total_loss = 0.0
    # total_tokens = 0
    #
    # with torch.no_grad():
    #     for input_ids, target_ids in data_loader:
    #         input_ids = input_ids.to(device)
    #         target_ids = target_ids.to(device)
    #
    #         # Forward pass
    #         logits, _ = model(input_ids)
    #
    #         # Compute loss
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             target_ids.view(-1),
    #             reduction='sum'  # Sum instead of mean
    #         )
    #
    #         total_loss += loss.item()
    #         total_tokens += target_ids.numel()

    # Step 4: Compute average loss and perplexity
    # avg_loss = total_loss / total_tokens
    # perplexity = compute_perplexity(avg_loss)

    # return avg_loss, perplexity
    pass


# ==============================================================================
# EXERCISE 9: Warmup Schedule from Scratch
# Difficulty: Hard
# ==============================================================================

class Exercise09_CustomWarmupScheduler(LRScheduler):
    """
    EXERCISE 09: Implement warmup + cosine decay schedule from scratch.

    Learning Objectives:
    - Understand the math behind learning rate schedules
    - Learn to implement custom schedulers
    - Practice cosine annealing formula

    Formula:
        if step <= warmup_steps:
            lr = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        peak_lr: Peak learning rate
        min_lr: Minimum learning rate

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = Exercise09_CustomWarmupScheduler(
        ...     optimizer, warmup_steps=100, total_steps=1000, peak_lr=1e-3
        ... )
        >>> for step in range(1000):
        ...     optimizer.step()
        ...     scheduler.step()

    Theory reference: theory.md, Section "Learning Rate Schedules"

    Self-Assessment Questions:
    1. Why does cosine decay work well for transformers?
    2. What's the shape of the cosine curve?
    3. How would you modify this for linear decay?
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0
    ):
        # TODO: Implement initialization

        # Step 1: Store parameters
        # self.warmup_steps = warmup_steps
        # self.total_steps = total_steps
        # self.peak_lr = peak_lr
        # self.min_lr = min_lr

        # Step 2: Call parent constructor
        # super().__init__(optimizer)

        pass

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.

        Returns:
            List of learning rates (one per parameter group)
        """
        # TODO: Implement learning rate computation

        # Step 1: Get current step (stored in self.last_epoch by PyTorch)
        # step = self.last_epoch

        # Step 2: Warmup phase
        # if step <= self.warmup_steps:
        #     if self.warmup_steps == 0:
        #         lr = self.peak_lr
        #     else:
        #         # Linear warmup
        #         lr = self.peak_lr * (step / self.warmup_steps)

        # Step 3: Decay phase
        # else:
        #     # Compute progress [0, 1]
        #     decay_steps = self.total_steps - self.warmup_steps
        #     progress = (step - self.warmup_steps) / decay_steps
        #     progress = min(progress, 1.0)  # Clamp to [0, 1]
        #
        #     # Cosine annealing
        #     lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
        #         1.0 + math.cos(math.pi * progress)
        #     )

        # Step 4: Return list of LRs
        # return [lr for _ in self.optimizer.param_groups]

        pass


# ==============================================================================
# EXERCISE 10: Custom Trainer with Logging
# Difficulty: Hard
# ==============================================================================

class Exercise10_CustomTrainer:
    """
    EXERCISE 10: Build a custom trainer with comprehensive logging.

    Learning Objectives:
    - Integrate all training components
    - Implement proper logging and monitoring
    - Create reusable training infrastructure

    The trainer should support:
    - Training loop with progress tracking
    - Validation every N steps
    - Learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Comprehensive logging

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        warmup_steps: Warmup steps for LR schedule
        grad_clip: Max gradient norm (None = no clipping)
        eval_interval: Evaluate every N steps
        device: Training device

    Example:
        >>> trainer = Exercise10_CustomTrainer(
        ...     model, train_loader, val_loader,
        ...     learning_rate=1e-3, num_epochs=10
        ... )
        >>> metrics = trainer.train()
        >>> print(f"Final loss: {metrics['final_loss']:.4f}")

    Theory reference: theory.md, Section "Training Loop"

    Self-Assessment Questions:
    1. When should you evaluate on validation set?
    2. How do you track the best model during training?
    3. What metrics should you log for debugging?
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        grad_clip: Optional[float] = 1.0,
        eval_interval: int = 100,
        device: str = "cpu"
    ):
        # TODO: Implement initialization

        # Step 1: Store attributes
        # self.model = model.to(device)
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.num_epochs = num_epochs
        # self.grad_clip = grad_clip
        # self.eval_interval = eval_interval
        # self.device = device

        # Step 2: Create optimizer
        # self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Step 3: Create learning rate scheduler
        # total_steps = num_epochs * len(train_loader)
        # self.scheduler = Exercise09_CustomWarmupScheduler(
        #     self.optimizer,
        #     warmup_steps=warmup_steps,
        #     total_steps=total_steps,
        #     peak_lr=learning_rate,
        #     min_lr=learning_rate * 0.1
        # )

        # Step 4: Initialize tracking
        # self.step = 0
        # self.best_val_loss = float('inf')

        pass

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        # TODO: Implement epoch training

        # Step 1: Set model to train mode
        # self.model.train()
        # total_loss = 0.0
        # num_batches = 0

        # Step 2: Training loop
        # for input_ids, target_ids in self.train_loader:
        #     input_ids = input_ids.to(self.device)
        #     target_ids = target_ids.to(self.device)
        #
        #     # Forward pass
        #     logits, _ = self.model(input_ids)
        #     loss = F.cross_entropy(
        #         logits.view(-1, logits.size(-1)),
        #         target_ids.view(-1)
        #     )
        #
        #     # Backward pass
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #
        #     # Gradient clipping
        #     if self.grad_clip is not None:
        #         from tiny_transformer.training.utils import clip_gradient_norm
        #         clip_gradient_norm(self.model, self.grad_clip)
        #
        #     # Optimizer step
        #     self.optimizer.step()
        #     self.scheduler.step()
        #
        #     # Track metrics
        #     total_loss += loss.item()
        #     num_batches += 1
        #     self.step += 1
        #
        #     # Periodic evaluation
        #     if self.val_loader and self.step % self.eval_interval == 0:
        #         val_loss = self.evaluate()
        #         print(f"Step {self.step} | Val Loss: {val_loss:.4f}")
        #         self.model.train()

        # Step 3: Return epoch metrics
        # return {'loss': total_loss / num_batches}

        pass

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        # TODO: Implement evaluation

        # Similar to exercise_08_compute_perplexity

        pass

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Dictionary of final metrics
        """
        # TODO: Implement main training loop

        # Step 1: Print training info
        # print("=" * 70)
        # print("Starting Training")
        # print("=" * 70)
        # print(f"Device: {self.device}")
        # print(f"Training samples: {len(self.train_loader.dataset)}")
        # print()

        # Step 2: Training loop
        # for epoch in range(self.num_epochs):
        #     print(f"Epoch {epoch + 1}/{self.num_epochs}")
        #     epoch_metrics = self.train_one_epoch()
        #     print(f"  Train Loss: {epoch_metrics['loss']:.4f}")

        # Step 3: Final evaluation
        # if self.val_loader:
        #     final_val_loss = self.evaluate()
        #     print(f"\nFinal Validation Loss: {final_val_loss:.4f}")

        # Step 4: Return metrics
        # return {'final_loss': epoch_metrics['loss']}

        pass


# ==============================================================================
# EXERCISE 11: Early Stopping
# Difficulty: Hard
# ==============================================================================

class Exercise11_EarlyStopping:
    """
    EXERCISE 11: Implement early stopping to prevent overfitting.

    Learning Objectives:
    - Understand early stopping criteria
    - Learn patience-based stopping
    - Practice model selection

    Early stopping monitors validation loss and stops training if:
    - Loss doesn't improve for `patience` evaluations
    - Helps prevent overfitting
    - Saves training time

    Args:
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better) or 'max' for accuracy

    Example:
        >>> early_stopping = Exercise11_EarlyStopping(patience=5)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered!")
        ...         break

    Theory reference: theory.md, Section "Early Stopping"

    Self-Assessment Questions:
    1. Why does early stopping help prevent overfitting?
    2. How do you choose the patience value?
    3. What's the trade-off between patience and training time?
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        # TODO: Implement initialization

        # Step 1: Store parameters
        # self.patience = patience
        # self.min_delta = min_delta
        # self.mode = mode

        # Step 2: Initialize tracking variables
        # self.best_score = float('inf') if mode == 'min' else float('-inf')
        # self.counter = 0
        # self.should_stop = False

        pass

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric (loss or accuracy)

        Returns:
            True if training should stop, False otherwise
        """
        # TODO: Implement early stopping logic

        # Step 1: Check if score improved
        # if self.mode == 'min':
        #     improved = score < (self.best_score - self.min_delta)
        # else:
        #     improved = score > (self.best_score + self.min_delta)

        # Step 2: Update based on improvement
        # if improved:
        #     self.best_score = score
        #     self.counter = 0
        # else:
        #     self.counter += 1

        # Step 3: Check if we should stop
        # if self.counter >= self.patience:
        #     self.should_stop = True
        #     return True

        # return False

        pass

    def reset(self):
        """Reset early stopping state."""
        # TODO: Reset all tracking variables
        pass


# ==============================================================================
# EXERCISE 12: End-to-End Training Pipeline
# Difficulty: Very Hard
# ==============================================================================

def exercise_12_shakespeare_training(
    seq_len: int = 128,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    EXERCISE 12: Complete end-to-end training on Shakespeare dataset.

    Learning Objectives:
    - Integrate all components into working system
    - Train on real text data
    - Evaluate and generate text
    - Practice complete ML pipeline

    This exercise brings together everything:
    1. Data loading and preprocessing
    2. Model creation and configuration
    3. Training with schedulers and clipping
    4. Validation and perplexity tracking
    5. Text generation from trained model

    Args:
        seq_len: Sequence length
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        device: Training device

    Returns:
        Dictionary with:
        - model: Trained model
        - train_losses: List of training losses
        - val_losses: List of validation losses
        - perplexities: List of validation perplexities
        - generated_text: Sample generated text

    Example:
        >>> results = exercise_12_shakespeare_training(
        ...     seq_len=128, batch_size=32, num_epochs=5
        ... )
        >>> print(f"Final perplexity: {results['perplexities'][-1]:.2f}")
        >>> print(f"Generated text: {results['generated_text']}")

    Theory reference: theory.md, all sections

    Self-Assessment Questions:
    1. How do you know if your model is learning?
    2. What perplexity indicates good performance?
    3. How would you improve the model further?
    """
    # TODO: Implement complete training pipeline

    # Step 1: Create sample Shakespeare-like text (or load from file)
    # text = """
    # To be, or not to be, that is the question:
    # Whether 'tis nobler in the mind to suffer
    # The slings and arrows of outrageous fortune,
    # Or to take arms against a sea of troubles
    # And by opposing end them.
    # """ * 100  # Repeat for more data

    # Step 2: Create data loaders
    # train_loader, val_loader, vocab_size = exercise_04_create_data_loaders(
    #     text, seq_len=seq_len, batch_size=batch_size
    # )

    # Step 3: Create model
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(
    #     vocab_size=vocab_size,
    #     d_model=128,
    #     n_heads=4,
    #     n_layers=3,
    #     d_ff=512,
    #     max_len=seq_len,
    #     tie_weights=True
    # ).to(device)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 4: Create trainer
    # trainer = Exercise10_CustomTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     learning_rate=learning_rate,
    #     num_epochs=num_epochs,
    #     warmup_steps=warmup_steps,
    #     grad_clip=1.0,
    #     eval_interval=50,
    #     device=device
    # )

    # Step 5: Train
    # metrics = trainer.train()

    # Step 6: Evaluate final perplexity
    # final_loss, final_ppl = exercise_08_compute_perplexity(model, val_loader, device)

    # Step 7: Generate sample text
    # model.eval()
    # # Create tokenizer for decoding
    # tokenizer = Exercise02_CharTokenizer()
    # tokenizer.fit(text)
    #
    # # Generate from a prompt
    # prompt = "To be, or not to be"
    # prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #     # Simple greedy generation
    #     generated_tokens = prompt_tokens.clone()
    #     for _ in range(100):
    #         logits, _ = model(generated_tokens)
    #         next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #         generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
    #
    # generated_text = tokenizer.decode(generated_tokens[0].cpu().tolist())

    # Step 8: Return all results
    # return {
    #     'model': model,
    #     'final_loss': final_loss,
    #     'final_perplexity': final_ppl,
    #     'generated_text': generated_text,
    #     'vocab_size': vocab_size,
    # }

    pass


# ==============================================================================
# Testing and Validation
# ==============================================================================

def run_all_tests():
    """
    Run tests for all exercises.

    Uncomment each test as you complete the corresponding exercise.
    """
    print("=" * 70)
    print("Module 06: Training - Exercise Tests")
    print("=" * 70)

    # Test Exercise 1
    # print("\n[TEST 1] TextDataset")
    # tokens = list(range(100))
    # dataset = Exercise01_TextDataset(tokens, seq_len=10, stride=10)
    # assert len(dataset) == 9, f"Expected 9 windows, got {len(dataset)}"
    # input_ids, target_ids = dataset[0]
    # assert input_ids.shape == (10,), f"Expected (10,), got {input_ids.shape}"
    # assert torch.equal(target_ids[:-1], input_ids[1:]), "Target should be input shifted by 1"
    # print("✓ Exercise 1 passed!")

    # Test Exercise 2
    # print("\n[TEST 2] CharTokenizer")
    # tokenizer = Exercise02_CharTokenizer()
    # tokenizer.fit("hello world")
    # assert tokenizer.vocab_size > 0, "Vocab size should be positive"
    # tokens = tokenizer.encode("hello")
    # decoded = tokenizer.decode(tokens)
    # assert decoded == "hello", f"Expected 'hello', got '{decoded}'"
    # print("✓ Exercise 2 passed!")

    # Test Exercise 3
    # print("\n[TEST 3] Learning Rate Scheduler")
    # model = nn.Linear(10, 10)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler, lrs = exercise_03_create_scheduler(optimizer, warmup_steps=100, total_steps=1000)
    # assert len(lrs) == 1001, f"Expected 1001 LRs, got {len(lrs)}"
    # assert lrs[0] < lrs[100], "LR should increase during warmup"
    # assert lrs[100] > lrs[-1], "LR should decrease during decay"
    # print("✓ Exercise 3 passed!")

    # Test Exercise 4
    # print("\n[TEST 4] Create Data Loaders")
    # text = "hello world! " * 100
    # train_loader, val_loader, vocab_size = exercise_04_create_data_loaders(
    #     text, seq_len=32, batch_size=8
    # )
    # inputs, targets = next(iter(train_loader))
    # assert inputs.shape[0] <= 8, f"Batch size should be ≤8, got {inputs.shape[0]}"
    # assert inputs.shape[1] == 32, f"Seq len should be 32, got {inputs.shape[1]}"
    # print("✓ Exercise 4 passed!")

    # Test Exercise 5
    # print("\n[TEST 5] Basic Training Loop")
    # from tiny_transformer.model import TinyTransformerLM
    # model = TinyTransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # # Create dummy data loader
    # dummy_tokens = list(range(1000))
    # dummy_dataset = Exercise01_TextDataset(dummy_tokens, seq_len=32, stride=32)
    # dummy_loader = DataLoader(dummy_dataset, batch_size=8)
    # losses = exercise_05_basic_training_loop(model, dummy_loader, num_steps=10)
    # assert len(losses) == 10, f"Expected 10 losses, got {len(losses)}"
    # print("✓ Exercise 5 passed!")

    # Test Exercise 6
    # print("\n[TEST 6] Gradient Clipping")
    # losses, grad_norms = exercise_06_training_with_grad_clip(
    #     model, dummy_loader, num_steps=10, max_grad_norm=1.0
    # )
    # assert len(grad_norms) == 10, f"Expected 10 grad norms, got {len(grad_norms)}"
    # print("✓ Exercise 6 passed!")

    # Test Exercise 7
    # print("\n[TEST 7] Checkpointing")
    # checkpoint_paths = exercise_07_training_with_checkpoints(
    #     model, dummy_loader, num_steps=50, save_interval=25
    # )
    # assert len(checkpoint_paths) == 2, f"Expected 2 checkpoints, got {len(checkpoint_paths)}"
    # print("✓ Exercise 7 passed!")

    # Test Exercise 8
    # print("\n[TEST 8] Perplexity")
    # loss, ppl = exercise_08_compute_perplexity(model, dummy_loader)
    # assert ppl > 0, "Perplexity should be positive"
    # assert ppl == math.exp(loss), "Perplexity should equal exp(loss)"
    # print("✓ Exercise 8 passed!")

    # Test Exercise 9
    # print("\n[TEST 9] Custom Warmup Scheduler")
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    # scheduler = Exercise09_CustomWarmupScheduler(
    #     optimizer, warmup_steps=10, total_steps=100, peak_lr=1e-3
    # )
    # lrs = []
    # for _ in range(101):
    #     lrs.append(scheduler.get_lr()[0])
    #     scheduler.step()
    # assert lrs[0] < lrs[10], "Warmup should increase LR"
    # assert lrs[10] > lrs[-1], "Decay should decrease LR"
    # print("✓ Exercise 9 passed!")

    # Test Exercise 10
    # print("\n[TEST 10] Custom Trainer")
    # trainer = Exercise10_CustomTrainer(
    #     model, dummy_loader, dummy_loader,
    #     learning_rate=1e-3, num_epochs=1, warmup_steps=5
    # )
    # metrics = trainer.train()
    # assert 'final_loss' in metrics, "Should return final_loss"
    # print("✓ Exercise 10 passed!")

    # Test Exercise 11
    # print("\n[TEST 11] Early Stopping")
    # early_stopping = Exercise11_EarlyStopping(patience=3)
    # # Simulate improving then plateauing
    # assert not early_stopping(5.0), "Should not stop on first eval"
    # assert not early_stopping(4.0), "Should not stop when improving"
    # assert not early_stopping(4.0), "Should not stop (patience=3)"
    # assert not early_stopping(4.0), "Should not stop (patience=3)"
    # assert early_stopping(4.0), "Should stop after patience exhausted"
    # print("✓ Exercise 11 passed!")

    # Test Exercise 12
    # print("\n[TEST 12] Shakespeare Training")
    # results = exercise_12_shakespeare_training(
    #     seq_len=32, batch_size=8, num_epochs=1, device="cpu"
    # )
    # assert 'model' in results, "Should return trained model"
    # assert 'generated_text' in results, "Should return generated text"
    # assert len(results['generated_text']) > 0, "Should generate some text"
    # print("✓ Exercise 12 passed!")

    print("\n" + "=" * 70)
    print("All tests passed! Excellent work!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Module 06: Training - Exercises")
    print("=" * 70)
    print("\nWelcome! These exercises teach you to train transformer language models.")
    print("\nExercise Structure:")
    print("  01. Create TextDataset from Tokens (Easy)")
    print("  02. Character-Level Tokenizer (Easy)")
    print("  03. Learning Rate Scheduler Configuration (Easy)")
    print("  04. Create Data Loaders with Sliding Windows (Medium)")
    print("  05. Basic Training Loop (Medium)")
    print("  06. Add Gradient Clipping (Medium)")
    print("  07. Implement Checkpointing (Medium)")
    print("  08. Calculate Perplexity (Medium)")
    print("  09. Warmup Schedule from Scratch (Hard)")
    print("  10. Custom Trainer with Logging (Hard)")
    print("  11. Early Stopping (Hard)")
    print("  12. End-to-End Training Pipeline (Very Hard)")
    print("\nInstructions:")
    print("1. Work through exercises 1-12 in order")
    print("2. Read each docstring carefully")
    print("3. Implement the TODO sections")
    print("4. Uncomment test assertions to verify")
    print("5. Run this file to test your solutions")
    print("\nTips:")
    print("- Refer to theory.md for concepts")
    print("- Check tiny_transformer/training/*.py for reference")
    print("- Use print() to debug shapes and values")
    print("- Monitor loss curves for training progress")
    print("=" * 70)

    # Uncomment when ready to test:
    # run_all_tests()

    print("\nGood luck with your training! 🚀")
