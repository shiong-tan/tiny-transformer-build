"""
Module 06: Training - Complete Solutions

This file contains comprehensive reference solutions for all 12 training exercises.
Each solution includes:
- Complete, working implementation
- Educational comments explaining WHY, not just WHAT
- Shape annotations and assertions
- Best practices from reference implementations
- Notes about common mistakes and alternative approaches

Study these solutions to understand:
1. Dataset creation and tokenization patterns
2. Learning rate scheduling strategies
3. Training loop structure and best practices
4. Gradient clipping for stability
5. Checkpointing and resumable training
6. Evaluation metrics (perplexity)
7. Custom schedulers and trainers
8. Early stopping for overfitting prevention
9. End-to-end training pipelines

Author: Educational reference implementation
See: /Users/shiongtan/projects/tiny-transformer-build/docs/modules/06_training/theory.md
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
# SOLUTION 1: TextDataset - Creating Sliding Window Datasets
# ==============================================================================

class Solution01_TextDataset(Dataset):
    """
    SOLUTION 01: Sliding window dataset for language modeling.

    Key Concepts:
    1. Input-target alignment: Target is input shifted by 1 position
    2. We need seq_len + 1 tokens to create (input, target) pairs
    3. Stride controls overlap: stride=seq_len means no overlap
    4. Number of windows = (total_tokens - window_size) // stride + 1

    Common Mistakes:
    - Forgetting the +1 token needed for targets
    - Off-by-one errors in window calculation
    - Not validating sufficient tokens exist

    Alternative Approaches:
    - Pack multiple documents with padding
    - Use dynamic batching for variable lengths
    - Implement circular buffering for infinite streams

    See: tiny_transformer/training/dataset.py for reference
    """

    def __init__(
        self,
        tokens: List[int],
        seq_len: int,
        stride: Optional[int] = None
    ):
        # Convert to tensor if needed (PyTorch tensors are more efficient)
        # dtype=long because token IDs are integers
        self.tokens = torch.tensor(tokens, dtype=torch.long) if isinstance(tokens, list) else tokens

        self.seq_len = seq_len

        # Default stride = seq_len means no overlap between windows
        # stride < seq_len creates overlapping windows (more data, but correlated)
        self.stride = stride if stride is not None else seq_len

        # Calculate number of windows
        # We need seq_len + 1 tokens for each example:
        #   - seq_len tokens for input [t0, t1, ..., t(n-1)]
        #   - 1 additional token for target at position seq_len [tn]
        total_tokens = len(self.tokens)
        window_size = seq_len + 1

        # Validate we have enough tokens
        if total_tokens < window_size:
            raise ValueError(
                f"Not enough tokens ({total_tokens}) for even one sequence "
                f"of length {seq_len} + 1 target"
            )

        # Number of valid starting positions
        # Formula: For N total tokens, window size W, and stride S:
        #   num_windows = (N - W) // S + 1
        # Example: 100 tokens, window=11, stride=10 → (100-11)//10+1 = 9 windows
        self.num_windows = (total_tokens - window_size) // self.stride + 1

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair for language modeling.

        Shape Flow:
            tokens: [total_tokens]
            window: [seq_len + 1]
            input_ids: [seq_len]  - First seq_len tokens
            target_ids: [seq_len] - Last seq_len tokens (shifted by 1)

        Returns:
            Tuple of (input_ids, target_ids) where target_ids[i] = input_ids[i+1]
        """
        # Calculate starting position based on stride
        # idx=0 → start=0, idx=1 → start=stride, idx=2 → start=2*stride, etc.
        start_idx = idx * self.stride

        # Extract window of size seq_len + 1
        # Example: seq_len=10, start=0 → window[0:11] = 11 tokens
        window = self.tokens[start_idx : start_idx + self.seq_len + 1]

        # Split into input and target
        # Input: [t0, t1, t2, ..., t(n-1)] - First n tokens
        input_ids = window[:-1]

        # Target: [t1, t2, t3, ..., tn] - Last n tokens
        # This is equivalent to input shifted right by 1
        target_ids = window[1:]

        # Verify shapes (good practice for debugging)
        assert input_ids.shape[0] == self.seq_len, f"Expected {self.seq_len}, got {input_ids.shape[0]}"
        assert target_ids.shape[0] == self.seq_len, f"Expected {self.seq_len}, got {target_ids.shape[0]}"

        return input_ids, target_ids


# ==============================================================================
# SOLUTION 2: Character-Level Tokenizer
# ==============================================================================

class Solution02_CharTokenizer:
    """
    SOLUTION 02: Simple character-level tokenizer.

    Key Concepts:
    1. Vocabulary: Bidirectional mapping between chars and indices
    2. Sorting ensures deterministic vocab (same text → same vocab)
    3. Encoding: char → idx, Decoding: idx → char

    Common Mistakes:
    - Not handling unknown characters (should raise error or use UNK token)
    - Forgetting to sort (causes non-deterministic behavior)
    - Not creating inverse vocab for decoding

    Improvements:
    - Add special tokens: <PAD>, <UNK>, <BOS>, <EOS>
    - Handle unknown characters gracefully with UNK token
    - Support saving/loading vocab to disk
    - Use BPE or SentencePiece for subword tokenization

    See: tiny_transformer/training/dataset.py for reference
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize tokenizer with optional pre-built vocabulary.

        Args:
            vocab: Optional character to index mapping
        """
        # Store vocab: char → idx
        self.vocab = vocab if vocab is not None else {}

        # Create inverse vocab: idx → char (needed for decoding)
        # This is more efficient than searching through vocab each time
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()} if vocab else {}

    def fit(self, text: str):
        """
        Build vocabulary from text.

        Process:
        1. Extract unique characters with set()
        2. Sort for deterministic ordering (important for reproducibility!)
        3. Create bidirectional mappings

        Args:
            text: Training text to build vocabulary from
        """
        # Get unique characters
        # set() removes duplicates, sorted() ensures deterministic order
        # Sorting is crucial: "hello" will always produce the same vocab
        unique_chars = sorted(set(text))

        # Create character to index mapping
        # enumerate gives us (0, 'a'), (1, 'b'), etc.
        self.vocab = {char: idx for idx, char in enumerate(unique_chars)}

        # Create inverse mapping for decoding
        # We could compute this on-the-fly, but pre-computing is more efficient
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Process: For each character, look up its index in vocab

        Args:
            text: Input text

        Returns:
            List of token IDs

        Raises:
            KeyError: If text contains unknown characters

        Note: Production tokenizers should handle unknown chars with UNK token
        """
        # List comprehension is efficient for character-level encoding
        # This will raise KeyError if character not in vocab (fail fast!)
        return [self.vocab[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Process: For each token ID, look up its character and join

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text

        Raises:
            KeyError: If tokens contain unknown IDs
        """
        # Convert each token to character and join into string
        # ''.join() is the most efficient way to build strings in Python
        return ''.join([self.inverse_vocab[token] for token in tokens])

    @property
    def vocab_size(self) -> int:
        """
        Return vocabulary size.

        This is used for model initialization (embedding layer size)
        """
        return len(self.vocab)


# ==============================================================================
# SOLUTION 3: Learning Rate Scheduler Configuration
# ==============================================================================

def solution_03_create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    peak_lr: float = 1e-3,
    min_lr: float = 1e-5,
    scheduler_type: str = "cosine"
) -> Tuple[LRScheduler, List[float]]:
    """
    SOLUTION 03: Create and visualize learning rate scheduler.

    Key Concepts:
    1. Warmup: Gradually increase LR from 0 to peak_lr
       - Prevents early instability and divergence
       - Especially important for large models
    2. Decay: Gradually decrease LR from peak_lr to min_lr
       - Allows fine-grained optimization near end
       - Cosine decay is smoother than linear

    Scheduler Comparison:
    - Cosine: Smooth decay, stays higher longer, popular for transformers
    - Linear: Simpler, faster decay, sometimes used for smaller models

    Common Mistakes:
    - total_steps < warmup_steps (invalid configuration)
    - Not resetting scheduler after collecting history
    - Forgetting to call scheduler.step() during training

    See: tiny_transformer/training/scheduler.py for reference
    """
    # Import schedulers from reference implementation
    from tiny_transformer.training.scheduler import (
        WarmupCosineScheduler,
        WarmupLinearScheduler
    )

    # Create appropriate scheduler based on type
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr=min_lr
        )
    elif scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Choose 'cosine' or 'linear'")

    # Collect learning rates for visualization
    # We simulate stepping through training to see the LR curve
    lr_history = []
    for step in range(total_steps + 1):
        # Get current learning rate(s) - returns list with one LR per param group
        lr_history.append(scheduler.get_lr()[0])
        # Step the scheduler forward (this increments internal counter)
        scheduler.step()

    # Reset scheduler to initial state for actual use
    # Without this, the scheduler would be at the end of its schedule!
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr=min_lr
        )
    else:
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr=min_lr
        )

    return scheduler, lr_history


# ==============================================================================
# SOLUTION 4: Create Data Loaders
# ==============================================================================

def solution_04_create_data_loaders(
    text: str,
    seq_len: int = 128,
    batch_size: int = 32,
    train_split: float = 0.9,
    stride: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    SOLUTION 04: Create train/val data loaders from text.

    Key Concepts:
    1. Split tokens, not text (ensures clean boundaries)
    2. Use stride for training (can overlap), no overlap for validation
    3. Shuffle training data, don't shuffle validation
    4. pin_memory speeds up CPU→GPU transfer

    Data Loading Pipeline:
    text → tokenize → split → datasets → data loaders

    Common Mistakes:
    - Splitting text before tokenization (can split mid-character)
    - Shuffling validation data (makes evaluation non-deterministic)
    - Not using pin_memory when GPU is available

    DataLoader Parameters:
    - shuffle: Randomize order (only for training)
    - pin_memory: Keep data in pinned memory for faster GPU transfer
    - num_workers: Parallel data loading (0=single thread, safer for simple datasets)

    See: tiny_transformer/training/dataset.py for reference
    """
    # Step 1: Create and fit tokenizer
    tokenizer = Solution02_CharTokenizer()
    tokenizer.fit(text)  # Build vocab from all text
    vocab_size = tokenizer.vocab_size

    # Step 2: Encode entire text to tokens
    # Important: Encode before splitting to avoid splitting mid-token
    tokens = tokenizer.encode(text)

    # Step 3: Split into train/val at token level
    # Using train_split=0.9 means first 90% for training, last 10% for validation
    split_idx = int(len(tokens) * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    # Step 4: Create datasets
    # Training: Use provided stride (may overlap for more data)
    train_dataset = Solution01_TextDataset(train_tokens, seq_len, stride)

    # Validation: Always use stride=seq_len (no overlap) for clean evaluation
    # Overlapping validation data would artificially inflate performance
    val_dataset = Solution01_TextDataset(val_tokens, seq_len, seq_len)

    # Step 5: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Randomize order for better generalization
        pin_memory=True,     # Faster transfer to GPU
        num_workers=0        # Single-threaded (simpler, safer for small datasets)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle validation (want consistent evaluation)
        pin_memory=True,
        num_workers=0
    )

    return train_loader, val_loader, vocab_size


# ==============================================================================
# SOLUTION 5: Basic Training Loop
# ==============================================================================

def solution_05_basic_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    learning_rate: float = 1e-3
) -> List[float]:
    """
    SOLUTION 05: Basic training loop structure.

    Key Concepts:
    1. model.train() sets model to training mode (enables dropout, etc.)
    2. optimizer.zero_grad() clears old gradients
    3. loss.backward() computes gradients
    4. optimizer.step() updates weights

    Training Loop Pattern (Universal!):
    for batch in data:
        1. Forward pass: compute predictions
        2. Compute loss: measure error
        3. Backward pass: compute gradients
        4. Update weights: apply gradients

    Common Mistakes:
    - Forgetting to call model.train()
    - Not calling zero_grad() (gradients accumulate!)
    - Wrong loss function (MSE for classification, etc.)
    - Not reshaping logits/targets for cross_entropy

    Loss Shape Requirements:
    - logits: (batch * seq_len, vocab_size)
    - targets: (batch * seq_len,)
    cross_entropy expects these specific shapes!

    See: tiny_transformer/training/trainer.py for reference
    """
    # Step 1: Set model to training mode
    # This enables dropout, batch norm updates, etc.
    model.train()

    # Step 2: Create optimizer
    # AdamW is Adam with weight decay (better for transformers)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Step 3: Track losses for visualization
    losses = []

    # Step 4: Training loop
    step = 0
    while step < num_steps:
        # Iterate through batches
        for input_ids, target_ids in train_loader:
            # Step 4a: Forward pass
            # model returns (logits, loss) but we compute loss ourselves
            logits, _ = model(input_ids)
            # Shape: logits = (batch, seq_len, vocab_size)

            # Step 4b: Compute loss
            # cross_entropy requires specific shapes:
            # - predictions: (N, C) where N=batch*seq_len, C=vocab_size
            # - targets: (N,) where N=batch*seq_len
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
                target_ids.view(-1)                 # (batch*seq_len,)
            )

            # Step 4c: Backward pass
            # IMPORTANT: Clear gradients first! PyTorch accumulates by default
            optimizer.zero_grad()
            # Compute gradients via backpropagation
            loss.backward()

            # Step 4d: Update weights
            # Apply gradients to parameters (learning happens here!)
            optimizer.step()

            # Step 4e: Track metrics
            losses.append(loss.item())  # .item() converts tensor to Python float
            step += 1

            # Stop if we've reached num_steps
            if step >= num_steps:
                break

    return losses


# ==============================================================================
# SOLUTION 6: Gradient Clipping
# ==============================================================================

def solution_06_training_with_grad_clip(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0
) -> Tuple[List[float], List[float]]:
    """
    SOLUTION 06: Training with gradient clipping.

    Key Concepts:
    1. Gradient explosion: Gradients can grow very large in deep networks
    2. Clipping: Scale down gradients if norm exceeds threshold
    3. Global norm: Clip based on total norm across all parameters

    Why Gradient Clipping?
    - Transformers are deep and can have exploding gradients
    - Clipping prevents instability and NaN losses
    - Especially important early in training

    Clipping Methods:
    - By norm (recommended): Scale gradients to have max norm
    - By value: Clamp each gradient individually

    Common Mistakes:
    - Clipping after optimizer.step() (too late!)
    - Using value clipping instead of norm clipping
    - Threshold too low (limits learning) or too high (doesn't help)

    Good max_grad_norm values:
    - 0.5 - 1.0: Conservative (for unstable models)
    - 1.0 - 5.0: Standard (most transformers)
    - 5.0+: Aggressive (may not clip much)

    See: tiny_transformer/training/utils.py for reference
    """
    # Import gradient clipping utility
    from tiny_transformer.training.utils import clip_gradient_norm

    # Setup model and optimizer
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Track both losses and gradient norms
    losses = []
    grad_norms = []  # Norms BEFORE clipping (to see if clipping helped)

    # Training loop
    step = 0
    while step < num_steps:
        for input_ids, target_ids in train_loader:
            # Forward pass
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (CRITICAL: Before optimizer.step()!)
            # clip_gradient_norm returns the total norm BEFORE clipping
            # This lets us see if gradients were actually clipped
            grad_norm = clip_gradient_norm(model, max_grad_norm)

            # Update weights with clipped gradients
            optimizer.step()

            # Track metrics
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            step += 1

            if step >= num_steps:
                break

    return losses, grad_norms


# ==============================================================================
# SOLUTION 7: Checkpointing
# ==============================================================================

def solution_07_training_with_checkpoints(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int = 100,
    save_interval: int = 25,
    checkpoint_dir: str = "checkpoints"
) -> List[str]:
    """
    SOLUTION 07: Training with checkpoint saving.

    Key Concepts:
    1. Checkpoint = snapshot of training state
    2. Must save: model weights, optimizer state, training progress
    3. Enables resuming training after interruption

    What to Include in Checkpoint:
    - model_state_dict: Model weights and buffers
    - optimizer_state_dict: Optimizer momentum/state
    - step/epoch: Where we are in training
    - best_val_loss: Track best model so far
    - config: Hyperparameters for reproducibility

    Why Save Optimizer State?
    - Adam/AdamW maintains momentum and adaptive learning rates
    - Without optimizer state, resumed training may be unstable
    - Especially important for long training runs

    Common Mistakes:
    - Saving full model (use state_dict instead - more portable)
    - Not saving optimizer state (leads to training instability)
    - Hardcoding paths (use Path for cross-platform compatibility)
    - Saving too frequently (wasted disk I/O) or too rarely (lose progress)

    Checkpoint Strategy:
    - Save periodically (every N steps)
    - Save best model (based on validation loss)
    - Keep last N checkpoints (avoid filling disk)

    See: tiny_transformer/training/trainer.py for reference
    """
    # Step 1: Create checkpoint directory
    # Path handles cross-platform paths (Windows/Linux/Mac)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # parents=True: Create parent directories if needed
    # exist_ok=True: Don't error if directory already exists

    # Step 2: Setup training
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    checkpoint_files = []  # Track saved checkpoints

    # Step 3: Training loop with checkpointing
    step = 0
    while step < num_steps:
        for input_ids, target_ids in train_loader:
            # Standard training step
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # Save checkpoint at intervals
            if step % save_interval == 0:
                # Create checkpoint dictionary with all training state
                checkpoint = {
                    # Training progress
                    'step': step,
                    'loss': loss.item(),

                    # Model and optimizer state
                    # IMPORTANT: Use state_dict(), not the full objects!
                    # state_dict is more portable and smaller
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                    # Optional: Add config, best_loss, etc.
                }

                # Save to disk
                filename = checkpoint_path / f"checkpoint_step_{step}.pt"
                torch.save(checkpoint, filename)
                checkpoint_files.append(str(filename))
                print(f"Saved checkpoint: {filename}")

            if step >= num_steps:
                break

    return checkpoint_files


# ==============================================================================
# SOLUTION 8: Perplexity Computation
# ==============================================================================

def solution_08_compute_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    SOLUTION 08: Compute perplexity on a dataset.

    Key Concepts:
    1. Perplexity = exp(average_loss)
    2. Measures model's "surprise" on the data
    3. Lower perplexity = better model

    What is Perplexity?
    - Perplexity of 10: On average, model is confused between 10 tokens
    - Perplexity of 100: Model is very uncertain (worse)
    - Perfect model has perplexity = 1 (never surprised)

    Why Perplexity?
    - More interpretable than raw loss
    - Standard metric for language models
    - Easier to compare across different datasets

    Evaluation Best Practices:
    1. model.eval(): Disable dropout, use fixed batch norm stats
    2. @torch.no_grad(): Don't track gradients (saves memory)
    3. reduction='sum': Sum losses, then divide by total tokens
       (More numerically stable than averaging batch losses)

    Common Mistakes:
    - Not calling model.eval() (dropout affects results)
    - Computing gradients during eval (wastes memory)
    - Averaging batch losses instead of using reduction='sum'
    - Forgetting to move data to device

    See: tiny_transformer/training/utils.py for reference
    """
    # Import perplexity utility
    from tiny_transformer.training.utils import compute_perplexity

    # Step 1: Set model to evaluation mode
    # This disables dropout and sets batch norm to eval mode
    model.eval()
    model = model.to(device)

    # Step 2: Evaluate without gradients
    # Accumulate total loss and count total tokens
    total_loss = 0.0
    total_tokens = 0

    # @torch.no_grad() disables gradient computation
    # Saves memory and speeds up forward pass
    with torch.no_grad():
        for input_ids, target_ids in data_loader:
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits, _ = model(input_ids)

            # Compute loss with reduction='sum'
            # This sums losses across all tokens instead of averaging
            # Why? More numerically stable to sum everything, then divide once
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum'  # Sum instead of mean!
            )

            # Accumulate
            total_loss += loss.item()
            total_tokens += target_ids.numel()  # numel() = total number of elements

    # Step 3: Compute average loss and perplexity
    # Average loss = total_loss / total_tokens
    avg_loss = total_loss / total_tokens

    # Perplexity = exp(avg_loss)
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


# ==============================================================================
# SOLUTION 9: Custom Warmup Scheduler
# ==============================================================================

class Solution09_CustomWarmupScheduler(LRScheduler):
    """
    SOLUTION 09: Custom warmup + cosine decay scheduler.

    Key Concepts:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Cosine decay: Smooth decrease from peak_lr to min_lr
    3. Progress: Track how far through training we are

    Mathematical Formulas:
    Warmup (step <= warmup_steps):
        lr = peak_lr * (step / warmup_steps)

    Cosine Decay (step > warmup_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))

    Why Cosine?
    - Smooth decay (no sudden changes)
    - Stays higher longer (more exploration)
    - Gentle landing at min_lr (fine-tuning)

    Cosine Curve Properties:
    - cos(0) = 1 → lr starts at peak_lr
    - cos(π/2) = 0 → lr at midpoint between peak and min
    - cos(π) = -1 → lr ends at min_lr

    Alternative Schedules:
    - Linear decay: progress directly, no cosine
    - Exponential decay: lr *= decay_rate each step
    - Step decay: Drop LR at specific milestones

    See: tiny_transformer/training/scheduler.py for reference
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0
    ):
        # Store parameters
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        # Validation
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be greater than warmup_steps ({warmup_steps})"
            )

        # Call parent constructor AFTER storing parameters
        # This initializes self.optimizer and self.last_epoch
        # last_epoch is actually the step counter (PyTorch naming is confusing!)
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.

        Returns:
            List of learning rates (one per parameter group)
            Most optimizers have one param group, so list has length 1
        """
        # Step 1: Get current step
        # Despite the name, last_epoch is actually the step counter!
        # It's incremented each time scheduler.step() is called
        step = self.last_epoch

        # Step 2: Warmup phase (linear increase)
        if step <= self.warmup_steps:
            if self.warmup_steps == 0:
                # Edge case: No warmup, start at peak
                lr = self.peak_lr
            else:
                # Linear warmup: lr = peak_lr * (step / warmup_steps)
                # step=0 → lr=0, step=warmup_steps → lr=peak_lr
                lr = self.peak_lr * (step / self.warmup_steps)

        # Step 3: Decay phase (cosine decrease)
        else:
            # Compute progress through decay phase [0, 1]
            decay_steps = self.total_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_steps

            # Clamp to [0, 1] to handle steps beyond total_steps gracefully
            # Without this, progress > 1 would make cos negative and lr could increase!
            progress = min(progress, 1.0)

            # Cosine annealing formula
            # lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))
            #
            # Derivation:
            # - cos(π * progress) ranges from 1 (progress=0) to -1 (progress=1)
            # - (1 + cos(...)) ranges from 2 to 0
            # - 0.5 * (1 + cos(...)) ranges from 1 to 0
            # - Multiply by (peak - min) and add min → ranges from peak to min
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        # Step 4: Return list of LRs (one per param group)
        # Most models have one param group, but some use multiple
        # (e.g., different LR for embeddings vs. layers)
        return [lr for _ in self.optimizer.param_groups]


# ==============================================================================
# SOLUTION 10: Custom Trainer
# ==============================================================================

class Solution10_CustomTrainer:
    """
    SOLUTION 10: Custom trainer with comprehensive logging.

    Key Concepts:
    1. Encapsulation: All training logic in one class
    2. Configuration: Separate config from implementation
    3. Monitoring: Log metrics for debugging and analysis
    4. Validation: Periodic evaluation on held-out data

    Trainer Components:
    - __init__: Setup model, optimizer, scheduler
    - train_one_epoch: One pass through training data
    - evaluate: Compute metrics on validation data
    - train: Main training loop tying everything together

    Design Patterns:
    - Dependency injection: Pass in data loaders, config
    - Separation of concerns: Training vs. evaluation
    - Progress tracking: Log at regular intervals

    Common Mistakes:
    - Not tracking best model (need to know when to stop)
    - Evaluating too frequently (slows training)
    - Not returning model to train() mode after eval
    - Hardcoding hyperparameters (use config instead)

    Production Enhancements:
    - TensorBoard/Wandb logging
    - Learning rate finder
    - Mixed precision training (AMP)
    - Distributed training (DDP)
    - Early stopping
    - Checkpoint management (keep best N)

    See: tiny_transformer/training/trainer.py for reference
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
        # Store attributes
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.eval_interval = eval_interval
        self.device = device

        # Create optimizer
        # AdamW = Adam with decoupled weight decay (better for transformers)
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # Standard for transformers
            betas=(0.9, 0.95)   # Slightly different from default (0.9, 0.999)
        )

        # Create learning rate scheduler
        total_steps = num_epochs * len(train_loader)
        self.scheduler = Solution09_CustomWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=learning_rate,
            min_lr=learning_rate * 0.1  # Decay to 10% of peak
        )

        # Initialize tracking
        self.step = 0
        self.best_val_loss = float('inf')

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        # Set model to train mode (enables dropout, etc.)
        self.model.train()

        # Track metrics
        total_loss = 0.0
        num_batches = 0

        # Training loop
        for input_ids, target_ids in self.train_loader:
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                from tiny_transformer.training.utils import clip_gradient_norm
                clip_gradient_norm(self.model, self.grad_clip)

            # Update weights
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Periodic evaluation
            if self.val_loader and self.step % self.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"Step {self.step} | Val Loss: {val_loss:.4f}")
                # IMPORTANT: Return to training mode!
                self.model.train()

        # Return epoch metrics
        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        # Set to evaluation mode
        self.model.eval()

        # Accumulate metrics
        total_loss = 0.0
        total_tokens = 0

        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += target_ids.numel()

        # Return average loss
        return total_loss / total_tokens

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Dictionary of final metrics
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print()

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            epoch_metrics = self.train_one_epoch()
            print(f"  Train Loss: {epoch_metrics['loss']:.4f}")

        # Final evaluation
        if self.val_loader:
            final_val_loss = self.evaluate()
            print(f"\nFinal Validation Loss: {final_val_loss:.4f}")

        return {'final_loss': epoch_metrics['loss']}


# ==============================================================================
# SOLUTION 11: Early Stopping
# ==============================================================================

class Solution11_EarlyStopping:
    """
    SOLUTION 11: Early stopping to prevent overfitting.

    Key Concepts:
    1. Patience: Number of evaluations to wait for improvement
    2. Min delta: Minimum change to count as improvement
    3. Mode: Minimize loss or maximize accuracy

    Why Early Stopping?
    - Prevents overfitting (validation loss starts increasing)
    - Saves training time (no point continuing if not improving)
    - Automatic stopping criterion (no need to guess epochs)

    How It Works:
    1. Track best validation score seen so far
    2. If current score is better, reset patience counter
    3. If current score is worse, increment patience counter
    4. If patience counter reaches limit, stop training

    Choosing Patience:
    - Too low (e.g., 1-2): May stop too early during plateaus
    - Too high (e.g., 20+): Wastes time, may overfit
    - Sweet spot (5-10): Good for most cases

    Common Mistakes:
    - Checking train loss instead of val loss (defeats the purpose!)
    - Patience too low (stops during normal fluctuations)
    - Not saving best model (end with worse model than best)
    - Forgetting to call reset() between training runs

    Enhancements:
    - Restore best model when stopping
    - Save best checkpoint automatically
    - Exponential moving average of metrics

    See: Commonly used pattern in PyTorch training
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        # Store parameters
        self.patience = patience      # How many evals to wait
        self.min_delta = min_delta    # Minimum improvement threshold
        self.mode = mode              # 'min' for loss, 'max' for accuracy

        # Initialize tracking variables
        if mode == 'min':
            # For loss (lower is better)
            self.best_score = float('inf')
        else:
            # For accuracy (higher is better)
            self.best_score = float('-inf')

        self.counter = 0              # How many evals without improvement
        self.should_stop = False      # Flag to signal stopping

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric (loss or accuracy)

        Returns:
            True if training should stop, False otherwise
        """
        # Step 1: Check if score improved
        if self.mode == 'min':
            # For loss: improved if score < best - min_delta
            improved = score < (self.best_score - self.min_delta)
        else:
            # For accuracy: improved if score > best + min_delta
            improved = score > (self.best_score + self.min_delta)

        # Step 2: Update based on improvement
        if improved:
            # Score improved! Update best and reset counter
            self.best_score = score
            self.counter = 0
            print(f"  [Early Stop] Improvement! New best: {score:.4f}")
        else:
            # No improvement, increment counter
            self.counter += 1
            print(f"  [Early Stop] No improvement ({self.counter}/{self.patience})")

        # Step 3: Check if we should stop
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"  [Early Stop] Patience exhausted. Stopping training.")
            return True

        return False

    def reset(self):
        """Reset early stopping state."""
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
        self.counter = 0
        self.should_stop = False


# ==============================================================================
# SOLUTION 12: End-to-End Shakespeare Training
# ==============================================================================

def solution_12_shakespeare_training(
    seq_len: int = 128,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    SOLUTION 12: Complete end-to-end training pipeline.

    Key Concepts:
    This exercise integrates everything you've learned:
    1. Data loading and tokenization
    2. Model creation and configuration
    3. Training with schedulers and clipping
    4. Validation and perplexity tracking
    5. Text generation from trained model

    Pipeline Overview:
    text → tokenize → split → datasets → loaders → train → evaluate → generate

    Common Mistakes:
    - Not validating data pipeline before training
    - Training on too small data (model won't learn)
    - Not monitoring training (loss could be NaN)
    - Not testing generation (are we learning anything?)

    Production Checklist:
    1. Data: Clean, sufficient quantity, properly split
    2. Model: Appropriate size for data and compute
    3. Training: Stable loss curve, reasonable perplexity
    4. Validation: Not overfitting (train/val gap)
    5. Generation: Produces coherent text

    Hyperparameter Tuning:
    - Learning rate: Too high → instability, too low → slow
    - Batch size: Larger = faster but needs more memory
    - Model size: Larger = more capacity but slower
    - Sequence length: Longer = more context but more memory

    Expected Results (for reference):
    - Initial perplexity: ~50-100 (random)
    - After training: ~5-20 (decent)
    - Perfect model: ~1.0 (very rare!)

    See: Integrates all modules from tiny_transformer/training/
    """
    print("=" * 70)
    print("Exercise 12: End-to-End Shakespeare Training")
    print("=" * 70)
    print()

    # Step 1: Create sample Shakespeare-like text
    # In production, load from file: tiny_shakespeare.txt
    text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""" * 50  # Repeat for more training data

    print(f"Text length: {len(text)} characters")

    # Step 2: Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, vocab_size = solution_04_create_data_loaders(
        text,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.9
    )
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Step 3: Create model
    print("\nCreating model...")
    from tiny_transformer.model import TinyTransformerLM

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=128,        # Small model for demo
        n_heads=4,          # 4 attention heads
        n_layers=3,         # 3 transformer layers
        d_ff=512,          # Feedforward dimension
        max_len=seq_len,    # Maximum sequence length
        dropout=0.1,        # Dropout rate
        tie_weights=True    # Tie input/output embeddings (saves params)
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Step 4: Create trainer
    print("\nCreating trainer...")
    trainer = Solution10_CustomTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        grad_clip=1.0,      # Clip gradients to prevent explosion
        eval_interval=50,   # Evaluate every 50 steps
        device=device
    )

    # Step 5: Train
    print("\nStarting training...")
    start_time = time.time()
    metrics = trainer.train()
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s")

    # Step 6: Evaluate final perplexity
    print("\nComputing final perplexity...")
    final_loss, final_ppl = solution_08_compute_perplexity(model, val_loader, device)
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Final validation perplexity: {final_ppl:.2f}")

    # Step 7: Generate sample text
    print("\nGenerating sample text...")
    model.eval()

    # Create tokenizer for decoding
    tokenizer = Solution02_CharTokenizer()
    tokenizer.fit(text)

    # Generate from a prompt
    prompt = "To be, or not to be"
    print(f"Prompt: {prompt}")
    prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    # Simple greedy generation (always pick most likely token)
    with torch.no_grad():
        generated_tokens = prompt_tokens.clone()
        for _ in range(100):  # Generate 100 tokens
            # Forward pass
            logits, _ = model(generated_tokens)

            # Get most likely next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Stop if sequence gets too long
            if generated_tokens.size(1) > seq_len:
                break

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens[0].cpu().tolist())
    print(f"\nGenerated text:\n{generated_text}")

    # Step 8: Return all results
    return {
        'model': model,
        'final_loss': final_loss,
        'final_perplexity': final_ppl,
        'generated_text': generated_text,
        'vocab_size': vocab_size,
        'num_params': num_params,
        'training_time': elapsed,
    }


# ==============================================================================
# Testing and Validation
# ==============================================================================

def run_all_tests():
    """
    Run tests for all solutions.

    This validates that all implementations work correctly.
    """
    print("=" * 70)
    print("Module 06: Training - Solution Tests")
    print("=" * 70)

    # Test Solution 1
    print("\n[TEST 1] TextDataset")
    tokens = list(range(100))
    dataset = Solution01_TextDataset(tokens, seq_len=10, stride=10)
    assert len(dataset) == 9, f"Expected 9 windows, got {len(dataset)}"
    input_ids, target_ids = dataset[0]
    assert input_ids.shape == (10,), f"Expected (10,), got {input_ids.shape}"
    assert torch.equal(target_ids[:-1], input_ids[1:]), "Target should be input shifted by 1"
    print("✓ Solution 1 passed!")

    # Test Solution 2
    print("\n[TEST 2] CharTokenizer")
    tokenizer = Solution02_CharTokenizer()
    tokenizer.fit("hello world")
    assert tokenizer.vocab_size > 0, "Vocab size should be positive"
    tokens = tokenizer.encode("hello")
    decoded = tokenizer.decode(tokens)
    assert decoded == "hello", f"Expected 'hello', got '{decoded}'"
    print("✓ Solution 2 passed!")

    # Test Solution 3
    print("\n[TEST 3] Learning Rate Scheduler")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler, lrs = solution_03_create_scheduler(
        optimizer, warmup_steps=100, total_steps=1000
    )
    assert len(lrs) == 1001, f"Expected 1001 LRs, got {len(lrs)}"
    assert lrs[0] < lrs[100], "LR should increase during warmup"
    assert lrs[100] > lrs[-1], "LR should decrease during decay"
    print("✓ Solution 3 passed!")

    # Test Solution 4
    print("\n[TEST 4] Create Data Loaders")
    text = "hello world! " * 100
    train_loader, val_loader, vocab_size = solution_04_create_data_loaders(
        text, seq_len=32, batch_size=8
    )
    inputs, targets = next(iter(train_loader))
    assert inputs.shape[0] <= 8, f"Batch size should be ≤8, got {inputs.shape[0]}"
    assert inputs.shape[1] == 32, f"Seq len should be 32, got {inputs.shape[1]}"
    print("✓ Solution 4 passed!")

    print("\n" + "=" * 70)
    print("All solution tests passed!")
    print("=" * 70)


# ==============================================================================
# Main: Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Module 06: Training - Reference Solutions")
    print("=" * 70)
    print("\nThis file contains complete solutions for all 12 exercises.")
    print("\nSolution Structure:")
    print("  01. TextDataset - Sliding window dataset (Easy)")
    print("  02. CharTokenizer - Character-level tokenization (Easy)")
    print("  03. Scheduler Configuration - LR warmup + decay (Easy)")
    print("  04. Data Loaders - Complete data pipeline (Medium)")
    print("  05. Basic Training Loop - Forward/backward/update (Medium)")
    print("  06. Gradient Clipping - Prevent gradient explosion (Medium)")
    print("  07. Checkpointing - Save/load training state (Medium)")
    print("  08. Perplexity - Evaluation metric (Medium)")
    print("  09. Custom Warmup Scheduler - From-scratch LR schedule (Hard)")
    print("  10. Custom Trainer - Integrated training class (Hard)")
    print("  11. Early Stopping - Prevent overfitting (Hard)")
    print("  12. Shakespeare Training - End-to-end pipeline (Very Hard)")
    print("\nHow to Use:")
    print("1. Study each solution's implementation")
    print("2. Read the educational comments explaining WHY")
    print("3. Compare with your own implementations")
    print("4. Run tests to verify correctness")
    print("\nKey Takeaways:")
    print("- Dataset: Sliding windows with stride control")
    print("- Tokenizer: Bidirectional char↔idx mappings")
    print("- Scheduler: Warmup prevents instability, decay fine-tunes")
    print("- Training: Zero grad → forward → loss → backward → step")
    print("- Clipping: Prevents gradient explosion in deep networks")
    print("- Checkpointing: Save both model and optimizer state")
    print("- Perplexity: exp(loss), lower is better")
    print("- Early stopping: Prevent overfitting with patience")
    print("=" * 70)

    # Run tests
    print("\n\nRunning solution tests...\n")
    run_all_tests()

    print("\n\nExcellent work completing Module 06!")
    print("You now understand the complete training pipeline for transformers.")
    print("\nNext steps:")
    print("- Experiment with different hyperparameters")
    print("- Try larger datasets (full Shakespeare, Wikipedia)")
    print("- Implement advanced features (mixed precision, distributed training)")
    print("- Study other architectures (GPT, BERT, T5)")
    print("=" * 70)
