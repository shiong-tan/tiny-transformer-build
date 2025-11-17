"""
Comprehensive Tests for Training Package (Module 06).

Tests cover:
- TextDataset: sliding windows, stride, edge cases
- CharTokenizer: fit, encode, decode
- Learning rate schedulers: warmup and decay phases
- Trainer: training loop, checkpointing, resume
- Training utilities: perplexity, gradient clipping, etc.
- Integration tests: end-to-end training

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import torch.nn as nn
import math
import tempfile
import os
from pathlib import Path

from tiny_transformer.training import (
    TextDataset,
    CharTokenizer,
    create_data_loaders,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    get_scheduler,
    Trainer,
    TrainerConfig,
    compute_perplexity,
    get_gradient_norm,
    clip_gradient_norm,
    set_seed,
    get_device,
    AverageMeter,
)
from tiny_transformer.model import TinyTransformerLM


# ============================================================================
# Test TextDataset
# ============================================================================

class TestTextDataset:
    """Test TextDataset for language modeling."""

    def test_basic_functionality(self):
        """Test basic dataset creation and access."""
        tokens = list(range(100))  # 0, 1, 2, ..., 99
        seq_len = 10
        dataset = TextDataset(tokens, seq_len=seq_len)

        assert len(dataset) > 0
        input_ids, target_ids = dataset[0]

        # Check shapes
        assert input_ids.shape == (seq_len,)
        assert target_ids.shape == (seq_len,)

        # Check that targets are shifted by 1
        assert torch.allclose(target_ids, input_ids + 1)

    def test_sliding_window_default_stride(self):
        """Test sliding window with default stride (no overlap)."""
        tokens = list(range(50))
        seq_len = 10
        dataset = TextDataset(tokens, seq_len=seq_len)

        # With stride=seq_len (default), windows don't overlap
        # Each window needs seq_len+1 tokens
        # Expected windows: (50 - 11) // 10 + 1 = 4
        expected_len = (50 - (seq_len + 1)) // seq_len + 1
        assert len(dataset) == expected_len

        # Check no overlap
        input_0, _ = dataset[0]
        input_1, _ = dataset[1]

        # First window: [0:10], second window: [10:20]
        assert not torch.equal(input_0, input_1)

    def test_sliding_window_custom_stride(self):
        """Test sliding window with custom stride (overlap)."""
        tokens = list(range(100))
        seq_len = 10
        stride = 5  # 50% overlap

        dataset = TextDataset(tokens, seq_len=seq_len, stride=stride)

        # With stride=5, windows overlap by 5 tokens
        # Expected windows: (100 - 11) // 5 + 1 = 18
        expected_len = (100 - (seq_len + 1)) // stride + 1
        assert len(dataset) == expected_len

        # Check overlap
        input_0, _ = dataset[0]
        input_1, _ = dataset[1]

        # First window: [0:10], second window: [5:15]
        # Last 5 of input_0 should equal first 5 of input_1
        assert torch.equal(input_0[-5:], input_1[:5])

    def test_target_shifting(self):
        """Test that targets are correctly shifted versions of inputs."""
        tokens = list(range(20))
        seq_len = 5
        dataset = TextDataset(tokens, seq_len=seq_len)

        input_ids, target_ids = dataset[0]

        # input: [0, 1, 2, 3, 4]
        # target: [1, 2, 3, 4, 5]
        expected_input = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        expected_target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

        assert torch.equal(input_ids, expected_input)
        assert torch.equal(target_ids, expected_target)

    def test_edge_case_minimum_data(self):
        """Test with minimum possible data (seq_len + 1 tokens)."""
        seq_len = 5
        tokens = list(range(seq_len + 1))  # Exactly 6 tokens

        dataset = TextDataset(tokens, seq_len=seq_len)

        # Should have exactly 1 sample
        assert len(dataset) == 1

        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (seq_len,)
        assert target_ids.shape == (seq_len,)

    def test_edge_case_insufficient_data(self):
        """Test with insufficient data (< seq_len + 1 tokens)."""
        tokens = list(range(5))  # Only 5 tokens
        seq_len = 10  # Need 11 tokens minimum

        # Should raise ValueError
        with pytest.raises(ValueError, match="Not enough tokens"):
            TextDataset(tokens, seq_len=seq_len)

    def test_index_out_of_bounds(self):
        """Test that accessing invalid index raises IndexError."""
        tokens = list(range(50))
        seq_len = 10
        dataset = TextDataset(tokens, seq_len=seq_len)

        with pytest.raises(IndexError):
            dataset[len(dataset)]  # Access beyond bounds

        with pytest.raises(IndexError):
            dataset[999]  # Far beyond bounds

    def test_tensor_conversion(self):
        """Test that input tokens are converted to tensors."""
        tokens = [10, 20, 30, 40, 50, 60]
        seq_len = 2

        dataset = TextDataset(tokens, seq_len=seq_len)

        input_ids, target_ids = dataset[0]

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long


class TestCharTokenizer:
    """Test CharTokenizer for character-level tokenization."""

    def test_fit_creates_vocabulary(self):
        """Test that fit creates proper vocabulary."""
        text = "hello world"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        # Check vocab size (unique chars in "hello world")
        unique_chars = set(text)
        assert tokenizer.vocab_size == len(unique_chars)

        # Check that all characters are in vocab
        for char in unique_chars:
            assert char in tokenizer.vocab

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode returns original text."""
        text = "the quick brown fox"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        # Encode
        tokens = tokenizer.encode(text)

        # Decode
        decoded_text = tokenizer.decode(tokens)

        assert decoded_text == text

    def test_encode_returns_list_of_ints(self):
        """Test that encode returns list of integers."""
        text = "abc"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        tokens = tokenizer.encode(text)

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == len(text)

    def test_vocabulary_sorted_order(self):
        """Test that vocabulary is created in sorted order."""
        text = "dcba"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        # Vocab should be sorted: a, b, c, d
        assert tokenizer.vocab['a'] < tokenizer.vocab['b']
        assert tokenizer.vocab['b'] < tokenizer.vocab['c']
        assert tokenizer.vocab['c'] < tokenizer.vocab['d']

    def test_encode_unknown_character_fails(self):
        """Test that encoding unknown character raises error."""
        tokenizer = CharTokenizer()
        tokenizer.fit("abc")

        with pytest.raises(ValueError, match="Unknown character"):
            tokenizer.encode("abcX")  # 'X' not in vocab

    def test_decode_invalid_token_fails(self):
        """Test that decoding invalid token raises error."""
        tokenizer = CharTokenizer()
        tokenizer.fit("abc")

        with pytest.raises(ValueError, match="Unknown token ID"):
            tokenizer.decode([0, 1, 999])  # 999 not in inverse_vocab

    def test_special_characters(self):
        """Test tokenization with special characters."""
        text = "hello\nworld\t!"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text


class TestCreateDataLoaders:
    """Test create_data_loaders utility function."""

    def test_basic_functionality(self):
        """Test basic dataloader creation."""
        train_tokens = list(range(100))
        val_tokens = list(range(100, 150))

        train_loader, val_loader = create_data_loaders(
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            seq_len=10,
            batch_size=4
        )

        assert train_loader is not None
        assert val_loader is not None

        # Check batch shape
        for input_ids, target_ids in train_loader:
            assert input_ids.shape[0] <= 4  # batch_size
            assert input_ids.shape[1] == 10  # seq_len
            assert target_ids.shape == input_ids.shape
            break

    def test_no_validation_data(self):
        """Test creation without validation data."""
        train_tokens = list(range(100))

        train_loader, val_loader = create_data_loaders(
            train_tokens=train_tokens,
            val_tokens=None,
            seq_len=10,
            batch_size=4
        )

        assert train_loader is not None
        assert val_loader is None


# ============================================================================
# Test Learning Rate Schedulers
# ============================================================================

class TestWarmupCosineScheduler:
    """Test WarmupCosineScheduler."""

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1.0,
            min_lr=0.0
        )

        # At step 0, lr should be 0 (or very small)
        lr_0 = scheduler.get_last_lr()[0]
        assert lr_0 == 0.0

        # At step 50 (halfway through warmup), lr should be ~0.5
        for _ in range(50):
            scheduler.step()

        lr_50 = scheduler.get_last_lr()[0]
        assert 0.4 < lr_50 < 0.6

        # At step 100 (end of warmup), lr should be peak_lr
        for _ in range(50):
            scheduler.step()

        lr_100 = scheduler.get_last_lr()[0]
        assert abs(lr_100 - 1.0) < 0.01

    def test_cosine_decay_phase(self):
        """Test cosine decay after warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1.0,
            min_lr=0.1
        )

        # Skip warmup
        for _ in range(101):
            scheduler.step()

        # After warmup, lr should start decaying
        lr_after_warmup = scheduler.get_last_lr()[0]

        # Step to middle of decay
        for _ in range(400):
            scheduler.step()

        lr_mid_decay = scheduler.get_last_lr()[0]

        # LR should have decreased
        assert lr_mid_decay < lr_after_warmup

        # Step to end
        for _ in range(499):
            scheduler.step()

        lr_end = scheduler.get_last_lr()[0]

        # LR should approach min_lr
        assert abs(lr_end - 0.1) < 0.05

    def test_zero_warmup_steps(self):
        """Test with warmup_steps=0 (should start at peak_lr)."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,
            total_steps=1000,
            peak_lr=1.0,
            min_lr=0.0
        )

        # At step 0, lr should be peak_lr
        lr_0 = scheduler.get_last_lr()[0]
        assert abs(lr_0 - 1.0) < 0.01

    def test_invalid_warmup_steps(self):
        """Test that invalid warmup_steps raises error."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        with pytest.raises(ValueError):
            WarmupCosineScheduler(
                optimizer,
                warmup_steps=-10,  # Invalid
                total_steps=1000,
                peak_lr=1.0
            )

    def test_invalid_total_steps(self):
        """Test that total_steps <= warmup_steps raises error."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        with pytest.raises(ValueError):
            WarmupCosineScheduler(
                optimizer,
                warmup_steps=100,
                total_steps=50,  # Invalid: <= warmup_steps
                peak_lr=1.0
            )

    def test_invalid_peak_lr(self):
        """Test that peak_lr <= min_lr raises error."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        with pytest.raises(ValueError, match="peak_lr.*must be greater than min_lr"):
            WarmupCosineScheduler(
                optimizer,
                warmup_steps=10,
                total_steps=100,
                peak_lr=0.1,
                min_lr=1.0  # Invalid: min_lr > peak_lr
            )


class TestWarmupLinearScheduler:
    """Test WarmupLinearScheduler."""

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1.0,
            min_lr=0.0
        )

        # At step 0, lr should be 0
        lr_0 = scheduler.get_last_lr()[0]
        assert lr_0 == 0.0

        # At step 50, lr should be ~0.5
        for _ in range(50):
            scheduler.step()

        lr_50 = scheduler.get_last_lr()[0]
        assert 0.4 < lr_50 < 0.6

    def test_linear_decay_phase(self):
        """Test linear decay after warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1.0,
            min_lr=0.0
        )

        # Skip warmup
        for _ in range(101):
            scheduler.step()

        lr_after_warmup = scheduler.get_last_lr()[0]

        # Step to middle of decay
        for _ in range(400):
            scheduler.step()

        lr_mid_decay = scheduler.get_last_lr()[0]

        # LR should have decreased linearly
        assert lr_mid_decay < lr_after_warmup

        # Step to end
        for _ in range(499):
            scheduler.step()

        lr_end = scheduler.get_last_lr()[0]

        # LR should approach min_lr
        assert abs(lr_end - 0.0) < 0.05


class TestGetScheduler:
    """Test get_scheduler factory function."""

    def test_get_warmup_cosine(self):
        """Test getting warmup cosine scheduler."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = get_scheduler(
            "warmup_cosine",
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1e-3
        )

        assert isinstance(scheduler, WarmupCosineScheduler)

    def test_get_warmup_linear(self):
        """Test getting warmup linear scheduler."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = get_scheduler(
            "warmup_linear",
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            peak_lr=1e-3
        )

        assert isinstance(scheduler, WarmupLinearScheduler)

    def test_invalid_scheduler_name(self):
        """Test that invalid scheduler name raises error."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError):
            get_scheduler(
                "invalid_name",
                optimizer,
                warmup_steps=100,
                total_steps=1000
            )


# ============================================================================
# Test Training Utilities
# ============================================================================

class TestTrainingUtilities:
    """Test training utility functions."""

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        # Perplexity = exp(loss)
        loss = 2.0
        perplexity = compute_perplexity(loss)
        expected = math.exp(2.0)

        assert abs(perplexity - expected) < 1e-5

    def test_get_gradient_norm(self):
        """Test gradient norm computation."""
        model = nn.Linear(10, 10)
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        # Forward and backward
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        # Get gradient norm
        grad_norm = get_gradient_norm(model)

        assert grad_norm > 0.0
        assert isinstance(grad_norm, float)

    def test_clip_gradient_norm(self):
        """Test gradient clipping."""
        model = nn.Linear(10, 10)
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        # Forward and backward
        output = model(x)
        loss = ((output - y) ** 2).mean() * 1000  # Large loss
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        clipped_norm = clip_gradient_norm(model, max_norm)

        # After clipping, norm should be <= max_norm
        current_norm = get_gradient_norm(model)
        assert current_norm <= max_norm + 1e-5

    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        assert torch.allclose(tensor1, tensor2)

    def test_get_device(self):
        """Test device detection."""
        device = get_device()

        assert isinstance(device, torch.device)
        # Should return cuda, mps, or cpu
        assert device.type in ['cuda', 'mps', 'cpu']

    def test_average_meter(self):
        """Test AverageMeter utility."""
        meter = AverageMeter()

        meter.update(10.0)
        assert meter.avg == 10.0
        assert meter.sum == 10.0
        assert meter.count == 1

        meter.update(20.0)
        assert meter.avg == 15.0
        assert meter.sum == 30.0
        assert meter.count == 2

        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0


# ============================================================================
# Test Trainer
# ============================================================================

class TestTrainer:
    """Test Trainer class."""

    def test_trainer_initialization(self):
        """Test basic trainer initialization."""
        # Create tiny model and dataset
        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        tokens = list(range(100))
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=4
        )

        config = TrainerConfig(learning_rate=1e-3, log_interval=10)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device is not None

    def test_train_one_epoch(self):
        """Test training for one epoch."""
        set_seed(42)

        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        tokens = list(range(200))
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=4
        )

        config = TrainerConfig(
            learning_rate=1e-3,
            log_interval=100,
            grad_clip=1.0
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # Train one epoch
        metrics = trainer.train_one_epoch()

        # Should return metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'lr' in metrics

        # Loss should be positive
        assert metrics['loss'] > 0

    def test_validation(self):
        """Test validation loop."""
        set_seed(42)

        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        tokens = list(range(200))
        train_loader, val_loader = create_data_loaders(
            train_tokens=tokens,
            val_tokens=tokens,
            seq_len=8,
            batch_size=4
        )

        config = TrainerConfig(learning_rate=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )

        # Evaluate
        val_metrics = trainer.evaluate()

        # Should return metrics
        assert 'loss' in val_metrics
        assert 'perplexity' in val_metrics

        # Loss should be positive
        assert val_metrics['loss'] > 0

    def test_save_and_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        set_seed(42)

        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        tokens = list(range(200))
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=4
        )

        config = TrainerConfig(learning_rate=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # Train a bit
        trainer.train_one_epoch()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path)

            assert os.path.exists(checkpoint_path)

            # Create new trainer and load checkpoint
            new_model = TinyTransformerLM(
                vocab_size=50,
                d_model=32,
                n_heads=2,
                n_layers=2,
                d_ff=64,
                max_len=16,
                dropout=0.0
            )

            new_trainer = Trainer(
                model=new_model,
                train_loader=train_loader,
                config=config
            )

            new_trainer.load_checkpoint(checkpoint_path)

            # Check that states match
            assert new_trainer.step == trainer.step
            assert new_trainer.epoch == trainer.epoch

            # Check model weights match
            for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_empty_dataloader_raises_error(self):
        """Test that empty dataloader raises error."""
        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16
        )

        # Create dataset with insufficient data
        tokens = list(range(5))  # Too few tokens
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=10,
            batch_size=4
        )

        config = TrainerConfig(learning_rate=1e-3, max_steps=100)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # Should raise RuntimeError for empty dataloader
        with pytest.raises(RuntimeError, match="train_loader is empty"):
            trainer.train(max_steps=100)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    def test_end_to_end_training(self):
        """Test complete training pipeline from data to trained model."""
        set_seed(42)

        # 1. Create tokenizer and data
        text = "hello world this is a test of the training pipeline"
        tokenizer = CharTokenizer()
        tokenizer.fit(text)
        tokens = tokenizer.encode(text)

        # 2. Create dataloaders
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=2
        )

        # 3. Create model
        model = TinyTransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.1
        )

        # 4. Create trainer
        config = TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            grad_clip=1.0,
            log_interval=10
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # 5. Train for a few steps
        initial_loss = None
        final_loss = None

        for step in range(20):
            metrics = trainer.train_one_epoch()
            if step == 0:
                initial_loss = metrics['loss']
            final_loss = metrics['loss']

        # Loss should decrease with training
        assert final_loss < initial_loss

    def test_training_with_scheduler(self):
        """Test training with learning rate scheduler."""
        set_seed(42)

        tokens = list(range(200))
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=4
        )

        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        config = TrainerConfig(learning_rate=1e-3, log_interval=50)

        # Trainer creates its own optimizer, then we create scheduler with it
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # Create scheduler with trainer's optimizer
        scheduler = WarmupCosineScheduler(
            trainer.optimizer,
            warmup_steps=10,
            total_steps=100,
            peak_lr=1e-3,
            min_lr=1e-5
        )

        # Attach scheduler to trainer
        trainer.scheduler = scheduler

        # Train with scheduler
        initial_lr = scheduler.get_last_lr()[0]
        trainer.train_one_epoch()
        final_lr = scheduler.get_last_lr()[0]

        # LR should have changed
        assert final_lr != initial_lr

    def test_checkpointing_and_resume(self):
        """Test saving checkpoint and resuming training."""
        set_seed(42)

        tokens = list(range(200))
        train_loader, _ = create_data_loaders(
            train_tokens=tokens,
            val_tokens=None,
            seq_len=8,
            batch_size=4
        )

        model = TinyTransformerLM(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_len=16,
            dropout=0.0
        )

        config = TrainerConfig(learning_rate=1e-3)

        trainer1 = Trainer(
            model=model,
            train_loader=train_loader,
            config=config
        )

        # Train for 10 steps
        for _ in range(10):
            trainer1.train_one_epoch()

        step_before_save = trainer1.step

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            trainer1.save_checkpoint(checkpoint_path)

            # Create new trainer and resume
            new_model = TinyTransformerLM(
                vocab_size=50,
                d_model=32,
                n_heads=2,
                n_layers=2,
                d_ff=64,
                max_len=16,
                dropout=0.0
            )

            trainer2 = Trainer(
                model=new_model,
                train_loader=train_loader,
                config=config
            )

            trainer2.load_checkpoint(checkpoint_path)

            # Step should match
            assert trainer2.step == step_before_save

            # Continue training
            trainer2.train_one_epoch()

            # Step should have advanced
            assert trainer2.step > step_before_save


if __name__ == "__main__":
    print("=" * 70)
    print("Running Training Package Tests")
    print("=" * 70)
    print()
    print("Run with: pytest tests/test_training.py -v")
    print()
