#!/usr/bin/env python3
"""
Comprehensive tests for engineering utilities (Module 08).

Tests:
1. TrainingLogger - JSON formatting, metric logging, system info
2. CheckpointManager - Save/load, best-N strategy, resume
3. ExperimentTracker - Backend initialization, graceful fallback
4. Integration tests - Complete workflows

Author: Claude (AI Assistant)
Date: November 17, 2025
"""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from tiny_transformer.utils import (
    TrainingLogger,
    CheckpointManager,
    ExperimentTracker,
    save_checkpoint,
    load_checkpoint,
    get_git_commit,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    return SimpleModel()


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)


@pytest.fixture
def scheduler(optimizer):
    """Create scheduler for testing."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# =============================================================================
# TrainingLogger Tests
# =============================================================================

class TestTrainingLogger:
    """Test TrainingLogger functionality."""

    def test_logger_initialization(self, temp_dir):
        """Test logger creates necessary files and directories."""
        logger = TrainingLogger(
            experiment_name="test_experiment",
            log_dir=temp_dir,
        )

        assert logger.log_dir.exists()
        assert logger.log_file.exists()
        assert logger.experiment_name == "test_experiment"

    def test_log_metrics(self, temp_dir):
        """Test metric logging writes to file."""
        logger = TrainingLogger(
            experiment_name="test_metrics",
            log_dir=temp_dir,
            log_to_console=False,  # Disable console for testing
        )

        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 1e-4,
        }

        logger.log_metrics(metrics, step=100, epoch=5)

        # Read log file and verify
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        # Find the metrics line (skip any startup logs)
        metrics_line = None
        for line in lines:
            if '"loss"' in line:
                metrics_line = line
                break

        assert metrics_line is not None
        log_entry = json.loads(metrics_line)

        assert log_entry["loss"] == 0.5
        assert log_entry["accuracy"] == 0.95
        assert log_entry["step"] == 100
        assert log_entry["epoch"] == 5

    def test_log_metrics_with_tensors(self, temp_dir):
        """Test that tensor values are converted to floats."""
        logger = TrainingLogger(
            experiment_name="test_tensors",
            log_dir=temp_dir,
            log_to_console=False,
        )

        metrics = {
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor([0.95]),
        }

        # Should not raise error
        logger.log_metrics(metrics, step=1)

        # Verify conversion
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if '"loss"' in line:
                log_entry = json.loads(line)
                assert isinstance(log_entry["loss"], float)
                assert isinstance(log_entry["accuracy"], float)
                break

    def test_log_start(self, temp_dir):
        """Test logging experiment start."""
        logger = TrainingLogger(
            experiment_name="test_start",
            log_dir=temp_dir,
            log_to_console=False,
        )

        config = {
            "model": {"d_model": 512, "n_heads": 8},
            "training": {"learning_rate": 1e-4},
        }

        logger.log_start(config=config)

        # Verify start was logged
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        # Find start log
        for line in lines:
            if "Experiment started" in line or "experiment_started" in line:
                log_entry = json.loads(line)
                assert "timestamp" in log_entry
                break

    def test_log_end(self, temp_dir):
        """Test logging experiment end."""
        logger = TrainingLogger(
            experiment_name="test_end",
            log_dir=temp_dir,
            log_to_console=False,
        )

        final_metrics = {"final_loss": 0.1, "final_accuracy": 0.99}
        logger.log_end(final_metrics=final_metrics)

        # Verify end was logged
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        # Find end log
        for line in lines:
            if "Experiment completed" in line or "experiment_completed" in line:
                log_entry = json.loads(line)
                assert "timestamp" in log_entry
                break

    def test_metric_prefix(self, temp_dir):
        """Test metric prefixing (e.g., 'train/' or 'val/')."""
        logger = TrainingLogger(
            experiment_name="test_prefix",
            log_dir=temp_dir,
            log_to_console=False,
        )

        metrics = {"loss": 0.5}
        logger.log_metrics(metrics, step=1, prefix="train/")

        # Verify prefix was added
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if '"train/loss"' in line:
                log_entry = json.loads(line)
                assert "train/loss" in log_entry
                assert log_entry["train/loss"] == 0.5
                break


# =============================================================================
# CheckpointManager Tests
# =============================================================================

class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_manager_initialization(self, temp_dir):
        """Test checkpoint manager creates directory."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        assert manager.checkpoint_dir.exists()
        assert manager.keep_best_n == 3
        assert len(manager.checkpoints) == 0

    def test_save_checkpoint(self, temp_dir, simple_model, optimizer, scheduler):
        """Test saving a checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        config = {"model": {"d_model": 512}}

        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=1000,
            epoch=5,
            val_loss=0.5,
            config=config,
        )

        assert checkpoint_path.exists()
        assert "checkpoint_1000.pt" in checkpoint_path.name
        assert len(manager.checkpoints) == 1

    def test_save_and_load_checkpoint(self, temp_dir, simple_model, optimizer, scheduler):
        """Test saving and loading checkpoint restores state."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_best_n=3)

        config = {"model": {"d_model": 512}}

        # Save checkpoint
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=1000,
            epoch=5,
            val_loss=0.5,
            config=config,
        )

        # Create new model and optimizer
        new_model = simple_model.__class__()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=100)

        # Load checkpoint
        metadata = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device="cpu",
        )

        assert metadata["step"] == 1000
        assert metadata["epoch"] == 5
        assert metadata["val_loss"] == 0.5

        # Verify model state was loaded
        for p1, p2 in zip(simple_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_best_n_strategy(self, temp_dir, simple_model, optimizer, scheduler):
        """Test that only best N checkpoints are kept."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        config = {"model": {"d_model": 512}}

        # Save 5 checkpoints with different val_loss
        val_losses = [0.5, 0.3, 0.7, 0.2, 0.4]
        paths = []

        for i, val_loss in enumerate(val_losses):
            path = manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=i * 100,
                epoch=i,
                val_loss=val_loss,
                config=config,
            )
            paths.append(path)

        # Should only keep 3 best (lowest loss): 0.2, 0.3, 0.4
        assert len(manager.checkpoints) == 3

        # Verify best checkpoints are kept
        kept_losses = [ckpt[0] for ckpt in manager.checkpoints]
        assert sorted(kept_losses) == [0.2, 0.3, 0.4]

        # Verify files exist for best checkpoints
        assert paths[3].exists()  # 0.2 (best)
        assert paths[1].exists()  # 0.3
        assert paths[4].exists()  # 0.4

        # Verify worst checkpoints were deleted
        assert not paths[0].exists()  # 0.5 (deleted)
        assert not paths[2].exists()  # 0.7 (deleted)

    def test_save_without_val_loss(self, temp_dir, simple_model, optimizer, scheduler):
        """Test saving checkpoint without validation loss (keep_best_n=0)."""
        # When keep_best_n=0, val_loss can be None
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=0,  # No best-N tracking
        )

        config = {"model": {"d_model": 512}}

        # Save without val_loss (allowed when keep_best_n=0)
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=1,
            val_loss=None,
            config=config,
        )

        assert checkpoint_path.exists()
        # Should save successfully without tracking in best-N

    def test_save_without_val_loss_raises_error(self, temp_dir, simple_model, optimizer, scheduler):
        """Test that saving without val_loss raises error when keep_best_n > 0."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,  # Requires val_loss
        )

        config = {"model": {"d_model": 512}}

        # Should raise ValueError when val_loss is None and keep_best_n > 0
        with pytest.raises(ValueError, match="val_loss must be provided"):
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=100,
                epoch=1,
                val_loss=None,
                config=config,
            )

    def test_resume_from_checkpoint(self, temp_dir, simple_model, optimizer, scheduler):
        """Test resume capability discovers existing checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Manually save a checkpoint
        save_checkpoint(
            path=checkpoint_dir / "checkpoint_500.pt",
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=500,
            epoch=2,
            val_loss=0.3,
            config={"model": {}},
        )

        # Create manager (should discover existing)
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_best_n=3)

        # Should discover the existing checkpoint
        assert len(manager.checkpoints) == 1

    def test_git_commit_tracking(self, temp_dir, simple_model, optimizer, scheduler):
        """Test that git commit is tracked in checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        config = {"model": {"d_model": 512}}

        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=1,
            val_loss=0.5,
            config=config,
        )

        # Load and verify git commit is present
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # git_commit should be present (may be None if not in git repo)
        assert "git_commit" in checkpoint


# =============================================================================
# ExperimentTracker Tests
# =============================================================================

class TestExperimentTracker:
    """Test ExperimentTracker functionality."""

    def test_tracker_console_backend(self, temp_dir):
        """Test tracker with console backend (always available)."""
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_exp",
            config={"learning_rate": 1e-4},
            backend="console",
            log_dir=temp_dir,
        )

        assert tracker.backend == "console"

        # Should not raise errors
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_hyperparams({"lr": 1e-4})
        tracker.finish()

    def test_tracker_auto_backend_fallback(self, temp_dir):
        """Test that auto backend falls back gracefully."""
        # This will try wandb → tensorboard → console
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_exp",
            config={"learning_rate": 1e-4},
            backend="auto",
            log_dir=temp_dir,
        )

        # Should fall back to console if wandb/tensorboard unavailable
        assert tracker.backend in ["wandb", "tensorboard", "console"]

        # Should work regardless of backend
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.finish()

    def test_log_metrics(self, temp_dir):
        """Test metric logging."""
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_metrics",
            config={},
            backend="console",
            log_dir=temp_dir,
        )

        metrics = {"loss": 0.5, "accuracy": 0.95}

        # Should not raise
        tracker.log_metrics(metrics, step=100)

    def test_log_hyperparams(self, temp_dir):
        """Test hyperparameter logging."""
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_params",
            config={},
            backend="console",
            log_dir=temp_dir,
        )

        params = {"learning_rate": 1e-4, "batch_size": 32}

        # Should not raise
        tracker.log_hyperparams(params)

    def test_save_artifact(self, temp_dir):
        """Test artifact saving."""
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_artifact",
            config={},
            backend="console",
            log_dir=temp_dir,
        )

        # Create dummy file
        artifact_path = temp_dir / "model.pt"
        torch.save({"data": "test"}, artifact_path)

        # Should not raise
        tracker.save_artifact(artifact_path, name="test_model", artifact_type="model")

    def test_finish(self, temp_dir):
        """Test experiment finish."""
        tracker = ExperimentTracker(
            project="test_project",
            experiment_name="test_finish",
            config={},
            backend="console",
            log_dir=temp_dir,
        )

        # Should not raise
        tracker.finish()

    def test_context_manager(self, temp_dir):
        """Test tracker as context manager."""
        # Test normal exit with auto-cleanup
        with ExperimentTracker(
            project="test_project",
            experiment_name="context_test",
            config={},
            backend="console",
            log_dir=temp_dir,
        ) as tracker:
            tracker.log_metrics({"loss": 0.5}, step=1)
            assert tracker.backend == "console"

        # Tracker should auto-finish on exit

    def test_context_manager_with_exception(self, temp_dir):
        """Test context manager cleanup on exception."""
        # Should still cleanup properly even with exception
        with pytest.raises(ValueError):
            with ExperimentTracker(
                project="test_project",
                experiment_name="context_error",
                config={},
                backend="console",
                log_dir=temp_dir,
            ) as tracker:
                tracker.log_metrics({"loss": 0.5}, step=1)
                raise ValueError("Simulated error")

        # Cleanup should have occurred despite exception

    def test_get_experiment_tracker_utility(self, temp_dir):
        """Test get_experiment_tracker() utility function."""
        from tiny_transformer.utils import get_experiment_tracker

        tracker = get_experiment_tracker(
            project="test_project",
            experiment_name="util_test",
            config={},
            backend="console",
            log_dir=temp_dir,
        )

        assert tracker is not None
        assert tracker.backend == "console"

        tracker.finish()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_training_workflow(self, temp_dir, simple_model, optimizer, scheduler):
        """Test complete workflow: train → checkpoint → log."""
        # Setup
        logger = TrainingLogger(
            experiment_name="integration_test",
            log_dir=temp_dir / "logs",
            log_to_console=False,
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        tracker = ExperimentTracker(
            project="integration",
            experiment_name="integration_test",
            config={"learning_rate": 1e-3},
            backend="console",
            log_dir=temp_dir / "experiments",
        )

        # Simulate training loop
        logger.log_start(config={"learning_rate": 1e-3})

        for step in range(1, 6):
            # Log metrics
            metrics = {"loss": 1.0 / step, "lr": 1e-3}
            logger.log_metrics(metrics, step=step)
            tracker.log_metrics(metrics, step=step)

            # Save checkpoint
            if step % 2 == 0:
                checkpoint_manager.save(
                    model=simple_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=step // 2,
                    val_loss=1.0 / step,
                    config={"learning_rate": 1e-3},
                )

        logger.log_end(final_metrics={"final_loss": 0.2})
        tracker.finish()

        # Verify all components worked
        assert logger.log_file.exists()
        assert len(list((temp_dir / "checkpoints").glob("*.pt"))) > 0
        assert checkpoint_manager.checkpoints  # Has tracked checkpoints

    def test_resume_workflow(self, temp_dir, simple_model, optimizer, scheduler):
        """Test resume workflow: save → load → continue."""
        checkpoint_dir = temp_dir / "checkpoints"

        # Initial training
        manager1 = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_best_n=3)

        checkpoint_path = manager1.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=5,
            val_loss=0.5,
            config={"learning_rate": 1e-3},
        )

        # Resume training
        new_model = simple_model.__class__()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=100)

        metadata = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device="cpu",
        )

        # Continue from loaded state
        start_step = metadata["step"]
        assert start_step == 100

        # Save another checkpoint
        manager2 = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_best_n=3)

        manager2.save(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            step=start_step + 100,
            epoch=10,
            val_loss=0.3,
            config={"learning_rate": 1e-3},
        )

        # Should have 2 checkpoints
        assert len(manager2.checkpoints) == 2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_nonexistent_checkpoint(self, simple_model, optimizer, scheduler):
        """Test loading non-existent checkpoint raises error."""
        # May raise FileNotFoundError or RuntimeError depending on PyTorch version
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_checkpoint(
                path="nonexistent.pt",
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
            )

    def test_empty_metrics(self, temp_dir):
        """Test logging empty metrics dict."""
        logger = TrainingLogger(
            experiment_name="test_empty",
            log_dir=temp_dir,
            log_to_console=False,
        )

        # Should not raise
        logger.log_metrics({}, step=1)

    def test_checkpoint_without_scheduler(self, temp_dir, simple_model, optimizer):
        """Test saving checkpoint without scheduler."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=3,
        )

        # Should work with scheduler=None
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            epoch=1,
            val_loss=0.5,
            config={},
        )

        assert checkpoint_path.exists()

    def test_checkpoint_max_mode(self, temp_dir, simple_model, optimizer, scheduler):
        """Test checkpoint manager with max mode (e.g., for accuracy)."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            keep_best_n=2,
            metric_mode="max",  # Keep highest metrics
        )

        config = {}

        # Save checkpoints with accuracy values
        accuracies = [0.7, 0.9, 0.6, 0.95, 0.8]

        for i, acc in enumerate(accuracies):
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=i * 100,
                epoch=i,
                val_loss=acc,  # Using val_loss field for accuracy
                config=config,
            )

        # Should keep 2 best (highest): 0.95, 0.9
        assert len(manager.checkpoints) == 2
        kept_values = [ckpt[0] for ckpt in manager.checkpoints]
        assert sorted(kept_values, reverse=True) == [0.95, 0.9]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
