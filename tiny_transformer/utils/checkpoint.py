"""
Checkpoint management utilities for saving and resuming training.

Provides CheckpointManager for saving model checkpoints, keeping best N checkpoints,
and resuming training from saved state.

Example:
    >>> manager = CheckpointManager("checkpoints", keep_best_n=3)
    >>> manager.save(model, optimizer, scheduler, step=1000, val_loss=3.42)
    >>> state = manager.load_best()
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Git commit hash or None if not in a git repository
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    step: int = 0,
    epoch: int = 0,
    val_loss: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    **extra_data
) -> None:
    """
    Save training checkpoint with full state.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        step: Current training step
        epoch: Current epoch
        val_loss: Validation loss (for tracking best model)
        config: Configuration dictionary
        **extra_data: Any additional data to save

    Example:
        >>> save_checkpoint(
        ...     "checkpoint.pt",
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     step=1000,
        ...     val_loss=3.42,
        ...     config={"d_model": 512}
        ... )
    """
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
    }

    # Add optional components
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    if config is not None:
        checkpoint['config'] = config

    # Add any extra data
    checkpoint.update(extra_data)

    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map checkpoint to

    Returns:
        Dictionary with checkpoint metadata (step, epoch, val_loss, etc.)

    Example:
        >>> metadata = load_checkpoint(
        ...     "checkpoint.pt",
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device='cuda'
        ... )
        >>> print(f"Resuming from step {metadata['step']}")
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Return metadata
    metadata = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss'),
        'timestamp': checkpoint.get('timestamp'),
        'git_commit': checkpoint.get('git_commit'),
        'config': checkpoint.get('config'),
    }

    return metadata


class CheckpointManager:
    """
    Manage model checkpoints with automatic pruning of worst checkpoints.

    Keeps track of the best N checkpoints by validation loss and automatically
    removes worse checkpoints to save disk space.

    Example:
        >>> manager = CheckpointManager("checkpoints", keep_best_n=3)
        >>>
        >>> # During training
        >>> manager.save(model, optimizer, scheduler, step=1000, val_loss=3.42)
        >>> manager.save(model, optimizer, scheduler, step=2000, val_loss=3.20)
        >>>
        >>> # Load best checkpoint
        >>> best_state = manager.load_best()
        >>> model.load_state_dict(best_state['model_state_dict'])
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_best_n: int = 3,
        metric_mode: str = 'min'
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
            metric_mode: 'min' or 'max' (for loss, use 'min'; for accuracy, use 'max')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_n = keep_best_n
        self.metric_mode = metric_mode

        # Track checkpoints: List of (metric_value, path, step) tuples
        self.checkpoints = []

        # Load existing checkpoints if resuming
        self._discover_existing_checkpoints()

    def _discover_existing_checkpoints(self):
        """Discover existing checkpoints in checkpoint directory."""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                val_loss = checkpoint.get('val_loss')
                step = checkpoint.get('step', 0)

                if val_loss is not None:
                    self.checkpoints.append((val_loss, str(checkpoint_file), step))
            except Exception:
                # Skip corrupted checkpoints
                continue

        # Sort checkpoints
        self._sort_checkpoints()

    def _sort_checkpoints(self):
        """Sort checkpoints by metric (best first)."""
        reverse = (self.metric_mode == 'max')
        self.checkpoints.sort(reverse=reverse)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        val_loss: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "checkpoint",
        **extra_data
    ) -> Path:
        """
        Save checkpoint and manage best N checkpoints.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            step: Training step
            epoch: Epoch number
            val_loss: Validation loss (or other metric)
            config: Configuration dict
            prefix: Checkpoint filename prefix
            **extra_data: Additional data to save

        Returns:
            Path to saved checkpoint

        Raises:
            ValueError: If val_loss is None and keep_best_n > 0
        """
        if self.keep_best_n > 0 and val_loss is None:
            raise ValueError("val_loss must be provided when keep_best_n > 0")

        # Create checkpoint filename
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{step}.pt"

        # Save checkpoint
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            val_loss=val_loss,
            config=config,
            **extra_data
        )

        # Track this checkpoint
        if val_loss is not None:
            self.checkpoints.append((val_loss, str(checkpoint_path), step))
            self._sort_checkpoints()

            # Prune worst checkpoints
            if len(self.checkpoints) > self.keep_best_n:
                self._prune_checkpoints()

        return checkpoint_path

    def _prune_checkpoints(self):
        """Remove worst checkpoints beyond keep_best_n."""
        while len(self.checkpoints) > self.keep_best_n:
            # Remove worst checkpoint (last in sorted list)
            _, worst_path, _ = self.checkpoints.pop()

            # Delete file
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass  # Already deleted

    def load_best(self, device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.

        Args:
            device: Device to map checkpoint to

        Returns:
            Checkpoint dictionary or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None

        # Best checkpoint is first in sorted list
        _, best_path, _ = self.checkpoints[0]

        return torch.load(best_path, map_location=device)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load specific checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to map checkpoint to

        Returns:
            Checkpoint metadata
        """
        return load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        if not self.checkpoints:
            return None
        _, best_path, _ = self.checkpoints[0]
        return Path(best_path)

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to most recent checkpoint (by step number).

        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            return None

        # Find checkpoint with highest step
        latest = max(self.checkpoints, key=lambda x: x[2])  # x[2] is step
        _, latest_path, _ = latest
        return Path(latest_path)

    def list_checkpoints(self) -> list[Tuple[float, Path, int]]:
        """
        List all tracked checkpoints.

        Returns:
            List of (metric_value, path, step) tuples, sorted by metric (best first)
        """
        return [(metric, Path(path), step) for metric, path, step in self.checkpoints]

    def __len__(self) -> int:
        """Number of tracked checkpoints."""
        return len(self.checkpoints)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CheckpointManager(dir={self.checkpoint_dir}, "
            f"keep_best_n={self.keep_best_n}, "
            f"num_checkpoints={len(self.checkpoints)})"
        )


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    from torch.optim import AdamW

    # Create simple model for demo
    model = nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Create checkpoint manager
    manager = CheckpointManager("demo_checkpoints", keep_best_n=3)

    print(f"Checkpoint manager: {manager}\n")

    # Simulate training with improving validation loss
    val_losses = [4.5, 4.2, 3.9, 3.7, 3.5, 3.3, 3.4, 3.2]  # Last one is best

    for step, val_loss in enumerate(val_losses, start=1):
        # Save checkpoint
        saved_path = manager.save(
            model=model,
            optimizer=optimizer,
            step=step * 1000,
            val_loss=val_loss,
            config={"d_model": 512}
        )
        print(f"Step {step * 1000}: Saved checkpoint with val_loss={val_loss:.2f}")

    print(f"\n✓ Saved {len(val_losses)} checkpoints")
    print(f"✓ Kept best {len(manager)} checkpoints\n")

    # List checkpoints
    print("Tracked checkpoints (best first):")
    for metric, path, step in manager.list_checkpoints():
        print(f"  Step {step}: val_loss={metric:.2f}, path={path.name}")

    # Load best checkpoint
    best_checkpoint = manager.load_best()
    print(f"\n✓ Best checkpoint: step={best_checkpoint['step']}, "
          f"val_loss={best_checkpoint['val_loss']:.2f}")

    # Load latest checkpoint (by step number)
    latest_path = manager.get_latest_checkpoint_path()
    print(f"✓ Latest checkpoint: {latest_path}")

    # Cleanup
    import shutil
    shutil.rmtree("demo_checkpoints")
    print("\n✓ Demo complete, cleaned up demo checkpoints")
