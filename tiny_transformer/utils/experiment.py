"""
Experiment tracking utilities with Weights & Biases and TensorBoard support.

Provides unified interface for experiment tracking that works with multiple backends
(wandb, tensorboard) with graceful fallback if dependencies are not installed.

Example:
    >>> tracker = ExperimentTracker(
    ...     project="tiny-transformer",
    ...     experiment_name="shakespeare",
    ...     config={"lr": 1e-4, "batch_size": 32},
    ...     backend="wandb"
    ... )
    >>> tracker.log_metrics({"loss": 3.42}, step=100)
    >>> tracker.finish()
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Supports multiple backends (Weights & Biases, TensorBoard) with automatic
    fallback to console logging if dependencies are missing.

    Example:
        >>> # With Weights & Biases
        >>> tracker = ExperimentTracker(
        ...     project="tiny-transformer",
        ...     experiment_name="shakespeare_small",
        ...     config={"d_model": 512, "lr": 1e-4},
        ...     backend="wandb"
        ... )
        >>>
        >>> # Log metrics during training
        >>> for step in range(1000):
        ...     tracker.log_metrics({"train/loss": 3.42}, step=step)
        >>>
        >>> # Finish tracking
        >>> tracker.finish()
    """

    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        backend: str = "wandb",
        log_dir: Optional[Union[str, Path]] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            project: Project name (e.g., "tiny-transformer")
            experiment_name: Experiment name/run name
            config: Configuration dictionary (hyperparameters)
            backend: Backend to use ("wandb", "tensorboard", or "console")
            log_dir: Directory for logs (used by tensorboard)
            tags: List of tags for experiment
            notes: Notes/description for experiment
        """
        self.project = project
        self.experiment_name = experiment_name or "experiment"
        self.config = config or {}
        self.backend = backend
        self.log_dir = Path(log_dir) if log_dir else Path("runs")
        self.tags = tags
        self.notes = notes

        # Initialize backend
        self._wandb = None
        self._tb_writer = None
        self._initialized = False

        self._init_backend()

    def _init_backend(self):
        """Initialize the selected backend."""
        if self.backend == "wandb":
            self._init_wandb()
        elif self.backend == "tensorboard":
            self._init_tensorboard()
        elif self.backend == "console":
            self._init_console()
        else:
            warnings.warn(
                f"Unknown backend '{self.backend}', falling back to console logging"
            )
            self._init_console()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb

            self._wandb = wandb
            self._wandb.init(
                project=self.project,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
            )
            self._initialized = True
            print(f"✓ Initialized Weights & Biases tracking for '{self.experiment_name}'")

        except ImportError:
            warnings.warn(
                "Weights & Biases (wandb) not installed. "
                "Install with: pip install wandb\n"
                "Falling back to console logging."
            )
            self._init_console()

    def _init_tensorboard(self):
        """Initialize TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_path = self.log_dir / self.project / self.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)

            self._tb_writer = SummaryWriter(log_dir=str(log_path))
            self._initialized = True

            # Log config as text
            if self.config:
                config_str = "\n".join(f"{k}: {v}" for k, v in self.config.items())
                self._tb_writer.add_text("config", config_str, 0)

            print(f"✓ Initialized TensorBoard logging at: {log_path}")

        except ImportError:
            warnings.warn(
                "TensorBoard not installed (requires torch.utils.tensorboard). "
                "Falling back to console logging."
            )
            self._init_console()

    def _init_console(self):
        """Initialize console-only logging (fallback)."""
        self.backend = "console"
        self._initialized = True
        print(f"ℹ Using console logging for '{self.experiment_name}'")
        if self.config:
            print(f"ℹ Config: {self.config}")

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics (e.g., {"train/loss": 3.42, "lr": 1e-4})
            step: Global step number
            commit: Whether to commit (wandb only, affects when metrics are synced)

        Example:
            >>> tracker.log_metrics({"train/loss": 3.42, "train/ppl": 30.5}, step=100)
        """
        # Convert torch tensors to Python floats
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        if self.backend == "wandb" and self._wandb and self._wandb.run:
            self._wandb.log(metrics, step=step, commit=commit)

        elif self.backend == "tensorboard" and self._tb_writer:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)

        else:  # console
            step_str = f"Step {step}" if step is not None else "Step ?"
            metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in metrics.items())
            print(f"{step_str} | {metric_str}")

    def log_hyperparams(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters (useful if not provided at init).

        Args:
            hyperparams: Dictionary of hyperparameters

        Example:
            >>> tracker.log_hyperparams({"d_model": 512, "n_heads": 8})
        """
        self.config.update(hyperparams)

        if self.backend == "wandb" and self._wandb and self._wandb.run:
            self._wandb.config.update(hyperparams)

        elif self.backend == "tensorboard" and self._tb_writer:
            config_str = "\n".join(f"{k}: {v}" for k, v in hyperparams.items())
            self._tb_writer.add_text("hyperparameters", config_str, 0)

        else:  # console
            print(f"ℹ Hyperparams: {hyperparams}")

    def log_model(self, model: nn.Module, name: str = "model"):
        """
        Log model architecture and optionally save checkpoint.

        Args:
            model: PyTorch model to log
            name: Model name

        Example:
            >>> tracker.log_model(model, name="best_model")
        """
        if self.backend == "wandb" and self._wandb and self._wandb.run:
            # Log model architecture as text
            self._wandb.config.update({"model_summary": str(model)})

        elif self.backend == "tensorboard" and self._tb_writer:
            # TensorBoard can visualize model graph (requires example input)
            # For now, just log as text
            self._tb_writer.add_text("model_architecture", str(model), 0)

        else:  # console
            print(f"ℹ Model '{name}' logged")

    def save_artifact(self, artifact_path: Union[str, Path], name: str, artifact_type: str = "model"):
        """
        Save artifact (e.g., model checkpoint, dataset).

        Args:
            artifact_path: Path to artifact file
            name: Artifact name
            artifact_type: Type of artifact ("model", "dataset", etc.)

        Example:
            >>> tracker.save_artifact("checkpoint.pt", "best_checkpoint", "model")
        """
        if self.backend == "wandb" and self._wandb and self._wandb.run:
            artifact = self._wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(artifact_path))
            self._wandb.log_artifact(artifact)
            print(f"✓ Saved artifact '{name}' to Weights & Biases")

        elif self.backend == "tensorboard":
            # TensorBoard doesn't have native artifact support
            print(f"ℹ Artifact '{name}' saved locally at: {artifact_path}")

        else:  # console
            print(f"ℹ Artifact '{name}' at: {artifact_path}")

    def watch_model(self, model: nn.Module, log_freq: int = 100):
        """
        Watch model parameters and gradients (wandb only).

        Args:
            model: Model to watch
            log_freq: Logging frequency

        Example:
            >>> tracker.watch_model(model, log_freq=100)
        """
        if self.backend == "wandb" and self._wandb and self._wandb.run:
            self._wandb.watch(model, log="all", log_freq=log_freq)
            print("✓ Watching model parameters and gradients")
        else:
            print("ℹ Model watching only supported with Weights & Biases")

    def finish(self):
        """Finish experiment tracking and cleanup."""
        if self.backend == "wandb" and self._wandb and self._wandb.run:
            self._wandb.finish()
            print("✓ Finished Weights & Biases tracking")

        elif self.backend == "tensorboard" and self._tb_writer:
            self._tb_writer.close()
            print("✓ Closed TensorBoard writer")

        else:  # console
            print("✓ Finished console logging")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExperimentTracker(project='{self.project}', "
            f"experiment='{self.experiment_name}', "
            f"backend='{self.backend}')"
        )


def get_experiment_tracker(
    project: str,
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    backend: str = "auto",
    **kwargs
) -> ExperimentTracker:
    """
    Create an experiment tracker with automatic backend selection.

    Args:
        project: Project name
        experiment_name: Experiment name
        config: Configuration dict
        backend: Backend ("auto", "wandb", "tensorboard", "console")
        **kwargs: Additional arguments for ExperimentTracker

    Returns:
        Initialized ExperimentTracker

    Example:
        >>> tracker = get_experiment_tracker(
        ...     project="tiny-transformer",
        ...     experiment_name="shakespeare",
        ...     config={"lr": 1e-4}
        ... )
    """
    # Auto-detect available backend
    if backend == "auto":
        try:
            import wandb
            backend = "wandb"
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
                backend = "tensorboard"
            except ImportError:
                backend = "console"

    return ExperimentTracker(
        project=project,
        experiment_name=experiment_name,
        config=config,
        backend=backend,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create tracker (will use console if wandb not installed)
    tracker = ExperimentTracker(
        project="tiny-transformer",
        experiment_name="demo",
        config={"d_model": 512, "lr": 1e-4, "batch_size": 32},
        backend="console"  # Use console for demo
    )

    print(f"\nTracker: {tracker}\n")

    # Log some metrics
    for step in range(10):
        metrics = {
            "train/loss": 4.0 - step * 0.2,
            "train/perplexity": 50.0 - step * 3.0,
            "learning_rate": 1e-4 * (1 - step / 10),
        }
        tracker.log_metrics(metrics, step=step)

    # Log hyperparameters
    tracker.log_hyperparams({"warmup_steps": 100, "max_steps": 1000})

    # Finish tracking
    tracker.finish()

    print("\n" + "="*70)
    print("Example with context manager:")
    print("="*70 + "\n")

    # Using context manager
    with ExperimentTracker(
        project="tiny-transformer",
        experiment_name="demo_context",
        config={"lr": 1e-4},
        backend="console"
    ) as tracker:
        tracker.log_metrics({"loss": 3.42}, step=1)
        print("✓ Inside context manager")

    print("✓ Outside context manager (auto-closed)")
