"""
Structured logging utilities for training runs.

Provides JSON-formatted logging for easy parsing and analysis of training metrics,
hyperparameters, and system information.

Example:
    >>> logger = get_training_logger("my_experiment")
    >>> logger.log_metrics({"loss": 3.42, "perplexity": 30.5}, step=100)
    >>> logger.log_hyperparams({"learning_rate": 1e-4, "batch_size": 32})
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add any extra fields passed via logging.info(..., extra={...})
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        if hasattr(record, "step"):
            log_data["step"] = record.step
        if hasattr(record, "epoch"):
            log_data["epoch"] = record.epoch

        return json.dumps(log_data)


class TrainingLogger:
    """
    Structured logger for training runs.

    Provides convenient methods for logging metrics, hyperparameters, and system info
    in JSON format for easy parsing and analysis.

    Example:
        >>> logger = TrainingLogger("my_experiment", log_dir="logs")
        >>> logger.log_start()
        >>> logger.log_hyperparams({"lr": 1e-4, "batch_size": 32})
        >>>
        >>> for step in range(100):
        >>>     logger.log_metrics({"loss": 3.42}, step=step)
        >>>
        >>> logger.log_end()
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Union[str, Path]] = None,
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        """Initialize training logger.

        Args:
            experiment_name: Name of experiment (used in log filename)
            log_dir: Directory to save log files (default: "logs")
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file

        # Create logger
        self.logger = logging.getLogger(f"training.{experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear any existing handlers

        # Console handler (human-readable)
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler (JSON format)
        if log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

            self.log_file = log_file
        else:
            self.log_file = None

    def log_start(self, config: Optional[Dict[str, Any]] = None):
        """Log experiment start with optional configuration.

        Args:
            config: Configuration dictionary to log
        """
        msg = f"Starting experiment: {self.experiment_name}"
        self.logger.info(msg)

        if config:
            self.log_hyperparams(config)

        # Log system info
        self.log_system_info()

    def log_hyperparams(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.logger.info(
            f"Hyperparameters: {json.dumps(hyperparams, indent=2)}",
            extra={"metrics": hyperparams}
        )

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ):
        """Log training/validation metrics.

        Args:
            metrics: Dictionary of metrics (e.g., {"loss": 3.42, "perplexity": 30.5})
            step: Training step number
            epoch: Epoch number
            prefix: Prefix for metric names (e.g., "train/" or "val/")
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Convert torch tensors to Python floats
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        # Format message
        msg_parts = []
        if step is not None:
            msg_parts.append(f"Step {step}")
        if epoch is not None:
            msg_parts.append(f"Epoch {epoch}")

        metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in metrics.items()]
        msg_parts.extend(metric_strs)

        msg = " | ".join(msg_parts)

        # Log with extra data for JSON formatting
        extra = {"metrics": metrics}
        if step is not None:
            extra["step"] = step
        if epoch is not None:
            extra["epoch"] = epoch

        self.logger.info(msg, extra=extra)

    def log_system_info(self):
        """Log system information (PyTorch version, CUDA, etc.)."""
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
        }

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["device"] = "MPS (Apple Silicon)"

        self.logger.info(
            f"System info: {json.dumps(info, indent=2)}",
            extra={"metrics": info}
        )

    def log_checkpoint(self, checkpoint_path: Union[str, Path], step: int, metrics: Dict[str, float]):
        """Log checkpoint save event.

        Args:
            checkpoint_path: Path where checkpoint was saved
            step: Training step
            metrics: Metrics at checkpoint (e.g., validation loss)
        """
        msg = f"Saved checkpoint: {checkpoint_path} | Step {step} | Metrics: {metrics}"
        self.logger.info(
            msg,
            extra={"step": step, "metrics": metrics}
        )

    def log_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """Log experiment end with optional final metrics.

        Args:
            final_metrics: Final metrics to log
        """
        msg = f"Experiment {self.experiment_name} completed"

        if final_metrics:
            msg += f" | Final metrics: {final_metrics}"
            self.logger.info(msg, extra={"metrics": final_metrics})
        else:
            self.logger.info(msg)

    def log_error(self, error: Exception, step: Optional[int] = None):
        """Log error during training.

        Args:
            error: Exception that occurred
            step: Step where error occurred
        """
        msg = f"Error: {str(error)}"
        extra = {}
        if step is not None:
            extra["step"] = step
            msg = f"Step {step} | {msg}"

        self.logger.error(msg, extra=extra, exc_info=True)

    def info(self, msg: str, **kwargs):
        """Log general info message.

        Args:
            msg: Message to log
            **kwargs: Additional fields for JSON logging
        """
        self.logger.info(msg, extra={"metrics": kwargs} if kwargs else {})


def get_training_logger(
    experiment_name: str,
    log_dir: Optional[Union[str, Path]] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> TrainingLogger:
    """
    Create a training logger.

    Convenience function for creating a TrainingLogger instance.

    Args:
        experiment_name: Name of experiment
        log_dir: Directory to save log files
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file

    Returns:
        TrainingLogger instance

    Example:
        >>> logger = get_training_logger("shakespeare_small")
        >>> logger.log_start({"learning_rate": 1e-4, "batch_size": 32})
        >>> logger.log_metrics({"loss": 3.42}, step=100)
    """
    return TrainingLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
    )


def parse_json_logs(log_file: Union[str, Path]) -> list[Dict[str, Any]]:
    """
    Parse JSON log file into list of dictionaries.

    Args:
        log_file: Path to .jsonl log file

    Returns:
        List of parsed log entries

    Example:
        >>> logs = parse_json_logs("logs/experiment_20231116.jsonl")
        >>> steps = [log["step"] for log in logs if "step" in log]
        >>> losses = [log["metrics"]["loss"] for log in logs if "loss" in log.get("metrics", {})]
    """
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return logs


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_training_logger("demo_experiment", log_dir="logs")

    # Log experiment start
    logger.log_start(config={
        "model": {"d_model": 512, "n_heads": 8},
        "training": {"lr": 1e-4, "batch_size": 32}
    })

    # Simulate training
    for step in range(10):
        metrics = {
            "loss": 4.0 - step * 0.1,
            "perplexity": 50.0 - step * 2.0,
        }
        logger.log_metrics(metrics, step=step, prefix="train/")

        if step % 5 == 0:
            val_metrics = {"loss": 3.8 - step * 0.08, "perplexity": 45.0}
            logger.log_metrics(val_metrics, step=step, prefix="val/")
            logger.log_checkpoint(f"checkpoint_{step}.pt", step=step, metrics=val_metrics)

    # Log end
    logger.log_end(final_metrics={"loss": 3.0, "perplexity": 20.0})

    # Parse logs
    if logger.log_file:
        logs = parse_json_logs(logger.log_file)
        print(f"\n✓ Created {len(logs)} log entries")
        print(f"✓ Log file: {logger.log_file}")
