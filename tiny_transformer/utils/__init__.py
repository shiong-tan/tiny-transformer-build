"""
Utilities for Tiny Transformer

Provides shape checking, visualization, logging, checkpointing,
and experiment tracking utilities.
"""

from tiny_transformer.utils.shape_checker import (
    check_shape,
    assert_batch_consistency,
    print_shape_info,
    check_attention_shapes,
    check_transformer_block_shapes,
    ShapeTracer,
    shape_check,
)

from tiny_transformer.utils.visualization import (
    plot_attention_pattern,
    plot_multi_head_attention,
    plot_training_curves,
    plot_token_embeddings_2d,
    plot_gradient_flow,
)

from tiny_transformer.utils.logging import (
    TrainingLogger,
    get_training_logger,
    parse_json_logs,
    JSONFormatter,
)

from tiny_transformer.utils.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    get_git_commit,
)

from tiny_transformer.utils.experiment import (
    ExperimentTracker,
    get_experiment_tracker,
)

__all__ = [
    # Shape checking
    "check_shape",
    "assert_batch_consistency",
    "print_shape_info",
    "check_attention_shapes",
    "check_transformer_block_shapes",
    "ShapeTracer",
    "shape_check",
    # Visualization
    "plot_attention_pattern",
    "plot_multi_head_attention",
    "plot_training_curves",
    "plot_token_embeddings_2d",
    "plot_gradient_flow",
    # Logging
    "TrainingLogger",
    "get_training_logger",
    "parse_json_logs",
    "JSONFormatter",
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "get_git_commit",
    # Experiment tracking
    "ExperimentTracker",
    "get_experiment_tracker",
]
