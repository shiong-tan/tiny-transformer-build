"""
Utilities for Tiny Transformer

Provides shape checking, visualization, and other helper functions.
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
    plot_attention_heads,
    plot_training_curves,
    plot_token_embeddings_2d,
    plot_gradient_flow,
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
    "plot_attention_heads",
    "plot_training_curves",
    "plot_token_embeddings_2d",
    "plot_gradient_flow",
]
