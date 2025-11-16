"""
Shape Checking Utilities

Critical for debugging transformer implementations where tensor shapes
can be complex and errors often manifest as shape mismatches.

Philosophy: "Shapes first, values second" - if shapes are wrong, 
nothing else matters.
"""

import torch
from typing import Tuple, Optional, Union


def check_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Union[int, str], ...],
    name: str = "tensor"
) -> None:
    """
    Verify a tensor has the expected shape.
    
    Supports named dimensions using strings (e.g., 'B' for batch, 'T' for time).
    
    Args:
        tensor: Tensor to check
        expected_shape: Tuple of expected dimensions. Use strings for named dims.
        name: Name of tensor for error messages
        
    Example:
        >>> x = torch.randn(32, 10, 512)
        >>> check_shape(x, ('B', 'T', 'd_model'), 'input')  # Passes
        >>> check_shape(x, (32, 10, 512), 'input')  # Passes
        >>> check_shape(x, (32, 10, 256), 'input')  # Raises AssertionError
    """
    actual_shape = tuple(tensor.shape)
    
    # Convert named dimensions to 'any' for comparison
    expected_numeric = []
    for i, dim in enumerate(expected_shape):
        if isinstance(dim, str):
            # Named dimension - just check it matches the actual dimension
            expected_numeric.append(actual_shape[i])
        else:
            expected_numeric.append(dim)
    
    expected_numeric = tuple(expected_numeric)
    
    if actual_shape != expected_numeric:
        # Create helpful error message
        dim_names = []
        for dim in expected_shape:
            if isinstance(dim, str):
                dim_names.append(dim)
            else:
                dim_names.append(str(dim))
        
        expected_str = f"({', '.join(dim_names)})"
        actual_str = f"{actual_shape}"
        
        raise AssertionError(
            f"\nShape mismatch for {name}:\n"
            f"  Expected: {expected_str}\n"
            f"  Got:      {actual_str}\n"
            f"  Tensor shape: {tensor.shape}"
        )


def assert_batch_consistency(*tensors: torch.Tensor) -> None:
    """
    Verify all tensors have the same batch size.
    
    Args:
        *tensors: Variable number of tensors to check
        
    Raises:
        AssertionError: If batch sizes don't match
    """
    if len(tensors) == 0:
        return
    
    batch_size = tensors[0].shape[0]
    
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape[0] != batch_size:
            raise AssertionError(
                f"Batch size mismatch:\n"
                f"  Tensor 0: batch_size={batch_size}\n"
                f"  Tensor {i}: batch_size={tensor.shape[0]}"
            )


def print_shape_info(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Print detailed shape information for debugging.
    
    Args:
        tensor: Tensor to inspect
        name: Name of tensor
    """
    print(f"\n{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dims:  {tensor.ndim}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    if tensor.ndim > 0:
        print(f"  Min/Max: {tensor.min():.4f} / {tensor.max():.4f}")


def check_attention_shapes(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    batch_size: int,
    seq_len: int,
    d_k: int
) -> None:
    """
    Specialized check for attention mechanism shapes.
    
    Args:
        Q, K, V: Query, Key, Value tensors
        batch_size: Expected batch size
        seq_len: Expected sequence length
        d_k: Expected dimension per head
    """
    expected = (batch_size, seq_len, d_k)
    
    check_shape(Q, expected, "Queries (Q)")
    check_shape(K, expected, "Keys (K)")
    check_shape(V, expected, "Values (V)")


def check_transformer_block_shapes(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    batch_size: int,
    seq_len: int,
    d_model: int
) -> None:
    """
    Verify transformer block input and output shapes match.
    
    Transformer blocks should maintain shape: (B, T, d_model) -> (B, T, d_model)
    
    Args:
        x_in: Input tensor
        x_out: Output tensor
        batch_size, seq_len, d_model: Expected dimensions
    """
    expected = (batch_size, seq_len, d_model)
    
    check_shape(x_in, expected, "Block input")
    check_shape(x_out, expected, "Block output")
    
    # Verify they match each other
    if x_in.shape != x_out.shape:
        raise AssertionError(
            f"Transformer block changed shape!\n"
            f"  Input:  {x_in.shape}\n"
            f"  Output: {x_out.shape}\n"
            f"  Blocks should preserve shape."
        )


class ShapeTracer:
    """
    Context manager for tracing shapes through a forward pass.
    
    Useful for debugging complex models.
    
    Example:
        >>> with ShapeTracer() as tracer:
        ...     x = model.embed(tokens)  # tracer records shape
        ...     x = model.block1(x)      # tracer records shape
        ...     x = model.block2(x)      # tracer records shape
        ...     logits = model.head(x)   # tracer records shape
        >>> tracer.print_trace()
    """
    
    def __init__(self):
        self.shapes = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def record(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Record tensor shape and return tensor unchanged."""
        self.shapes.append((name, tuple(tensor.shape)))
        return tensor
    
    def print_trace(self) -> None:
        """Print all recorded shapes."""
        print("\n" + "="*60)
        print("Shape Trace:")
        print("="*60)
        for name, shape in self.shapes:
            print(f"  {name:30s} {str(shape)}")
        print("="*60 + "\n")
    
    def clear(self) -> None:
        """Clear recorded shapes."""
        self.shapes = []


# Convenience function for inline shape checking
def shape_check(tensor: torch.Tensor, *dims: Union[int, str]) -> torch.Tensor:
    """
    Check shape and return tensor (for use in forward pass).
    
    Example:
        >>> x = shape_check(x, 'B', 'T', 512)  # Checks but doesn't interrupt flow
    """
    check_shape(tensor, dims, "inline check")
    return tensor
