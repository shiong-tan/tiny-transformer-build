"""
Scaled Dot-Product Attention Implementation

The fundamental building block of transformer models.

Key equation:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

This module implements both basic and masked versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    This is the core attention mechanism used in transformers.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)  
        value: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask tensor of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
              Values should be 0 (keep) or -inf (mask out)
        dropout: Optional dropout layer to apply to attention weights
        
    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention probabilities of shape (batch_size, seq_len, seq_len)
        
    Shape annotations:
        B = batch_size
        T = seq_len (sequence length)
        d_k = key/query dimension
        d_v = value dimension (often equal to d_k)
    """
    # Get dimensions
    # query: (B, T, d_k)
    # key: (B, T, d_k)
    # value: (B, T, d_v)
    
    d_k = query.size(-1)  # Key dimension for scaling
    
    # Step 1: Compute attention scores
    # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
    scores = query @ key.transpose(-2, -1)  # Matrix multiplication
    
    # Step 2: Scale by sqrt(d_k)
    # Why? Prevents dot products from growing too large, which would lead to
    # very small gradients after softmax (vanishing gradient problem)
    scores = scores / math.sqrt(d_k)  # (B, T, T)
    
    # Step 3: Apply mask if provided (for causal attention)
    if mask is not None:
        # Add mask (which contains -inf for positions to mask out)
        # After softmax, these will become 0
        scores = scores + mask  # Broadcasting: mask can be (T, T) or (B, T, T)
    
    # Step 4: Apply softmax to get attention probabilities
    # Softmax is applied over the last dimension (key positions)
    # Each query position gets a probability distribution over all key positions
    attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
    
    # Step 5: Apply dropout if provided (during training)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Step 6: Apply attention weights to values
    # (B, T, T) @ (B, T, d_v) -> (B, T, d_v)
    output = attention_weights @ value
    
    return output, attention_weights


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a causal (lower-triangular) mask for autoregressive generation.
    
    This ensures that position i can only attend to positions <= i.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        mask: Causal mask of shape (seq_len, seq_len) with 0s and -inf
        
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[ 0., -inf, -inf, -inf],
                [ 0.,  0., -inf, -inf],
                [ 0.,  0.,  0., -inf],
                [ 0.,  0.,  0.,  0.]])
    """
    # Create lower triangular matrix of ones
    # tril creates: [[1, 0, 0],
    #                [1, 1, 0],
    #                [1, 1, 1]]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Convert to 0 and -inf
    # 1 -> 0 (keep), 0 -> -inf (mask out)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    
    return mask


class Attention(nn.Module):
    """
    Attention module that can be used as a layer in a network.
    
    This wraps the functional attention implementation with learnable
    parameters for Q, K, V projections (added in multi-head attention).
    
    For now, this is a simple wrapper for the functional version.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize attention module.
        
        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention.
        
        Args:
            query, key, value: Input tensors
            mask: Optional attention mask
            
        Returns:
            output: Attention output
            attention_weights: Attention probabilities
        """
        return scaled_dot_product_attention(query, key, value, mask, self.dropout)


# Example usage and testing
if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    print("="*70)
    print("Attention Mechanism - Basic Example")
    print("="*70)
    
    # Small example: batch_size=2, seq_len=4, d_k=8
    batch_size = 2
    seq_len = 4
    d_k = 8
    
    # Create random Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    print(f"\nInput shapes:")
    print(f"  Q: {tuple(Q.shape)} - Queries")
    print(f"  K: {tuple(K.shape)} - Keys")
    print(f"  V: {tuple(V.shape)} - Values")
    
    # Compute attention without mask
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  Attention weights: {tuple(attn_weights.shape)}")
    
    print(f"\nAttention weights for first example:")
    print(attn_weights[0].detach().numpy())
    print(f"\nRow sums (should all be 1.0):")
    print(attn_weights[0].sum(dim=-1).detach().numpy())
    
    # Example with causal mask
    print("\n" + "="*70)
    print("Causal Attention - Autoregressive Example")
    print("="*70)
    
    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask shape: {tuple(mask.shape)}")
    print(f"Causal mask:\n{mask.numpy()}")
    
    output_causal, attn_weights_causal = scaled_dot_product_attention(Q, K, V, mask)
    
    print(f"\nCausal attention weights for first example:")
    print(attn_weights_causal[0].detach().numpy())
    print("\nNotice: Upper triangle is now zeros (can't attend to future)")
    
    print(f"\nRow sums (should still all be 1.0):")
    print(attn_weights_causal[0].sum(dim=-1).detach().numpy())
