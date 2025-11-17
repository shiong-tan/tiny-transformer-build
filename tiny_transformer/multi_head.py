"""
Multi-Head Attention Implementation

Extends the single-head attention mechanism to multiple parallel attention heads.

Key equation:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

This allows the model to jointly attend to information from different
representation subspaces at different positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from tiny_transformer.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Instead of performing a single attention function with d_model-dimensional keys,
    values and queries, we linearly project the queries, keys and values h times
    with different learned linear projections to d_k, d_k and d_v dimensions, respectively.

    On each of these projected versions of queries, keys and values we then perform
    the attention function in parallel, yielding d_v-dimensional output values.
    These are concatenated and once again projected, resulting in the final values.

    Args:
        d_model: Model dimension (input/output dimension)
        n_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        bias: Whether to use bias in linear projections

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention module.

        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of parallel attention heads
            dropout: Dropout probability
            bias: Whether to include bias in projections
        """
        super().__init__()

        # Validate inputs
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Usually same as d_k

        # Linear projections for Q, K, V
        # Note: We use a single linear layer that projects to all heads at once,
        # then split into heads. This is more efficient than separate projections.
        self.W_q = nn.Linear(d_model, d_model, bias=bias)  # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)  # (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)  # (d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)  # (d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).

        This reshapes the input to separate out the different attention heads,
        allowing them to be computed in parallel.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_k)

        Shape transformations:
            (B, T, d_model) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: (B, T, d_model) -> (B, T, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose: (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        # This puts the head dimension before sequence dimension for parallel computation
        x = x.transpose(1, 2)

        return x  # (B, n_heads, T, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads: merge the head dimension back.

        Concatenates all attention heads back into a single tensor.

        Args:
            x: Tensor of shape (batch_size, n_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)

        Shape transformations:
            (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()

        # Transpose: (B, n_heads, T, d_k) -> (B, T, n_heads, d_k)
        x = x.transpose(1, 2)

        # Reshape: (B, T, n_heads, d_k) -> (B, T, d_model)
        # contiguous() is needed because transpose() doesn't create a new tensor
        x = x.contiguous().view(batch_size, seq_len, self.d_model)

        return x  # (B, T, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
                  or (batch_size, 1, seq_len, seq_len) for multi-head broadcasting

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len_q, seq_len_k)

        Shape flow:
            Input:  (B, T, d_model)
            After projection: (B, T, d_model)
            After split_heads: (B, n_heads, T, d_k)
            After attention: (B, n_heads, T, d_v)
            After combine_heads: (B, T, d_model)
            After output projection: (B, T, d_model)
        """
        batch_size = query.size(0)

        # Validate input dimensions
        assert query.size(-1) == self.d_model, \
            f"query dimension {query.size(-1)} != d_model {self.d_model}"
        assert key.size(-1) == self.d_model, \
            f"key dimension {key.size(-1)} != d_model {self.d_model}"
        assert value.size(-1) == self.d_model, \
            f"value dimension {value.size(-1)} != d_model {self.d_model}"

        # Step 1: Linear projections
        # (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
        Q = self.W_q(query)  # (B, T, d_model)
        K = self.W_k(key)    # (B, T, d_model)
        V = self.W_v(value)  # (B, T, d_model)

        # Step 2: Split into multiple heads
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        Q = self.split_heads(Q)  # (B, n_heads, T, d_k)
        K = self.split_heads(K)  # (B, n_heads, T, d_k)
        V = self.split_heads(V)  # (B, n_heads, T, d_v) where d_v == d_k

        # Step 3: Apply scaled dot-product attention on each head in parallel
        # The attention function expects (B, T, d_k) but we have (B, n_heads, T, d_k)
        # We can reshape to (B*n_heads, T, d_k) and process as larger batch

        # Reshape for batched attention: (B, n_heads, T, d_k) -> (B*n_heads, T, d_k)
        B_n, T_q, d_k = batch_size * self.n_heads, Q.size(2), self.d_k
        Q_batched = Q.reshape(B_n, T_q, d_k)  # (B*n_heads, T, d_k)
        K_batched = K.reshape(B_n, K.size(2), d_k)  # (B*n_heads, T, d_k)
        V_batched = V.reshape(B_n, V.size(2), self.d_v)  # (B*n_heads, T, d_v)

        # Handle mask broadcasting
        # Mask can be (T, T), (B, T, T), or (B, 1, T, T)
        # Values should be 0 (keep) or -inf (mask out)
        mask_batched = None
        if mask is not None:
            if mask.dim() == 2:  # (T, T)
                # Validate mask shape
                assert mask.shape[0] == T_q and mask.shape[1] == K.size(2), \
                    f"2D mask shape {mask.shape} doesn't match sequence lengths ({T_q}, {K.size(2)})"
                # Broadcast to (B*n_heads, T, T)
                mask_batched = mask.unsqueeze(0).expand(B_n, -1, -1)
            elif mask.dim() == 3:  # (B, T, T)
                # Validate mask shape
                assert mask.shape[0] == batch_size and mask.shape[1] == T_q and mask.shape[2] == K.size(2), \
                    f"3D mask shape {mask.shape} incompatible with batch_size={batch_size}, seq_lens=({T_q}, {K.size(2)})"
                # Repeat for each head: (B, T, T) -> (B*n_heads, T, T)
                mask_batched = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                mask_batched = mask_batched.reshape(B_n, mask.size(1), mask.size(2))
            elif mask.dim() == 4:  # (B, 1, T, T) - already head-broadcasted
                # Validate mask shape
                assert mask.shape[0] == batch_size and mask.shape[2] == T_q and mask.shape[3] == K.size(2), \
                    f"4D mask shape {mask.shape} incompatible with batch_size={batch_size}, seq_lens=({T_q}, {K.size(2)})"
                # Reshape: (B, 1, T, T) -> (B*n_heads, T, T)
                mask_batched = mask.expand(-1, self.n_heads, -1, -1)
                mask_batched = mask_batched.reshape(B_n, mask.size(2), mask.size(3))

        # Apply attention
        # (B*n_heads, T, d_k) -> (B*n_heads, T, d_v)
        attn_output, attn_weights = scaled_dot_product_attention(
            Q_batched, K_batched, V_batched,
            mask=mask_batched,
            dropout=self.dropout
        )

        # Reshape back: (B*n_heads, T, d_v) -> (B, n_heads, T, d_v)
        attn_output = attn_output.view(batch_size, self.n_heads, T_q, self.d_v)
        attn_weights = attn_weights.view(batch_size, self.n_heads, T_q, K.size(2))

        # Step 4: Concatenate heads
        # (B, n_heads, T, d_v) -> (B, T, d_model)
        output = self.combine_heads(attn_output)  # (B, T, d_model)

        # Step 5: Final linear projection
        # (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
        output = self.W_o(output)  # (B, T, d_model)

        return output, attn_weights


# Example usage and testing
if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    print("="*70)
    print("Multi-Head Attention - Basic Example")
    print("="*70)

    # Model dimensions
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 8

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k = d_v = d_model / n_heads = {d_model // n_heads}")

    # Create multi-head attention module
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)

    # Create input (for self-attention, Q=K=V)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {tuple(x.shape)}")

    # Forward pass (self-attention)
    output, attn_weights = mha(x, x, x)

    print(f"\nOutput shapes:")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  Attention weights: {tuple(attn_weights.shape)}")
    print(f"  (batch_size, n_heads, seq_len, seq_len)")

    # Verify shapes
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len), "Attention shape mismatch!"

    print(f"\n✓ Shape verification passed!")

    # Show attention pattern for first example, first head
    print(f"\nAttention weights for batch 0, head 0:")
    print(attn_weights[0, 0].detach().numpy())
    print(f"\nRow sums (should all be ~1.0):")
    print(attn_weights[0, 0].sum(dim=-1).detach().numpy())

    # Example with causal mask
    print("\n" + "="*70)
    print("Multi-Head Attention with Causal Mask")
    print("="*70)

    from tiny_transformer.attention import create_causal_mask

    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask shape: {tuple(mask.shape)}")

    # Forward pass with mask
    output_masked, attn_weights_masked = mha(x, x, x, mask=mask)

    print(f"\nMasked attention weights for batch 0, head 0:")
    print(attn_weights_masked[0, 0].detach().numpy())
    print("\nNotice: Upper triangle is now zeros (can't attend to future)")

    # Count parameters
    print("\n" + "="*70)
    print("Parameter Count")
    print("="*70)

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print(f"\nBreakdown:")
    for name, param in mha.named_parameters():
        print(f"  {name:20s}: {tuple(param.shape):30s} = {param.numel():,} params")

    # Expected: 4 * (d_model * d_model) for W_q, W_k, W_v, W_o
    expected = 4 * (d_model * d_model)
    print(f"\nExpected (4 × d_model²): {expected:,}")
    print(f"Actual: {total_params:,}")
    print(f"Match: {'✓' if total_params == expected else '✗'}")

    # Different Q, K, V (cross-attention example)
    print("\n" + "="*70)
    print("Cross-Attention Example")
    print("="*70)

    # Create different tensors for Q (decoder) and K,V (encoder)
    decoder_input = torch.randn(batch_size, 5, d_model)  # Shorter sequence
    encoder_output = torch.randn(batch_size, 10, d_model)  # Longer sequence

    print(f"\nDecoder input (Q): {tuple(decoder_input.shape)}")
    print(f"Encoder output (K, V): {tuple(encoder_output.shape)}")

    # Cross-attention: Q from decoder, K and V from encoder
    cross_attn_output, cross_attn_weights = mha(
        query=decoder_input,
        key=encoder_output,
        value=encoder_output
    )

    print(f"\nCross-attention output: {tuple(cross_attn_output.shape)}")
    print(f"Cross-attention weights: {tuple(cross_attn_weights.shape)}")
    print(f"(batch_size, n_heads, decoder_len, encoder_len)")

    # Verify output matches query sequence length
    assert cross_attn_output.shape == (batch_size, 5, d_model)
    assert cross_attn_weights.shape == (batch_size, n_heads, 5, 10)

    print(f"\n✓ Cross-attention shapes correct!")

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
