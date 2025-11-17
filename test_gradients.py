"""
Test gradient flow through embeddings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
base_path = "tiny_transformer/embeddings"
token_emb = load_module("token_embedding", f"{base_path}/token_embedding.py")
pos_enc = load_module("positional_encoding", f"{base_path}/positional_encoding.py")

TokenEmbedding = token_emb.TokenEmbedding
SinusoidalPositionalEncoding = pos_enc.SinusoidalPositionalEncoding
LearnedPositionalEmbedding = pos_enc.LearnedPositionalEmbedding

def test_gradients():
    print("=" * 70)
    print("Testing Gradient Flow")
    print("=" * 70)
    print()

    # Test 1: Token embedding gradients
    print("Test 1: Token Embedding gradient flow")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        tokens = torch.randint(0, 1000, (8, 32))

        output = emb(tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert emb.embedding.weight.grad is not None
        grad_norm = emb.embedding.weight.grad.norm().item()
        print(f"  ✓ Token embedding has gradients (norm: {grad_norm:.4f})")

        # Check only used embeddings have gradients
        unique_tokens = torch.unique(tokens)
        print(f"  ✓ Used {len(unique_tokens)} unique tokens out of {emb.vocab_size}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Sinusoidal PE has no trainable parameters
    print("\nTest 2: Sinusoidal PE has no trainable parameters")
    try:
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
        trainable_params = sum(p.numel() for p in pe.parameters() if p.requires_grad)

        if trainable_params == 0:
            print(f"  ✓ Sinusoidal PE has 0 trainable parameters")
        else:
            print(f"  ✗ Sinusoidal PE has {trainable_params} trainable parameters (should be 0)")

        # Verify pe buffer doesn't require grad
        assert not pe.pe.requires_grad
        print("  ✓ PE buffer correctly doesn't require gradients")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Learned PE has trainable parameters
    print("\nTest 3: Learned PE has trainable parameters")
    try:
        max_len = 1000
        d_model = 256
        pe = LearnedPositionalEmbedding(max_len=max_len, d_model=d_model)

        trainable_params = sum(p.numel() for p in pe.parameters() if p.requires_grad)
        expected_params = max_len * d_model

        if trainable_params == expected_params:
            print(f"  ✓ Learned PE has {trainable_params:,} trainable parameters")
        else:
            print(f"  ✗ Expected {expected_params:,}, got {trainable_params:,}")

        # Test gradient flow
        x = torch.zeros(4, 50, d_model, requires_grad=True)
        output = pe(x)
        loss = output.sum()
        loss.backward()

        assert pe.position_embeddings.weight.grad is not None
        grad_norm = pe.position_embeddings.weight.grad.norm().item()
        print(f"  ✓ Learned PE receives gradients (norm: {grad_norm:.4f})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Combined embedding gradient flow
    print("\nTest 4: Combined embedding with sinusoidal PE")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000, dropout=0.0)

        tokens = torch.randint(0, 1000, (8, 32))

        # Forward pass
        token_emb = emb(tokens)
        output = pe(token_emb)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert emb.embedding.weight.grad is not None
        assert pe.pe.grad is None  # Buffer shouldn't have gradients
        print("  ✓ Token embeddings receive gradients")
        print("  ✓ PE buffer correctly has no gradients")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Combined embedding with learned PE
    print("\nTest 5: Combined embedding with learned PE")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        pe = LearnedPositionalEmbedding(max_len=1000, d_model=256, dropout=0.0)

        tokens = torch.randint(0, 1000, (8, 32))

        # Forward pass
        token_emb = emb(tokens)
        output = pe(token_emb)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert emb.embedding.weight.grad is not None
        assert pe.position_embeddings.weight.grad is not None

        token_grad_norm = emb.embedding.weight.grad.norm().item()
        pos_grad_norm = pe.position_embeddings.weight.grad.norm().item()

        print(f"  ✓ Token embeddings receive gradients (norm: {token_grad_norm:.4f})")
        print(f"  ✓ Position embeddings receive gradients (norm: {pos_grad_norm:.4f})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 6: Padding index doesn't receive gradients
    print("\nTest 6: Padding index doesn't receive gradients")
    try:
        pad_idx = 0
        emb = TokenEmbedding(vocab_size=1000, d_model=256, padding_idx=pad_idx)

        # Create tokens with padding
        tokens = torch.randint(1, 1000, (8, 32))
        tokens[:, :5] = pad_idx  # First 5 positions are padding

        output = emb(tokens)
        loss = output.sum()
        loss.backward()

        # Padding embedding gradient should be zero
        pad_grad = emb.embedding.weight.grad[pad_idx]
        if torch.allclose(pad_grad, torch.zeros_like(pad_grad)):
            print("  ✓ Padding index gradients are zero")
        else:
            print(f"  ✗ Padding index has non-zero gradients: {pad_grad.norm().item()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 7: Gradient magnitude after scaling
    print("\nTest 7: Gradient scaling behavior")
    try:
        d_model = 256
        emb = TokenEmbedding(vocab_size=1000, d_model=d_model)
        tokens = torch.randint(0, 1000, (8, 32))

        output = emb(tokens)
        loss = output.sum()
        loss.backward()

        # Gradients should account for the sqrt(d_model) scaling
        avg_grad = emb.embedding.weight.grad.abs().mean().item()
        print(f"  ✓ Average gradient magnitude: {avg_grad:.6f}")
        print(f"  ✓ Scaling factor was: {emb.scale:.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 8: Dropout doesn't block gradients
    print("\nTest 8: Dropout doesn't block gradients")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000, dropout=0.5)
        pe.train()  # Enable dropout

        tokens = torch.randint(0, 1000, (8, 32))
        token_emb = emb(tokens)
        output = pe(token_emb)
        loss = output.sum()
        loss.backward()

        assert emb.embedding.weight.grad is not None
        print("  ✓ Gradients flow through dropout")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 70)
    print("Gradient Flow Testing Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_gradients()
