"""
Test edge cases for embeddings module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import math

# Import directly from files to avoid __init__ issues
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

def test_edge_cases():
    print("=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    print()

    # Test 1: Empty-like sequences (seq_len=1)
    print("Test 1: Minimum sequence length (seq_len=1)")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=512)
        tokens = torch.randint(0, 1000, (4, 1))
        output = emb(tokens)
        assert output.shape == (4, 1, 512)
        print("  ✓ seq_len=1 works correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Sequence at max_len boundary
    print("\nTest 2: Sequence exactly at max_len")
    try:
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=100)
        x = torch.randn(2, 100, 512)  # Exactly max_len
        output = pe(x)
        assert output.shape == (2, 100, 512)
        print("  ✓ Sequence at max_len boundary works")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Sequence exceeding max_len
    print("\nTest 3: Sequence exceeding max_len (should fail)")
    try:
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=100)
        x = torch.randn(2, 101, 512)  # Exceeds max_len
        output = pe(x)
        print("  ✗ Should have raised an error!")
    except AssertionError as e:
        print("  ✓ Correctly raises AssertionError")

    # Test 4: vocab_size boundary
    print("\nTest 4: Token ID at vocab_size-1 boundary")
    try:
        vocab_size = 1000
        emb = TokenEmbedding(vocab_size=vocab_size, d_model=256)
        tokens = torch.full((4, 10), vocab_size - 1)  # All tokens at max valid ID
        output = emb(tokens)
        assert output.shape == (4, 10, 256)
        print("  ✓ Token ID = vocab_size-1 works")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Token ID >= vocab_size (should fail)
    print("\nTest 5: Token ID >= vocab_size (should fail)")
    try:
        vocab_size = 1000
        emb = TokenEmbedding(vocab_size=vocab_size, d_model=256)
        tokens = torch.full((4, 10), vocab_size)  # Invalid: equals vocab_size
        output = emb(tokens)
        print("  ✗ Should have raised an error!")
    except AssertionError as e:
        print("  ✓ Correctly raises AssertionError")

    # Test 6: Negative token IDs (should fail)
    print("\nTest 6: Negative token IDs (should fail)")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        tokens = torch.tensor([[-1, 0, 1], [2, 3, 4]])
        output = emb(tokens)
        print("  ✗ Should have raised an error!")
    except AssertionError as e:
        print("  ✓ Correctly raises AssertionError")

    # Test 7: Position 0 handling
    print("\nTest 7: Position 0 encoding")
    try:
        pe = SinusoidalPositionalEncoding(d_model=8, max_len=100)
        # Position 0 should have pattern: [sin(0), cos(0), sin(0), cos(0), ...]
        # = [0, 1, 0, 1, 0, 1, ...]
        pos_0 = pe.pe[0]
        for i in range(8):
            expected = 0.0 if i % 2 == 0 else 1.0
            if abs(pos_0[i].item() - expected) > 1e-6:
                print(f"  ✗ Position 0, index {i}: expected {expected}, got {pos_0[i].item()}")
                break
        else:
            print("  ✓ Position 0 encoding correct")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 8: Device consistency - positional encoding
    print("\nTest 8: Device consistency (sinusoidal PE)")
    try:
        if torch.cuda.is_available():
            pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
            x_cpu = torch.randn(4, 50, 256)
            x_cuda = x_cpu.cuda()

            # PE should work with input device
            pe_cpu = pe(x_cpu)
            assert pe_cpu.device == x_cpu.device

            pe_module_cuda = pe.cuda()
            pe_cuda = pe_module_cuda(x_cuda)
            assert pe_cuda.device == x_cuda.device
            print("  ✓ Device handling correct (CPU & CUDA)")
        else:
            # Test CPU only
            pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
            x = torch.randn(4, 50, 256)
            output = pe(x)
            assert output.device == x.device
            print("  ✓ Device handling correct (CPU only)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 9: Device consistency - learned PE
    print("\nTest 9: Device consistency (learned PE)")
    try:
        pe = LearnedPositionalEmbedding(max_len=1000, d_model=256)
        x = torch.randn(4, 50, 256)
        output = pe(x)
        assert output.device == x.device
        print("  ✓ Learned PE device handling correct")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 10: Padding index = 0 embeddings are zero
    print("\nTest 10: Padding index embeddings are zero")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256, padding_idx=0)
        pad_emb = emb.embedding.weight[0]
        # Before scaling, padding embeddings should be zero
        if torch.allclose(pad_emb, torch.zeros(256)):
            print("  ✓ Padding embeddings are zero (before scaling)")
        else:
            print(f"  ✗ Padding embeddings not zero: norm={pad_emb.norm()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 11: Embedding scaling magnitude
    print("\nTest 11: Embedding scaling magnitude")
    try:
        d_model = 512
        emb = TokenEmbedding(vocab_size=10000, d_model=d_model)
        expected_scale = math.sqrt(d_model)

        if abs(emb.scale - expected_scale) < 1e-6:
            print(f"  ✓ Scaling factor correct: √{d_model} = {emb.scale:.4f}")
        else:
            print(f"  ✗ Scaling factor wrong: expected {expected_scale}, got {emb.scale}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 12: Odd d_model for sinusoidal (should fail)
    print("\nTest 12: Odd d_model for sinusoidal PE (should fail)")
    try:
        pe = SinusoidalPositionalEncoding(d_model=513, max_len=1000)
        print("  ✗ Should have raised an error!")
    except AssertionError as e:
        print("  ✓ Correctly raises AssertionError for odd d_model")

    # Test 13: Batch size = 1
    print("\nTest 13: Batch size = 1")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)

        tokens = torch.randint(0, 1000, (1, 50))
        embedded = emb(tokens)
        with_pe = pe(embedded)

        assert with_pe.shape == (1, 50, 256)
        print("  ✓ Batch size = 1 works correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 14: Large batch size
    print("\nTest 14: Large batch size (256)")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=128)
        tokens = torch.randint(0, 1000, (256, 32))
        output = emb(tokens)
        assert output.shape == (256, 32, 128)
        print("  ✓ Large batch size works correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 15: Wrong input dimensions (should fail)
    print("\nTest 15: Wrong input dimensions (should fail)")
    try:
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        tokens = torch.randint(0, 1000, (32,))  # 1D instead of 2D
        output = emb(tokens)
        print("  ✗ Should have raised an error!")
    except AssertionError as e:
        print("  ✓ Correctly raises AssertionError for wrong dimensions")

    print("\n" + "=" * 70)
    print("Edge Case Testing Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_edge_cases()
