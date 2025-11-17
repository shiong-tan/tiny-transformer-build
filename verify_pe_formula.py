"""
Standalone script to verify positional encoding formula correctness.
"""
import torch
import math

def verify_positional_encoding_formula():
    """
    Verify that the positional encoding follows the exact formula from
    "Attention is All You Need":

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    print("=" * 70)
    print("Verifying Positional Encoding Formula")
    print("=" * 70)
    print()

    max_len = 100
    d_model = 8  # Small for easy verification

    # Implementation from the code
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    print(f"Testing with d_model={d_model}, max_len={max_len}")
    print()

    # Test 1: Position 0 should be [0, 1, 0, 1, 0, 1, ...]
    print("Test 1: Position 0")
    print("  Expected: sin(0)=0 for even indices, cos(0)=1 for odd indices")
    pos0 = pe[0]
    for i in range(d_model):
        expected = 0.0 if i % 2 == 0 else 1.0
        actual = pos0[i].item()
        match = "✓" if abs(expected - actual) < 1e-6 else "✗"
        print(f"  Index {i}: expected={expected:.6f}, actual={actual:.6f} {match}")

    # Test 2: Manual calculation for specific positions
    print("\nTest 2: Manual formula verification for position 5, index 0")
    pos = 5
    i = 0
    # For even index 2i=0, i=0
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    #             = sin(5 / 10000^(0/8))
    #             = sin(5 / 1)
    #             = sin(5)
    expected = math.sin(5.0)
    actual = pe[pos, i].item()
    match = "✓" if abs(expected - actual) < 1e-6 else "✗"
    print(f"  Expected: sin(5) = {expected:.6f}")
    print(f"  Actual: {actual:.6f} {match}")

    print("\nTest 3: Manual formula verification for position 5, index 1")
    pos = 5
    i = 1
    # For odd index 2i+1=1, i=0
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    #               = cos(5 / 10000^(0/8))
    #               = cos(5 / 1)
    #               = cos(5)
    expected = math.cos(5.0)
    actual = pe[pos, i].item()
    match = "✓" if abs(expected - actual) < 1e-6 else "✗"
    print(f"  Expected: cos(5) = {expected:.6f}")
    print(f"  Actual: {actual:.6f} {match}")

    print("\nTest 4: Manual formula verification for position 10, index 2")
    pos = 10
    idx = 2
    # For even index 2i=2, i=1
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    #             = sin(10 / 10000^(2*1/8))
    #             = sin(10 / 10000^(0.25))
    #             = sin(10 / 10^1)
    #             = sin(1.0)
    freq = 10000.0 ** (2 * 1 / d_model)
    expected = math.sin(10.0 / freq)
    actual = pe[pos, idx].item()
    match = "✓" if abs(expected - actual) < 1e-6 else "✗"
    print(f"  Expected: sin(10 / 10000^(0.25)) = {expected:.6f}")
    print(f"  Actual: {actual:.6f} {match}")

    print("\nTest 5: Manual formula verification for position 10, index 3")
    pos = 10
    idx = 3
    # For odd index 2i+1=3, i=1
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    #               = cos(10 / 10000^(2*1/8))
    #               = cos(10 / 10000^(0.25))
    freq = 10000.0 ** (2 * 1 / d_model)
    expected = math.cos(10.0 / freq)
    actual = pe[pos, idx].item()
    match = "✓" if abs(expected - actual) < 1e-6 else "✗"
    print(f"  Expected: cos(10 / 10000^(0.25)) = {expected:.6f}")
    print(f"  Actual: {actual:.6f} {match}")

    # Test 6: Verify div_term calculation
    print("\nTest 6: Verify div_term calculation")
    print("  div_term represents: 1 / 10000^(2i/d_model) for i in [0, d_model/2)")
    for idx, i in enumerate(range(0, d_model, 2)):
        expected_div = 1.0 / (10000.0 ** (i / d_model))
        actual_div = div_term[idx].item()
        match = "✓" if abs(expected_div - actual_div) < 1e-6 else "✗"
        print(f"  i={i//2}: 1/10000^({i}/{d_model}) = {expected_div:.6f}, actual={actual_div:.6f} {match}")

    # Test 7: Check PE values are in valid range
    print("\nTest 7: PE values in range [-1, 1]")
    min_val = pe.min().item()
    max_val = pe.max().item()
    in_range = -1.0 <= min_val and max_val <= 1.0
    match = "✓" if in_range else "✗"
    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f} {match}")

    # Test 8: Alternating sin/cos pattern
    print("\nTest 8: Alternating sin/cos pattern verification")
    print("  Checking that even columns use sin, odd columns use cos")
    # For a specific position, verify the relationship
    pos = 7
    all_correct = True
    for i in range(0, d_model, 2):
        freq = 10000.0 ** (i / d_model)
        sin_val = pe[pos, i].item()
        cos_val = pe[pos, i+1].item()
        expected_sin = math.sin(pos / freq)
        expected_cos = math.cos(pos / freq)

        sin_match = abs(sin_val - expected_sin) < 1e-6
        cos_match = abs(cos_val - expected_cos) < 1e-6

        if not (sin_match and cos_match):
            all_correct = False
            print(f"  Dimension {i} (sin): {sin_val:.6f} vs {expected_sin:.6f} {'✓' if sin_match else '✗'}")
            print(f"  Dimension {i+1} (cos): {cos_val:.6f} vs {expected_cos:.6f} {'✓' if cos_match else '✗'}")

    if all_correct:
        print(f"  All sin/cos pairs correct for position {pos} ✓")

    print("\n" + "=" * 70)
    print("Formula Verification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    verify_positional_encoding_formula()
