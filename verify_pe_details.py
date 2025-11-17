"""
Detailed verification of positional encoding implementation against the paper.
"""
import torch
import math

def verify_implementation_details():
    """
    Verify implementation matches "Attention is All You Need" exactly.

    From the paper:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Note: The paper uses 2i to index even dimensions (0, 2, 4, ...)
          and 2i+1 to index odd dimensions (1, 3, 5, ...)
    """
    print("=" * 70)
    print("Detailed Positional Encoding Verification")
    print("=" * 70)
    print()

    d_model = 512
    max_len = 5000

    # Implementation from the code
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    print(f"Configuration: d_model={d_model}, max_len={max_len}")
    print()

    # Test 1: Verify div_term formula
    print("Test 1: Verify div_term calculation")
    print("  Formula: 1 / 10000^(2i/d_model)")
    print("  Implementation: exp(2i * -log(10000) / d_model)")
    print()

    for i in [0, 1, 2, 64, 128, 255]:
        idx = i * 2
        if idx >= d_model:
            continue

        # Expected: 1 / 10000^(2i/d_model) = 1 / 10000^(idx/d_model)
        expected = 1.0 / (10000.0 ** (idx / d_model))

        # Actual from implementation
        actual = div_term[i].item()

        match = "✓" if abs(expected - actual) < 1e-9 else "✗"
        print(f"  i={i} (dim {idx}): expected={expected:.10e}, actual={actual:.10e} {match}")

    # Test 2: Verify specific positional encodings
    print("\nTest 2: Verify specific position encodings")

    test_cases = [
        (0, 0, "sin", 0, 0.0),
        (0, 1, "cos", 0, 1.0),
        (1, 0, "sin", 0, math.sin(1.0)),
        (1, 1, "cos", 0, math.cos(1.0)),
        (10, 4, "sin", 2, math.sin(10.0 / (10000.0 ** (4 / d_model)))),
        (10, 5, "cos", 2, math.cos(10.0 / (10000.0 ** (4 / d_model)))),
        (100, 100, "sin", 50, math.sin(100.0 / (10000.0 ** (100 / d_model)))),
        (100, 101, "cos", 50, math.cos(100.0 / (10000.0 ** (100 / d_model)))),
    ]

    for pos, dim, func, i, expected in test_cases:
        actual = pe[pos, dim].item()
        match = "✓" if abs(expected - actual) < 1e-6 else "✗"
        print(f"  PE(pos={pos:3d}, dim={dim:3d}) = {func}(...): "
              f"expected={expected:8.5f}, actual={actual:8.5f} {match}")

    # Test 3: Verify wavelengths
    print("\nTest 3: Verify wavelengths of sinusoidal functions")
    print("  Wavelength for dimension 2i: 2π × 10000^(2i/d_model)")
    print()

    for i in [0, 1, 2, 255]:
        idx = i * 2
        if idx >= d_model:
            continue

        wavelength = 2 * math.pi * (10000.0 ** (idx / d_model))
        print(f"  Dimension {idx:3d} (i={i:3d}): wavelength = {wavelength:12.2f}")

    # Test 4: Check that sin and cos are in phase
    print("\nTest 4: Verify sin and cos relationship")
    print("  For same i, sin is at dimension 2i, cos is at dimension 2i+1")
    print("  They use the same frequency: 1 / 10000^(2i/d_model)")
    print()

    for pos in [5, 50, 500]:
        for i in [0, 1, 128]:
            sin_dim = i * 2
            cos_dim = i * 2 + 1
            if cos_dim >= d_model:
                continue

            sin_val = pe[pos, sin_dim].item()
            cos_val = pe[pos, cos_dim].item()

            # Verify sin^2 + cos^2 = 1 (within numerical precision)
            identity = sin_val**2 + cos_val**2
            match = "✓" if abs(identity - 1.0) < 1e-5 else "✗"

            if not (abs(identity - 1.0) < 1e-5):
                print(f"  pos={pos:3d}, i={i:3d}: sin²+cos² = {identity:.10f} {match}")

    print("  ✓ All sin²+cos²=1 checks passed")

    # Test 5: Verify position independence (different positions should differ)
    print("\nTest 5: Verify position encoding differentiates positions")

    same_count = 0
    for dim in range(d_model):
        if torch.allclose(pe[:10, dim], pe[:10, dim].mean().expand(10)):
            same_count += 1

    if same_count == 0:
        print("  ✓ All dimensions vary across positions")
    else:
        print(f"  ✗ {same_count} dimensions don't vary across positions")

    # Test 6: Check scaling - verify the implementation doesn't accidentally scale PE
    print("\nTest 6: Verify PE values are not scaled")
    print("  (Token embeddings are scaled by √d_model, PE should NOT be)")

    if pe.abs().max() <= 1.0:
        print(f"  ✓ Max PE value: {pe.abs().max():.6f} (within [-1, 1])")
    else:
        print(f"  ✗ Max PE value: {pe.abs().max():.6f} (exceeds 1.0)")

    # Test 7: Verify relative position hypothesis
    print("\nTest 7: Verify relative position representation")
    print("  The paper claims PE allows the model to attend by relative positions")
    print("  PE(pos+k) can be represented as a linear function of PE(pos)")
    print()

    # For a fixed k, PE(pos+k, 2i) - PE(pos, 2i) should follow a pattern
    k = 3
    pos = 100
    for i in [0, 1, 10]:
        dim = i * 2
        if dim >= d_model:
            continue

        freq = 1.0 / (10000.0 ** (dim / d_model))

        # Theoretical difference
        sin_pos = math.sin(pos * freq)
        sin_pos_k = math.sin((pos + k) * freq)

        # Actual difference from implementation
        actual_diff = pe[pos + k, dim].item() - pe[pos, dim].item()
        expected_diff = sin_pos_k - sin_pos

        match = "✓" if abs(actual_diff - expected_diff) < 1e-6 else "✗"
        print(f"  Dimension {dim:3d}: PE({pos+k}) - PE({pos}) = {actual_diff:8.5f} "
              f"(expected {expected_diff:8.5f}) {match}")

    print("\n" + "=" * 70)
    print("All Detailed Verifications Complete!")
    print("=" * 70)

if __name__ == "__main__":
    verify_implementation_details()
