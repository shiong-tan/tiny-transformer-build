#!/usr/bin/env python3
"""
Shakespeare Text Generation Script

Generate Shakespeare-style text using various sampling strategies.
Supports batch generation with different prompts and parameters.

Usage:
    # Basic generation
    python tools/shakespeare_generate.py --checkpoint checkpoints/shakespeare/best.pt

    # Custom prompt
    python tools/shakespeare_generate.py --checkpoint checkpoints/shakespeare/best.pt --prompt "HAMLET:"

    # Compare sampling strategies
    python tools/shakespeare_generate.py --checkpoint checkpoints/shakespeare/best.pt --compare

    # Batch generation with multiple prompts
    python tools/shakespeare_generate.py --checkpoint checkpoints/shakespeare/best.pt --batch

Features:
    - Multiple sampling strategies (greedy, temperature, top-k, top-p)
    - Comparison mode to evaluate different settings
    - Batch generation with character-specific prompts
    - Save outputs to file
    - Diversity analysis

Example:
    python tools/shakespeare_generate.py \
        --checkpoint checkpoints/shakespeare/best.pt \
        --prompt "ROMEO:" \
        --temperature 0.8 \
        --max-tokens 200 \
        --output romeo_speech.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_transformer.model import TinyTransformerLM
from tiny_transformer.training import CharTokenizer
from tiny_transformer.sampling import TextGenerator, GeneratorConfig


# Shakespeare character prompts
SHAKESPEARE_PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "HAMLET:",
    "KING LEAR:",
    "MACBETH:",
    "LADY MACBETH:",
    "PROSPERO:",
    "ARIEL:",
    "First Citizen:",
    "BRUTUS:",
]


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Infer vocab size
    state_dict = checkpoint['model_state_dict']
    if 'embedding.token_embedding.embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.token_embedding.embedding.weight'].shape[0]
    else:
        vocab_size = state_dict['token_embedding.weight'].shape[0]

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model: d_model={model_config.get('d_model')}, "
          f"n_layers={model_config.get('n_layers')}, "
          f"n_heads={model_config.get('n_heads')}")

    # Create model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 384),
        n_heads=model_config.get('n_heads', 6),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 1536),
        max_len=model_config.get('max_seq_len', 256),
        dropout=0.0,
        tie_weights=model_config.get('tie_weights', True),
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create tokenizer from saved vocabulary
    tokenizer = CharTokenizer()
    if 'tokenizer_vocab' in checkpoint:
        # Load saved vocabulary
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.reverse_vocab = {i: c for c, i in tokenizer.vocab.items()}
        print(f"  Loaded tokenizer with {len(tokenizer.vocab)} characters")
    else:
        # Fallback: reconstruct from ASCII (legacy checkpoints)
        print("  Warning: No tokenizer vocabulary in checkpoint, using ASCII fallback")
        chars = [chr(i) for i in range(vocab_size)]
        tokenizer.vocab = {c: i for i, c in enumerate(chars)}
        tokenizer.reverse_vocab = {i: c for i, c in enumerate(chars)}

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}\n")

    return model, tokenizer


def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str,
    temperature: float = 0.8,
    top_k: int = None,
    top_p: float = 0.95,
    max_tokens: int = 200,
    greedy: bool = False,
) -> str:
    """Generate text from prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        greedy: Use greedy sampling

    Returns:
        Generated text
    """
    gen_config = GeneratorConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=not greedy,
    )

    generator = TextGenerator(model, gen_config, device=device)

    try:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)
    except (KeyError, ValueError) as e:
        print(f"Error: Prompt contains unknown characters: {e}")
        return ""

    with torch.no_grad():
        output_tokens = generator.generate(prompt_tokens)

    return tokenizer.decode(output_tokens[0].tolist())


def compare_sampling_strategies(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 200,
):
    """Compare different sampling strategies.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device
        max_tokens: Maximum tokens
    """
    strategies = [
        ("Greedy (deterministic)", {"greedy": True}),
        ("Low temperature (0.5)", {"temperature": 0.5, "top_p": 0.9}),
        ("Medium temperature (0.8)", {"temperature": 0.8, "top_p": 0.95}),
        ("High temperature (1.2)", {"temperature": 1.2, "top_k": 50}),
        ("Top-k only (k=20)", {"temperature": 1.0, "top_k": 20}),
        ("Top-p only (p=0.9)", {"temperature": 1.0, "top_p": 0.9}),
    ]

    print("="*70)
    print(f"Comparing Sampling Strategies")
    print(f"Prompt: {prompt}")
    print("="*70 + "\n")

    for name, params in strategies:
        print(f"Strategy: {name}")
        print("-"*70)

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_tokens=max_tokens,
            **params
        )

        print(generated)
        print("-"*70 + "\n")


def batch_generation(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    device: str,
    prompts: List[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 200,
    output_file: str = None,
):
    """Generate text for multiple prompts.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device
        prompts: List of prompts
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        output_file: Optional output file
    """
    prompts = prompts or SHAKESPEARE_PROMPTS[:5]  # Use first 5 by default

    print("="*70)
    print(f"Batch Generation ({len(prompts)} prompts)")
    print("="*70 + "\n")

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating for: {prompt}")

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        results.append((prompt, generated))

        print("-"*70)
        print(generated)
        print("-"*70 + "\n")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Shakespeare Text Generation - Batch Output\n")
            f.write("="*70 + "\n\n")

            for prompt, generated in results:
                f.write(f"Prompt: {prompt}\n")
                f.write("-"*70 + "\n")
                f.write(generated + "\n")
                f.write("="*70 + "\n\n")

        print(f"✓ Saved {len(results)} generations to: {output_path}")


def analyze_diversity(texts: List[str]):
    """Analyze diversity of generated texts.

    Args:
        texts: List of generated texts
    """
    print("\n" + "="*70)
    print("Diversity Analysis")
    print("="*70 + "\n")

    # Character diversity
    all_chars = set()
    for text in texts:
        all_chars.update(set(text))

    print(f"Unique characters across all generations: {len(all_chars)}")

    # Word diversity (rough approximation for character-level)
    all_words = set()
    for text in texts:
        words = text.split()
        all_words.update(words)

    print(f"Unique words: {len(all_words)}")

    # Average length
    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"Average length: {avg_len:.1f} characters")

    # Repetition analysis
    for i, text in enumerate(texts, 1):
        # Simple repetition check: look for repeated n-grams
        repeated = 0
        n = 10  # Check 10-character sequences
        seen = set()

        for j in range(len(text) - n):
            ngram = text[j:j+n]
            if ngram in seen:
                repeated += 1
            seen.add(ngram)

        repetition_rate = repeated / max(len(text) - n, 1)
        print(f"Text {i} repetition rate: {repetition_rate:.2%}")


def main(args: argparse.Namespace):
    """Main generation function."""

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # Mode selection
    if args.compare:
        # Compare sampling strategies
        prompt = args.prompt or "ROMEO:"
        compare_sampling_strategies(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_tokens=args.max_tokens,
        )

    elif args.batch:
        # Batch generation
        batch_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=None,  # Uses default Shakespeare prompts
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_file=args.output,
        )

    else:
        # Single generation
        prompt = args.prompt or "ROMEO:"

        print("="*70)
        print("Shakespeare Text Generation")
        print("="*70)
        print(f"\nPrompt: {prompt}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-k: {args.top_k}")
        print(f"Top-p: {args.top_p}")
        print(f"Max tokens: {args.max_tokens}")
        print()

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            greedy=args.greedy,
        )

        print("-"*70)
        print(generated)
        print("-"*70)

        # Save if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(f"Prompt: {prompt}\n")
                f.write("-"*70 + "\n")
                f.write(generated + "\n")

            print(f"\n✓ Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Shakespeare-style text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:",
        help="Input prompt for generation"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p/nucleus sampling"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy sampling (deterministic)"
    )

    # Modes
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different sampling strategies"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate for multiple character prompts"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Save generated text to file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
