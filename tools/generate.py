#!/usr/bin/env python3
"""
Text generation script for Tiny Transformer.

Load a trained checkpoint and generate text using various sampling strategies.

Usage:
    # Basic generation
    python tools/generate.py --checkpoint checkpoints/best.pt --prompt "Hello world"

    # Advanced sampling
    python tools/generate.py \\
        --checkpoint checkpoints/best.pt \\
        --prompt "To be or not to be" \\
        --temperature 0.8 \\
        --top-k 50 \\
        --top-p 0.95 \\
        --max-tokens 200

    # Generate from file with multiple prompts
    python tools/generate.py --checkpoint checkpoints/best.pt --prompts-file prompts.txt

Example:
    python tools/generate.py --checkpoint checkpoints/shakespeare/best.pt --prompt "ROMEO:"
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_transformer.model import TinyTransformerLM
from tiny_transformer.training import CharTokenizer
from tiny_transformer.sampling import TextGenerator, GeneratorConfig


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        (model, tokenizer, config) tuple
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Get vocab size from checkpoint
    # Try to extract from model state dict
    state_dict = checkpoint['model_state_dict']

    # Infer vocab_size from embedding layer
    if 'embedding.token_embedding.embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.token_embedding.embedding.weight'].shape[0]
    elif 'token_embedding.weight' in state_dict:
        vocab_size = state_dict['token_embedding.weight'].shape[0]
    else:
        raise ValueError("Cannot infer vocab_size from checkpoint")

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model: d_model={model_config.get('d_model', '?')}, "
          f"n_heads={model_config.get('n_heads', '?')}, "
          f"n_layers={model_config.get('n_layers', '?')}")

    # Create model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 512),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 2048),
        max_len=model_config.get('max_seq_len', 512),
        dropout=0.0,  # No dropout for inference
        tie_weights=model_config.get('tie_weights', True),
    )

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create tokenizer (need to reconstruct vocab)
    # For character tokenizer, we need the original text to fit
    # In a real system, you'd save the tokenizer with the checkpoint
    tokenizer = CharTokenizer()

    # Try to reconstruct vocab from embedding layer
    # This is a workaround - in production, save tokenizer separately
    print("\n⚠  Note: Reconstructing tokenizer from checkpoint")
    print("   For production use, save tokenizer separately")

    # Simple reconstruction: use ASCII printable characters
    # This works for Shakespeare and similar character-level models
    chars = [chr(i) for i in range(vocab_size)]
    tokenizer.vocab = {c: i for i, c in enumerate(chars)}
    tokenizer.reverse_vocab = {i: c for i, c in enumerate(chars)}

    return model, tokenizer, config


def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: str,
) -> str:
    """Generate text from prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        args: Generation arguments
        device: Device

    Returns:
        Generated text
    """
    # Create generator config
    gen_config = GeneratorConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.vocab.get('\n') if args.stop_at_newline else None,
    )

    # Create generator
    generator = TextGenerator(model, gen_config, device=device)

    # Encode prompt
    try:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)
    except (KeyError, ValueError) as e:
        print(f"Error encoding prompt: {e}")
        print("This prompt contains characters not in the model's vocabulary.")
        return ""

    # Generate
    with torch.no_grad():
        output_tokens = generator.generate(prompt_tokens)

    # Decode
    generated_text = tokenizer.decode(output_tokens[0].tolist())

    return generated_text


def main(args: argparse.Namespace):
    """Main generation function."""
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}\n")

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)

    print(f"\n{'='*70}")
    print("Text Generation")
    print(f"{'='*70}\n")

    # Get prompts
    if args.prompts_file:
        # Load prompts from file
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}\n")
    else:
        # Single prompt from command line
        prompts = [args.prompt]

    # Generate for each prompt
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}/{len(prompts)}: {prompt[:50]}...")

        generated = generate_text(model, tokenizer, prompt, args, device)

        if generated:
            results.append({
                'prompt': prompt,
                'generated': generated
            })

            # Print result
            print(f"\nGenerated ({len(generated)} chars):")
            print("-" * 70)
            print(generated)
            print("-" * 70 + "\n")
        else:
            print("  (generation failed)\n")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"PROMPT: {result['prompt']}\n")
                f.write(f"{'-'*70}\n")
                f.write(result['generated'])
                f.write(f"\n{'='*70}\n\n")

        print(f"✓ Saved {len(results)} generations to: {output_path}")

    print(f"\n{'='*70}")
    print(f"Generated {len(results)} samples")
    print(f"{'='*70}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with trained Tiny Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    # Prompts
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Input prompt for generation"
    )
    prompt_group.add_argument(
        "--prompts-file",
        type=str,
        help="File with prompts (one per line)"
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate"
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    gen_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (e.g., 50)"
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p/nucleus sampling (e.g., 0.95)"
    )
    gen_group.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy sampling (deterministic)"
    )
    gen_group.add_argument(
        "--stop-at-newline",
        action="store_true",
        help="Stop generation at newline character"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Save generated text to file"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for generation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
