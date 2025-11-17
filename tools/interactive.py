#!/usr/bin/env python3
"""
Interactive text generation REPL for Tiny Transformer.

Load a trained model and interactively generate text with adjustable parameters.

Usage:
    python tools/interactive.py --checkpoint checkpoints/shakespeare/best.pt

Commands:
    /help          - Show help message
    /temp <value>  - Set temperature (e.g., /temp 0.8)
    /topk <value>  - Set top-k (e.g., /topk 50)
    /topp <value>  - Set top-p (e.g., /topp 0.95)
    /len <value>   - Set max tokens (e.g., /len 200)
    /greedy        - Toggle greedy sampling
    /settings      - Show current settings
    /quit or /exit - Exit interactive mode

Example:
    $ python tools/interactive.py --checkpoint checkpoints/shakespeare/best.pt
    > ROMEO:
    [generated text appears]
    > /temp 1.2
    Temperature set to 1.2
    > JULIET:
    [more creative generation]
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


class InteractiveGenerator:
    """Interactive text generation session."""

    def __init__(self, checkpoint_path: str, device: str):
        """Initialize interactive generator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """
        self.device = device
        self.model, self.tokenizer = self._load_model(checkpoint_path)

        # Default generation settings
        self.temperature = 0.8
        self.top_k = None
        self.top_p = 0.95
        self.max_tokens = 200
        self.greedy = False

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        state_dict = checkpoint['model_state_dict']

        # Infer vocab size
        if 'embedding.token_embedding.embedding.weight' in state_dict:
            vocab_size = state_dict['embedding.token_embedding.embedding.weight'].shape[0]
        else:
            vocab_size = state_dict['token_embedding.weight'].shape[0]

        # Create model
        model = TinyTransformerLM(
            vocab_size=vocab_size,
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6),
            d_ff=model_config.get('d_ff', 2048),
            max_len=model_config.get('max_seq_len', 512),
            dropout=0.0,
            tie_weights=model_config.get('tie_weights', True),
        )

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        # Create tokenizer
        tokenizer = CharTokenizer()
        chars = [chr(i) for i in range(vocab_size)]
        tokenizer.vocab = {c: i for i, c in enumerate(chars)}
        tokenizer.reverse_vocab = {i: c for i, c in enumerate(chars)}

        print(f"✓ Loaded model (vocab_size={vocab_size}, "
              f"d_model={model_config.get('d_model')}, "
              f"params={sum(p.numel() for p in model.parameters()):,})")

        return model, tokenizer

    def generate(self, prompt: str) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Create config
        config = GeneratorConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=not self.greedy,
        )

        # Create generator
        generator = TextGenerator(self.model, config, device=self.device)

        # Encode prompt
        try:
            prompt_tokens = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        except (KeyError, ValueError):
            return "[Error: Prompt contains unknown characters]"

        # Generate
        with torch.no_grad():
            output_tokens = generator.generate(prompt_tokens)

        # Decode
        return self.tokenizer.decode(output_tokens[0].tolist())

    def show_settings(self):
        """Display current generation settings."""
        print("\nCurrent Settings:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-k: {self.top_k if self.top_k else 'disabled'}")
        print(f"  Top-p: {self.top_p if self.top_p else 'disabled'}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Greedy: {self.greedy}")
        print()

    def show_help(self):
        """Display help message."""
        print("\nInteractive Generation Commands:")
        print("  /help          - Show this help message")
        print("  /temp <value>  - Set temperature (e.g., /temp 0.8)")
        print("  /topk <value>  - Set top-k (e.g., /topk 50, /topk none)")
        print("  /topp <value>  - Set top-p (e.g., /topp 0.95, /topp none)")
        print("  /len <value>   - Set max tokens (e.g., /len 200)")
        print("  /greedy        - Toggle greedy sampling")
        print("  /settings      - Show current settings")
        print("  /quit, /exit   - Exit interactive mode")
        print("\nJust type your prompt to generate text!")
        print()

    def handle_command(self, command: str) -> bool:
        """Handle special commands.

        Args:
            command: Command string

        Returns:
            True to continue, False to exit
        """
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd in ['/quit', '/exit']:
            return False

        elif cmd == '/help':
            self.show_help()

        elif cmd == '/settings':
            self.show_settings()

        elif cmd == '/temp':
            if len(parts) > 1:
                try:
                    self.temperature = float(parts[1])
                    print(f"Temperature set to {self.temperature}")
                except ValueError:
                    print("Error: Temperature must be a number")
            else:
                print("Usage: /temp <value>")

        elif cmd == '/topk':
            if len(parts) > 1:
                if parts[1].lower() == 'none':
                    self.top_k = None
                    print("Top-k disabled")
                else:
                    try:
                        self.top_k = int(parts[1])
                        print(f"Top-k set to {self.top_k}")
                    except ValueError:
                        print("Error: Top-k must be an integer or 'none'")
            else:
                print("Usage: /topk <value|none>")

        elif cmd == '/topp':
            if len(parts) > 1:
                if parts[1].lower() == 'none':
                    self.top_p = None
                    print("Top-p disabled")
                else:
                    try:
                        self.top_p = float(parts[1])
                        print(f"Top-p set to {self.top_p}")
                    except ValueError:
                        print("Error: Top-p must be a number or 'none'")
            else:
                print("Usage: /topp <value|none>")

        elif cmd == '/len':
            if len(parts) > 1:
                try:
                    self.max_tokens = int(parts[1])
                    print(f"Max tokens set to {self.max_tokens}")
                except ValueError:
                    print("Error: Max tokens must be an integer")
            else:
                print("Usage: /len <value>")

        elif cmd == '/greedy':
            self.greedy = not self.greedy
            print(f"Greedy sampling: {'ON' if self.greedy else 'OFF'}")

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")

        return True

    def run(self):
        """Run interactive generation loop."""
        print("\n" + "="*70)
        print("Interactive Text Generation")
        print("="*70)
        print("\nType /help for commands, or just enter a prompt to generate text.")
        print("Press Ctrl+C or type /quit to exit.\n")

        self.show_settings()

        try:
            while True:
                # Get input
                try:
                    prompt = input("> ").strip()
                except EOFError:
                    print()
                    break

                if not prompt:
                    continue

                # Handle commands
                if prompt.startswith('/'):
                    if not self.handle_command(prompt):
                        break
                    continue

                # Generate text
                print("\nGenerating...\n")
                try:
                    generated = self.generate(prompt)
                    print(generated)
                    print()
                except Exception as e:
                    print(f"Error during generation: {e}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")

        print("\n✓ Session ended")


def main(args: argparse.Namespace):
    """Main function."""
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

    # Create and run interactive generator
    generator = InteractiveGenerator(args.checkpoint, device)
    generator.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive text generation with Tiny Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
