#!/usr/bin/env python3
"""
Shakespeare Demo - Minimal Example

Quick demonstration of the Tiny Transformer for Shakespeare text generation.
Perfect for README examples and quick testing.

Usage:
    # Train a small model (5 minutes)
    python examples/shakespeare_demo.py --train

    # Generate text from pre-trained model
    python examples/shakespeare_demo.py --generate --checkpoint checkpoints/shakespeare/best.pt

    # Interactive mode
    python examples/shakespeare_demo.py --interactive --checkpoint checkpoints/shakespeare/best.pt

Requirements:
    - data/tiny_shakespeare.txt (run: bash data/download_shakespeare.sh)
    - PyTorch installed
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_transformer.model import TinyTransformerLM
from tiny_transformer.training import (
    CharTokenizer,
    TextDataset,
    Trainer,
    TrainerConfig,
    WarmupCosineScheduler,
)
from tiny_transformer.sampling import TextGenerator, GeneratorConfig
from torch.utils.data import DataLoader


def quick_train(data_path: str = "data/tiny_shakespeare.txt", max_steps: int = 2000):
    """Quick training demo with tiny model.

    Args:
        data_path: Path to Shakespeare text
        max_steps: Number of training steps
    """
    print("\n" + "="*60)
    print("Shakespeare Demo - Quick Training")
    print("="*60 + "\n")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Load data
    print(f"Loading data: {data_path}")
    with open(data_path, 'r') as f:
        text = f.read()

    print(f"  Characters: {len(text):,}")

    # Tokenize
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    tokens = tokenizer.encode(text)

    print(f"  Vocabulary: {tokenizer.vocab_size} characters")

    # Split data
    val_size = len(tokens) // 10
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_len=128, stride=64)
    val_dataset = TextDataset(val_tokens, seq_len=128, stride=128)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")

    # Create tiny model (fast training)
    print("Creating model...")
    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,      # Small for demo
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_len=128,
        dropout=0.1,
        tie_weights=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}\n")

    # Create trainer
    config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,
        warmup_steps=200,
        max_steps=max_steps,
        eval_interval=500,
        log_interval=100,
        device=str(device),
    )

    trainer = Trainer(model, train_loader, val_loader, config)

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=trainer.optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        peak_lr=config.learning_rate,
        min_lr=1e-5,
    )
    trainer.scheduler = scheduler

    # Train
    print("="*60)
    print(f"Training for {max_steps} steps...")
    print("="*60 + "\n")

    try:
        trainer.train()

        # Save checkpoint
        checkpoint_dir = Path("checkpoints/demo")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "demo_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': trainer.step,
            'epoch': trainer.epoch,
            'config': {
                'model': {
                    'd_model': 256,
                    'n_heads': 4,
                    'n_layers': 4,
                    'd_ff': 1024,
                    'max_seq_len': 128,
                    'tie_weights': True,
                }
            },
        }, checkpoint_path)

        print(f"\n✓ Model saved to: {checkpoint_path}")
        print(f"\nGenerate text with:")
        print(f"  python examples/shakespeare_demo.py --generate --checkpoint {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")


def quick_generate(checkpoint_path: str, prompt: str = "ROMEO:", max_tokens: int = 150):
    """Quick generation demo.

    Args:
        checkpoint_path: Path to checkpoint
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    """
    print("\n" + "="*60)
    print("Shakespeare Demo - Quick Generation")
    print("="*60 + "\n")

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Loading: {checkpoint_path}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {}).get('model', {})

    # Get vocab size from state dict
    state_dict = checkpoint['model_state_dict']
    if 'embedding.token_embedding.embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.token_embedding.embedding.weight'].shape[0]
    else:
        vocab_size = state_dict['token_embedding.weight'].shape[0]

    # Create model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', 1024),
        max_len=config.get('max_seq_len', 128),
        dropout=0.0,
        tie_weights=config.get('tie_weights', True),
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create tokenizer from saved vocabulary
    tokenizer = CharTokenizer()
    if 'tokenizer_vocab' in checkpoint:
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.reverse_vocab = {i: c for c, i in tokenizer.vocab.items()}
    else:
        # Fallback for legacy checkpoints
        chars = [chr(i) for i in range(vocab_size)]
        tokenizer.vocab = {c: i for i, c in enumerate(chars)}
        tokenizer.reverse_vocab = {i: c for i, c in enumerate(chars)}

    # Generate
    print(f"Prompt: {prompt}")
    print("-"*60)

    gen_config = GeneratorConfig(
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
    )

    generator = TextGenerator(model, gen_config, device=device)

    try:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)

        with torch.no_grad():
            output_tokens = generator.generate(prompt_tokens)

        generated = tokenizer.decode(output_tokens[0].tolist())

        print(generated)
        print("-"*60)

    except Exception as e:
        print(f"Error: {e}")


def interactive_mode(checkpoint_path: str):
    """Interactive generation mode.

    Args:
        checkpoint_path: Path to checkpoint
    """
    print("\n" + "="*60)
    print("Shakespeare Demo - Interactive Mode")
    print("="*60 + "\n")

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Loading: {checkpoint_path}\n")

    # Load model (same as quick_generate)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {}).get('model', {})

    state_dict = checkpoint['model_state_dict']
    if 'embedding.token_embedding.embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.token_embedding.embedding.weight'].shape[0]
    else:
        vocab_size = state_dict['token_embedding.weight'].shape[0]

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', 1024),
        max_len=config.get('max_seq_len', 128),
        dropout=0.0,
        tie_weights=config.get('tie_weights', True),
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create tokenizer from saved vocabulary
    tokenizer = CharTokenizer()
    if 'tokenizer_vocab' in checkpoint:
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.reverse_vocab = {i: c for c, i in tokenizer.vocab.items()}
    else:
        # Fallback for legacy checkpoints
        chars = [chr(i) for i in range(vocab_size)]
        tokenizer.vocab = {c: i for i, c in enumerate(chars)}
        tokenizer.reverse_vocab = {i: c for i, c in enumerate(chars)}

    # Interactive loop
    temperature = 0.8
    max_tokens = 150

    print("Commands:")
    print("  /temp <value>  - Set temperature")
    print("  /len <value>   - Set max tokens")
    print("  /quit          - Exit")
    print("\nJust type a prompt to generate!\n")

    try:
        while True:
            prompt = input("> ").strip()

            if not prompt:
                continue

            if prompt in ['/quit', '/exit']:
                break

            if prompt.startswith('/temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: /temp <value>")
                continue

            if prompt.startswith('/len '):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except:
                    print("Usage: /len <value>")
                continue

            # Generate
            gen_config = GeneratorConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
            )

            generator = TextGenerator(model, gen_config, device=device)

            try:
                prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)

                with torch.no_grad():
                    output_tokens = generator.generate(prompt_tokens)

                generated = tokenizer.decode(output_tokens[0].tolist())

                print("\n" + "-"*60)
                print(generated)
                print("-"*60 + "\n")

            except Exception as e:
                print(f"Error: {e}\n")

    except KeyboardInterrupt:
        print("\n")

    print("✓ Session ended")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Shakespeare Transformer Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--train", action="store_true", help="Train a small model")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--data", type=str, default="data/tiny_shakespeare.txt",
                       help="Path to training data")
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")

    args = parser.parse_args()

    if args.train:
        quick_train(data_path=args.data, max_steps=args.steps)

    elif args.generate:
        if not args.checkpoint:
            print("Error: --checkpoint required for generation")
            sys.exit(1)
        quick_generate(checkpoint_path=args.checkpoint, prompt=args.prompt,
                      max_tokens=args.max_tokens)

    elif args.interactive:
        if not args.checkpoint:
            print("Error: --checkpoint required for interactive mode")
            sys.exit(1)
        interactive_mode(checkpoint_path=args.checkpoint)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
