#!/usr/bin/env python3
"""
Shakespeare Training Script - Specialized for Tiny Shakespeare Dataset

Optimized training script for character-level Shakespeare text generation.
Includes generation callbacks during training to monitor progress.

Usage:
    # Basic training (uses shakespeare.yaml config)
    python tools/shakespeare_train.py

    # Custom parameters
    python tools/shakespeare_train.py --max-steps 10000 --batch-size 32

    # Resume training
    python tools/shakespeare_train.py --resume checkpoints/shakespeare/checkpoint_5000.pt

Features:
    - Automatic shakespeare.yaml config loading
    - Generation callbacks every N steps
    - Character distribution analysis
    - Progress tracking with sample outputs
    - Optimized for character-level modeling

Example Output:
    Step 1000 | Train Loss: 1.85 | Val Loss: 1.92
    Sample Generation:
    ----------------------------------------
    ROMEO:
    What say you to my lady's love, and not
    The hearts of men are made of stone.
    ----------------------------------------
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_transformer.model import TinyTransformerLM
from tiny_transformer.training import CharTokenizer, Trainer, TrainerConfig
from tiny_transformer.sampling import TextGenerator, GeneratorConfig
from tiny_transformer.utils import (
    TrainingLogger,
    CheckpointManager,
    ExperimentTracker,
    load_checkpoint,
)
from tools.train import (
    load_config,
    prepare_data,
    build_model,
)


class ShakespeareCallback:
    """Callback for generating Shakespeare samples during training."""

    def __init__(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        device: str,
        prompts: Optional[list] = None,
        temperature: float = 0.8,
        max_tokens: int = 150,
    ):
        """Initialize Shakespeare generation callback.

        Args:
            model: The model being trained
            tokenizer: Character tokenizer
            device: Device to use
            prompts: List of prompts to generate from
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Default Shakespeare prompts
        self.prompts = prompts or [
            "ROMEO:",
            "JULIET:",
            "First Citizen:",
            "KING LEAR:",
        ]

        # Generator config
        self.gen_config = GeneratorConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )

    def __call__(self, step: int, metrics: dict):
        """Generate samples at specified steps.

        Args:
            step: Current training step
            metrics: Current training metrics
        """
        print(f"\n{'='*70}")
        print(f"Step {step} - Sample Generation")
        print(f"Train Loss: {metrics.get('train_loss', 0.0):.3f} | "
              f"Val Loss: {metrics.get('val_loss', 0.0):.3f}")
        print(f"{'='*70}\n")

        # Generate from first prompt only (to save time)
        prompt = self.prompts[0]

        try:
            # Set to eval mode
            was_training = self.model.training
            self.model.eval()

            # Create generator
            generator = TextGenerator(self.model, self.gen_config, device=self.device)

            # Encode prompt
            prompt_tokens = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

            # Generate
            with torch.no_grad():
                output_tokens = generator.generate(prompt_tokens)

            # Decode
            generated = self.tokenizer.decode(output_tokens[0].tolist())

            # Display
            print(f"Prompt: {prompt}")
            print(f"{'-'*70}")
            print(generated)
            print(f"{'-'*70}\n")

            # Restore training mode
            if was_training:
                self.model.train()

        except Exception as e:
            print(f"Error during generation: {e}\n")


def analyze_dataset(text: str):
    """Analyze Shakespeare dataset statistics.

    Args:
        text: Full Shakespeare text
    """
    print("\n" + "="*70)
    print("Shakespeare Dataset Analysis")
    print("="*70 + "\n")

    # Basic stats
    print(f"Total characters: {len(text):,}")
    print(f"Total lines: {text.count(chr(10)):,}")

    # Character distribution
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    vocab_size = len(char_counts)
    print(f"Vocabulary size: {vocab_size}")

    # Most common characters
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 characters:")
    for char, count in sorted_chars[:10]:
        char_display = repr(char) if char in ['\n', '\t', ' '] else char
        pct = 100 * count / len(text)
        print(f"  {char_display:8s}: {count:8,} ({pct:5.2f}%)")

    # Character categories
    letters = sum(1 for c in text if c.isalpha())
    spaces = sum(1 for c in text if c.isspace())
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())

    print(f"\nCharacter categories:")
    print(f"  Letters:      {letters:8,} ({100*letters/len(text):5.2f}%)")
    print(f"  Whitespace:   {spaces:8,} ({100*spaces/len(text):5.2f}%)")
    print(f"  Punctuation:  {punct:8,} ({100*punct/len(text):5.2f}%)")
    print()


def main(args: argparse.Namespace):
    """Main Shakespeare training function."""

    # Load Shakespeare config
    config_path = args.config or "configs/shakespeare.yaml"
    config = load_config(config_path)

    # Override with CLI args
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate

    # Ensure using Shakespeare data
    data_path = args.data or "data/tiny_shakespeare.txt"
    config['data']['train_file'] = data_path

    # Set experiment name
    if args.experiment_name:
        config['experiment']['experiment_name'] = args.experiment_name
    else:
        config['experiment']['experiment_name'] = "shakespeare"

    print("="*70)
    print("Shakespeare Language Model Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: d_model={config['model']['d_model']}, "
          f"n_layers={config['model']['n_layers']}, "
          f"n_heads={config['model']['n_heads']}")
    print(f"  Training: lr={config['training']['learning_rate']}, "
          f"batch_size={config['training']['batch_size']}, "
          f"max_steps={config['training']['max_steps']}")
    print(f"  Data: {data_path}")
    print()

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}\n")

    # Load and analyze data
    with open(data_path, 'r') as f:
        text = f.read()

    if args.analyze:
        analyze_dataset(text)

    # Prepare data
    tokenizer = CharTokenizer()
    train_loader, val_loader = prepare_data(config, tokenizer)

    # Build model
    model = build_model(config, tokenizer.vocab_size, str(device))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters\n")

    # Setup logging and tracking
    exp_name = config['experiment']['experiment_name']

    logger = TrainingLogger(
        experiment_name=exp_name,
        log_dir=config['experiment'].get('log_dir', 'logs'),
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(config['paths']['checkpoint_dir']),
        keep_best_n=config['training'].get('keep_best_n', 5),
    )

    tracker = ExperimentTracker(
        project=config['experiment']['project'],
        experiment_name=exp_name,
        config=config,
        backend=config['experiment'].get('backend', 'auto'),
        log_dir=config['experiment'].get('log_dir', 'logs'),
    )

    # Create trainer
    from tiny_transformer.training import WarmupCosineScheduler

    trainer_config = TrainerConfig(
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        grad_clip=config['training'].get('grad_clip', 1.0),
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        eval_interval=config['training'].get('eval_interval', 500),
        log_interval=config['training'].get('log_interval', 100),
        device=str(device),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
    )

    # Setup scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=trainer.optimizer,
        warmup_steps=trainer_config.warmup_steps,
        total_steps=trainer_config.max_steps,
        peak_lr=trainer_config.learning_rate,
        min_lr=config['training'].get('min_lr', 1e-5),
    )
    trainer.scheduler = scheduler

    # Create generation callback
    if args.no_generation:
        generation_callback = None
    else:
        generation_callback = ShakespeareCallback(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            temperature=args.gen_temperature,
            max_tokens=args.gen_max_tokens,
        )

    # Resume if requested
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        metadata = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            device=str(device),
        )
        trainer.step = metadata['step']
        trainer.epoch = metadata['epoch']
        print(f"✓ Resumed from step {trainer.step}\n")

    # Training loop
    logger.log_start(config=config)

    print("="*70)
    print("Training Started")
    print("="*70 + "\n")

    try:
        # Custom training loop with generation callbacks
        while trainer.step < trainer_config.max_steps:
            # Train for eval_interval steps
            metrics = trainer.train_one_epoch()

            # Evaluate
            if val_loader:
                val_metrics = trainer.evaluate()
                metrics.update(val_metrics)

            # Log
            logger.log_metrics(metrics, step=trainer.step)
            tracker.log_metrics(metrics, step=trainer.step)

            # Generate samples
            if generation_callback and trainer.step % trainer_config.eval_interval == 0:
                generation_callback(trainer.step, metrics)

            # Checkpoint
            if trainer.step % config['training'].get('checkpoint_interval', 1000) == 0:
                checkpoint_path = checkpoint_manager.save(
                    model=model,
                    optimizer=trainer.optimizer,
                    scheduler=scheduler,
                    step=trainer.step,
                    epoch=trainer.epoch,
                    val_loss=metrics.get('val_loss'),
                    config=config,
                    tokenizer_vocab=tokenizer.vocab,
                )
                print(f"✓ Checkpoint saved: {checkpoint_path.name}\n")

        # Final checkpoint
        final_path = checkpoint_manager.save(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            step=trainer.step,
            epoch=trainer.epoch,
            val_loss=metrics.get('val_loss'),
            config=config,
            prefix="final",
            tokenizer_vocab=tokenizer.vocab,
        )

        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"\nFinal checkpoint: {final_path}")
        print(f"Logs: {logger.log_file}")
        print(f"\nGenerate text with:")
        print(f"  python tools/shakespeare_generate.py --checkpoint {final_path}")
        print()

        logger.log_end(final_metrics=metrics)
        tracker.finish()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        emergency_path = checkpoint_manager.checkpoint_dir / "checkpoint_interrupted.pt"
        checkpoint_manager.save(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            step=trainer.step,
            epoch=trainer.epoch,
            val_loss=None,
            config=config,
            prefix="interrupted",
            tokenizer_vocab=tokenizer.vocab,
        )
        print(f"✓ Emergency checkpoint saved: {emergency_path}\n")
        logger.log_end()
        tracker.finish()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Shakespeare language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="data/tiny_shakespeare.txt",
        help="Path to Shakespeare data file"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shakespeare.yaml",
        help="Path to config file"
    )

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    # Experiment
    parser.add_argument("--experiment-name", type=str, help="Experiment name")

    # Resume
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Generation during training
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable generation callbacks during training"
    )
    parser.add_argument(
        "--gen-temperature",
        type=float,
        default=0.8,
        help="Temperature for generation callbacks"
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=150,
        help="Max tokens for generation callbacks"
    )

    # Analysis
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show dataset analysis before training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
