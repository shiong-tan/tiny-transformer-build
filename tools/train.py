#!/usr/bin/env python3
"""
Main training script for Tiny Transformer.

Supports YAML configuration files, CLI overrides, experiment tracking,
checkpointing, and resuming training.

Usage:
    # Basic training
    python tools/train.py --config configs/base.yaml

    # Override hyperparameters
    python tools/train.py --config configs/base.yaml --learning-rate 1e-4 --batch-size 64

    # Resume from checkpoint
    python tools/train.py --config configs/base.yaml --resume checkpoints/checkpoint_5000.pt

Example:
    python tools/train.py --config configs/shakespeare.yaml --data-train data/tiny_shakespeare.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_transformer.model import TinyTransformerLM
from tiny_transformer.training import (
    TextDataset,
    CharTokenizer,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    Trainer,
    TrainerConfig,
    set_seed,
    get_device,
)
from tiny_transformer.utils import (
    TrainingLogger,
    CheckpointManager,
    ExperimentTracker,
    load_checkpoint,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config with CLI arguments.

    Args:
        config: Configuration dictionary
        args: Parsed CLI arguments

    Returns:
        Updated configuration
    """
    # Model overrides
    if args.d_model is not None:
        config['model']['d_model'] = args.d_model
    if args.n_heads is not None:
        config['model']['n_heads'] = args.n_heads
    if args.n_layers is not None:
        config['model']['n_layers'] = args.n_layers

    # Training overrides
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.warmup_steps is not None:
        config['training']['warmup_steps'] = args.warmup_steps
    if args.grad_clip is not None:
        config['training']['grad_clip'] = args.grad_clip

    # Data overrides
    if args.data_train is not None:
        config['data']['train_file'] = args.data_train
    if args.data_val is not None:
        config['data']['val_file'] = args.data_val
    if args.seq_len is not None:
        config['data']['seq_len'] = args.seq_len

    # Experiment overrides
    if args.experiment_name is not None:
        config['experiment']['experiment_name'] = args.experiment_name
    if args.backend is not None:
        config['experiment']['backend'] = args.backend

    # Device override
    if args.device is not None:
        config['training']['device'] = args.device

    return config


def build_model(config: Dict[str, Any], vocab_size: int, device: str) -> nn.Module:
    """Build model from configuration.

    Args:
        config: Model configuration
        vocab_size: Vocabulary size
        device: Device to place model on

    Returns:
        Initialized model
    """
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model'].get('d_ff', config['model']['d_model'] * 4),
        max_len=config['model']['max_seq_len'],
        dropout=config['model'].get('dropout', 0.1),
        tie_weights=config['model'].get('tie_weights', True),
    )

    model = model.to(device)
    return model


def prepare_data(
    config: Dict[str, Any],
    tokenizer: CharTokenizer
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Prepare training and validation data loaders.

    Args:
        config: Data configuration
        tokenizer: Tokenizer instance

    Returns:
        (train_loader, val_loader) tuple
    """
    # Load training data
    train_file = config['data']['train_file']
    if train_file is None:
        raise ValueError("train_file must be specified in config or via --data-train")

    with open(train_file, 'r') as f:
        text = f.read()

    # Fit tokenizer
    tokenizer.fit(text)
    tokens = tokenizer.encode(text)

    print(f"✓ Loaded data from {train_file}")
    print(f"  Total characters: {len(text):,}")
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    seq_len = config['data']['seq_len']
    stride = config['data'].get('stride')

    # Split into train/val if no separate val file
    val_file = config['data'].get('val_file')
    if val_file is None:
        # Use 10% for validation
        val_size = len(tokens) // 10
        train_tokens = tokens[:-val_size]
        val_tokens = tokens[-val_size:]

        print(f"  Split: {len(train_tokens):,} train, {len(val_tokens):,} val")
    else:
        train_tokens = tokens
        with open(val_file, 'r') as f:
            val_text = f.read()
        val_tokens = tokenizer.encode(val_text)
        print(f"  Validation file: {val_file} ({len(val_tokens):,} tokens)")

    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_len=seq_len, stride=stride)
    val_dataset = TextDataset(val_tokens, seq_len=seq_len, stride=seq_len) if val_tokens else None

    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    print(f"✓ Created data loaders (batch_size={batch_size})")
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


def main(args: argparse.Namespace):
    """Main training function.

    Args:
        args: Parsed CLI arguments
    """
    # Load configuration
    config = load_config(args.config)
    config = override_config(config, args)

    # Set seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Get device
    device_str = config['training'].get('device', 'auto')
    if device_str == 'auto':
        device = get_device()
    else:
        device = torch.device(device_str)
    print(f"✓ Using device: {device}")

    # Create tokenizer and prepare data
    tokenizer = CharTokenizer()
    train_loader, val_loader = prepare_data(config, tokenizer)

    # Build model
    model = build_model(config, tokenizer.vocab_size, str(device))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created model with {num_params:,} parameters")

    # Create experiment name if not provided
    experiment_name = config['experiment'].get('experiment_name')
    if experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"
        config['experiment']['experiment_name'] = experiment_name

    # Set up experiment tracking
    tracker = ExperimentTracker(
        project=config['experiment']['project'],
        experiment_name=experiment_name,
        config=config,
        backend=config['experiment'].get('backend', 'auto'),
        log_dir=config['experiment'].get('log_dir', 'logs'),
        tags=config['experiment'].get('tags', []),
        notes=config['experiment'].get('notes', ''),
    )

    # Set up logging
    logger = TrainingLogger(
        experiment_name=experiment_name,
        log_dir=config['experiment'].get('log_dir', 'logs'),
    )
    logger.log_start(config=config)

    # Set up checkpointing
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_best_n=config['training'].get('keep_best_n', 3),
    )

    # Create trainer config
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

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
    )

    # Create scheduler
    scheduler_type = config['training'].get('scheduler', 'cosine')
    min_lr = config['training'].get('min_lr', 1e-5)

    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer=trainer.optimizer,
            warmup_steps=trainer_config.warmup_steps,
            total_steps=trainer_config.max_steps,
            peak_lr=trainer_config.learning_rate,
            min_lr=min_lr,
        )
    else:  # linear
        scheduler = WarmupLinearScheduler(
            optimizer=trainer.optimizer,
            warmup_steps=trainer_config.warmup_steps,
            total_steps=trainer_config.max_steps,
            peak_lr=trainer_config.learning_rate,
            min_lr=min_lr,
        )

    trainer.scheduler = scheduler

    # Resume from checkpoint if requested
    if args.resume:
        print(f"\n{'='*70}")
        print(f"Resuming from checkpoint: {args.resume}")
        print(f"{'='*70}\n")

        metadata = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            device=str(device),
        )

        trainer.step = metadata['step']
        trainer.epoch = metadata['epoch']

        print(f"✓ Resumed from step {trainer.step}, epoch {trainer.epoch}")

    # Training loop with integrated logging
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")

    try:
        # Run training
        final_metrics = trainer.train()

        # Save final checkpoint
        checkpoint_path = checkpoint_manager.save(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            step=trainer.step,
            epoch=trainer.epoch,
            val_loss=final_metrics.get('val_loss'),
            config=config,
        )

        print(f"\n✓ Saved final checkpoint: {checkpoint_path}")

        # Log to experiment tracker
        tracker.log_metrics(final_metrics, step=trainer.step)
        tracker.save_artifact(checkpoint_path, "final_checkpoint", "model")

        # Finish logging
        logger.log_end(final_metrics=final_metrics)
        tracker.finish()

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Final metrics: {final_metrics}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"Logs saved to: {logger.log_file}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

        # Save emergency checkpoint
        emergency_path = checkpoint_dir / "checkpoint_interrupted.pt"
        checkpoint_manager.save(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            step=trainer.step,
            epoch=trainer.epoch,
            val_loss=None,
            config=config,
            prefix="checkpoint_interrupted",
        )
        print(f"✓ Saved emergency checkpoint: {emergency_path}")

        logger.log_end()
        tracker.finish()

    except Exception as e:
        print(f"\n\nError during training: {e}")
        logger.log_error(e, step=trainer.step)
        tracker.finish()
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Tiny Transformer language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # Model architecture overrides
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--d-model", type=int, help="Model dimension")
    model_group.add_argument("--n-heads", type=int, help="Number of attention heads")
    model_group.add_argument("--n-layers", type=int, help="Number of transformer layers")

    # Training overrides
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--learning-rate", type=float, help="Learning rate")
    train_group.add_argument("--batch-size", type=int, help="Batch size")
    train_group.add_argument("--max-steps", type=int, help="Maximum training steps")
    train_group.add_argument("--warmup-steps", type=int, help="Warmup steps")
    train_group.add_argument("--grad-clip", type=float, help="Gradient clipping threshold")

    # Data overrides
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-train", type=str, help="Training data file")
    data_group.add_argument("--data-val", type=str, help="Validation data file")
    data_group.add_argument("--seq-len", type=int, help="Sequence length")

    # Experiment overrides
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument("--experiment-name", type=str, help="Experiment name")
    exp_group.add_argument("--backend", type=str, choices=["wandb", "tensorboard", "console"],
                          help="Experiment tracking backend")

    # Other
    parser.add_argument("--device", type=str, help="Device (cuda, mps, cpu, or auto)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
