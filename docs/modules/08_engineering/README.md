# Module 08: Engineering & Production Practices

Professional engineering practices for transformer development - logging, experiment tracking, configuration management, and production deployment.

## What You'll Learn

1. **Logging Systems** - Structured logging for training runs
2. **Experiment Tracking** - Managing hyperparameters and results
3. **Configuration Management** - YAML configs and CLI arguments
4. **Checkpointing Best Practices** - Save/resume strategies
5. **Model Export** - ONNX, TorchScript for deployment
6. **Monitoring** - TensorBoard, Weights & Biases integration

## Prerequisites

- ✓ Completed Modules 01-07
- ✓ Have trained models
- ✓ Ready for production

## Engineering Stack

```
┌─────────────────────────────────────┐
│ Configuration (YAML + CLI)          │
│  - Model config                     │
│  - Training config                  │
│  - Data config                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Experiment Tracking                 │
│  - wandb / tensorboard              │
│  - Log hyperparameters              │
│  - Track metrics                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Training Loop with Logging          │
│  - Structured logs (JSON)           │
│  - Progress bars                    │
│  - Metric tracking                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Checkpointing                       │
│  - Periodic saves                   │
│  - Best model tracking              │
│  - Resume capability                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Model Export & Deployment           │
│  - ONNX export                      │
│  - TorchScript                      │
│  - Quantization                     │
└─────────────────────────────────────┘
```

## Key Concepts

### 1. Configuration Management

**Problem**: Hard-coded hyperparameters are hard to track and reproduce

**Solution**: Configuration files + CLI overrides

```python
# config.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 12
  d_ff: 2048
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 3e-4
  epochs: 10
  warmup_steps: 1000

# Load and override
config = load_config("config.yaml")
# CLI: --model.d_model 768 --training.batch_size 64
```

### 2. Experiment Tracking

**Track everything**:
- Hyperparameters
- Training metrics (loss, perplexity)
- Validation metrics
- Model checkpoints
- System info (Git commit, timestamp, hardware)

```python
import wandb

wandb.init(
    project="tiny-transformer",
    config={
        "d_model": config.model.d_model,
        "learning_rate": config.training.learning_rate,
        # ... all hyperparameters
    }
)

# During training
wandb.log({
    "train/loss": loss.item(),
    "train/perplexity": perplexity,
    "learning_rate": current_lr,
}, step=step)
```

### 3. Structured Logging

```python
import logging
import json

# JSON logging for easy parsing
logger = logging.getLogger(__name__)
logger.info(json.dumps({
    "step": step,
    "loss": loss.item(),
    "lr": scheduler.get_last_lr()[0],
    "tokens_per_sec": tokens_per_sec
}))

# Output: {"step": 1000, "loss": 3.42, "lr": 0.0003, "tokens_per_sec": 50000}
```

### 4. Checkpointing Strategy

**Best practices**:
- Save every N steps (e.g., 1000)
- Keep best N checkpoints by validation loss
- Save optimizer state for resuming
- Include full config in checkpoint

```python
def save_checkpoint(model, optimizer, scheduler, step, val_loss, path):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
    }
    torch.save(checkpoint, path)
```

### 5. Model Export

**ONNX**: For cross-framework compatibility
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['tokens'],
    output_names=['logits'],
    dynamic_axes={'tokens': {0: 'batch_size', 1: 'seq_len'}}
)
```

**TorchScript**: For PyTorch deployment
```python
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")
```

## What You'll Implement

```python
# 1. Configuration System
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def override_from_cli(self, args):
        # Apply CLI overrides
        pass

# 2. Experiment Tracker
class ExperimentTracker:
    def __init__(self, project_name, config, backend="wandb"):
        if backend == "wandb":
            wandb.init(project=project_name, config=config)
        # Also support tensorboard

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)

    def log_model(self, model, name):
        wandb.save(name)

# 3. Checkpoint Manager
class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_best_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_best_n = keep_best_n
        self.checkpoints = []  # (val_loss, path) sorted by loss

    def save(self, model, optimizer, scheduler, step, val_loss):
        path = f"{self.checkpoint_dir}/checkpoint_{step}.pt"
        save_checkpoint(model, optimizer, scheduler, step, val_loss, path)

        # Track and prune
        self.checkpoints.append((val_loss, path))
        self.checkpoints.sort()  # Best first
        if len(self.checkpoints) > self.keep_best_n:
            _, old_path = self.checkpoints.pop()
            os.remove(old_path)  # Delete worst checkpoint

    def load_best(self):
        if not self.checkpoints:
            return None
        best_loss, best_path = self.checkpoints[0]
        return torch.load(best_path)

# 4. Training Runner
class TrainingRunner:
    def __init__(self, config_path, cli_args=None):
        # Load config
        self.config = Config.from_yaml(config_path)
        if cli_args:
            self.config.override_from_cli(cli_args)

        # Setup experiment tracking
        self.tracker = ExperimentTracker(
            project_name="tiny-transformer",
            config=self.config
        )

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir
        )

        # Build model, optimizer, etc.
        self.model = build_model(self.config.model)
        self.optimizer = build_optimizer(self.model, self.config.training)

    def train(self):
        # Training loop with all integrations
        pass
```

## CLI Interface

```bash
# Basic training
python train.py --config configs/base.yaml

# Override hyperparameters
python train.py \\
    --config configs/base.yaml \\
    --model.d_model 768 \\
    --training.batch_size 64 \\
    --training.learning_rate 1e-4

# Resume from checkpoint
python train.py \\
    --config configs/base.yaml \\
    --resume checkpoints/checkpoint_10000.pt

# Evaluate only
python train.py \\
    --config configs/base.yaml \\
    --checkpoint checkpoints/best.pt \\
    --eval-only
```

## Module Contents

- `theory.md` - Engineering best practices
- `../../tools/train.py` - Complete training script with all features
- `../../configs/` - Example configuration files
- `../../tiny_transformer/utils/logging.py` - Logging utilities
- `../../tiny_transformer/utils/checkpoint.py` - Checkpoint management

## Production Checklist

- [ ] Configuration management with YAML + CLI
- [ ] Experiment tracking (Weights & Biases or TensorBoard)
- [ ] Structured logging (JSON format)
- [ ] Checkpoint management (save best N models)
- [ ] Resume training capability
- [ ] Git commit tracking in checkpoints
- [ ] Model export (ONNX/TorchScript)
- [ ] CLI interface
- [ ] Documentation and examples

## Next Module

**Module 09: Capstone Project** - End-to-end project training a transformer on Shakespeare and generating text.
