# Module 08: Engineering & Production Practices

## Table of Contents

1. [Introduction](#introduction)
2. [Structured Logging](#structured-logging)
3. [Experiment Tracking](#experiment-tracking)
4. [Checkpoint Management](#checkpoint-management)
5. [Configuration Management](#configuration-management)
6. [Production Training Loops](#production-training-loops)
7. [Model Export & Deployment](#model-export--deployment)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Summary](#summary)

---

## Introduction

Training a transformer from scratch (Modules 01-07) teaches you the fundamentals: attention mechanisms, optimization, sampling strategies. But **shipping a transformer to production** requires an entirely different skill set.

**The gap between research code and production:**

```python
# Research/tutorial code (Module 06)
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item()}")

# Production code (Module 08)
with ExperimentTracker("project", config) as tracker:
    for step in train_loop():
        try:
            metrics = train_step(batch)
            tracker.log_metrics(metrics, step)

            if should_checkpoint(step):
                checkpoint_manager.save(model, optimizer, step, metrics)

            if should_evaluate(step):
                val_metrics = evaluate(model, val_loader)
                logger.log_metrics(val_metrics, step, prefix="val/")

        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            checkpoint_manager.save(model, optimizer, step, metrics)
            raise
```

The difference? **Robustness, observability, reproducibility.**

**What you'll learn:**

- Why JSON logging beats print statements for production
- How experiment tracking (Weights & Biases, TensorBoard) enables collaboration
- Checkpoint strategies: save everything, or just the best?
- Configuration management: YAML files vs. hard-coded hyperparameters
- Production training patterns that survive crashes, OOM errors, and interruptions
- Model export for deployment (ONNX, TorchScript)
- Monitoring techniques for catching gradient explosions, memory leaks, and training failures

**Key Insight Preview:**

Production ML engineering follows these principles:

```
1. Everything is logged (metrics, hyperparameters, system info)
2. Everything is versioned (code, data, configs, checkpoints)
3. Everything is reproducible (seeds, git commits, environment specs)
4. Everything can be resumed (checkpoints with full state)
5. Everything fails gracefully (error handling, cleanup, notifications)
```

These aren't optional niceties—they're **essential** for:
- **Debugging:** "Why did loss spike at step 5000?"
- **Collaboration:** "What hyperparameters did you use?"
- **Reproducibility:** "Can you replicate your result?"
- **Production:** "Can this run unsupervised for 3 days?"

**Prerequisites:**
- Completed Modules 01-07 (especially Module 06: Training)
- Understanding of software engineering best practices
- Familiarity with command-line tools
- Basic knowledge of experiment tracking concepts

---

## Structured Logging

### Why Logging Matters

Training a transformer for production involves:
- Hours or days of unattended execution
- Thousands of training steps
- Multiple experiments running in parallel
- Debugging failures that occur at step 10,000+ (after hours of training)

**Without proper logging:**

```python
# Bad: Print statements
print(f"Step 100: loss = 3.42")
print(f"Step 200: loss = 3.38")
print(f"Step 300: loss = 3.35")
# ... thousands more lines ...
print(f"Step 5000: loss = NaN")  # WHAT WENT WRONG?
```

Problems:
- No timestamps (when did step 5000 occur?)
- No context (what were the hyperparameters?)
- Can't parse programmatically (analysis requires manual reading)
- Lost when terminal closes
- No way to compare across experiments

### The JSON Logging Solution

**Structured logging** means logging in a machine-readable format (JSON) with consistent schema.

```python
# Good: Structured JSON logging
{
    "timestamp": "2024-11-17T14:32:15.123456",
    "level": "INFO",
    "message": "Training step completed",
    "step": 100,
    "metrics": {
        "loss": 3.42,
        "perplexity": 30.5,
        "learning_rate": 0.0003
    }
}
```

**Benefits:**
1. **Parseable:** Load all logs with `json.loads()` and analyze programmatically
2. **Queryable:** Filter logs by step, metric, timestamp
3. **Persistent:** Saved to file, survives crashes
4. **Structured:** Consistent schema across all experiments
5. **Timestamped:** Know exactly when events occurred

### JSON Logging Implementation

**Core Components:**

```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields (metrics, step, epoch, etc.)
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        if hasattr(record, "step"):
            log_data["step"] = record.step

        return json.dumps(log_data)
```

**How it works:**

1. Python's `logging` module creates `LogRecord` objects for each log call
2. Our custom `JSONFormatter` converts records to JSON strings
3. Handlers write JSON strings to console and/or file

**The data flow:**

```
logger.info(msg, extra={"metrics": {...}, "step": 100})
    ↓
LogRecord created with message and extra fields
    ↓
JSONFormatter.format(record)
    ↓
JSON string: {"timestamp": ..., "metrics": {...}, "step": 100}
    ↓
Handler writes to file (logs/experiment_20241117.jsonl)
```

### TrainingLogger: Production-Ready Interface

**Implementation** (from `tiny_transformer/utils/logging.py`):

```python
class TrainingLogger:
    """
    Structured logger for training runs.

    Features:
    - Dual output: human-readable console + JSON file
    - Automatic timestamping
    - Metric tracking with type conversion
    - System info logging (GPU, PyTorch version, etc.)
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        self.logger = logging.getLogger(f"training.{experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Console handler: human-readable
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler: JSON format
        if log_to_file:
            log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.jsonl"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ):
        """Log training/validation metrics."""
        # Add prefix (e.g., "train/" or "val/")
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Convert torch tensors to Python floats
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        # Format message
        msg = f"Step {step}" if step else "?"
        if epoch is not None:
            msg += f" | Epoch {epoch}"

        metric_strs = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ]
        msg += " | " + " | ".join(metric_strs)

        # Log with extra data for JSON formatting
        self.logger.info(
            msg,
            extra={"metrics": metrics, "step": step, "epoch": epoch}
        )
```

**Why two output formats?**

```
Console output (for humans):
2024-11-17 14:32:15 - INFO - Step 100 | Epoch 0 | train/loss=3.42 | train/ppl=30.5

JSON file output (for machines):
{"timestamp": "2024-11-17T14:32:15.123", "level": "INFO", "message": "...",
 "step": 100, "epoch": 0, "metrics": {"train/loss": 3.42, "train/ppl": 30.5}}
```

Humans read console during training. Machines parse JSON files for analysis.

### Complete Usage Example

```python
from tiny_transformer.utils import TrainingLogger

# Create logger
logger = TrainingLogger("shakespeare_experiment", log_dir="logs")

# Log experiment start
logger.log_start(config={
    "model": {"d_model": 512, "n_heads": 8},
    "training": {"lr": 1e-4, "batch_size": 32}
})

# Log system info (GPU, PyTorch version, etc.)
logger.log_system_info()

# Training loop
for step in range(10000):
    # ... train step ...

    # Log metrics
    metrics = {
        "loss": loss.item(),
        "perplexity": math.exp(loss.item()),
        "learning_rate": optimizer.param_groups[0]['lr'],
        "grad_norm": total_grad_norm,
    }
    logger.log_metrics(metrics, step=step, prefix="train/")

    # Validation
    if step % 500 == 0:
        val_metrics = evaluate(model, val_loader)
        logger.log_metrics(val_metrics, step=step, prefix="val/")

        # Log checkpoint save
        logger.log_checkpoint(
            checkpoint_path=f"checkpoint_{step}.pt",
            step=step,
            metrics=val_metrics
        )

# Log experiment end
logger.log_end(final_metrics={"best_val_loss": 2.1, "final_ppl": 8.2})
```

**Output structure:**

```
logs/
├── shakespeare_experiment_20241117_143215.jsonl
└── (each line is a JSON object)
```

### Log Analysis and Post-Processing

**Parsing JSON logs:**

```python
def parse_json_logs(log_file: str) -> list[Dict]:
    """Parse JSONL log file into list of dictionaries."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return logs

# Load logs
logs = parse_json_logs("logs/shakespeare_20241117_143215.jsonl")

# Extract training loss over time
train_steps = [log["step"] for log in logs if "step" in log]
train_losses = [
    log["metrics"]["train/loss"]
    for log in logs
    if "train/loss" in log.get("metrics", {})
]

# Plot
import matplotlib.pyplot as plt
plt.plot(train_steps, train_losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

**Comparing experiments:**

```python
def compare_experiments(log_files: list[str]):
    """Compare loss curves across multiple experiments."""
    plt.figure(figsize=(12, 6))

    for log_file in log_files:
        logs = parse_json_logs(log_file)
        steps = [log["step"] for log in logs if "step" in log]
        losses = [
            log["metrics"]["train/loss"]
            for log in logs
            if "train/loss" in log.get("metrics", {})
        ]

        # Extract experiment name from filename
        exp_name = Path(log_file).stem
        plt.plot(steps, losses, label=exp_name)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Experiment Comparison")
    plt.show()

# Compare three experiments
compare_experiments([
    "logs/exp1_lr1e3.jsonl",
    "logs/exp2_lr1e4.jsonl",
    "logs/exp3_lr1e5.jsonl",
])
```

### Best Practices for Logging

**1. Log at consistent intervals**

```python
# Good: Log every N steps
if step % log_interval == 0:
    logger.log_metrics(metrics, step=step)

# Bad: Log randomly
if random.random() < 0.1:  # ???
    logger.log_metrics(metrics, step=step)
```

**2. Use prefixes to namespace metrics**

```python
# Good: Clear namespacing
logger.log_metrics({"loss": 3.42}, step=100, prefix="train/")
logger.log_metrics({"loss": 3.20}, step=100, prefix="val/")

# Result:
# train/loss: 3.42
# val/loss: 3.20

# Bad: Ambiguous names
logger.log_metrics({"loss": 3.42}, step=100)  # train or val?
```

**3. Log hyperparameters at start**

```python
logger.log_hyperparams({
    "learning_rate": 1e-4,
    "batch_size": 32,
    "warmup_steps": 1000,
    "grad_clip": 1.0,
    "model_size": num_parameters,
})
```

**4. Log system info for debugging**

```python
logger.log_system_info()
# Logs: PyTorch version, CUDA availability, GPU name, OS, etc.
```

**5. Handle errors gracefully**

```python
try:
    metrics = train_step(batch)
    logger.log_metrics(metrics, step=step)
except Exception as e:
    logger.log_error(e, step=step)
    raise  # Re-raise after logging
```

### Log Aggregation and Analysis Tools

For large-scale production systems, consider log aggregation services:

**1. ELK Stack (Elasticsearch, Logstash, Kibana)**
- Aggregate logs from multiple machines
- Query with Elasticsearch syntax
- Visualize with Kibana dashboards

**2. Splunk**
- Enterprise log management
- Powerful search and analytics
- Real-time monitoring

**3. CloudWatch / Stackdriver**
- Cloud-native logging (AWS, GCP)
- Integration with cloud services
- Automatic retention and archival

**For most research use cases, JSON files + simple Python analysis suffice.**

---

## Experiment Tracking

### The Reproducibility Problem

**Scenario:** You train a model and achieve 85% accuracy.

**One month later:** Colleague asks, "Can you reproduce that 85% result?"

**You:** "Uh... I think I used learning_rate=1e-4? Or was it 1e-3? And batch_size=32, probably? Oh, and I modified the attention code but forgot what I changed..."

**Result:** Cannot reproduce. Experiment is lost forever.

### What is Experiment Tracking?

**Experiment tracking** systematically records:
1. **Hyperparameters:** All configuration values (LR, batch size, architecture, etc.)
2. **Metrics:** Training/validation loss, accuracy, perplexity over time
3. **Artifacts:** Model checkpoints, predictions, visualizations
4. **Environment:** Python version, PyTorch version, hardware, git commit
5. **Code:** Exact code state (via git commit hash)

**Goal:** Anyone (including future you) can **exactly reproduce** your results.

### Experiment Tracking Frameworks

**Three popular options:**

1. **Weights & Biases (wandb)**
   - Cloud-based (free for individuals, paid for teams)
   - Beautiful web UI with real-time charts
   - Automatic comparison across experiments
   - Model artifact versioning
   - Most popular in research

2. **TensorBoard**
   - Open-source, runs locally
   - Built into PyTorch
   - Great for visualizing training curves
   - Limited collaboration features
   - Good for solo development

3. **MLflow**
   - Open-source experiment tracking platform
   - Self-hostable
   - Supports multiple frameworks
   - More DevOps-focused

**Our implementation** provides a **unified interface** supporting wandb and TensorBoard, with automatic fallback to console logging if neither is available.

### Weights & Biases Integration

**Installation:**

```bash
pip install wandb
wandb login  # One-time setup
```

**Basic usage:**

```python
import wandb

# Initialize run
wandb.init(
    project="tiny-transformer",
    name="shakespeare_lr1e4",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "d_model": 512,
        "n_heads": 8,
    }
)

# Training loop
for step in range(max_steps):
    metrics = train_step(batch)

    # Log metrics
    wandb.log({
        "train/loss": metrics["loss"],
        "train/perplexity": math.exp(metrics["loss"]),
        "learning_rate": get_lr(optimizer),
    }, step=step)

# Finish run
wandb.finish()
```

**What wandb does automatically:**
- Uploads metrics to cloud in real-time
- Creates interactive plots (loss curves, histograms, etc.)
- Saves hyperparameters for comparison
- Records git commit hash
- Logs system info (GPU, CPU, memory usage)

**Web dashboard:**

```
https://wandb.ai/your-username/tiny-transformer/runs/abc123

Interactive UI showing:
├── Training curves (loss, perplexity, LR)
├── System metrics (GPU utilization, memory)
├── Hyperparameters (lr, batch_size, etc.)
├── Code version (git commit)
├── Logs and stdout
└── Artifacts (saved models)
```

### TensorBoard Integration

**Basic usage:**

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/shakespeare_experiment')

# Training loop
for step in range(max_steps):
    metrics = train_step(batch)

    # Log scalar metrics
    writer.add_scalar('train/loss', metrics['loss'], step)
    writer.add_scalar('train/ppl', math.exp(metrics['loss']), step)
    writer.add_scalar('learning_rate', get_lr(optimizer), step)

# Close writer
writer.close()
```

**Launch TensorBoard:**

```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```

**TensorBoard UI features:**
- Scalar plots (loss, accuracy over time)
- Histogram plots (weight distributions)
- Image/text logging
- Model graph visualization
- Embedding projections (t-SNE, PCA)

### ExperimentTracker: Unified Interface

**Our implementation** (from `tiny_transformer/utils/experiment.py`) provides a backend-agnostic interface:

```python
class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Supports multiple backends (wandb, tensorboard) with automatic
    fallback to console logging.
    """

    def __init__(
        self,
        project: str,
        experiment_name: str,
        config: Dict[str, Any],
        backend: str = "auto",  # "wandb", "tensorboard", or "console"
        log_dir: str = "runs",
    ):
        self.backend = backend
        if backend == "auto":
            backend = self._detect_backend()

        if backend == "wandb":
            self._init_wandb()
        elif backend == "tensorboard":
            self._init_tensorboard()
        else:
            self._init_console()

    def _detect_backend(self) -> str:
        """Auto-detect available backend."""
        try:
            import wandb
            return "wandb"
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
                return "tensorboard"
            except ImportError:
                return "console"

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics (backend-agnostic)."""
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)
        else:  # console
            print(f"Step {step} | {metrics}")

    def save_artifact(self, artifact_path: str, name: str):
        """Save model checkpoint or other artifact."""
        if self.backend == "wandb":
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        else:
            print(f"Artifact '{name}' saved at: {artifact_path}")
```

**Usage:**

```python
from tiny_transformer.utils import ExperimentTracker

# Automatic backend selection
tracker = ExperimentTracker(
    project="tiny-transformer",
    experiment_name="shakespeare_experiment",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "d_model": 512,
    },
    backend="auto"  # Uses wandb if installed, else tensorboard, else console
)

# Training loop (same code regardless of backend!)
for step in range(max_steps):
    metrics = train_step(batch)
    tracker.log_metrics(metrics, step=step)

# Save model checkpoint
tracker.save_artifact("checkpoint.pt", "best_model")

# Finish tracking
tracker.finish()
```

**Benefits:**
- ✓ Write once, run with any backend
- ✓ Easy to switch backends (change one config line)
- ✓ Graceful fallback if dependencies missing
- ✓ Same interface for local development and cloud deployment

### Complete Tracking Example

```python
from tiny_transformer.utils import ExperimentTracker

# Initialize tracking
with ExperimentTracker(
    project="tiny-transformer",
    experiment_name="shakespeare_gpt2_arch",
    config={
        "model": {"d_model": 512, "n_heads": 8, "n_layers": 6},
        "training": {"lr": 1e-4, "batch_size": 32, "grad_clip": 1.0},
        "data": {"dataset": "shakespeare", "seq_len": 256},
    },
    backend="wandb",
    tags=["gpt2", "shakespeare", "baseline"],
    notes="Testing GPT-2 architecture on Shakespeare dataset",
) as tracker:

    # Log model architecture
    tracker.log_model(model, name="shakespeare_gpt2")

    # Watch gradients and parameters (wandb only)
    tracker.watch_model(model, log_freq=100)

    # Training loop
    best_val_loss = float('inf')

    for step in range(max_steps):
        # Train step
        train_metrics = train_step(model, batch, optimizer)
        tracker.log_metrics(train_metrics, step=step)

        # Validation
        if step % eval_interval == 0:
            val_metrics = evaluate(model, val_loader)
            tracker.log_metrics(val_metrics, step=step)

            # Save best checkpoint
            if val_metrics['val/loss'] < best_val_loss:
                best_val_loss = val_metrics['val/loss']
                torch.save(model.state_dict(), 'best_model.pt')
                tracker.save_artifact('best_model.pt', 'best_checkpoint')

        # Generate sample text periodically
        if step % 1000 == 0:
            sample_text = generate(model, prompt="To be or not to be")
            tracker.log_text("generated_samples", sample_text, step)

    # Log final metrics
    tracker.log_metrics({
        "final/best_val_loss": best_val_loss,
        "final/total_steps": step,
    }, step=step)

# tracker.finish() called automatically via context manager
```

### Hyperparameter Tracking Best Practices

**1. Track everything that affects model behavior**

```python
config = {
    # Model architecture
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 512,
    },

    # Optimization
    "training": {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 32,
        "grad_clip": 1.0,
        "warmup_steps": 1000,
        "max_steps": 50000,
        "optimizer": "adamw",
        "lr_schedule": "cosine",
    },

    # Data
    "data": {
        "dataset": "shakespeare",
        "seq_len": 256,
        "vocab_size": 1000,
        "train_split": 0.9,
    },

    # Environment
    "environment": {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "num_gpus": torch.cuda.device_count(),
        "git_commit": get_git_commit(),  # Crucial!
    },
}

tracker = ExperimentTracker(..., config=config)
```

**2. Use descriptive experiment names**

```python
# Good: Descriptive names
experiment_name = "gpt2_shakespeare_lr1e4_bs32_6layers"

# Bad: Generic names
experiment_name = "experiment_1"
experiment_name = "test"
experiment_name = "final_final_v2_really_final"
```

**3. Add tags for organization**

```python
tracker = ExperimentTracker(
    project="tiny-transformer",
    experiment_name="...",
    tags=["baseline", "gpt2", "shakespeare", "ablation-study"],
    notes="Ablation study: effect of model depth on perplexity"
)
```

**4. Track git commit for reproducibility**

```python
import subprocess

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
    except:
        return None

config["environment"]["git_commit"] = get_git_commit()
```

**Why git commit matters:**

```
Experiment A: 85% accuracy
config: {lr: 1e-4, batch_size: 32, git_commit: "abc123"}

One month later, trying to reproduce:
$ git checkout abc123
$ python train.py --config exp_a_config.yaml
# EXACT code state restored → reproducible!
```

### Comparing Experiments in W&B

**Scenario:** You ran 5 experiments with different learning rates. Which is best?

**W&B parallel coordinates plot:**

```
Learning Rate    Val Loss    Final Perplexity
1e-3            3.2         24.5  (worst)
5e-4            2.8         16.4
3e-4            2.5         12.2  (best!)
1e-4            2.7         14.9
5e-5            3.0         20.1
```

**W&B automatically creates:**
- Parallel coordinate plots (hyperparams vs metrics)
- Loss curve overlays (all experiments on one chart)
- Scatter plots (any metric vs any hyperparameter)
- Tables sorted by best performance

**Code for comparing experiments:**

```python
import wandb

# Initialize API
api = wandb.Api()

# Get all runs from project
runs = api.runs("your-username/tiny-transformer")

# Extract hyperparameters and metrics
summary_list = []
for run in runs:
    summary_list.append({
        "name": run.name,
        "learning_rate": run.config["learning_rate"],
        "batch_size": run.config["batch_size"],
        "best_val_loss": run.summary.get("best_val_loss"),
        "final_ppl": run.summary.get("final/perplexity"),
    })

# Convert to DataFrame for analysis
import pandas as pd
df = pd.DataFrame(summary_list)
print(df.sort_values("best_val_loss"))
```

---

## Checkpoint Management

### Why Checkpointing Is Critical

**Training a large transformer:**
- Takes hours or days
- Costs money (GPU/cloud compute)
- Can crash at any time (OOM, hardware failure, power outage)

**Without checkpointing:**

```
Training for 8 hours...
Step 9500/10000
Step 9600/10000
Step 9700/10000
[CRASH - CUDA out of memory]

Result: 8 hours of training LOST FOREVER
Must restart from scratch
```

**With checkpointing:**

```
Training for 8 hours...
Step 5000: Saved checkpoint
Step 6000: Saved checkpoint
Step 7000: Saved checkpoint
[CRASH - CUDA out of memory]

Resume training from step 7000
Only lost 30 minutes of progress!
```

### What to Save in Checkpoints

**Minimal checkpoint:**

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')
```

**Problem:** Can't resume training! Missing optimizer state, scheduler state, training step, etc.

**Complete checkpoint for resuming training:**

```python
checkpoint = {
    # Model weights
    'model_state_dict': model.state_dict(),

    # Optimizer state (momentum, adaptive learning rates)
    'optimizer_state_dict': optimizer.state_dict(),

    # LR scheduler state
    'scheduler_state_dict': scheduler.state_dict(),

    # Training progress
    'step': current_step,
    'epoch': current_epoch,

    # Metrics
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,

    # Configuration (for rebuilding model)
    'config': model_config,

    # Versioning and reproducibility
    'timestamp': datetime.now().isoformat(),
    'git_commit': get_git_commit(),
    'pytorch_version': torch.__version__,

    # Optional: for analysis
    'learning_rate': optimizer.param_groups[0]['lr'],
    'grad_norm': last_grad_norm,
}

torch.save(checkpoint, f'checkpoint_{step}.pt')
```

**Why save optimizer state?**

AdamW maintains per-parameter moving averages (momentum and adaptive LR):

```python
# AdamW optimizer state for each parameter
optimizer_state = {
    'layer.weight': {
        'step': 5000,
        'exp_avg': tensor([...]),      # First moment (momentum)
        'exp_avg_sq': tensor([...]),   # Second moment (adaptive LR)
    },
    'layer.bias': { ... },
    # ... for all parameters
}
```

**Without optimizer state:**

```python
# Resume from checkpoint WITHOUT optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = AdamW(model.parameters())  # Fresh optimizer!

# Optimizer forgets all momentum
# Effective learning rate resets
# Training is disrupted!
```

**With optimizer state:**

```python
# Resume WITH optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Optimizer remembers momentum
# Adaptive learning rates preserved
# Training continues smoothly!
```

### Checkpoint Strategies

**Strategy 1: Save every N steps**

```python
save_interval = 1000

for step in range(max_steps):
    train_step(...)

    if step % save_interval == 0:
        save_checkpoint(
            f'checkpoints/checkpoint_{step}.pt',
            model, optimizer, scheduler, step
        )
```

**Pros:**
- ✓ Never lose more than N steps of progress
- ✓ Can resume from any checkpoint

**Cons:**
- ✗ Uses lots of disk space (each checkpoint ~100MB-1GB)
- ✗ May keep bad checkpoints

**Strategy 2: Save only best N checkpoints**

```python
checkpoint_manager = CheckpointManager(keep_best_n=3)

for step in range(max_steps):
    train_step(...)

    if step % eval_interval == 0:
        val_loss = evaluate(model, val_loader)

        # Automatically keeps only best 3 checkpoints
        checkpoint_manager.save(
            model, optimizer, scheduler,
            step=step, val_loss=val_loss
        )
```

**Pros:**
- ✓ Saves disk space (only keeps best models)
- ✓ Always have access to best model
- ✓ Automatic pruning of worse checkpoints

**Cons:**
- ✗ May delete checkpoints you wanted to keep
- ✗ Can't resume from arbitrary step

**Strategy 3: Hybrid (best N + periodic)**

```python
for step in range(max_steps):
    train_step(...)

    # Save periodically (every 5000 steps)
    if step % 5000 == 0:
        save_checkpoint(f'periodic_{step}.pt', ...)

    # Evaluate and save best
    if step % eval_interval == 0:
        val_loss = evaluate(model, val_loader)
        checkpoint_manager.save(..., val_loss=val_loss)
```

**Pros:**
- ✓ Best of both: can resume + keeps best models
- ✓ Periodic checkpoints for debugging

**Cons:**
- ✗ Uses more disk space

**Recommendation:** Use Strategy 3 (hybrid) for production.

### CheckpointManager Implementation

**Our implementation** (from `tiny_transformer/utils/checkpoint.py`):

```python
class CheckpointManager:
    """
    Manage model checkpoints with automatic pruning.

    Keeps track of the best N checkpoints by validation loss and
    automatically deletes worst checkpoints to save disk space.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        metric_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_n = keep_best_n
        self.metric_mode = metric_mode

        # Track checkpoints: List of (metric, path, step) tuples
        self.checkpoints = []

        # Discover existing checkpoints (for resuming)
        self._discover_existing_checkpoints()

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        step: int,
        val_loss: float,
        config: Optional[Dict] = None,
    ) -> Path:
        """
        Save checkpoint and manage best N checkpoints.

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint path
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"

        # Save checkpoint
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            val_loss=val_loss,
            config=config,
        )

        # Track this checkpoint
        self.checkpoints.append((val_loss, str(checkpoint_path), step))

        # Sort by metric (best first)
        reverse = (self.metric_mode == 'max')
        self.checkpoints.sort(reverse=reverse)

        # Prune worst checkpoints
        if len(self.checkpoints) > self.keep_best_n:
            self._prune_checkpoints()

        return checkpoint_path

    def _prune_checkpoints(self):
        """Remove worst checkpoints beyond keep_best_n."""
        while len(self.checkpoints) > self.keep_best_n:
            # Remove worst checkpoint (last in sorted list)
            _, worst_path, _ = self.checkpoints.pop()

            # Delete file
            try:
                os.remove(worst_path)
                print(f"Deleted checkpoint: {worst_path}")
            except FileNotFoundError:
                pass  # Already deleted

    def load_best(self, device='cpu') -> Optional[Dict]:
        """Load the best checkpoint."""
        if not self.checkpoints:
            return None

        # Best checkpoint is first in sorted list
        _, best_path, _ = self.checkpoints[0]
        return torch.load(best_path, map_location=device)

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.checkpoints:
            return None
        _, best_path, _ = self.checkpoints[0]
        return Path(best_path)
```

### Complete Checkpointing Example

```python
from tiny_transformer.utils import CheckpointManager

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints/shakespeare",
    keep_best_n=3,  # Keep top 3 checkpoints by val loss
    metric_mode='min'
)

# Training loop
best_val_loss = float('inf')

for step in range(max_steps):
    # Training
    train_metrics = train_step(model, batch, optimizer, scheduler)

    # Evaluation and checkpointing
    if step % eval_interval == 0:
        val_metrics = evaluate(model, val_loader)
        val_loss = val_metrics['val/loss']

        # Save checkpoint (automatically manages best N)
        saved_path = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            val_loss=val_loss,
            config=model_config,
        )

        print(f"Saved checkpoint: {saved_path}")
        print(f"Val loss: {val_loss:.4f}")

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best val loss: {best_val_loss:.4f}")

# After training, load best checkpoint
best_checkpoint = checkpoint_manager.load_best()
model.load_state_dict(best_checkpoint['model_state_dict'])

print(f"Loaded best checkpoint from step {best_checkpoint['step']}")
print(f"Best val loss: {best_checkpoint['val_loss']:.4f}")
```

**Checkpoint directory structure:**

```
checkpoints/shakespeare/
├── checkpoint_1000.pt  (val_loss=3.8)  ← Worst, will be deleted
├── checkpoint_2000.pt  (val_loss=3.5)  ← Middle
├── checkpoint_3000.pt  (val_loss=3.2)  ← Best
└── (older checkpoints deleted automatically)
```

### Resuming Training from Checkpoint

**Scenario:** Training crashed at step 7000. Resume from last checkpoint.

```python
from tiny_transformer.utils import load_checkpoint

# Build model and optimizer
model = TinyTransformerLM(...)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, ...)

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_7000.pt"
metadata = load_checkpoint(
    path=checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda'
)

# Extract metadata
start_step = metadata['step']
start_epoch = metadata['epoch']
best_val_loss = metadata.get('val_loss', float('inf'))

print(f"Resuming from step {start_step}, epoch {start_epoch}")
print(f"Previous val loss: {best_val_loss:.4f}")

# Continue training from start_step
for step in range(start_step, max_steps):
    train_step(...)
```

**Key point:** Start training loop from `start_step`, not from 0!

### Checkpoint Versioning with Git Commits

**Problem:** Model behavior changed, but checkpoints don't know which code version created them.

**Solution:** Save git commit in checkpoint.

```python
import subprocess

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit
    except:
        return None

# Save checkpoint with git commit
checkpoint = {
    'model_state_dict': model.state_dict(),
    'git_commit': get_git_commit(),
    # ... rest of checkpoint
}
```

**Benefit:** Know exact code state for any checkpoint.

```python
# Load checkpoint
checkpoint = torch.load('checkpoint_5000.pt')
git_commit = checkpoint.get('git_commit')

if git_commit:
    print(f"Checkpoint created at git commit: {git_commit}")
    print(f"To reproduce: git checkout {git_commit}")
```

### Emergency Checkpoints (Signal Handling)

**Scenario:** You hit Ctrl+C to stop training, but want to save current state before exiting.

**Solution:** Catch interrupt signal and save emergency checkpoint.

```python
import signal
import sys

# Global flag for graceful shutdown
should_stop = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global should_stop
    print("\nReceived interrupt signal. Saving checkpoint...")
    should_stop = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Training loop
for step in range(start_step, max_steps):
    if should_stop:
        # Save emergency checkpoint
        save_checkpoint(
            f'checkpoints/emergency_{step}.pt',
            model, optimizer, scheduler, step
        )
        print("Emergency checkpoint saved. Exiting.")
        sys.exit(0)

    train_step(...)
```

**Result:** Hit Ctrl+C → Saves checkpoint → Exits cleanly

**No data loss!**

### Disk Space Considerations

**Checkpoint size breakdown:**

```python
# For a 50M parameter model (float32)
model_state:       50M params × 4 bytes = 200 MB
optimizer_state:   2 × 200 MB = 400 MB  (momentum + adaptive LR)
scheduler_state:   ~1 KB
metadata:          ~1 KB
────────────────
Total:             ~600 MB per checkpoint
```

**With 10 checkpoints:** 6 GB disk space

**Mitigation strategies:**

1. **Keep fewer checkpoints** (keep_best_n=3 instead of 10)
2. **Save less frequently** (every 1000 steps instead of 100)
3. **Compress checkpoints** (not recommended, slows loading)
4. **Use cloud storage** (AWS S3, GCP Cloud Storage)
5. **Save model-only** for some checkpoints (no optimizer state)

```python
# Full checkpoint (for resuming training)
if step % 5000 == 0:
    save_checkpoint_full(...)

# Model-only checkpoint (for inference)
elif step % 1000 == 0:
    torch.save(model.state_dict(), f'model_only_{step}.pt')
    # Much smaller: 200MB instead of 600MB
```

---

## Configuration Management

### The Hard-Coded Hyperparameter Problem

**Bad practice: Hyperparameters scattered throughout code**

```python
# In model.py
model = TinyTransformerLM(
    vocab_size=1000,
    d_model=512,      # Hard-coded!
    n_heads=8,        # Hard-coded!
    n_layers=6,       # Hard-coded!
)

# In train.py
optimizer = AdamW(model.parameters(), lr=1e-4)  # Hard-coded!
batch_size = 32  # Hard-coded!

# In data.py
seq_len = 256  # Hard-coded!
```

**Problems:**
- ✗ Must edit code to change hyperparameters
- ✗ Can't compare experiments (code changes required)
- ✗ Can't version control experiments (git tracks code, not runs)
- ✗ Can't reproduce experiments (which values did you use?)
- ✗ Difficult to run sweeps (must manually edit and re-run)

### Configuration Files: The Solution

**Good practice: Single YAML/JSON file with all configuration**

```yaml
# configs/shakespeare_baseline.yaml

experiment:
  name: shakespeare_baseline
  project: tiny-transformer
  tags: [baseline, shakespeare]

model:
  vocab_size: 1000
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  grad_clip: 1.0
  max_steps: 50000
  warmup_steps: 5000
  eval_interval: 500
  log_interval: 100
  checkpoint_interval: 1000

data:
  train_file: data/shakespeare_train.txt
  val_file: data/shakespeare_val.txt
  seq_len: 256

paths:
  checkpoint_dir: checkpoints/shakespeare_baseline
  log_dir: logs/shakespeare_baseline
```

**Benefits:**
- ✓ All hyperparameters in one place
- ✓ Version control configs, not code
- ✓ Easy to create variants (copy and modify)
- ✓ Machine-readable (can parse programmatically)
- ✓ Human-readable (YAML is intuitive)

### Using Configs in Training Scripts

**Load and use config:**

```python
import yaml

# Load config
with open('configs/shakespeare_baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model from config
model = TinyTransformerLM(
    vocab_size=config['model']['vocab_size'],
    d_model=config['model']['d_model'],
    n_heads=config['model']['n_heads'],
    n_layers=config['model']['n_layers'],
    d_ff=config['model']['d_ff'],
    max_seq_len=config['model']['max_seq_len'],
    dropout=config['model']['dropout'],
)

# Build optimizer from config
optimizer = AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
)

# Get training params from config
batch_size = config['training']['batch_size']
max_steps = config['training']['max_steps']
```

**Creating experiments is now easy:**

```bash
# Experiment 1: Baseline
python train.py --config configs/shakespeare_baseline.yaml

# Experiment 2: Larger model
python train.py --config configs/shakespeare_large.yaml

# Experiment 3: Higher learning rate
python train.py --config configs/shakespeare_lr5e4.yaml
```

### CLI Overrides for Quick Experiments

**Problem:** Config files are great, but sometimes you want to quickly test one hyperparameter change without editing the file.

**Solution:** CLI arguments override config values.

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Config file (required)
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')

    # Override: model architecture
    parser.add_argument('--d-model', type=int,
                       help='Override model d_model')
    parser.add_argument('--n-heads', type=int,
                       help='Override number of attention heads')
    parser.add_argument('--n-layers', type=int,
                       help='Override number of transformer layers')

    # Override: training hyperparams
    parser.add_argument('--learning-rate', type=float,
                       help='Override learning rate')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--max-steps', type=int,
                       help='Override max training steps')

    # Override: data paths
    parser.add_argument('--data-train', type=str,
                       help='Override train data path')

    return parser.parse_args()

def override_config(config, args):
    """Apply CLI overrides to config."""
    if args.d_model is not None:
        config['model']['d_model'] = args.d_model
    if args.n_heads is not None:
        config['model']['n_heads'] = args.n_heads
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    # ... etc

    return config

# Main script
args = parse_args()
config = load_config(args.config)
config = override_config(config, args)
```

**Usage:**

```bash
# Use config as-is
python train.py --config configs/baseline.yaml

# Override learning rate
python train.py --config configs/baseline.yaml --learning-rate 5e-4

# Override multiple params
python train.py --config configs/baseline.yaml \
    --learning-rate 5e-4 \
    --batch-size 64 \
    --n-layers 8
```

**Benefits:**
- ✓ Quick experimentation without editing files
- ✓ Base config stays clean
- ✓ Easy to run parameter sweeps

**CLI override precedence:**

```
CLI arguments > Config file > Default values
```

### Type-Safe Configs with Dataclasses

**Problem:** Config dictionaries have no type checking. Typos cause runtime errors.

```python
# Typo in config file:
model:
  d_modl: 512  # Typo! Should be "d_model"

# Python loads it as dictionary
config['model']['d_modl']  # No error!
config['model']['d_model']  # KeyError at runtime!
```

**Solution:** Use Python dataclasses for type-safe configs.

**Our implementation** (from `tiny_transformer/config.py`):

```python
from dataclasses import dataclass, asdict
from typing import Literal, Optional

@dataclass
class ModelConfig:
    """Type-safe model configuration."""

    # Architecture
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Type-safe training configuration."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 10000
    warmup_steps: int = 500

    # Constrain optimizer to valid choices
    optimizer: Literal['adam', 'adamw'] = 'adamw'

    weight_decay: float = 0.01
    grad_clip: Optional[float] = 1.0

    # Constrain LR schedule to valid choices
    lr_schedule: Literal['constant', 'cosine', 'linear'] = 'cosine'

    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig
    training: TrainingConfig

    # Metadata
    name: str = "experiment"
    description: str = ""
    tags: list = field(default_factory=list)
```

**Usage:**

```python
# Create config
config = ExperimentConfig(
    model=ModelConfig(
        vocab_size=1000,
        d_model=512,
        n_heads=8,
        n_layers=6,
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        max_steps=50000,
    ),
    name="shakespeare_baseline"
)

# Save config
config.save('configs/experiment.yaml')

# Load config
loaded_config = ExperimentConfig.load('configs/experiment.yaml')

# Access with type safety and autocomplete!
print(config.model.d_model)  # IDE provides autocomplete
print(config.training.learning_rate)
print(config.model.d_k)  # Computed property
```

**Benefits:**
- ✓ Type checking at definition time
- ✓ IDE autocomplete
- ✓ Automatic validation (`__post_init__`)
- ✓ No typos (attribute errors caught immediately)
- ✓ Documentation via type hints

### Preset Configurations

**Create preset configs for common use cases:**

```python
def get_tiny_config() -> ExperimentConfig:
    """Ultra-small config for quick testing (~1 minute training)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=500,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=256,
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1e-3,
            max_steps=500,
        ),
        name="tiny_test"
    )

def get_small_config() -> ExperimentConfig:
    """Small config for learning (~10 minutes)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=3,
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=3e-4,
            max_steps=5000,
        ),
        name="small_model"
    )

def get_medium_config() -> ExperimentConfig:
    """Medium config for better results (~1 hour)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=5000,
            d_model=256,
            n_heads=8,
            n_layers=6,
        ),
        training=TrainingConfig(
            batch_size=64,
            learning_rate=1e-4,
            max_steps=20000,
        ),
        name="medium_model"
    )
```

**Usage:**

```python
# Quick smoke test
config = get_tiny_config()
model = build_model(config.model)
train(model, config.training)  # Runs in 1 minute

# Actual training
config = get_medium_config()
model = build_model(config.model)
train(model, config.training)  # Runs in 1 hour
```

### Environment-Specific Configs

**Scenario:** Development laptop vs. production cloud GPU server.

**configs/dev.yaml:**

```yaml
training:
  batch_size: 8  # Small for laptop
  max_steps: 1000  # Quick test
  device: cpu  # No GPU

data:
  train_file: data/tiny_shakespeare_sample.txt  # Small dataset
```

**configs/prod.yaml:**

```yaml
training:
  batch_size: 64  # Large for GPU
  max_steps: 50000  # Full training
  device: cuda  # Use GPU

data:
  train_file: data/shakespeare_full.txt  # Full dataset
```

**Usage:**

```bash
# Development
python train.py --config configs/dev.yaml

# Production
python train.py --config configs/prod.yaml
```

---

## Production Training Loops

### Anatomy of a Production Training Loop

**Research training loop (simplified):**

```python
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Production training loop (realistic):**

```python
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    logger,
    tracker,
    checkpoint_manager,
):
    """Production-grade training loop."""

    # Setup
    model.train()
    global_step = 0
    best_val_loss = float('inf')

    # Wrap in try-finally for cleanup
    try:
        for epoch in range(config.max_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids, target_ids = batch
                input_ids = input_ids.to(config.device)
                target_ids = target_ids.to(config.device)

                # Forward pass
                try:
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.log_error(e, step=global_step)
                        torch.cuda.empty_cache()
                        continue
                    raise

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip
                )

                # Check for gradient anomalies
                if not torch.isfinite(grad_norm):
                    logger.warning(f"Gradient norm is {grad_norm}, skipping step")
                    optimizer.zero_grad()
                    continue

                # Optimizer step
                optimizer.step()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % config.log_interval == 0:
                    metrics = {
                        "train/loss": loss.item(),
                        "train/perplexity": math.exp(loss.item()),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "grad_norm": grad_norm.item(),
                    }
                    logger.log_metrics(metrics, step=global_step, prefix="train/")
                    tracker.log_metrics(metrics, step=global_step)

                # Evaluation
                if global_step % config.eval_interval == 0:
                    val_metrics = evaluate(model, val_loader, config)
                    logger.log_metrics(val_metrics, step=global_step, prefix="val/")
                    tracker.log_metrics(val_metrics, step=global_step)

                    # Checkpointing
                    val_loss = val_metrics['val/loss']
                    checkpoint_manager.save(
                        model, optimizer, scheduler,
                        step=global_step,
                        val_loss=val_loss,
                        config=config,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"New best val loss: {best_val_loss:.4f}")

                # Early stopping
                if global_step >= config.max_steps:
                    logger.info("Reached max steps, stopping training")
                    return

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        checkpoint_manager.save(
            model, optimizer, scheduler,
            step=global_step,
            val_loss=float('inf'),  # Unknown val loss
            config=config,
        )
        logger.info("Saved emergency checkpoint")
        raise

    except Exception as e:
        logger.log_error(e, step=global_step)
        raise

    finally:
        # Cleanup
        logger.log_end(final_metrics={"best_val_loss": best_val_loss})
        tracker.finish()
```

**Key differences from research code:**

1. **Error handling:** Try-except blocks for OOM, NaN gradients, interrupts
2. **Device management:** Explicit `.to(device)`
3. **Gradient monitoring:** Check for anomalies before optimizer step
4. **Structured logging:** Log to files, not print
5. **Experiment tracking:** Send metrics to W&B/TensorBoard
6. **Checkpointing:** Regular saves with best-model tracking
7. **Cleanup:** Finally block ensures logs are flushed

### Error Handling Strategies

**1. CUDA Out of Memory (OOM)**

```python
try:
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.warning(f"OOM at step {global_step}, clearing cache")
        torch.cuda.empty_cache()

        # Option 1: Skip batch
        continue

        # Option 2: Reduce batch size dynamically
        # batch_size = batch_size // 2
        # continue

        # Option 3: Crash and let user fix
        # raise
    else:
        raise  # Other RuntimeError, re-raise
```

**2. NaN Loss**

```python
if not torch.isfinite(loss):
    logger.warning(f"NaN/Inf loss at step {global_step}, skipping batch")
    optimizer.zero_grad()  # Clear gradients
    continue  # Skip this batch

# Proceed with backward pass
loss.backward()
```

**3. Exploding Gradients**

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if grad_norm > 100.0:  # Threshold
    logger.warning(f"Large gradient norm: {grad_norm:.2f} at step {global_step}")

if not torch.isfinite(grad_norm):
    logger.warning(f"Gradient norm is {grad_norm}, skipping optimizer step")
    optimizer.zero_grad()
    continue

optimizer.step()
```

**4. Keyboard Interrupts (Ctrl+C)**

```python
try:
    # Training loop
    for step in range(max_steps):
        train_step(...)

except KeyboardInterrupt:
    logger.info("Training interrupted, saving checkpoint...")
    checkpoint_manager.save(model, optimizer, scheduler, step, ...)
    logger.info("Checkpoint saved, exiting gracefully")
    sys.exit(0)
```

### Gradient Accumulation for Large Effective Batch Sizes

**Problem:** Want batch_size=128, but GPU only fits batch_size=32.

**Solution:** Accumulate gradients over multiple forward-backward passes.

```python
accumulation_steps = 4  # Effective batch_size = 32 * 4 = 128

optimizer.zero_grad()

for batch_idx, batch in enumerate(dataloader):
    # Forward pass
    logits = model(input_ids)
    loss = F.cross_entropy(...)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps

    # Backward pass (accumulates gradients)
    loss.backward()

    # Optimizer step every N batches
    if (batch_idx + 1) % accumulation_steps == 0:
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
```

**Why scale loss?**

Without scaling:

```
Batch 1: loss=3.0 → gradients ≈ 3.0
Batch 2: loss=3.0 → gradients ≈ 6.0 (accumulated)
Batch 3: loss=3.0 → gradients ≈ 9.0 (accumulated)
Batch 4: loss=3.0 → gradients ≈ 12.0 (accumulated)
optimizer.step()  # Gradients are 4x too large!
```

With scaling (divide by 4):

```
Batch 1: loss=0.75 → gradients ≈ 0.75
Batch 2: loss=0.75 → gradients ≈ 1.5 (accumulated)
Batch 3: loss=0.75 → gradients ≈ 2.25 (accumulated)
Batch 4: loss=0.75 → gradients ≈ 3.0 (accumulated)
optimizer.step()  # Gradients are correct!
```

### Mixed Precision Training (FP16)

**Why mixed precision?**

- Faster training (2-3× speedup on modern GPUs)
- Lower memory usage (can fit larger batches)
- Minimal accuracy impact (FP16 for forward/backward, FP32 for master weights)

**Implementation with PyTorch AMP:**

```python
from torch.cuda.amp import autocast, GradScaler

# Create gradient scaler
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16 (autocast)
    with autocast():
        logits = model(input_ids)
        loss = F.cross_entropy(...)

    # Backward pass (scales loss to prevent underflow)
    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step (updates FP32 master weights)
    scaler.step(optimizer)
    scaler.update()
```

**How it works:**

```
Model weights (FP32) → Copy to FP16 → Forward pass (FP16)
                                       ↓
                                   Loss (FP16)
                                       ↓
                                   Backward (FP16)
                                       ↓
Gradients (FP16) → Scale up (prevent underflow)
                                       ↓
                                   Unscale & clip
                                       ↓
Update master weights (FP32)
```

**Key points:**
- Master weights stay in FP32 (prevents rounding errors during optimization)
- Forward/backward computed in FP16 (speed + memory savings)
- Gradient scaling prevents underflow (FP16 range is smaller)

### Distributed Training (Multi-GPU)

**For very large models or datasets, train across multiple GPUs:**

**DataParallel (simple, but slow):**

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to('cuda')

# Training loop unchanged
for batch in dataloader:
    loss = model(input_ids)  # Automatically split across GPUs
    loss.backward()
    optimizer.step()
```

**DistributedDataParallel (faster, recommended):**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Create model and wrap with DDP
model = TinyTransformerLM(...)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)

# Training loop
for epoch in range(epochs):
    train_sampler.set_epoch(epoch)  # Important!

    for batch in train_loader:
        # Training step (gradients synchronized automatically)
        loss = model(input_ids)
        loss.backward()
        optimizer.step()
```

**Launch with torchrun:**

```bash
torchrun --nproc_per_node=4 train.py --config configs/base.yaml
```

---

## Model Export & Deployment

### From Training to Production

**Training:** Model runs in Python with PyTorch

**Production:** Model may need to run:
- On mobile devices (iOS, Android)
- In web browsers (JavaScript)
- In C++ servers (low-latency inference)
- Without Python dependency

**Export formats:**

1. **PyTorch Script (TorchScript):** Optimized PyTorch model, runs in C++
2. **ONNX:** Cross-framework format, runs in many inference engines
3. **TensorFlow SavedModel:** Convert to TensorFlow for serving
4. **Quantized models:** Reduce size and increase speed (INT8 instead of FP32)

### Saving PyTorch Models

**Three ways to save models:**

**Method 1: State dict (recommended)**

```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model = TinyTransformerLM(vocab_size=1000, d_model=512, ...)
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

**Pros:** Small file size, flexible (can change model code)
**Cons:** Must recreate model architecture in code

**Method 2: Entire model**

```python
# Save
torch.save(model, 'model.pt')

# Load
model = torch.load('model.pt')
model.eval()
```

**Pros:** Self-contained (includes architecture)
**Cons:** Larger file, less flexible

**Method 3: JIT scripting (see next section)**

### TorchScript Export

**TorchScript:** Optimized representation of PyTorch model, runs without Python.

**Two modes:**

1. **Tracing:** Run model once, record operations

```python
import torch.jit

# Load trained model
model = TinyTransformerLM(...)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Create example input
example_input = torch.randint(0, 1000, (1, 256))  # (batch, seq_len)

# Trace model
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save('model_traced.pt')
```

**Loading traced model:**

```python
# In Python
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(input_ids)

# In C++
# #include <torch/script.h>
# torch::jit::script::Module model = torch::jit::load("model_traced.pt");
# auto output = model.forward({input_tensor}).toTensor();
```

2. **Scripting:** Compile Python code to TorchScript

```python
# Annotate model with type hints
class TinyTransformerLM(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Type annotations required for scripting
        ...

# Script model
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

**Tracing vs Scripting:**

| Feature | Tracing | Scripting |
|---------|---------|-----------|
| Ease of use | Easy | Requires type annotations |
| Control flow | Limited (no conditionals) | Full support |
| Performance | Fast | Fast |
| Best for | Simple models | Models with if/else, loops |

### ONNX Export

**ONNX (Open Neural Network Exchange):** Universal model format supported by many frameworks.

**Export to ONNX:**

```python
import torch.onnx

model = TinyTransformerLM(...)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Example input
dummy_input = torch.randint(0, 1000, (1, 256))

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    },
    opset_version=14
)

print("Exported to ONNX format")
```

**Running ONNX model:**

```python
import onnxruntime as ort

# Create inference session
session = ort.InferenceSession('model.onnx')

# Run inference
outputs = session.run(
    ['logits'],
    {'input_ids': input_ids.numpy()}
)

logits = outputs[0]
```

**Benefits of ONNX:**
- ✓ Run on ONNX Runtime (optimized inference engine)
- ✓ Deploy to edge devices (ONNX Runtime Mobile)
- ✓ Convert to other frameworks (TensorFlow, TensorRT)
- ✓ Optimize with graph transformations

### Model Quantization

**Quantization:** Convert FP32 weights to INT8 (8-bit integers).

**Benefits:**
- 4× smaller model size (32 bits → 8 bits)
- 2-4× faster inference (INT8 ops are faster)
- Minimal accuracy loss (<1% typically)

**Types of quantization:**

1. **Dynamic quantization** (easiest)

```python
import torch.quantization

# Load model
model = TinyTransformerLM(...)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Quantize (dynamic)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized.pt')

# Inference (same API)
output = quantized_model(input_ids)
```

2. **Static quantization** (more accurate)

Requires calibration data to compute activation ranges.

```python
# Prepare model for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate (run on representative data)
for batch in calibration_loader:
    model(batch)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
```

3. **Quantization-aware training** (best accuracy)

Train with fake quantization to simulate INT8 behavior.

```python
# Add fake quantization during training
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model, inplace=False)

# Train normally (fake quantization applied)
train(model_prepared, ...)

# Convert to real quantized model
model_quantized = torch.quantization.convert(model_prepared, inplace=False)
```

**Quantization trade-offs:**

| Method | Accuracy | Speed | Effort |
|--------|----------|-------|--------|
| Dynamic | Good | Fast | Low (1 line) |
| Static | Better | Faster | Medium (calibration) |
| QAT | Best | Fastest | High (re-train) |

### Serving Models in Production

**Option 1: Flask API (simple)**

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load model once at startup
model = torch.jit.load('model_traced.pt')
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data['prompt']

    # Tokenize
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids])

    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100)

    # Decode
    output_text = tokenizer.decode(output_ids[0])

    return jsonify({'generated_text': output_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Option 2: TorchServe (production)**

TorchServe is PyTorch's official model serving framework.

```bash
# Install TorchServe
pip install torchserve torch-model-archiver

# Archive model
torch-model-archiver \
    --model-name tiny_transformer \
    --version 1.0 \
    --serialized-file model.pt \
    --handler text_handler.py \
    --export-path model_store/

# Start server
torchserve \
    --start \
    --model-store model_store \
    --models tiny_transformer=tiny_transformer.mar
```

**Option 3: ONNX Runtime (fast inference)**

```python
import onnxruntime as ort

# Create session (optimized for inference)
session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Inference
def predict(input_ids):
    outputs = session.run(
        ['logits'],
        {'input_ids': input_ids}
    )
    return outputs[0]
```

**Option 4: TensorRT (NVIDIA GPUs, fastest)**

Convert ONNX model to TensorRT for maximum speed.

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt
```

```python
import tensorrt as trt
import pycuda.driver as cuda

# Load TensorRT engine
with open('model.trt', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Inference (fastest option for NVIDIA GPUs)
context.execute_v2(bindings)
```

---

## Monitoring & Debugging

### Essential Metrics to Track

**1. Training Loss**

```python
# Track raw loss and smoothed loss
ema_loss = None
alpha = 0.9

for step in range(max_steps):
    loss = train_step(...)

    # Exponential moving average
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = alpha * ema_loss + (1 - alpha) * loss

    logger.log_metrics({
        "train/loss": loss,
        "train/loss_smoothed": ema_loss,
    }, step=step)
```

**Why track both?**
- Raw loss: Shows actual training dynamics (noisy)
- Smoothed loss: Shows overall trend (easier to read)

**2. Perplexity**

```python
perplexity = math.exp(loss)
logger.log_metrics({"train/perplexity": perplexity}, step=step)
```

**Interpretation:**
- Perplexity = 10 → Model is confused between ~10 tokens on average
- Perplexity = 5 → Model is more confident (good!)
- Lower is better

**3. Learning Rate**

```python
lr = optimizer.param_groups[0]['lr']
logger.log_metrics({"learning_rate": lr}, step=step)
```

**Why track?**
- Verify scheduler is working correctly
- Debug training issues (loss not decreasing → check if LR too low)

**4. Gradient Norm**

```python
# Before clipping
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5

logger.log_metrics({"grad_norm": total_norm}, step=step)

# Apply clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why track?**
- Detect exploding gradients (norm > 10 is suspicious)
- Verify gradient clipping is working
- Debug vanishing gradients (norm < 0.001)

**5. GPU Memory Usage**

```python
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
    memory_reserved = torch.cuda.memory_reserved() / 1e9    # GB

    logger.log_metrics({
        "gpu/memory_allocated_gb": memory_allocated,
        "gpu/memory_reserved_gb": memory_reserved,
    }, step=step)
```

**Why track?**
- Detect memory leaks (memory grows over time)
- Optimize batch size (maximize GPU utilization)

**6. Training Speed (tokens/sec)**

```python
import time

start_time = time.time()
tokens_processed = 0

for step, batch in enumerate(dataloader):
    # Training step
    train_step(batch)
    tokens_processed += batch.numel()

    # Log speed
    if step % log_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed

        logger.log_metrics({
            "training/tokens_per_sec": tokens_per_sec,
        }, step=step)
```

**Why track?**
- Detect slowdowns (data loading bottleneck?)
- Compare training efficiency across experiments

### Debugging Training Failures

**Symptom 1: Loss Not Decreasing**

**Possible causes and solutions:**

1. **Learning rate too low**

```python
# Check current LR
lr = optimizer.param_groups[0]['lr']
print(f"Current LR: {lr}")

# If LR < 1e-6, probably too low
# Solution: Increase peak LR in config
```

2. **Learning rate too high** (loss oscillates)

```python
# Check if loss oscillates wildly
# If loss goes: 3.5 → 4.2 → 3.8 → 5.1 → ...
# Solution: Reduce peak LR
```

3. **Gradient clipping too aggressive**

```python
# Check gradient norm before clipping
if grad_norm < 0.1:
    print("Gradients are very small!")
    # Solution: Reduce grad_clip or increase LR
```

4. **Model architecture issue**

```python
# Check if d_model is divisible by n_heads
assert config.d_model % config.n_heads == 0

# Check layer norm is applied correctly
# Check residual connections exist
```

**Symptom 2: Loss Becomes NaN**

**Possible causes:**

1. **Exploding gradients**

```python
# Monitor gradient norm
if grad_norm > 1000:
    print(f"WARNING: Large gradient norm: {grad_norm}")
    # Solution: Reduce LR or increase grad_clip
```

2. **Learning rate too high**

```python
# Try reducing peak LR by 10x
learning_rate = 1e-5  # Was 1e-4
```

3. **Numerical instability in attention**

```python
# Check attention scores
scores = Q @ K.T / math.sqrt(d_k)

if torch.any(torch.isnan(scores)):
    print("NaN in attention scores!")
    # Check for inf values in Q or K
    # Solution: Reduce model initialization scale
```

**Symptom 3: Memory Leaks**

**Detection:**

```python
import tracemalloc

tracemalloc.start()

for step in range(100):
    train_step(...)

    if step % 10 == 0:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Step {step}: Current={current/1e6}MB, Peak={peak/1e6}MB")
```

**Common causes:**

1. **Not detaching tensors from graph**

```python
# Bad: Keeps computation graph in memory
losses.append(loss)

# Good: Detach from graph
losses.append(loss.item())  # or loss.detach()
```

2. **Accumulating gradients**

```python
# Always clear gradients after step
optimizer.zero_grad()
```

3. **Not clearing CUDA cache**

```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Symptom 4: Training Slows Down Over Time**

**Possible causes:**

1. **Data loading bottleneck**

```python
# Use multiple workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Use 4 processes for data loading
    pin_memory=True,  # Speed up CPU→GPU transfer
)
```

2. **Gradient accumulation without clearing cache**

```python
# Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

3. **Logging too frequently**

```python
# Log every 100 steps, not every step
if step % 100 == 0:
    logger.log_metrics(...)
```

### Profiling Training Code

**PyTorch Profiler:**

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for step in range(10):  # Profile first 10 steps
        train_step(batch)

# Print report
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Output:**

```
-----------------------------------  ---------------  ---------------
Name                                 CPU Time        CUDA Time
-----------------------------------  ---------------  ---------------
aten::linear                         10.5ms          45.2ms
aten::matmul                         8.3ms           38.7ms
aten::softmax                        2.1ms           12.4ms
aten::layer_norm                     1.5ms           8.9ms
...
-----------------------------------  ---------------  ---------------
```

**Interpretation:**
- Identify bottlenecks (which operations take most time)
- Optimize hotspots first

**Memory profiling:**

```python
with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    train_step(batch)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

### Best Practices Checklist

**Before training:**
- [ ] Sanity check: Train on 1 batch, verify loss decreases
- [ ] Verify data loading (inspect batch shapes and values)
- [ ] Test full training loop for 10 steps
- [ ] Check GPU utilization (should be >80%)

**During training:**
- [ ] Monitor loss curves in real-time (W&B or TensorBoard)
- [ ] Track gradient norms (detect explosions early)
- [ ] Check memory usage (avoid OOM errors)
- [ ] Save checkpoints regularly (every 1000 steps)

**After training:**
- [ ] Load best checkpoint and verify performance
- [ ] Generate sample outputs (qualitative evaluation)
- [ ] Compare with baseline (did we improve?)
- [ ] Save final model and config

---

## Summary

### Key Takeaways

**1. Structured Logging: Record Everything**
- Use JSON logging for machine-readable logs
- Log metrics, hyperparameters, system info, and errors
- Dual output: human-readable console + JSON file
- Parse logs programmatically for analysis

**2. Experiment Tracking: Never Lose an Experiment**
- Track hyperparameters, metrics, artifacts, code version
- Use Weights & Biases or TensorBoard
- Compare experiments side-by-side
- Reproduce results months later

**3. Checkpoint Management: Never Lose Progress**
- Save full state: model, optimizer, scheduler, step, metrics
- Keep best N checkpoints to save disk space
- Resume training seamlessly after crashes
- Track git commits for reproducibility

**4. Configuration Management: No More Hard-Coded Values**
- Use YAML/JSON files for all hyperparameters
- Type-safe configs with dataclasses
- CLI overrides for quick experiments
- Preset configs for common use cases

**5. Production Training: Robust and Reliable**
- Error handling: OOM, NaN loss, gradient explosions
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16) for 2-3× speedup
- Distributed training for multi-GPU setups

**6. Model Export: From Training to Deployment**
- TorchScript for C++ deployment
- ONNX for cross-framework compatibility
- Quantization for 4× smaller models
- Serving with Flask, TorchServe, or ONNX Runtime

**7. Monitoring & Debugging: Catch Problems Early**
- Track loss, perplexity, LR, gradient norm, memory
- Profile training code to find bottlenecks
- Debug NaN loss, exploding gradients, memory leaks
- Use PyTorch Profiler for optimization

### Production Checklist

**Before starting a training run:**
- [ ] Config file created with all hyperparameters
- [ ] Experiment tracker initialized (W&B or TensorBoard)
- [ ] Structured logger set up (JSON file + console)
- [ ] Checkpoint manager configured (keep best N)
- [ ] Git repo is clean (commit all changes)
- [ ] Data loading tested (inspect batches)
- [ ] Model architecture verified (forward pass works)
- [ ] Training loop tested (10 steps without errors)

**During training:**
- [ ] Monitor loss curves in real-time
- [ ] Check gradient norms (detect explosions)
- [ ] Track GPU utilization (>80% is good)
- [ ] Save checkpoints periodically (every 1000 steps)
- [ ] Generate sample outputs (qualitative check)

**After training:**
- [ ] Load best checkpoint (by validation loss)
- [ ] Evaluate on test set (never train on test!)
- [ ] Export model (TorchScript or ONNX)
- [ ] Document results (what worked, what didn't)
- [ ] Archive experiment (logs, checkpoints, config)

### From Research to Production

**Research code:**
```python
for epoch in range(10):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
```

**Production code:**
```python
with ExperimentTracker("project", config) as tracker:
    checkpoint_manager = CheckpointManager(...)
    logger = TrainingLogger(...)

    try:
        for step in range(start_step, max_steps):
            try:
                metrics = train_step(batch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

            if not torch.isfinite(metrics['loss']):
                logger.warning(f"NaN loss, skipping")
                continue

            tracker.log_metrics(metrics, step)

            if step % checkpoint_interval == 0:
                checkpoint_manager.save(...)

            if step % eval_interval == 0:
                val_metrics = evaluate(...)
                tracker.log_metrics(val_metrics, step)

    except KeyboardInterrupt:
        checkpoint_manager.save(...)  # Emergency save
        raise

    finally:
        logger.log_end()
        tracker.finish()
```

**Difference:** Robustness, observability, reproducibility.

### Practical Implementation

**File structure for a production training repo:**

```
tiny-transformer-build/
├── configs/
│   ├── base.yaml                    # Base configuration
│   ├── shakespeare_small.yaml       # Preset configs
│   └── shakespeare_large.yaml
├── tiny_transformer/
│   ├── model.py                     # Model definition
│   ├── training.py                  # Trainer class
│   ├── config.py                    # Config dataclasses
│   └── utils/
│       ├── logging.py               # Structured logging
│       ├── checkpoint.py            # Checkpoint management
│       └── experiment.py            # Experiment tracking
├── tools/
│   ├── train.py                     # Main training script
│   ├── generate.py                  # Text generation script
│   └── evaluate.py                  # Evaluation script
├── data/
│   ├── shakespeare_train.txt
│   └── shakespeare_val.txt
├── checkpoints/                     # Model checkpoints
├── logs/                            # JSON logs
└── runs/                            # TensorBoard logs
```

**Typical workflow:**

```bash
# 1. Create config
vim configs/my_experiment.yaml

# 2. Start training
python tools/train.py --config configs/my_experiment.yaml

# 3. Monitor (in another terminal)
tensorboard --logdir runs/

# 4. Training finishes
# Best checkpoint saved automatically

# 5. Generate text
python tools/generate.py \
    --checkpoint checkpoints/my_experiment/best.pt \
    --prompt "To be or not to be"

# 6. Export for deployment
python tools/export.py \
    --checkpoint checkpoints/my_experiment/best.pt \
    --format onnx
```

### Next Steps

**You now have production-ready engineering practices!**

The tools and patterns in this module enable you to:
- Train models for days without babysitting
- Reproduce any experiment months later
- Collaborate with teammates effectively
- Deploy models to production
- Debug training issues systematically

**Further learning:**
- Read production ML code from major projects (HuggingFace Transformers, PyTorch examples)
- Set up CI/CD for model training
- Learn MLOps platforms (Kubeflow, MLflow, SageMaker)
- Study distributed training (DeepSpeed, Megatron-LM)
- Explore model compression (pruning, distillation, quantization)

---

**Created:** November 2024
**For:** TinyTransformerBuild Educational Series
**Module:** 08 - Engineering & Production Practices
**Target Audience:** Developers ready to train transformers at scale
