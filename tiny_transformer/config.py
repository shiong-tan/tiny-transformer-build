"""
Configuration Management for Tiny Transformer

Provides dataclasses and utilities for managing model and training configuration.
Follows best practices: type-safe, serializable, version-controlled configs.

Key principles:
1. Configuration in one place
2. Type checking at definition time
3. Easy to save/load
4. Version control friendly (YAML/JSON)
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Tiny Transformer model architecture."""
    
    # Model architecture
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512  # Feedforward dimension (usually 4 * d_model)
    max_seq_len: int = 256
    dropout: float = 0.1
    
    # Embeddings
    use_learned_positional: bool = True  # vs sinusoidal
    
    # Layer norm
    layer_norm_eps: float = 1e-5
    pre_ln: bool = True  # Pre-LN vs Post-LN
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
    
    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    def save(self, path: str) -> None:
        """Save configuration to file (JSON or YAML)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        print(f"Saved config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    
    # Optimization
    optimizer: Literal['adam', 'adamw'] = 'adamw'
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: Optional[float] = 1.0
    
    # Learning rate schedule
    lr_schedule: Literal['constant', 'cosine', 'linear'] = 'cosine'
    min_lr: float = 1e-5
    
    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 100
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 1000
    save_total_limit: int = 3  # Keep only N most recent checkpoints
    
    # Data
    seq_len: int = 256  # Training sequence length
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        print(f"Saved config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return cls(**config_dict)


@dataclass
class SamplingConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Sampling strategy
    strategy: Literal['greedy', 'temperature', 'top_k', 'nucleus'] = 'temperature'
    
    # Stopping criteria
    stop_tokens: list = field(default_factory=list)
    
    # Reproducibility
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"
        assert self.temperature > 0, "temperature must be positive"
        
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be positive"
        
        if self.top_p is not None:
            assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"


@dataclass
class ExperimentConfig:
    """
    Complete configuration for an experiment.
    
    Combines model, training, and sampling configs.
    """
    
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Experiment metadata
    name: str = "tiny_transformer_exp"
    description: str = ""
    tags: list = field(default_factory=list)
    
    def save(self, path: str) -> None:
        """Save complete experiment configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'sampling': asdict(self.sampling),
            'metadata': {
                'name': self.name,
                'description': self.description,
                'tags': self.tags
            }
        }
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        print(f"Saved experiment config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load complete experiment configuration."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            sampling=SamplingConfig(**config_dict['sampling']),
            name=config_dict['metadata']['name'],
            description=config_dict['metadata']['description'],
            tags=config_dict['metadata']['tags']
        )


# Preset configurations for quick experimentation
def get_tiny_config() -> ExperimentConfig:
    """Ultra-small config for quick testing (trains in ~1 minute)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=500,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1e-3,
            max_steps=500,
            warmup_steps=50,
            eval_interval=100,
            log_interval=10
        ),
        name="tiny_test"
    )


def get_small_config() -> ExperimentConfig:
    """Small config for learning (trains in ~10 minutes)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=512,
            max_seq_len=256,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=3e-4,
            max_steps=5000,
            warmup_steps=500,
            eval_interval=500,
            log_interval=50
        ),
        name="small_model"
    )


def get_medium_config() -> ExperimentConfig:
    """Medium config for better results (trains in ~1 hour)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=5000,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=64,
            learning_rate=1e-4,
            max_steps=20000,
            warmup_steps=2000,
            eval_interval=1000,
            log_interval=100
        ),
        name="medium_model"
    )
