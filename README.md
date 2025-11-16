# Tiny Transformer Build

Educational repository for implementing transformer architectures from scratch.

## Installation

```bash
git clone https://github.com/yourusername/tiny-transformer-build.git
cd tiny-transformer-build
make install
source .venv/bin/activate
make verify
```

## Usage

```python
import torch
from tiny_transformer import scaled_dot_product_attention

Q = K = V = torch.randn(2, 5, 64)
output, attn_weights = scaled_dot_product_attention(Q, K, V)
```

## Structure

```
tiny-transformer-build/
├── tiny_transformer/          # Main package
│   ├── attention.py          # Attention mechanism
│   ├── config.py             # Configuration
│   └── utils/                # Utilities
├── tests/                     # Test suite
├── tools/                     # Validation tools
└── docs/                      # Documentation
```

## Documentation

- [BUILD_STATUS.md](docs/BUILD_STATUS.md) - Implementation status
- [LEARNING_PATH.md](docs/LEARNING_PATH.md) - Learning progression
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## Commands

```bash
make test      # Run tests
make verify    # Verify environment
make example   # Run example
make clean     # Clean generated files
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## License

MIT
