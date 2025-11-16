# Contributing to Tiny Transformer Build

Thank you for your interest in contributing! This educational repository welcomes contributions that help others learn to build transformers from scratch.

## Ways to Contribute

### 1. Complete Modules
See [BUILD_STATUS.md](BUILD_STATUS.md) for incomplete modules.

**Module Completion Checklist**:
- [ ] Write `theory.md` with clear conceptual explanations
- [ ] Implement core functionality with shape annotations
- [ ] Create comprehensive tests (>90% coverage)
- [ ] Build exercises with detailed solutions
- [ ] Create interactive Jupyter notebook
- [ ] Add visualizations where applicable
- [ ] Update BUILD_STATUS.md

### 2. Improve Documentation
- Fix typos or unclear explanations
- Add more examples
- Create visual diagrams
- Write tutorials or blog posts

### 3. Add Features
- New visualization functions
- Additional configuration presets
- Performance optimizations
- Better error messages

### 4. Fix Bugs
- Report issues with detailed reproduction steps
- Submit fixes with tests

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/tiny-transformer-build.git
cd tiny-transformer-build

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Verify installation
python tools/verify_environment.py
```

### Before Submitting Pull Request

```bash
# 1. Install the package in editable mode
pip install -e ".[dev]"

# 2. Run all tests
pytest tests/ -v

# 3. Check code formatting (optional but recommended)
black tiny_transformer tests --check
flake8 tiny_transformer tests

# 4. Validate notebooks (if you modified any)
python tools/validate_all.py

# 5. Update documentation
# - Update relevant .md files
# - Add docstrings to new functions
# - Update examples if needed
```

## Code Style Guidelines

### 1. Shape Annotations

Always include shape comments for tensor operations:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Process input through layer.

    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)

    Returns:
        output: Shape (batch_size, seq_len, d_model)
    """
    # x: (B, T, d_model)
    batch_size, seq_len, d_model = x.shape

    # Compute attention
    # output: (B, T, d_model)
    output = self.attention(x)

    return output
```

### 2. Type Hints

Use type hints for all function parameters and returns:

```python
from typing import Optional, Tuple
import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention."""
    # ... implementation ...
```

### 3. Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.

    More detailed explanation if needed. Can span multiple lines
    and include mathematical formulas, examples, etc.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid

    Example:
        >>> result = function_name(42, "hello")
        >>> print(result)
        True
    """
    # ... implementation ...
```

### 4. Educational Code

Since this is an educational repository, prioritize clarity over brevity:

```python
# GOOD: Clear and educational
# Step 1: Compute attention scores
# Dot product measures similarity between query and each key
scores = query @ key.transpose(-2, -1)  # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)

# Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
# For large d_k, dot products grow too large
scaling_factor = math.sqrt(d_k)
scores = scores / scaling_factor

# BAD: Too terse for learning
scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
```

### 5. Tests

Every new function needs tests covering:
- Basic functionality
- Shape correctness
- Edge cases
- Error handling

```python
class TestMyFunction:
    """Tests for my_function."""

    def test_basic_functionality(self):
        """Test that function works with standard inputs."""
        result = my_function(input_data)
        assert result.shape == expected_shape

    def test_edge_case_empty(self):
        """Test behavior with empty input."""
        with pytest.raises(ValueError):
            my_function(torch.empty(0))
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/documentation-improvement
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation

### 3. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add multi-head attention implementation

- Implement MultiHeadAttention module
- Add comprehensive tests
- Create theory.md explaining concept
- Add visualization example"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description explaining what and why
- Reference to any related issues
- Screenshots/examples if relevant

### Pull Request Template

```markdown
## Description
[Describe your changes in detail]

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Module completion

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Manual testing performed
- [ ] Notebooks validated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Commented code, particularly complex areas
- [ ] Updated documentation
- [ ] Added tests proving fix/feature works
- [ ] New and existing tests pass locally
- [ ] Added docstrings to new functions
```

## Code Review Process

All submissions require review. Reviewers will check:
- Code quality and style
- Test coverage
- Documentation clarity
- Educational value

Be patient and responsive to feedback. Multiple rounds of review are normal!

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions about implementation
- Check existing issues/discussions first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make transformer models more accessible to learners worldwide. Thank you for being part of this educational mission!
