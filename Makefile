# Makefile for Tiny Transformer Course
# Provides convenient commands for setup, testing, and validation

.PHONY: help install test validate clean check notebook

# Default target
help:
	@echo "Tiny Transformer Course - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install       - Create venv and install all dependencies"
	@echo "  make verify        - Verify environment is correctly configured"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test          - Run all unit tests"
	@echo "  make validate      - Run full validation suite (tests + notebooks)"
	@echo "  make check         - Quick check before committing (tests only)"
	@echo "  make notebook      - Start Jupyter notebook server"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  make clean         - Remove generated files and caches"
	@echo "  make clean-all     - Remove everything including venv"
	@echo ""
	@echo "Examples:"
	@echo "  make example       - Run quick attention example"
	@echo ""

# Installation
install:
	@echo "Setting up Tiny Transformer Course environment..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3.11 -m venv .venv || python3 -m venv .venv; \
	fi
	@echo "Activating virtual environment and installing packages..."
	@. .venv/bin/activate && \
		pip install --upgrade pip wheel && \
		pip install -r requirements.txt
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate environment:  source .venv/bin/activate"
	@echo "  2. Verify installation:   make verify"
	@echo "  3. Start learning:        make example"

# Verify environment
verify:
	@. .venv/bin/activate && python tools/verify_environment.py

# Run all tests
test:
	@echo "Running all unit tests..."
	@. .venv/bin/activate && python -m pytest tests/ --tb=short -v

# Full validation (tests + notebooks)
validate:
	@echo "Running full validation suite..."
	@. .venv/bin/activate && python tools/validate_all.py

# Quick check (just tests, faster than validate)
check:
	@echo "Running quick check..."
	@. .venv/bin/activate && python -m pytest --tb=short -q

# Start Jupyter notebook
notebook:
	@. .venv/bin/activate && jupyter notebook

# Run example
example:
	@echo "Running attention example..."
	@. .venv/bin/activate && python -c "from tiny_transformer.attention import *; import torch; Q=K=V=torch.randn(2,5,64); out,attn=scaled_dot_product_attention(Q,K,V); print(f'Output shape: {out.shape}')"

# Cleanup
clean:
	@echo "Cleaning generated files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/
	@echo "✅ Cleanup complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf .venv
	@echo "✅ Deep cleanup complete!"

# Development: Auto-format code
format:
	@. .venv/bin/activate && black . --line-length 100 --exclude .venv

# Development: Lint code
lint:
	@. .venv/bin/activate && flake8 . --exclude .venv --max-line-length 100
