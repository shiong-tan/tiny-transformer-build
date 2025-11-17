# CLI Tools Validation Checklist

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python3 setup/verify_environment.py
```

## Module 08 Engineering - CLI Tools Validation

### 1. `tools/train.py` - Training Script

#### Basic Functionality
- [ ] **Help message displays correctly**
  ```bash
  python3 tools/train.py --help
  ```
  Expected: Shows all arguments (--config, --data-train, --learning-rate, etc.)

- [ ] **Config file loading works**
  ```bash
  python3 -c "from tools.train import load_config; print(load_config('configs/base.yaml'))"
  ```
  Expected: Returns config dictionary

- [ ] **Config override works**
  ```bash
  # Test that CLI arguments override config values
  ```

#### Training Execution (requires data)
- [ ] **Train with tiny config (fast test)**
  ```bash
  # Download sample data first
  mkdir -p data
  curl -o data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

  # Train for 100 steps
  python3 tools/train.py \
    --config configs/tiny.yaml \
    --data-train data/tiny_shakespeare.txt \
    --max-steps 100 \
    --experiment-name test_run
  ```
  Expected:
  - Creates checkpoints in `checkpoints/` directory
  - Creates logs in `logs/` directory
  - Prints training progress
  - No errors or warnings
  - Final checkpoint saved

- [ ] **Resume from checkpoint**
  ```bash
  python3 tools/train.py \
    --config configs/tiny.yaml \
    --data-train data/tiny_shakespeare.txt \
    --max-steps 200 \
    --resume checkpoints/checkpoint_100.pt
  ```
  Expected:
  - Resumes from step 100
  - Continues to step 200
  - Loads optimizer and scheduler state

- [ ] **Validation logging works**
  ```bash
  # Check that val_loss is logged
  grep "val_loss" logs/test_run/*.log
  ```

### 2. `tools/generate.py` - Text Generation Script

#### Basic Functionality
- [ ] **Help message displays correctly**
  ```bash
  python3 tools/generate.py --help
  ```
  Expected: Shows generation parameters (--prompt, --temperature, --top-k, etc.)

#### Generation (requires trained checkpoint)
- [ ] **Generate with default settings**
  ```bash
  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompt "ROMEO:" \
    --max-tokens 100
  ```
  Expected:
  - Generates text continuation
  - No errors
  - Output is readable

- [ ] **Generate with different sampling strategies**
  ```bash
  # Greedy
  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompt "ROMEO:" \
    --greedy

  # High temperature
  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompt "ROMEO:" \
    --temperature 1.5

  # Top-k sampling
  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompt "ROMEO:" \
    --top-k 50

  # Top-p sampling
  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompt "ROMEO:" \
    --top-p 0.95
  ```
  Expected: Different sampling produces different outputs

- [ ] **Batch generation from file**
  ```bash
  # Create prompts file
  echo -e "ROMEO:\nJULIET:\nKING:" > prompts.txt

  python3 tools/generate.py \
    --checkpoint checkpoints/checkpoint_100.pt \
    --prompts-file prompts.txt \
    --output generated.txt
  ```
  Expected:
  - Generates for all prompts
  - Saves to generated.txt
  - File contains all 3 generations

### 3. `tools/interactive.py` - Interactive REPL

#### Basic Functionality
- [ ] **Help message displays correctly**
  ```bash
  python3 tools/interactive.py --help
  ```

#### Interactive Mode (requires trained checkpoint)
- [ ] **REPL starts successfully**
  ```bash
  python3 tools/interactive.py --checkpoint checkpoints/checkpoint_100.pt
  ```
  Expected:
  - Shows welcome message
  - Shows current settings
  - Displays `>` prompt

- [ ] **Commands work**
  ```
  # In REPL, test:
  > /help           # Shows help
  > /settings       # Shows current settings
  > /temp 1.0       # Sets temperature
  > /topk 50        # Sets top-k
  > /topp 0.95      # Sets top-p
  > /len 200        # Sets max tokens
  > /greedy         # Toggles greedy
  > /quit           # Exits
  ```

- [ ] **Generation works**
  ```
  > ROMEO:
  [should generate text]

  > /temp 1.5
  > JULIET:
  [should generate more creative text]
  ```

### 4. Configuration Files

#### YAML Syntax Validation
- [ ] **base.yaml is valid**
  ```bash
  python3 -c "import yaml; print(yaml.safe_load(open('configs/base.yaml')))"
  ```

- [ ] **tiny.yaml is valid**
  ```bash
  python3 -c "import yaml; print(yaml.safe_load(open('configs/tiny.yaml')))"
  ```

- [ ] **shakespeare.yaml is valid**
  ```bash
  python3 -c "import yaml; print(yaml.safe_load(open('configs/shakespeare.yaml')))"
  ```

#### Config Values
- [ ] **Configs have required fields**
  - model: d_model, n_heads, n_layers, d_ff, max_seq_len
  - training: learning_rate, batch_size, max_steps, warmup_steps
  - data: train_file, seq_len
  - paths: checkpoint_dir, output_dir

### 5. Integration Tests

#### Complete End-to-End Pipeline
- [ ] **Full pipeline: train → generate → interactive**
  ```bash
  # 1. Train
  python3 tools/train.py \
    --config configs/tiny.yaml \
    --data-train data/tiny_shakespeare.txt \
    --max-steps 500 \
    --experiment-name e2e_test

  # 2. Generate
  python3 tools/generate.py \
    --checkpoint checkpoints/e2e_test/best.pt \
    --prompt "ROMEO:" \
    --max-tokens 100 \
    --output e2e_output.txt

  # 3. Interactive
  python3 tools/interactive.py \
    --checkpoint checkpoints/e2e_test/best.pt
  ```

- [ ] **Checkpoint compatibility**
  - Checkpoints from train.py load in generate.py
  - Checkpoints from train.py load in interactive.py
  - Resume works across sessions

### 6. Error Handling

#### Expected Errors
- [ ] **Missing config file**
  ```bash
  python3 tools/train.py --config nonexistent.yaml
  ```
  Expected: Clear error message about missing file

- [ ] **Missing checkpoint**
  ```bash
  python3 tools/generate.py --checkpoint nonexistent.pt --prompt "test"
  ```
  Expected: Clear error message

- [ ] **Invalid prompt (unknown characters)**
  ```bash
  python3 tools/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "🚀✨" \
    --max-tokens 10
  ```
  Expected: Error about unknown characters in vocabulary

- [ ] **Missing required arguments**
  ```bash
  python3 tools/train.py --config configs/base.yaml
  # Missing --data-train
  ```
  Expected: Error about required data_train argument

### 7. Logging and Checkpointing

#### Logging
- [ ] **Log files created**
  ```bash
  ls logs/test_run/
  ```
  Expected: Log files with timestamps

- [ ] **JSON logs parseable**
  ```bash
  python3 -c "
  import json
  with open('logs/test_run/train.log') as f:
      for line in f:
          if line.strip():
              json.loads(line)
  "
  ```
  Expected: No JSON parsing errors

- [ ] **Metrics logged**
  ```bash
  grep '"train_loss"' logs/test_run/*.log
  grep '"val_loss"' logs/test_run/*.log
  grep '"learning_rate"' logs/test_run/*.log
  ```
  Expected: Metrics present in logs

#### Checkpointing
- [ ] **Checkpoints contain all state**
  ```bash
  python3 -c "
  import torch
  ckpt = torch.load('checkpoints/checkpoint_100.pt', map_location='cpu')
  print('Keys:', ckpt.keys())
  assert 'model_state_dict' in ckpt
  assert 'optimizer_state_dict' in ckpt
  assert 'scheduler_state_dict' in ckpt
  assert 'step' in ckpt
  assert 'epoch' in ckpt
  assert 'config' in ckpt
  print('✓ All required keys present')
  "
  ```

- [ ] **Best-N strategy works**
  - Train for 1000 steps with checkpoint_interval=100
  - Check that only best N checkpoints are kept
  - Verify worst checkpoints were deleted

### 8. Performance

#### Memory Usage
- [ ] **Training doesn't leak memory**
  - Monitor memory during training
  - Memory should stabilize, not grow continuously

#### Speed
- [ ] **Training speed is reasonable**
  - Record steps/second
  - Should be >10 steps/sec for tiny model on CPU
  - Should be >50 steps/sec with MPS/CUDA

#### Generation Speed
- [ ] **Generation is fast**
  - 100 tokens should generate in <5 seconds for small model

## Validation Summary

After completing all checks:

```bash
# Run automated validation
bash tools/validate_cli.sh

# Run full test suite
pytest tests/test_engineering.py -v

# Check coverage
pytest tests/test_engineering.py --cov=tiny_transformer --cov-report=html
```

## Sign-off

- [ ] All CLI tools execute without errors
- [ ] All configurations are valid
- [ ] End-to-end pipeline works
- [ ] Error handling is robust
- [ ] Logging and checkpointing work correctly
- [ ] Performance is acceptable
- [ ] All tests pass

**Module 08 Engineering: COMPLETE** ✅

---

**Validated by**: _________________
**Date**: _________________
**Notes**: _________________
