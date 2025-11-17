"""
Module 06: Training - Exercises

Comprehensive exercises for learning transformer training.

Available Exercises:
- Exercise 01: TextDataset - Sliding window datasets (Easy)
- Exercise 02: CharTokenizer - Character-level tokenization (Easy)
- Exercise 03: Scheduler Configuration - Learning rate schedules (Easy)
- Exercise 04: Data Loaders - Creating train/val loaders (Medium)
- Exercise 05: Basic Training Loop - Forward/backward/update (Medium)
- Exercise 06: Gradient Clipping - Preventing gradient explosion (Medium)
- Exercise 07: Checkpointing - Saving model state (Medium)
- Exercise 08: Perplexity Computation - Evaluating language models (Medium)
- Exercise 09: Custom Warmup Scheduler - Implementing schedules from scratch (Hard)
- Exercise 10: Custom Trainer - Complete training infrastructure (Hard)
- Exercise 11: Early Stopping - Preventing overfitting (Hard)
- Exercise 12: Shakespeare Training - End-to-end pipeline (Very Hard)

Usage:
    >>> from docs.modules.06_training.exercises import exercises
    >>> # Work through exercises in order
    >>> dataset = exercises.Exercise01_TextDataset(tokens, seq_len=128)
    >>> tokenizer = exercises.Exercise02_CharTokenizer()
    >>> # ... continue with other exercises
"""

from .exercises import (
    Exercise01_TextDataset,
    Exercise02_CharTokenizer,
    exercise_03_create_scheduler,
    exercise_04_create_data_loaders,
    exercise_05_basic_training_loop,
    exercise_06_training_with_grad_clip,
    exercise_07_training_with_checkpoints,
    exercise_08_compute_perplexity,
    Exercise09_CustomWarmupScheduler,
    Exercise10_CustomTrainer,
    Exercise11_EarlyStopping,
    exercise_12_shakespeare_training,
    run_all_tests,
)

__all__ = [
    # Classes
    'Exercise01_TextDataset',
    'Exercise02_CharTokenizer',
    'Exercise09_CustomWarmupScheduler',
    'Exercise10_CustomTrainer',
    'Exercise11_EarlyStopping',

    # Functions
    'exercise_03_create_scheduler',
    'exercise_04_create_data_loaders',
    'exercise_05_basic_training_loop',
    'exercise_06_training_with_grad_clip',
    'exercise_07_training_with_checkpoints',
    'exercise_08_compute_perplexity',
    'exercise_12_shakespeare_training',
    'run_all_tests',
]
