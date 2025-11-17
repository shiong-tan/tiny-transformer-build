"""
Language Modeling Dataset Implementation.

This module provides dataset classes for training language models:
- TextDataset: Sliding window dataset for character or token-level LM
- Efficient batching with proper input-target alignment

See Also:
    - theory.md Section 2: Data Loading and Preprocessing
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, List, Callable
import os


class TextDataset(Dataset):
    """
    Dataset for language modeling with sliding window batching.

    Creates input-target pairs where target is input shifted by 1 position:
        Input:  [t0, t1, t2, ..., tn-1]
        Target: [t1, t2, t3, ..., tn]

    This is the standard causal language modeling setup.

    Args:
        tokens: List or tensor of token IDs
        seq_len: Sequence length for each example
        stride: Stride for sliding window (default: seq_len for no overlap)

    Shape:
        Input: (seq_len,) - Token IDs
        Target: (seq_len,) - Next token IDs (shifted by 1)

    Example:
        >>> tokens = list(range(1000))  # 1000 tokens
        >>> dataset = TextDataset(tokens, seq_len=128, stride=128)
        >>> len(dataset)
        7
        >>> input_ids, target_ids = dataset[0]
        >>> input_ids.shape, target_ids.shape
        (torch.Size([128]), torch.Size([128]))
        >>> # Verify target is input shifted by 1
        >>> assert torch.equal(input_ids[1:], target_ids[:-1])
    """

    def __init__(
        self,
        tokens: List[int],
        seq_len: int,
        stride: Optional[int] = None
    ):
        self.tokens = torch.tensor(tokens, dtype=torch.long) if isinstance(tokens, list) else tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Calculate number of windows
        # We need seq_len + 1 tokens for each example (seq_len input + 1 for target)
        total_tokens = len(self.tokens)
        window_size = seq_len + 1

        if total_tokens < window_size:
            raise ValueError(
                f"Not enough tokens ({total_tokens}) for even one sequence "
                f"of length {seq_len} + 1 target"
            )

        # Number of valid starting positions
        self.num_windows = (total_tokens - window_size) // self.stride + 1

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair for language modeling.

        Args:
            idx: Index of example

        Returns:
            Tuple of (input_ids, target_ids) where:
            - input_ids: Token IDs [t0, t1, ..., t(n-1)]
            - target_ids: Next tokens [t1, t2, ..., tn]

        Shape flow:
            Extract window of length seq_len + 1
            Split into input (first seq_len) and target (last seq_len)
            Target is input shifted right by 1
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Starting position in token sequence
        start_idx = idx * self.stride

        # Extract window: [start_idx, start_idx + seq_len + 1)
        window = self.tokens[start_idx : start_idx + self.seq_len + 1]

        # Split into input and target
        # Input: [t0, t1, t2, ..., t(n-1)]
        input_ids = window[:-1]

        # Target: [t1, t2, t3, ..., tn]
        target_ids = window[1:]

        # Verify shapes
        assert input_ids.shape[0] == self.seq_len
        assert target_ids.shape[0] == self.seq_len

        return input_ids, target_ids

    @classmethod
    def from_text_file(
        cls,
        file_path: str,
        tokenizer: Callable[[str], List[int]],
        seq_len: int,
        stride: Optional[int] = None
    ) -> 'TextDataset':
        """
        Create dataset from text file.

        Args:
            file_path: Path to text file
            tokenizer: Function to convert text to token IDs
            seq_len: Sequence length
            stride: Stride for windows (default: seq_len)

        Returns:
            TextDataset instance

        Example:
            >>> def char_tokenizer(text):
            ...     # Simple character-level tokenizer
            ...     char_to_idx = {ch: i for i, ch in enumerate(set(text))}
            ...     return [char_to_idx[ch] for ch in text]
            >>> dataset = TextDataset.from_text_file(
            ...     'data/shakespeare.txt',
            ...     char_tokenizer,
            ...     seq_len=128
            ... )
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = tokenizer(text)

        return cls(tokens, seq_len, stride)

    def get_stats(self) -> dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_tokens': len(self.tokens),
            'seq_len': self.seq_len,
            'stride': self.stride,
            'num_examples': len(self),
            'overlap_tokens': self.seq_len - self.stride if self.stride < self.seq_len else 0,
            'coverage': (len(self) * self.stride + self.seq_len) / len(self.tokens),
        }


class CharTokenizer:
    """
    Simple character-level tokenizer for text data.

    Creates a vocabulary from unique characters in the text.

    Args:
        vocab: Optional pre-defined vocabulary (char -> idx mapping)

    Example:
        >>> tokenizer = CharTokenizer()
        >>> text = "hello world"
        >>> tokenizer.fit(text)
        >>> tokens = tokenizer.encode(text)
        >>> decoded = tokenizer.decode(tokens)
        >>> assert decoded == text
    """

    def __init__(self, vocab: Optional[dict] = None):
        self.vocab = vocab if vocab is not None else {}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()} if vocab else {}

    def fit(self, text: str):
        """
        Build vocabulary from text.

        Args:
            text: Training text
        """
        # Get unique characters
        unique_chars = sorted(set(text))

        # Create character to index mapping
        self.vocab = {char: idx for idx, char in enumerate(unique_chars)}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs

        Raises:
            KeyError: If text contains unknown characters
        """
        try:
            return [self.vocab[char] for char in text]
        except KeyError as e:
            raise ValueError(f"Unknown character in text: {e}")

    def decode(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text

        Raises:
            KeyError: If tokens contain unknown IDs
        """
        try:
            return ''.join([self.inverse_vocab[token] for token in tokens])
        except KeyError as e:
            raise ValueError(f"Unknown token ID: {e}")

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def save(self, path: str):
        """Save vocabulary to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load vocabulary from file."""
        import json
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab)


def create_data_loaders(
    train_tokens: List[int],
    val_tokens: Optional[List[int]],
    seq_len: int,
    batch_size: int,
    stride: Optional[int] = None,
    num_workers: int = 0
):
    """
    Create train and validation data loaders.

    Args:
        train_tokens: Training tokens
        val_tokens: Validation tokens (optional)
        seq_len: Sequence length
        batch_size: Batch size
        stride: Stride for sliding window (default: seq_len)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader) where val_loader is None if no val_tokens

    Example:
        >>> train_tokens = list(range(10000))
        >>> val_tokens = list(range(10000, 11000))
        >>> train_loader, val_loader = create_data_loaders(
        ...     train_tokens, val_tokens,
        ...     seq_len=128, batch_size=32
        ... )
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_len, stride)
    val_dataset = TextDataset(val_tokens, seq_len, seq_len) if val_tokens else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if val_dataset else None

    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 70)
    print("TextDataset - Demo")
    print("=" * 70)
    print()

    # Create sample data
    tokens = list(range(1000))
    dataset = TextDataset(tokens, seq_len=128, stride=64)

    print(f"Dataset stats: {dataset.get_stats()}")
    print()

    # Get a sample
    input_ids, target_ids = dataset[0]
    print(f"Sample input:  {input_ids[:10].tolist()} ...")
    print(f"Sample target: {target_ids[:10].tolist()} ...")
    print()

    # Verify target is input shifted by 1
    print(f"Target matches input shifted? {torch.equal(input_ids[1:], target_ids[:-1])}")
    print()

    # Test character tokenizer
    print("=" * 70)
    print("CharTokenizer - Demo")
    print("=" * 70)
    print()

    text = "hello world! this is a test."
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")
    print()

    tokens = tokenizer.encode(text)
    print(f"Encoded: {tokens}")
    print()

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"Matches original? {decoded == text}")
    print()

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
