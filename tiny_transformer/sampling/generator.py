"""
Text Generation Interface.

This module provides a high-level interface for text generation:
- TextGenerator: Complete generation with all sampling strategies
- Autoregressive generation loop
- EOS token handling
- Batched generation support

See Also:
    - theory.md Section 10: Generation Loop
    - theory.md Section 3: Autoregressive Generation
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Callable

from tiny_transformer.sampling.strategies import (
    greedy_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    combined_sample
)


@dataclass
class GeneratorConfig:
    """
    Configuration for text generation.

    Args:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (1.0 = no change)
        top_k: Top-k filtering (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)
        do_sample: If False, use greedy decoding
        eos_token_id: Token ID for end-of-sequence (None = no EOS)
        pad_token_id: Token ID for padding (for batched generation)

    Example:
        >>> config = GeneratorConfig(
        ...     max_new_tokens=50,
        ...     temperature=0.8,
        ...     top_k=50,
        ...     top_p=0.9
        ... )
    """
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


class TextGenerator:
    """
    High-level interface for text generation.

    Args:
        model: TinyTransformerLM model
        config: Generator configuration
        device: Device for generation (cuda/cpu)

    Example:
        >>> from tiny_transformer.model import TinyTransformerLM
        >>> from tiny_transformer.sampling import TextGenerator, GeneratorConfig
        >>>
        >>> model = TinyTransformerLM(vocab_size=1000, ...)
        >>> config = GeneratorConfig(max_new_tokens=50, temperature=0.8)
        >>> generator = TextGenerator(model, config)
        >>>
        >>> start_tokens = torch.randint(0, 1000, (1, 10))
        >>> generated = generator.generate(start_tokens)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[GeneratorConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config if config else GeneratorConfig()

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        self.model.eval()  # Set to eval mode

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            start_tokens: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Override config max_new_tokens
            temperature: Override config temperature
            top_k: Override config top_k
            top_p: Override config top_p
            do_sample: Override config do_sample
            eos_token_id: Override config eos_token_id

        Returns:
            Generated tokens of shape (batch_size, seq_len + max_new_tokens)

        Example:
            >>> start_tokens = torch.randint(0, 1000, (2, 10))
            >>> generated = generator.generate(
            ...     start_tokens,
            ...     max_new_tokens=20,
            ...     temperature=0.8
            ... )
            >>> generated.shape
            torch.Size([2, 30])
        """
        # Override config if arguments provided
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Move to device
        start_tokens = start_tokens.to(self.device)
        batch_size, start_len = start_tokens.shape

        # Initialize generated sequence with start tokens
        generated = start_tokens.clone()

        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Autoregressive generation loop
        for step in range(max_new_tokens):
            # Get logits for next token
            # Model may truncate context if sequence exceeds max_len
            logits, _ = self.model(generated)

            # Take logits for last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Sample next token
            if do_sample:
                # Use specified sampling strategy
                if top_k is not None or top_p is not None:
                    next_token = combined_sample(
                        next_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                else:
                    next_token = temperature_sample(
                        next_token_logits,
                        temperature=temperature
                    )
            else:
                # Greedy decoding
                next_token = greedy_sample(next_token_logits)

            # For finished sequences, replace with pad token (if specified)
            if eos_token_id is not None and self.config.pad_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.config.pad_token_id),
                    next_token
                )

            # Append to generated sequence
            next_token = next_token.unsqueeze(-1)  # (batch_size, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Update finished status
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

                # If all sequences finished, stop early
                if finished.all():
                    break

        return generated

    @torch.no_grad()
    def generate_batch(
        self,
        start_tokens_list: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate for multiple starting sequences of different lengths.

        Args:
            start_tokens_list: List of starting token tensors (each 1D)
            **kwargs: Generation arguments (same as generate())

        Returns:
            List of generated token tensors

        Example:
            >>> start_1 = torch.tensor([1, 2, 3])
            >>> start_2 = torch.tensor([4, 5])
            >>> generated = generator.generate_batch([start_1, start_2])
        """
        if self.config.pad_token_id is None:
            raise ValueError(
                "pad_token_id must be set in config for batched generation"
            )

        # Pad all sequences to same length
        max_len = max(seq.size(0) for seq in start_tokens_list)
        batch_size = len(start_tokens_list)

        padded_tokens = torch.full(
            (batch_size, max_len),
            self.config.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        for i, seq in enumerate(start_tokens_list):
            seq_len = seq.size(0)
            padded_tokens[i, :seq_len] = seq.to(self.device)

        # Generate
        generated = self.generate(padded_tokens, **kwargs)

        # Split back into list (remove padding)
        result = []
        for i, seq in enumerate(start_tokens_list):
            original_len = seq.size(0)
            # Take from original position to end
            result.append(generated[i, :original_len + kwargs.get('max_new_tokens', self.config.max_new_tokens)])

        return result


if __name__ == "__main__":
    print("=" * 70)
    print("TextGenerator - Demo")
    print("=" * 70)
    print()

    from tiny_transformer.model import TinyTransformerLM, get_model_config

    # Create tiny model
    config = get_model_config("tiny")
    model = TinyTransformerLM(vocab_size=100, **config)
    model.eval()

    # Create generator
    gen_config = GeneratorConfig(
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    generator = TextGenerator(model, gen_config, device='cpu')

    print(f"Generator Config: {gen_config}")
    print()

    # Test single sequence generation
    print("1. Single Sequence Generation:")
    torch.manual_seed(42)
    start_tokens = torch.randint(0, 100, (1, 5))
    print(f"   Start: {start_tokens[0].tolist()}")

    generated = generator.generate(start_tokens)
    print(f"   Generated: {generated[0].tolist()}")
    print(f"   Length: {start_tokens.size(1)} → {generated.size(1)}")
    print()

    # Test batch generation
    print("2. Batch Generation:")
    torch.manual_seed(42)
    start_batch = torch.randint(0, 100, (3, 5))
    print(f"   Start shape: {start_batch.shape}")

    generated_batch = generator.generate(start_batch, max_new_tokens=10)
    print(f"   Generated shape: {generated_batch.shape}")
    for i, seq in enumerate(generated_batch):
        print(f"   Seq {i+1}: {seq.tolist()}")
    print()

    # Test greedy vs sampling
    print("3. Greedy vs Sampling:")
    start_single = torch.randint(0, 100, (1, 5))

    torch.manual_seed(123)
    greedy = generator.generate(start_single, do_sample=False, max_new_tokens=10)

    torch.manual_seed(123)
    sampled = generator.generate(start_single, do_sample=True, max_new_tokens=10)

    print(f"   Greedy:  {greedy[0].tolist()}")
    print(f"   Sampled: {sampled[0].tolist()}")
    print()

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
