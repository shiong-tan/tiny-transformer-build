"""
Visualization Utilities for Transformer Models

Provides functions to visualize:
- Attention patterns (heatmaps)
- Model training progress
- Token embeddings
- Sampling temperature effects
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import math


def plot_attention_pattern(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Pattern",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Shape (seq_len, seq_len) attention matrix
        tokens: Optional list of token strings for axis labels
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    if attention_weights.ndim == 3:
        # If batch dimension present, take first example
        attention_weights = attention_weights[0]
    
    # Convert to numpy
    attn = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Set labels
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention plot to {save_path}")
    
    plt.show()


def plot_multi_head_attention(
    attention_weights: torch.Tensor,
    n_heads: int,
    tokens: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize multiple attention heads.
    
    Args:
        attention_weights: Shape (n_heads, seq_len, seq_len)
        n_heads: Number of attention heads
        tokens: Optional token labels
        figsize: Figure size
        save_path: Optional save path
    """
    # Determine grid layout
    n_cols = min(4, n_heads)
    n_rows = math.ceil(n_heads / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        
        # Get attention for this head
        attn = attention_weights[head_idx].detach().cpu().numpy()
        
        # Plot
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {head_idx + 1}')
        
        # Add token labels if provided and space allows
        if tokens is not None and len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
    
    # Hide extra subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-head attention plot to {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        figsize: Figure size
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = range(1, len(train_losses) + 1)
    
    ax.plot(steps, train_losses, label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        val_steps = np.linspace(1, len(train_losses), len(val_losses))
        ax.plot(val_steps, val_losses, label='Validation Loss', 
                linewidth=2, linestyle='--')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_token_embeddings_2d(
    embeddings: torch.Tensor,
    tokens: List[str],
    method: str = 'pca',
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize token embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Shape (vocab_size, d_model) or (n_tokens, d_model)
        tokens: List of token labels
        method: 'pca' or 'tsne'
        figsize: Figure size
        save_path: Optional save path
    """
    # Convert to numpy
    emb = embeddings.detach().cpu().numpy()
    
    # Reduce to 2D
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    coords = reducer.fit_transform(emb)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=100)
    
    # Add labels
    for i, token in enumerate(tokens):
        ax.annotate(token, (coords[i, 0], coords[i, 1]),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Token Embeddings ({method.upper()})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embeddings plot to {save_path}")
    
    plt.show()


def plot_temperature_comparison(
    logits: torch.Tensor,
    temperatures: List[float] = [0.5, 1.0, 2.0],
    vocab: Optional[List[str]] = None,
    top_k: int = 20,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize effect of temperature on sampling probabilities.
    
    Args:
        logits: Shape (vocab_size,) raw logits
        temperatures: List of temperatures to compare
        vocab: Optional vocabulary for labels
        top_k: Show top K tokens
        figsize: Figure size
        save_path: Optional save path
    """
    n_temps = len(temperatures)
    fig, axes = plt.subplots(1, n_temps, figsize=figsize)
    
    if n_temps == 1:
        axes = [axes]
    
    for idx, temp in enumerate(temperatures):
        # Apply temperature
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Get top K
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Plot
        ax = axes[idx]
        positions = range(top_k)
        
        ax.bar(positions, top_probs.cpu().numpy())
        ax.set_title(f'Temperature = {temp}')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        
        # Add token labels if available
        if vocab is not None:
            labels = [vocab[i] for i in top_indices.cpu().numpy()]
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temperature comparison to {save_path}")
    
    plt.show()


def plot_gradient_flow(
    named_parameters,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot gradient flow through model layers.
    
    Useful for diagnosing vanishing/exploding gradients.
    
    Args:
        named_parameters: model.named_parameters()
        figsize: Figure size
        save_path: Optional save path
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in named_parameters:
        if param.grad is not None and param.requires_grad:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(len(layers)), max_grads, alpha=0.5, label='Max Gradient')
    ax.bar(range(len(layers)), ave_grads, alpha=0.5, label='Mean Gradient')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gradient flow plot to {save_path}")
    
    plt.show()


def create_learning_dashboard(
    train_losses: List[float],
    val_losses: List[float],
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive dashboard with training curves and attention.
    
    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        attention_weights: Recent attention pattern (seq_len, seq_len)
        tokens: Optional token labels
        save_path: Optional save path
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Training curves (left)
    ax1 = plt.subplot(1, 2, 1)
    steps = range(1, len(train_losses) + 1)
    ax1.plot(steps, train_losses, label='Train', linewidth=2)
    
    if val_losses:
        val_steps = np.linspace(1, len(train_losses), len(val_losses))
        ax1.plot(val_steps, val_losses, label='Validation', 
                linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Attention pattern (right)
    ax2 = plt.subplot(1, 2, 2)
    
    if attention_weights.ndim == 3:
        attention_weights = attention_weights[0]  # Take first example
    
    attn = attention_weights.detach().cpu().numpy()
    im = ax2.imshow(attn, cmap='viridis', aspect='auto')
    
    if tokens and len(tokens) <= 15:
        ax2.set_xticks(range(len(tokens)))
        ax2.set_yticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha='right')
        ax2.set_yticklabels(tokens)
    
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    ax2.set_title('Recent Attention Pattern')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")
    
    plt.show()
