"""
Neural Network Architectures for Market Signal Feature Prioritization.
Implements the 1D-CNN Market Signal Encoder and Dense Prediction Head
from 'A Deep Learning Framework for Predictive Feature Prioritization in Early-Stage Software Startups'
(MDPI Computers 2026, 15, 380).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class MarketSignalCNN(nn.Module):
    """
    Market Signal Encoder Architecture from the Research Paper:
      - Trainable / Initialized Embedding Layer (vocab_size, embedding_dim)
      - Parallel 1D Convolutions with filter kernel sizes [3, 4, 5]
      - 100 filters per kernel size (total 300 pooled feature maps)
      - ReLU non-linearity and Global 1D Max Pooling
      - Feature concatenation (300-dimensional market signal representation)
      - Dense classification head (Dense 128 + ReLU + Dropout 0.5 + Linear 1)
      - Binary classification output (Logits or Sigmoid probability)
    """

    def __init__(
        self,
        vocab_size: int = config.MAX_VOCAB_SIZE,
        embedding_dim: int = config.EMBEDDING_DIM,
        filter_sizes: List[int] = config.CNN_FILTER_SIZES,
        num_filters: int = config.NUM_FILTERS,
        dense_units: int = config.DENSE_UNITS,
        dropout_rate: float = config.DROPOUT_RATE,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dense_units = dense_units

        # Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        else:
            # Xavier / Uniform initialization strategy for embeddings
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            # Ensure padding index vector remains strictly 0
            with torch.no_grad():
                self.embedding.weight[0].fill_(0)

        # Parallel 1D Convolutions for multi-scale n-gram feature capture
        # Input to Conv1d: (batch_size, channels=embedding_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # keeps sequence length roughly consistent
            )
            for k in filter_sizes
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Classification / Fusion Head
        total_conv_out = num_filters * len(filter_sizes)  # 100 * 3 = 300
        self.fc1 = nn.Linear(total_conv_out, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 300-dimensional pooled market signal representation
        for potential multimodal fusion or feature inspection.
        """
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # Permute for Conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Apply convolutions, ReLU, and global max pooling
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_len)
            # Global max pooling over sequence dimension
            pooled, _ = torch.max(conv_out, dim=2)  # (batch_size, num_filters)
            pooled_outputs.append(pooled)

        # Concatenate along feature dimension -> (batch_size, 300)
        market_repr = torch.cat(pooled_outputs, dim=1)
        return market_repr

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits by default (for numerical stability with BCEWithLogitsLoss),
        or probabilities when return_probs=True.
        """
        features = self.extract_features(x)  # (batch_size, 300)
        hidden = F.relu(self.fc1(self.dropout(features)))  # (batch_size, 128)
        logits = self.fc2(self.dropout(hidden)).squeeze(-1)  # (batch_size,)

        if return_probs:
            return torch.sigmoid(logits)
        return logits
