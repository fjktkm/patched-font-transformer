"""Provides a class for token embedding in the transformer model."""

from __future__ import annotations

import math

from torch import Tensor, nn

from patched_font_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_MAX_ARGS_LEN,
)


class SegmentEmbedding(nn.Module):
    """Embedding layer for the transformer model."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the embedding layer with the specified embedding dimension."""
        super().__init__()

        self.command_type_to_num = POSTSCRIPT_COMMAND_TYPE_TO_NUM
        self.max_args_len = POSTSCRIPT_MAX_ARGS_LEN

        self.command_embedding = nn.Embedding(
            len(self.command_type_to_num),
            embedding_dim,
        )
        self.position_embedding = nn.Linear(self.max_args_len, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, glyphs_batch: tuple[Tensor, Tensor]) -> Tensor:
        """Forward pass for the embedding layer."""
        command_indices_tensor, positions_tensor = glyphs_batch
        embedded_commands = self.command_embedding(command_indices_tensor)
        embedded_positions = self.position_embedding(positions_tensor)
        combined_embeddings = (embedded_commands + embedded_positions) * math.sqrt(
            self.embedding_dim,
        )
        return self.dropout(combined_embeddings)


class SegmentUnembedding(nn.Module):
    """Output layer for the transformer model with Classifier."""

    def __init__(
        self,
        embedding_dim: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the output layer with the specified embedding dimension."""
        super().__init__()

        self.command_type_to_num = POSTSCRIPT_COMMAND_TYPE_TO_NUM
        self.max_args_len = POSTSCRIPT_MAX_ARGS_LEN

        self.command_classifier = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, len(self.command_type_to_num)),
        )
        self.position_output_layer = nn.Linear(embedding_dim, self.max_args_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, combined_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the output layer."""
        dropped_embeddings = self.dropout(combined_embeddings)
        command_logits = self.command_classifier(dropped_embeddings)
        position_output = self.position_output_layer(dropped_embeddings)

        return command_logits, position_output


class PatchedSegmentEmbedding(nn.Module):
    """Embedding layer for patched sequences in the transformer model."""

    def __init__(
        self,
        embedding_dim: int,
        patch_len: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the patched embedding layer."""
        super().__init__()
        self.segment_embedding = SegmentEmbedding(embedding_dim, dropout)
        self.patch_linear = nn.Linear(patch_len * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patched_glyphs_batch: tuple[Tensor, Tensor]) -> Tensor:
        """Forward pass for the patched embedding layer.

        Args:
            patched_glyphs_batch: Tuple of tensors
                - Commands: [batch_size, num_patches, patch_len]
                - Positions: [batch_size, num_patches, patch_len, 6]

        Returns:
            - Embedded tensor of shape [batch_size, num_patches, embedding_dim]

        """
        commands, positions = patched_glyphs_batch

        batch_size, num_patches, patch_len = commands.shape

        flat_commands = commands.view(batch_size * num_patches, patch_len)
        flat_positions = positions.view(batch_size * num_patches, patch_len, 6)

        embedded_patches = self.segment_embedding((flat_commands, flat_positions))

        embedded_patches = embedded_patches.view(
            batch_size,
            num_patches,
            patch_len * self.segment_embedding.embedding_dim,
        )

        patched_embeddings = self.patch_linear(embedded_patches)

        return self.dropout(patched_embeddings)
