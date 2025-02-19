"""Provides a class for token embedding in the transformer model."""

from __future__ import annotations

import math
from typing import Literal

from torch import Tensor, nn

from patched_font_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_MAX_ARGS_LEN,
    TRUETYPE_COMMAND_TYPE_TO_NUM,
    TRUETYPE_MAX_ARGS_LEN,
)


class SegmentEmbedding(nn.Module):
    """Embedding layer for the transformer model."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the embedding layer with the specified embedding dimension."""
        super().__init__()

        if outline_format == "truetype":
            self.command_type_to_num = TRUETYPE_COMMAND_TYPE_TO_NUM
            self.max_args_len = TRUETYPE_MAX_ARGS_LEN
        elif outline_format == "postscript":
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
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the output layer with the specified embedding dimension."""
        super().__init__()

        if outline_format == "truetype":
            self.command_type_to_num = TRUETYPE_COMMAND_TYPE_TO_NUM
            self.max_args_len = TRUETYPE_MAX_ARGS_LEN
        elif outline_format == "postscript":
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
