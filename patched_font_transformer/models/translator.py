"""Provides the FontTransformer model for generating fonts."""

from __future__ import annotations

from torch import Tensor, nn

from patched_font_transformer.modules.embedding import (
    SegmentEmbedding,
    SegmentUnembedding,
)
from patched_font_transformer.modules.positional_encoding import (
    LearnedPositionalEncoding,
)


class FontTranslator(nn.Module):
    """Font transformer model for generating fonts using nn.Transformer."""

    def __init__(
        self,
        num_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the FontTransformer model with specified parameters."""
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.generator = SegmentUnembedding(emb_size, dropout=dropout)
        self.src_tok_emb = SegmentEmbedding(emb_size, dropout=dropout)
        self.tgt_tok_emb = SegmentEmbedding(emb_size, dropout=dropout)
        self.positional_encoding = LearnedPositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        *,
        src: tuple[Tensor, Tensor],
        tgt: tuple[Tensor, Tensor],
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for the FontTransformer model."""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(memory)

    def encode(
        self,
        *,
        src: tuple[Tensor, Tensor],
        src_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode the source sequence."""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )

    def decode(
        self,
        *,
        tgt: tuple[Tensor, Tensor],
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode the target sequence."""
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
