"""Collate function for dataloaders."""

import torch
from torch import Tensor

from patched_font_transformer.torchfont.io.font import POSTSCRIPT_COMMAND_TYPE_TO_NUM


class FontPairPostScriptCollate:
    """Collate function wrapper for style transfer dataloaders."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Collate function for dataloaders that pads the batch to the same size."""
        src, target = zip(*batch, strict=True)
        target = _add_special_tokens_postscript_outline_batch(target)
        src_padded = _pad_postscript_outline(src, self.pad_size)
        target_padded = _pad_postscript_outline(target, self.pad_size)
        return src_padded, target_padded


class MultiFontPostScriptCollate:
    """Collate function wrapper for classification dataloaders."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor], int, int]],
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Collate function for dataloaders that pads the batch to the same size."""
        glyph, codepoint, font_index = zip(*batch, strict=True)
        glyph_padded = _pad_postscript_outline(glyph, self.pad_size)
        return glyph_padded, torch.tensor(codepoint), torch.tensor(font_index)


def _add_special_tokens_postscript_outline_batch(
    batch: list[tuple[Tensor, Tensor]],
) -> list[tuple[Tensor, Tensor]]:
    """Add <bos> and <eos> tokens to each sequence in the batch."""
    return [
        _add_special_tokens_postscript_outline(commands, coordinates)
        for commands, coordinates in batch
    ]


def _add_special_tokens_postscript_outline(
    commands: Tensor,
    coordinates: Tensor,
) -> tuple[Tensor, Tensor]:
    """Add <bos> and <eos> tokens to a single sequence."""
    bos_command = torch.tensor(
        [POSTSCRIPT_COMMAND_TYPE_TO_NUM["<bos>"]],
        dtype=commands.dtype,
        device=commands.device,
    )
    eos_command = torch.tensor(
        [POSTSCRIPT_COMMAND_TYPE_TO_NUM["<eos>"]],
        dtype=commands.dtype,
        device=commands.device,
    )
    bos_coordinates = torch.zeros(
        (1, coordinates.size(1)),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    eos_coordinates = torch.zeros(
        (1, coordinates.size(1)),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )

    commands = torch.cat([bos_command, commands, eos_command], dim=0)
    coordinates = torch.cat([bos_coordinates, coordinates, eos_coordinates], dim=0)

    return commands, coordinates


def _pad_postscript_outline(
    data: list[tuple[Tensor, Tensor]],
    pad_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pad commands and coordinates to the same length."""
    commands, coordinates = zip(*data, strict=True)

    if pad_size is None:
        pad_size = max(cmd.size(0) for cmd in commands)

    padded_commands = torch.stack(
        [
            torch.nn.functional.pad(
                cmd,
                (0, pad_size - cmd.size(0)),
                value=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
            )
            for cmd in commands
        ],
    )

    padded_coordinates = torch.stack(
        [
            torch.nn.functional.pad(
                coord,
                (0, 0, 0, pad_size - coord.size(0)),
                value=0.0,
            )
            for coord in coordinates
        ],
    )

    return padded_commands, padded_coordinates


class MultiFontPatchedPostScriptCollate:
    """Collate function wrapper for classification dataloaders with patches."""

    def __init__(self, patch_len: int) -> None:
        """Initialize the collate function with required patch length."""
        self.patch_len = patch_len

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor], int, int]],
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Collate function for dataloaders that pads the batch at the patch level."""
        glyph, codepoint, font_index = zip(*batch, strict=True)
        glyph_padded = _pad_patched_postscript_outline(glyph, self.patch_len)
        return glyph_padded, torch.tensor(codepoint), torch.tensor(font_index)


def _pad_patched_postscript_outline(
    data: list[tuple[Tensor, Tensor]],
    patch_len: int,
) -> tuple[Tensor, Tensor]:
    """Pad patched command sequences and coordinates to the same number of patches.

    Each missing patch is replaced by a fully padded patch.

    Args:
        data: List of tuples (commands, coordinates).
        patch_len: Fixed patch size.

    Returns:
        - Padded command sequences of shape [batch_size, max_patches, patch_len].
        - Padded coordinate sequences of shape [batch_size, max_patches, patch_len, 6].

    """
    commands, coordinates = zip(*data, strict=True)
    max_patches = max(cmd.size(0) for cmd in commands)
    batch_size = len(commands)

    padded_commands = torch.full(
        (batch_size, max_patches, patch_len),
        POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
        dtype=torch.int64,
    )
    padded_coordinates = torch.full(
        (batch_size, max_patches, patch_len, 6),
        -1.0,
        dtype=torch.float32,
    )

    for i, (cmd, coord) in enumerate(zip(commands, coordinates, strict=True)):
        num_patches = cmd.size(0)
        padded_commands[i, :num_patches] = cmd
        padded_coordinates[i, :num_patches] = coord

    return padded_commands, padded_coordinates
