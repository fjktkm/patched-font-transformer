"""PyTorch Lightning module for training FontTransformer models."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from torch import Tensor
from torch.nn import Transformer

from patched_font_transformer.models import FontTranslator
from patched_font_transformer.modules.loss import ReconstructionLoss
from patched_font_transformer.modules.mask import autoregressive_translation_mask
from patched_font_transformer.modules.scheduler import WarmupDecayLR
from patched_font_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_MAX_ARGS_LEN,
)
from patched_font_transformer.torchfont.utils.visualizer import save_combined_glyph_plot


class TranslatorLM(pl.LightningModule):
    """A PyTorch Lightning Module for training FontTransformer models."""

    def __init__(
        self,
        num_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        lr: float = 0.001,  # noqa: ARG002
        warmup_steps: int = 250,  # noqa: ARG002
    ) -> None:
        """Initialize the module with the model, loss function, and scheduler settings.

        Args:
            num_layers: Number of encoder and decoder layers.
            emb_size: Embedding size.
            nhead: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
            lr: Learning rate for the optimizer.
            warmup_steps: Number of warm-up steps for the scheduler.

        """
        super().__init__()
        self.save_hyperparameters()

        self.model = FontTranslator(
            num_layers=num_layers,
            emb_size=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        self.loss_fn = ReconstructionLoss(
            ignore_index=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
        )

    def forward(
        self,
        src: tuple[Tensor, Tensor],
        tgt: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Perform a forward pass through the FontTransformer."""
        (
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        ) = autoregressive_translation_mask(src, tgt)
        return self.model(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def training_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        _batch_idx: int,
    ) -> Tensor:
        """Execute a training step."""
        src, tgt = batch

        tgt_commands, tgt_coords = tgt
        tgt_commands_input = tgt_commands[:, :-1]
        tgt_coords_input = tgt_coords[:, :-1]
        tgt_commands_output = tgt_commands[:, 1:]
        tgt_coords_output = tgt_coords[:, 1:]

        tgt_input = (tgt_commands_input, tgt_coords_input)
        tgt_output = (tgt_commands_output, tgt_coords_output)

        predictions = self(src, tgt_input)

        loss = self.loss_fn(predictions, tgt_output)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )
        return loss

    def validation_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        _batch_idx: int,
    ) -> None:
        """Execute a validation step."""
        src, tgt = batch

        tgt_commands, tgt_coords = tgt
        tgt_commands_input = tgt_commands[:, :-1]
        tgt_coords_input = tgt_coords[:, :-1]
        tgt_commands_output = tgt_commands[:, 1:]
        tgt_coords_output = tgt_coords[:, 1:]

        tgt_input = (tgt_commands_input, tgt_coords_input)
        tgt_output = (tgt_commands_output, tgt_coords_output)

        predictions = self(src, tgt_input)

        loss = self.loss_fn(predictions, tgt_output)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

    def test_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        batch_idx: int,
    ) -> None:
        """Execute a test step by decoding the output and comparing with targets."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        src, target = batch

        src_commands, src_coords = src
        target_commands, target_coords = target

        batch_size = src_commands.size(0)

        for i in range(batch_size):
            input_tensor = (src_commands[i], src_coords[i])
            target_tensor = (target_commands[i], target_coords[i])
            output_tensor = self.greedy_decode(input_tensor)
            if self.trainer.log_dir:
                save_combined_glyph_plot(
                    input_tensor,
                    target_tensor,
                    output_tensor,
                    Path(self.trainer.log_dir) / "output",
                    f"output_rank{rank}_{batch_idx}_{i}",
                )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore[attr-defined]
        )
        scheduler: LRSchedulerConfigType = {
            "scheduler": WarmupDecayLR(
                optimizer,
                warmup_steps=self.hparams.warmup_steps,  # type: ignore[attr-defined]
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def greedy_decode(
        self,
        src: tuple[Tensor, Tensor],
        max_len: int = 256,
    ) -> tuple[Tensor, Tensor]:
        """Greedy decode function."""
        command_tensor, coords_tensor = src

        command_tensor = command_tensor.unsqueeze(0)
        coords_tensor = coords_tensor.unsqueeze(0)

        src_padding_mask = (
            command_tensor == POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"]
        ).to(
            self.device,
        )

        memory = self.model.encode(
            src=(command_tensor, coords_tensor),
            src_padding_mask=src_padding_mask,
        )

        ys_commands = torch.ones(1, 1, dtype=torch.long, device=self.device).fill_(
            POSTSCRIPT_COMMAND_TYPE_TO_NUM["<bos>"],
        )

        ys_coords = torch.zeros(1, 1, POSTSCRIPT_MAX_ARGS_LEN, device=self.device)

        for _ in range(max_len - 1):
            tgt_mask = Transformer.generate_square_subsequent_mask(
                ys_commands.size(1),
                dtype=torch.bool,
            )

            decoded_output = self.model.decode(
                tgt=(ys_commands, ys_coords),
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            command_logits, position_output = self.model.generator(
                decoded_output[:, -1],
            )

            next_command = torch.argmax(command_logits.detach(), dim=1)
            next_command = next_command.item()
            next_coords = position_output.detach()

            ys_commands = torch.cat(
                [ys_commands, torch.tensor([[next_command]], device=self.device)],
                dim=1,
            )
            ys_coords = torch.cat(
                [ys_coords, next_coords.unsqueeze(0)],
                dim=1,
            )

            if next_command == POSTSCRIPT_COMMAND_TYPE_TO_NUM["<eos>"]:
                break

        return ys_commands.squeeze(0), ys_coords.squeeze(0)
