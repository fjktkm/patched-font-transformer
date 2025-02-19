"""Main script for training the FontTransformer model."""

import warnings

import pytorch_lightning as pl
import torch
from fontTools.ttLib import TTFont
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from patched_font_transformer.lightning.data_modules import FontPairLDM
from patched_font_transformer.lightning.modules import AutoregressiveTranslatorLM

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")


def main() -> None:
    """Train the FontTranslator model."""
    src_font = TTFont("./fonts/ofl/notosansjp/NotoSansJP[wght].ttf")
    target_font = TTFont("./fonts/ofl/notoserifjp/NotoSerifJP[wght].ttf")

    data_module = FontPairLDM(
        src_font=src_font,
        target_font=target_font,
        batch_size=128,
        split_ratios=(1, 0.1, 0.01),
        seed=33114113,
    )

    model = AutoregressiveTranslatorLM(
        num_layers=3,
        emb_size=256,
        nhead=8,
        dim_feedforward=512,
        lr=0.001,
        warmup_steps=256,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="font-translator-{epoch:02d}-{val_loss:.2f}",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=64)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=2048,
        devices="auto",
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
