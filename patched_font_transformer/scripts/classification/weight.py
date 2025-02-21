"""Main script for training the FontClassifier model."""

import argparse
import warnings

import pytorch_lightning as pl
import torch
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from patched_font_transformer.lightning.data_modules import MultiFontLDM
from patched_font_transformer.lightning.modules import ClassifierLM

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")


def weight_classifier(
    patch_size: int = 4,
) -> None:
    """Train a FontClassifier model for weight classification."""
    variable_fonts = [
        TTFont("./fonts/ofl/notosansjp/NotoSansJP[wght].ttf"),
        TTFont("./fonts/ofl/notoserifjp/NotoSerifJP[wght].ttf"),
        TTFont("./fonts/ofl/mplus1/MPLUS1[wght].ttf"),
    ]
    weight_values = [300, 400, 500, 700]
    instantiated_variable_fonts = [
        instantiateVariableFont(
            variable_font,
            {"wght": weight},
            inplace=False,
            updateFontNames=True,
        )
        for variable_font in variable_fonts
        for weight in weight_values
    ]
    fonts = [
        *instantiated_variable_fonts,
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Light.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Regular.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Medium.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Bold.ttf"),
    ]

    class_labels = [
        f"{font['name'].getBestFamilyName()} {font['name'].getBestSubFamilyName()}"
        for font in fonts
    ]

    data_module = MultiFontLDM(
        fonts=fonts,
        batch_size=512,
        patch_size=patch_size,
        split_ratios=(0.8, 0.1, 0.1),
        seed=33114113,
    )

    model = ClassifierLM(
        num_layers=3,
        emb_size=128,
        patch_len=patch_size,
        nhead=4,
        class_labels=class_labels,
        dim_feedforward=256,
        dropout=0.1,
        lr=0.01,
        warmup_steps=256,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="font-classifier-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=64,
        devices="auto",
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FontClassifier model.")
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1,
        help="Patch size for the classifier (default: 1).",
    )

    args = parser.parse_args()

    weight_classifier(patch_size=args.patch_size)
