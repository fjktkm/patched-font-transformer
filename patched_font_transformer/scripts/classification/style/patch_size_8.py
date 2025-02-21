"""Main script for training the FontClassifier model."""

import warnings

import torch

from patched_font_transformer.scripts.classification.style.style import (
    style_classifier,
)

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    style_classifier(patch_size=8)
