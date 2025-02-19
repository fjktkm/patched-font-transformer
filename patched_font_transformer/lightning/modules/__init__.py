"""Initialize the lightning modules."""

from patched_font_transformer.lightning.modules.classifier import ClassifierLM
from patched_font_transformer.lightning.modules.translator import TranslatorLM

__all__ = [
    "ClassifierLM",
    "TranslatorLM",
]
