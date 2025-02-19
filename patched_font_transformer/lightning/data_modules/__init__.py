"""Initialize the lightning data modules."""

from patched_font_transformer.lightning.data_modules.font_pair import FontPairLDM
from patched_font_transformer.lightning.data_modules.multi_font import MultiFontLDM

__all__ = [
    "FontPairLDM",
    "MultiFontLDM",
]
