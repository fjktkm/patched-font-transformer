"""DataModule for managing FontPair datasets."""

from fontTools.ttLib import TTFont
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from patched_font_transformer.modules.collate_fn import FontPairPostScriptCollate
from patched_font_transformer.torchfont.datasets.font_pair import FontPairDataset
from patched_font_transformer.torchfont.transforms import (
    Compose,
    DecomposeSegment,
    NormalizeSegment,
    PostScriptSegmentToTensor,
    QuadToCubic,
)


class FontPairLDM(LightningDataModule):
    """DataModule for managing FontPair datasets."""

    def __init__(
        self,
        src_font: TTFont,
        target_font: TTFont,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = None,
        pad_size: int | None = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.src_font = src_font
        self.target_font = target_font
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.seed = seed
        self.pad_size = pad_size

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up datasets for train, val, and test splits."""
        transform = Compose(
            [
                DecomposeSegment(),
                NormalizeSegment(),
                QuadToCubic(),
                PostScriptSegmentToTensor("zeros"),
            ],
        )

        self.train_dataset = FontPairDataset(
            self.src_font,
            self.target_font,
            split="train",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.val_dataset = FontPairDataset(
            self.src_font,
            self.target_font,
            split="valid",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.test_dataset = FontPairDataset(
            self.src_font,
            self.target_font,
            split="test",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=FontPairPostScriptCollate(pad_size=self.pad_size),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=FontPairPostScriptCollate(pad_size=self.pad_size),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=FontPairPostScriptCollate(pad_size=self.pad_size),
            pin_memory=True,
        )
