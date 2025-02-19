"""DataModule for managing FontCollection datasets."""

from fontTools.ttLib import TTFont
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from patched_font_transformer.modules.collate_fn import MultiFontPostScriptCollate
from patched_font_transformer.torchfont.datasets.multi_font import MultiFontDataset
from patched_font_transformer.torchfont.transforms import (
    Compose,
    DecomposeSegment,
    NormalizeSegment,
    PostScriptSegmentToTensor,
    QuadToCubic,
)


class MultiFontLDM(LightningDataModule):
    """DataModule for managing FontCollection datasets."""

    def __init__(
        self,
        fonts: list[TTFont],
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        codepoints: list[int] | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = None,
        pad_size: int | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            fonts: List of fonts to include in the dataset.
            batch_size: Batch size for the dataloaders.
            num_workers: Number of workers for data loading.
            codepoints: List of codepoints to include in the dataset.
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            pad_size: Optional padding size for batching.

        """
        super().__init__()
        self.fonts = fonts
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.codepoints = codepoints
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

        self.train_dataset = MultiFontDataset(
            fonts=self.fonts,
            codepoints=self.codepoints,
            split="train",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.val_dataset = MultiFontDataset(
            fonts=self.fonts,
            codepoints=self.codepoints,
            split="valid",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.test_dataset = MultiFontDataset(
            fonts=self.fonts,
            codepoints=self.codepoints,
            split="test",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )

        self.collate_fn = MultiFontPostScriptCollate(pad_size=self.pad_size)

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
