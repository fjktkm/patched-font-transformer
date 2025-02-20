"""Transforms for processing glyphs."""

from collections.abc import Callable

from fontTools.ttLib import TTFont
from torch import Tensor

from patched_font_transformer.torchfont.io.font import (
    AtomicPostScriptOutline,
    AtomicSegmentOutline,
    SegmentOutline,
)
from patched_font_transformer.torchfont.transforms import functional as F


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms: list[Callable]) -> None:
        """Initialize the transform."""
        self.transforms = transforms

    def __call__(
        self,
        glyph: SegmentOutline,
        font: TTFont,
    ) -> SegmentOutline | tuple[Tensor, Tensor]:
        """Apply the transformations to the glyph."""
        for t in self.transforms:
            glyph = t(glyph, font)
        return glyph

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class DecomposeSegment:
    """Decompose complex Bezier segments in glyphs."""

    def __call__(self, glyph: SegmentOutline, _: TTFont) -> AtomicSegmentOutline:
        """Decompose the glyph into simpler segments."""
        return F.decompose_segment(glyph)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class QuadToCubic:
    """Convert quadratic Bézier curves to cubic Bézier curves."""

    def __call__(
        self,
        glyph: AtomicSegmentOutline,
        _: TTFont,
    ) -> AtomicPostScriptOutline:
        """Convert all `qCurvedTo` commands to `curveTo` commands."""
        return F.quad_to_cubic(glyph)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class NormalizeSegment:
    """Normalize the glyph path to fit within a standard coordinate range."""

    def __call__(self, glyph: SegmentOutline, font: TTFont) -> SegmentOutline:
        """Normalize the glyph using the font's units per em."""
        return F.normalize_segment(glyph, font)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class PostScriptSegmentToTensor:
    """Convert a glyph path to a PyTorch tensor."""

    def __init__(self, method: F.PadMethod) -> None:
        """Initialize the transform."""
        self.method: F.PadMethod = method

    def __call__(
        self,
        glyph: AtomicPostScriptOutline,
        _: TTFont,
    ) -> tuple[Tensor, Tensor]:
        """Convert the glyph to separate tensors for commands and arguments."""
        return F.postscript_segment_to_tensor(glyph, self.method)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return f"{self.__class__.__name__}(method='{self.method}')"


class SplitIntoPatches:
    """Split the tensor into patches."""

    def __init__(self, patch_size: int) -> None:
        """Initialize the transform."""
        self.patch_size = patch_size

    def __call__(
        self,
        tensor: tuple[Tensor, Tensor],
        _: TTFont,
    ) -> tuple[Tensor, Tensor]:
        """Split the tensor into patches."""
        return F.split_into_patches(tensor, self.patch_size)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return f"{self.__class__.__name__}(patch_size={self.patch_size})"


class MergePatches:
    """Merge the patches into a single tensor."""

    def __call__(
        self,
        tensor: tuple[Tensor, Tensor],
        _: TTFont,
    ) -> tuple[Tensor, Tensor]:
        """Merge the patches into a single tensor."""
        return F.merge_patches(tensor)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class TensorToSegment:
    """Convert a PyTorch tensor back to a glyph path."""

    def __call__(
        self,
        tensor: tuple[Tensor, Tensor],
        _: TTFont,
    ) -> AtomicPostScriptOutline:
        """Convert the tensors back to a glyph."""
        return F.tensor_to_segment(tensor)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"
