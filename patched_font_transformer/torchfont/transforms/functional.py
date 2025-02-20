"""Functional transformations for glyphs."""

from typing import Literal

import torch
from fontTools.pens.basePen import (
    decomposeQuadraticSegment,
    decomposeSuperBezierSegment,
)
from fontTools.ttLib import TTFont
from torch import Tensor

from patched_font_transformer.torchfont.io.font import (
    C_ARGS_LEN,
    M_L_ARGS_LEN,
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_NUM_TO_COMMAND_TYPE,
    Q_ARGS_LEN,
    AtomicPostScriptOutline,
    AtomicSegmentOutline,
    Command,
    Point,
    SegmentOutline,
)

PadMethod = Literal["trajectory", "zeros"]

ZERO_POINT = (-1.0, -1.0)


def _handle_curve_to(points: tuple[Point, ...]) -> AtomicSegmentOutline:
    """Handle decomposition for 'curveTo' commands."""
    n = len(points)

    if n < M_L_ARGS_LEN:
        return []
    if n == M_L_ARGS_LEN:
        return [("lineTo", (points[0],))]
    if n == Q_ARGS_LEN:
        return [("qCurveTo", (points[0], points[1]))]
    if n == C_ARGS_LEN:
        return [("curveTo", (points[0], points[1], points[2]))]

    decomposed_segments = decomposeSuperBezierSegment(points)
    return [("curveTo", segment) for segment in decomposed_segments]


def _handle_qcurve_to(points: tuple[Point, ...]) -> AtomicSegmentOutline:
    """Handle decomposition for 'qCurveTo' commands."""
    path = []

    if points[-1] is None:
        last_off_curve = points[-2]
        first_off_curve = points[0]
        implicit_on_curve = (
            (last_off_curve[0] + first_off_curve[0]) / 2,
            (last_off_curve[1] + first_off_curve[1]) / 2,
        )

        path.append(("moveTo", (implicit_on_curve,)))

        points = points[:-1] + (implicit_on_curve,)

    n = len(points)

    if n < M_L_ARGS_LEN:
        return path
    if n == M_L_ARGS_LEN:
        path.append(("lineTo", points))
        return path
    if n == Q_ARGS_LEN:
        path.append(("qCurveTo", points))
        return path

    decomposed_segments = decomposeQuadraticSegment(points)
    path.extend([("qCurveTo", segment) for segment in decomposed_segments])

    return path


def decompose_segment(glyph: SegmentOutline) -> AtomicSegmentOutline:
    """Decompose complex Bezier segments in a glyph into simpler segments.

    Args:
        glyph: A list of commands and their points,
        e.g., [('curveTo', [(x1, y1), (x2, y2), (x, y)])].

    Returns:
        A list of decomposed commands where all 'curveTo' and 'qCurveTo' segments
        are split into their atomic components.

    """
    decomposed_glyph = []

    for command, points in glyph:
        if command == "curveTo":
            decomposed_glyph.extend(_handle_curve_to(points))
        elif command == "qCurveTo":
            decomposed_glyph.extend(_handle_qcurve_to(points))
        else:
            decomposed_glyph.append((command, points))

    return decomposed_glyph


def quad_to_cubic(glyph: AtomicSegmentOutline) -> AtomicPostScriptOutline:
    """Convert quadratic B-spline curves (qCurveTo) to cubic Bézier curves (curveTo).

    Args:
        glyph: A list of commands representing the glyph path.

    Returns:
        A glyph where all `qCurveTo` commands are converted to `curveTo` commands.

    """
    converted_glyph = []
    current_point = ZERO_POINT
    path_start_point = ZERO_POINT

    for command, points in glyph:
        if command == "qCurveTo" and len(points) == Q_ARGS_LEN:
            control_point, end_point = points
            cp1 = (
                current_point[0] + 2 / 3 * (control_point[0] - current_point[0]),
                current_point[1] + 2 / 3 * (control_point[1] - current_point[1]),
            )
            cp2 = (
                end_point[0] + 2 / 3 * (control_point[0] - end_point[0]),
                end_point[1] + 2 / 3 * (control_point[1] - end_point[1]),
            )
            converted_glyph.append(("curveTo", (cp1, cp2, end_point)))
            current_point = end_point
        elif command == "moveTo" and len(points) == M_L_ARGS_LEN:
            path_start_point = points[-1]
            current_point = path_start_point
            converted_glyph.append((command, points))
        elif command == "closePath":
            converted_glyph.append((command, ()))
            current_point = path_start_point
        else:
            converted_glyph.append((command, points))
            if points:
                current_point = points[-1]

    return converted_glyph


def normalize_segment(glyph: SegmentOutline, font: TTFont) -> SegmentOutline:
    """Normalize the glyph path based on the font's unitsPerEm value.

    Args:
        glyph: A list of commands representing the glyph path.
        font: A `TTFont` object representing the font.

    Returns:
        A normalized glyph where all coordinates are divided by unitsPerEm.

    """
    normalized_glyph = []
    upem = font["head"].unitsPerEm  # type: ignore[attr-defined]

    for command, points in glyph:
        normalized_points = tuple((x / upem, y / upem) for x, y in points)
        normalized_glyph.append((command, normalized_points))

    return normalized_glyph


def _pad_postscript_with_trajectory(
    command: Command,
    points: tuple[Point, ...],
    current_point: Point,
    start_point: Point,
) -> tuple[tuple[Point, Point, Point], Point, Point]:
    """Pad points using trajectory-based interpolation."""
    if len(points) == M_L_ARGS_LEN:
        padded_points = (current_point, points[0], points[0])
        if command == "moveTo":
            start_point = points[0]
    elif len(points) == Q_ARGS_LEN:
        padded_points = (points[0], points[0], points[1])
    elif len(points) == C_ARGS_LEN:
        padded_points = points
    else:
        padded_points = (current_point, start_point, start_point)
    return padded_points, padded_points[-1], start_point


def _pad_postscript_with_zeros(points: tuple[Point, ...]) -> tuple[Point, Point, Point]:
    """Pad points using zero-based padding."""
    if len(points) == M_L_ARGS_LEN:
        return ZERO_POINT, ZERO_POINT, *points
    if len(points) == Q_ARGS_LEN:
        return ZERO_POINT, *points
    if len(points) == C_ARGS_LEN:
        return points
    return ZERO_POINT, ZERO_POINT, ZERO_POINT


def postscript_segment_to_tensor(
    glyph: AtomicPostScriptOutline,
    method: PadMethod,
) -> tuple[Tensor, Tensor]:
    """Pad the glyph path and convert it to tensors based on the specified method.

    Args:
        glyph: A list of commands representing the glyph path.
        method: The padding method to use. Can be "trajectory" or "zeros".

    Returns:
        A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the command arguments as floats.

    """
    command_types = []
    args = []
    current_point = ZERO_POINT
    start_point = ZERO_POINT

    for command, points in glyph:
        if method == "trajectory":
            (
                padded_points,
                current_point,
                start_point,
            ) = _pad_postscript_with_trajectory(
                command,
                points,
                current_point,
                start_point,
            )
        elif method == "zeros":
            padded_points = _pad_postscript_with_zeros(points)

        command_types.append(
            POSTSCRIPT_COMMAND_TYPE_TO_NUM.get(
                command,
                POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
            ),
        )
        args.append([coord for point in padded_points for coord in point])

    command_type_tensor = torch.tensor(command_types, dtype=torch.int64)
    args_tensor = torch.tensor(args, dtype=torch.float32).view(-1, 6)

    return command_type_tensor, args_tensor


def split_into_patches(
    tensor: tuple[Tensor, Tensor],
    patch_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split command sequences and arguments into patches of length `patch_len`.

    Args:
        tensor: A tuple of two tensors:
            - Command tensor of shape [seq_len]
            - Argument tensor of shape [seq_len, coord_dim] (coord_dim is always 6)
        patch_len (int): Length of each patch.

    Returns:
        Tuple[Tensor, Tensor]:
            - Command tensor of shape [num_patches, patch_len]
            - Argument tensor of shape [num_patches, patch_len, coord_dim].

    """
    command_type_tensor, args_tensor = tensor
    seq_len, coord_dim = args_tensor.shape

    pad_size = (patch_len - seq_len % patch_len) % patch_len

    command_type_tensor = torch.nn.functional.pad(
        command_type_tensor,
        (0, pad_size),
        value=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )
    args_tensor = torch.nn.functional.pad(args_tensor, (0, 0, 0, pad_size), value=-1)

    num_patches = (seq_len + pad_size) // patch_len
    padded_commands = command_type_tensor.view(num_patches, patch_len)
    padded_args = args_tensor.view(num_patches, patch_len, coord_dim)

    return padded_commands, padded_args


def merge_patches(
    patched_tensor: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    """Merge patched command sequences and arguments back to the original shape.

    Args:
        patched_tensor: A tuple of two tensors:
            - Command tensor of shape [num_patches, patch_len]
            - Argument tensor of shape [num_patches, patch_len, coord_dim]

    Returns:
        Tuple[Tensor, Tensor]:
            - Command tensor of shape [seq_len]
            - Argument tensor of shape [seq_len, coord_dim]

    """
    patched_commands, patched_args = patched_tensor

    merged_commands = patched_commands.view(-1)
    merged_args = patched_args.view(-1, 6)

    valid_indices = merged_commands != POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"]
    merged_commands = merged_commands[valid_indices]
    merged_args = merged_args[valid_indices]

    return merged_commands, merged_args


def tensor_to_segment(tensor: tuple[Tensor, Tensor]) -> AtomicPostScriptOutline:
    """Convert separate tensors back to a glyph path.

    Args:
        tensor: A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the flattened coordinates as a 2D tensor of shape.

    Returns:
        A list of commands representing the glyph path.

    """
    command_type_tensor, args_tensor = tensor
    glyph = []

    for command_type, coords in zip(
        command_type_tensor.tolist(),
        args_tensor.tolist(),
        strict=True,
    ):
        command = POSTSCRIPT_NUM_TO_COMMAND_TYPE.get(command_type)

        if command in ["moveTo", "lineTo"]:
            points = ((coords[4], coords[5]),)
        elif command == "curveTo":
            points = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
            )
        elif command == "closePath":
            points = ()
        else:
            continue

        glyph.append((command, tuple(points)))

    return glyph
