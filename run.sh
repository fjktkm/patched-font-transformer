#!/bin/sh

BASE_DIR="patched_font_transformer/scripts/classification"
TYPES="style weight"
PATCH_SIZES="1 2 4 8 16"

for type in $TYPES; do
    for patch_size in $PATCH_SIZES; do
        uv run "$BASE_DIR/$type/patch_size_$patch_size.py"
    done
done
