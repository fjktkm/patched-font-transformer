#!/bin/sh

BASE_DIR="patched_font_transformer/scripts/classification"
TYPES="style weight"

for type in $TYPES; do
    for patch_size in $(seq 1 16); do
        uv run "$BASE_DIR/$type.py" --patch_size "$patch_size"
    done
done
