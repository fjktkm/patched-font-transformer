#!/bin/sh

BASE_DIR="patched_font_transformer/scripts/classification"
TYPES="style weight"

for type in $TYPES; do
    uv run "$BASE_DIR/$type.py"
done
