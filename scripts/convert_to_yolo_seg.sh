#!/usr/bin/env bash
# ==============================================================================
# convert_to_yolo_seg.sh
#
# Converts a disaster-damage dataset from custom JSON format to YOLO segmentation
# format (.txt label files with normalized polygon coordinates).
#
# Source layout expected:
#   <INPUT_DIR>/
#     images/   *.png
#     labels/   *.json
#     targets/  *_target.png   (copied as-is; not used for label conversion)
#
# Output layout produced:
#   <OUTPUT_DIR>/
#     images/train/   *.png
#     labels/train/   *.txt
#     targets/train/  *_target.png
#     classes.txt
#
# YOLO segmentation label line format (one object per line):
#   <class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
# All coordinates are normalised to [0, 1] by the image width/height.
#
# Class mapping (damage level → YOLO class id):
#   0  no-damage
#   1  minor-damage
#   2  major-damage
#   3  destroyed
#   4  un-classified   (pre-disaster images have no subtype → class 4)
#
# Usage:
#   bash convert_to_yolo.sh [INPUT_DIR] [OUTPUT_DIR]
#
# Defaults:
#   INPUT_DIR  = ./train
#   OUTPUT_DIR = ./yolo_dataset
# ==============================================================================

set -euo pipefail

INPUT_DIR="${1:-./train}"
OUTPUT_DIR="${2:-./yolo_dataset}"

echo "=================================================="
echo "  Disaster Dataset → YOLO Segmentation Converter"
echo "=================================================="
echo "  Input  : $INPUT_DIR"
echo "  Output : $OUTPUT_DIR"
echo ""

# --------------------------------------------------------------------------
# 1. Create output directory structure
# --------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR/images/train"
mkdir -p "$OUTPUT_DIR/labels/train"
mkdir -p "$OUTPUT_DIR/targets/train"

# --------------------------------------------------------------------------
# 2. Write classes.txt
# --------------------------------------------------------------------------
cat > "$OUTPUT_DIR/classes.txt" <<'EOF'
no-damage
minor-damage
major-damage
destroyed
un-classified
EOF
echo "[1/3] Written $OUTPUT_DIR/classes.txt"

# --------------------------------------------------------------------------
# 3. Copy images
# --------------------------------------------------------------------------
img_count=0
if [ -d "$INPUT_DIR/images" ]; then
    for img in "$INPUT_DIR/images"/*.png; do
        [ -f "$img" ] || continue
        cp "$img" "$OUTPUT_DIR/images/train/"
        img_count=$((img_count + 1))
    done
fi
echo "[2/3] Copied $img_count image(s) → $OUTPUT_DIR/images/train/"

# --------------------------------------------------------------------------
# 4. Copy target masks
# --------------------------------------------------------------------------
tgt_count=0
if [ -d "$INPUT_DIR/targets" ]; then
    for tgt in "$INPUT_DIR/targets"/*_target.png; do
        [ -f "$tgt" ] || continue
        cp "$tgt" "$OUTPUT_DIR/targets/train/"
        tgt_count=$((tgt_count + 1))
    done
fi
echo "[2/3] Copied $tgt_count target mask(s) → $OUTPUT_DIR/targets/train/"

# --------------------------------------------------------------------------
# 5. Convert JSON labels → YOLO .txt via inline Python
# --------------------------------------------------------------------------
json_count=0
if [ -d "$INPUT_DIR/labels" ]; then
    for json_file in "$INPUT_DIR/labels"/*.json; do
        [ -f "$json_file" ] || continue

        # Derive output .txt filename (same stem as the json)
        base=$(basename "$json_file" .json)
        out_txt="$OUTPUT_DIR/labels/train/${base}.txt"

        python3 - "$json_file" "$out_txt" <<'PYTHON'
import sys, json, re

CLASS_MAP = {
    "no-damage":      0,
    "minor-damage":   1,
    "major-damage":   2,
    "destroyed":      3,
    "un-classified":  4,
}
DEFAULT_CLASS = 4   # for pre-disaster (no subtype field)

json_path, out_path = sys.argv[1], sys.argv[2]

with open(json_path) as f:
    data = json.load(f)

meta   = data.get("metadata", {})
width  = meta.get("width",  1024)
height = meta.get("height", 1024)

features_xy = data.get("features", {}).get("xy", [])

lines = []
for feat in features_xy:
    props   = feat.get("properties", {})
    subtype = props.get("subtype", "un-classified")
    class_id = CLASS_MAP.get(subtype, DEFAULT_CLASS)

    wkt = feat.get("wkt", "")
    # Extract all coordinate pairs from the WKT POLYGON string
    coords_str = re.search(r'\(\((.+)\)\)', wkt)
    if not coords_str:
        continue
    pairs = coords_str.group(1).split(",")
    norm_coords = []
    for pair in pairs:
        xy = pair.strip().split()
        if len(xy) != 2:
            continue
        x_norm = float(xy[0]) / width
        y_norm = float(xy[1]) / height
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        norm_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

    if len(norm_coords) >= 6:   # need at least 3 points for a polygon
        lines.append(f"{class_id} " + " ".join(norm_coords))

with open(out_path, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))

print(f"  Wrote {len(lines)} object(s) → {out_path}")
PYTHON

        json_count=$((json_count + 1))
    done
fi
echo "[3/3] Converted $json_count JSON file(s) → $OUTPUT_DIR/labels/train/"

# --------------------------------------------------------------------------
# 6. Summary
# --------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  Conversion complete!"
echo "  Output structure:"
echo "    $OUTPUT_DIR/"
echo "    ├── classes.txt"
echo "    ├── images/train/     ($img_count file(s))"
echo "    ├── labels/train/     ($json_count file(s))"
echo "    └── targets/train/    ($tgt_count file(s))"
echo "=================================================="
