#!/usr/bin/env bash
# =============================================================================
# convert_to_yolo.sh  (v3 — correct coordinate-space handling)
#
# Converts a dataset in the SpaceNet/Planet tile structure to YOLO format.
#
# Key fix (v3): label subdirectories use DIFFERENT coordinate spaces:
#
#   labels/           → WGS84 lon/lat  (geographic)
#   labels_match/     → WGS84 lon/lat  (geographic, temporally matched subset)
#   labels_match_pix/ → pixel coords   (x: 0→img_width, y: 0→img_height)
#
# The script detects coordinate space automatically from the values and applies
# the correct normalisation for each:
#   - Geographic → normalise against the tile's Web-Mercator bbox
#   - Pixel      → normalise against the image pixel dimensions
#
# Tile bbox priority (for geographic labels):
#   1. rasterio geotransform  – reads the GeoTIFF directly (most accurate)
#   2. Tile-name Web-Mercator – decodes _TX_TY_ZOOM from the tile ID
#   3. GeoJSON feature extent – last resort; only covers annotated buildings
#
# Output layout (YOLO):
#   <OUTPUT_ROOT>/
#     images/train/  *.tif  (symlinked or copied)
#     labels/train/  *.txt
#     dataset.yaml
#
# Dependencies: bash, python3 stdlib; rasterio optional but recommended
#
# Usage: bash convert_to_yolo.sh /path/to/dataset /path/to/yolo_output
# =============================================================================

set -euo pipefail

DATASET_ROOT="${1:-.}"
OUTPUT_ROOT="${2:-./yolo_dataset}"
SPLIT="train"
CLASS_ID=0
COPY_IMAGES=true

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 is required."

PYTHON_CONVERTER=$(cat <<'PYEOF'
import json, sys, os, math, re

# ── Coordinate-space detection ─────────────────────────────────────────────────

def detect_coord_space(geojson_path):
    """
    Return 'pixel' or 'geo' by inspecting coordinate magnitudes.

    Geographic coordinates: lon in [-180, 180], lat in [-90, 90]
    Pixel coordinates:      always positive, typically in [0, ~10000]

    We use a simple heuristic: if ALL x values are in [-180, 180] AND
    ALL y values are in [-90, 90], treat as geographic. Otherwise pixel.
    """
    with open(geojson_path) as f:
        fc = json.load(f)
    features = fc.get("features", [])
    if not features:
        return "geo"  # empty — doesn't matter, no coords to convert

    xs, ys = [], []
    for feat in features[:20]:  # sample first 20 features is enough
        geom = feat.get("geometry", {})
        def collect(c):
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for item in c: collect(item)
        collect(geom.get("coordinates", []))

    if not xs:
        return "geo"

    if all(-180 <= x <= 180 for x in xs) and all(-90 <= y <= 90 for y in ys):
        return "geo"
    return "pixel"

# ── Bbox strategies (for geographic labels) ────────────────────────────────────

def bbox_from_rasterio(tif_path):
    """Read exact geotransform from the GeoTIFF."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            b = src.bounds
            return b.left, b.bottom, b.right, b.top
    except Exception:
        return None

def bbox_from_tile_name(tile_id):
    """
    Decode Web-Mercator tile indices from the tile ID suffix _TX_TY_ZOOM.
    Returns the mathematically exact lon/lat extent of the full image tile.
    """
    m = re.search(r'_(\d+)_(\d+)_(\d+)$', tile_id)
    if not m:
        return None
    tx, ty, zoom = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n = 2 ** zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max

def bbox_from_geojson(geojson_path):
    """Fallback: bounding box of all feature coordinates."""
    with open(geojson_path) as f:
        fc = json.load(f)
    lons, lats = [], []
    for feat in fc.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        def collect(c):
            if isinstance(c[0], (int, float)):
                lons.append(c[0]); lats.append(c[1])
            else:
                for item in c: collect(item)
        collect(geom["coordinates"])
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)

# ── Image dimensions (for pixel labels) ───────────────────────────────────────

def image_dimensions(tif_path):
    """Return (width, height) in pixels."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            return src.width, src.height
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(tif_path) as img:
            return img.size  # (width, height)
    except Exception:
        pass
    return None, None

# ── YOLO conversion ────────────────────────────────────────────────────────────

def polygon_to_yolo_geo(coords_ring, tile_bbox):
    """Normalise a geographic polygon ring against the tile lon/lat bbox."""
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    tile_w = max_lon - min_lon
    tile_h = max_lat - min_lat
    if tile_w == 0 or tile_h == 0:
        return None
    lons = [c[0] for c in coords_ring]
    lats = [c[1] for c in coords_ring]
    bx_min, bx_max = min(lons), max(lons)
    by_min, by_max = min(lats), max(lats)
    x_center = ((bx_min + bx_max) / 2 - min_lon) / tile_w
    y_center = (max_lat - (by_min + by_max) / 2) / tile_h  # flip lat→row
    width    = (bx_max - bx_min) / tile_w
    height   = (by_max - by_min) / tile_h
    return (max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width)),
            max(0.0, min(1.0, height)))

def polygon_to_yolo_pixel(coords_ring, img_w, img_h):
    """Normalise a pixel-space polygon ring against image dimensions."""
    if img_w is None or img_h is None or img_w == 0 or img_h == 0:
        return None
    xs = [c[0] for c in coords_ring]
    ys = [c[1] for c in coords_ring]
    bx_min, bx_max = min(xs), max(xs)
    by_min, by_max = min(ys), max(ys)
    x_center = ((bx_min + bx_max) / 2) / img_w
    y_center = ((by_min + by_max) / 2) / img_h
    width    = (bx_max - bx_min) / img_w
    height   = (by_max - by_min) / img_h
    return (max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width)),
            max(0.0, min(1.0, height)))

def geojson_to_yolo(geojson_path, coord_space, class_id, out_path,
                    tile_bbox=None, img_w=None, img_h=None):
    with open(geojson_path) as f:
        fc = json.load(f)
    lines = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [])
        for poly in polys:
            ring = poly[0]
            if coord_space == "pixel":
                result = polygon_to_yolo_pixel(ring, img_w, img_h)
            else:
                result = polygon_to_yolo_geo(ring, tile_bbox)
            if result is None:
                continue
            x_c, y_c, w, h = result
            if w < 1e-6 or h < 1e-6:
                continue
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "detect":
        # argv: detect <geojson_path>
        print(detect_coord_space(sys.argv[2]))

    elif mode == "imgdims":
        # argv: imgdims <tif_path>
        w, h = image_dimensions(sys.argv[2])
        if w is None:
            sys.exit(1)
        print(w, h)

    elif mode == "bbox":
        # argv: bbox <tif_path> <tile_id> <geojson_path>
        tif_path, tile_id, geojson_path = sys.argv[2], sys.argv[3], sys.argv[4]
        for strategy, fn, args in [
            ("rasterio", bbox_from_rasterio, [tif_path]),
            ("tilename", bbox_from_tile_name, [tile_id]),
            ("geojson",  bbox_from_geojson,   [geojson_path]),
        ]:
            result = fn(*args)
            if result:
                print(strategy, *result)
                sys.exit(0)
        sys.exit(1)

    elif mode == "convert":
        # argv: convert <geojson_path> <coord_space> <class_id> <out_path>
        #              [geo: <min_lon> <min_lat> <max_lon> <max_lat>]
        #              [pixel: <img_w> <img_h>]
        gj_path     = sys.argv[2]
        coord_space = sys.argv[3]   # "geo" or "pixel"
        cls_id      = int(sys.argv[4])
        out_path    = sys.argv[5]
        if coord_space == "geo":
            tile_bbox = tuple(float(x) for x in sys.argv[6:10])
            n = geojson_to_yolo(gj_path, "geo", cls_id, out_path, tile_bbox=tile_bbox)
        else:
            img_w, img_h = int(sys.argv[6]), int(sys.argv[7])
            n = geojson_to_yolo(gj_path, "pixel", cls_id, out_path, img_w=img_w, img_h=img_h)
        print(n)
PYEOF
)

PY_SCRIPT=$(mktemp /tmp/geojson2yolo_XXXXXX.py)
trap 'rm -f "$PY_SCRIPT"' EXIT
echo "$PYTHON_CONVERTER" > "$PY_SCRIPT"

IMG_OUT="$OUTPUT_ROOT/images/$SPLIT"
LBL_OUT="$OUTPUT_ROOT/labels/$SPLIT"
mkdir -p "$IMG_OUT" "$LBL_OUT"

TRAIN_DIR="$DATASET_ROOT/$SPLIT"
[[ -d "$TRAIN_DIR" ]] || die "Split directory not found: $TRAIN_DIR"

total_tiles=0; total_images=0; total_labels=0; skipped=0
count_geo=0; count_pixel=0
bbox_rasterio=0; bbox_tilename=0; bbox_geojson=0

for tile_dir in "$TRAIN_DIR"/*/; do
    [[ -d "$tile_dir" ]] || continue
    tile_id=$(basename "$tile_dir")
    log "Processing tile: $tile_id"
    (( total_tiles++ )) || true

    images_dir="$tile_dir/images"
    [[ -d "$images_dir" ]] || { warn "No images/ in $tile_dir – skipping"; continue; }

    for img_path in "$images_dir"/*.tif; do
        [[ -f "$img_path" ]] || continue
        img_name=$(basename "$img_path")
        stem="${img_name%.tif}"
        geojson_stem="${stem}_Buildings.geojson"

        # Find best available GeoJSON (pix > match > labels)
        geojson_path=""
        geojson_source=""
        for ldir in "labels_match_pix" "labels_match" "labels"; do
            candidate="$tile_dir/$ldir/$geojson_stem"
            if [[ -f "$candidate" ]]; then
                geojson_path="$candidate"
                geojson_source="$ldir"
                break
            fi
        done

        if [[ -z "$geojson_path" ]]; then
            warn "No Buildings GeoJSON found for $img_name – writing empty label"
            touch "$LBL_OUT/${stem}.txt"
            continue
        fi

        # Symlink / copy image
        dest_img="$IMG_OUT/$img_name"
        if [[ ! -e "$dest_img" ]]; then
            if $COPY_IMAGES; then cp "$img_path" "$dest_img"
            else ln -s "$(realpath "$img_path")" "$dest_img"; fi
        fi
        (( total_images++ )) || true

        # Detect coordinate space of this GeoJSON
        coord_space=$(python3 "$PY_SCRIPT" detect "$geojson_path")
        out_txt="$LBL_OUT/${stem}.txt"

        if [[ "$coord_space" == "pixel" ]]; then
            # Pixel-space labels: normalise by image dimensions
            dims=$(python3 "$PY_SCRIPT" imgdims "$img_path" 2>/dev/null) || {
                warn "Could not read image dimensions for $img_name – skipping"
                touch "$out_txt"
                (( skipped++ )) || true
                continue
            }
            img_w=$(echo "$dims" | awk '{print $1}')
            img_h=$(echo "$dims" | awk '{print $2}')
            n=$(python3 "$PY_SCRIPT" convert "$geojson_path" pixel "$CLASS_ID" "$out_txt" "$img_w" "$img_h")
            log "  $img_name → ${stem}.txt  ($n annotations)  [pixel coords, ${img_w}x${img_h}, src: $geojson_source]"
            (( count_pixel++ )) || true

        else
            # Geographic labels: normalise by tile lon/lat bbox
            bbox_output=$(python3 "$PY_SCRIPT" bbox "$img_path" "$tile_id" "$geojson_path" 2>/dev/null) || {
                warn "Could not determine bbox for $tile_id/$img_name – skipping"
                touch "$out_txt"
                (( skipped++ )) || true
                continue
            }
            bbox_strategy=$(echo "$bbox_output" | awk '{print $1}')
            bbox=$(echo "$bbox_output" | cut -d' ' -f2-)
            case "$bbox_strategy" in
                rasterio) (( bbox_rasterio++ )) || true ;;
                tilename) (( bbox_tilename++  )) || true ;;
                geojson)
                    (( bbox_geojson++ )) || true
                    warn "GeoJSON extent fallback for $tile_id — boxes near edges may be inaccurate"
                    ;;
            esac
            n=$(python3 "$PY_SCRIPT" convert "$geojson_path" geo "$CLASS_ID" "$out_txt" $bbox)
            log "  $img_name → ${stem}.txt  ($n annotations)  [geo coords, bbox: $bbox_strategy, src: $geojson_source]"
            (( count_geo++ )) || true
        fi

        (( total_labels += n )) || true
    done
done

YAML="$OUTPUT_ROOT/dataset.yaml"
cat > "$YAML" <<YAML
# YOLO dataset configuration
path: $(realpath "$OUTPUT_ROOT")
train: images/train
val:   images/train   # update with a separate val split when available

nc: 1
names:
  0: building
YAML

log "────────────────────────────────────────────────────────"
log "Conversion complete."
log "  Tiles processed      : $total_tiles"
log "  Images processed     : $total_images"
log "  Total annotations    : $total_labels"
log "  Coord space breakdown:"
log "    pixel (labels_match_pix) : $count_pixel"
log "    geo   (labels/match)     : $count_geo"
log "  Geo bbox strategy:"
log "    rasterio (exact)         : $bbox_rasterio"
log "    tile-name Web-Mercator   : $bbox_tilename"
log "    GeoJSON extent (fallback): $bbox_geojson"
[[ $skipped       -gt 0 ]] && warn "  Skipped              : $skipped"
[[ $bbox_geojson  -gt 0 ]] && warn "  Install rasterio for accurate geo bbox: pip install rasterio"
log "  Output : $OUTPUT_ROOT"
log "────────────────────────────────────────────────────────"
