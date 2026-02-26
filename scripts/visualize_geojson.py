"""
visualize_geojson.py
--------------------
Display building footprints from *_Buildings.geojson overlaid on the
matching GeoTIFF satellite image, using the original (pre-YOLO) dataset layout.

Dataset layout expected
-----------------------
  <dataset_root>/
    <split>/                    e.g. train/
      <tile_id>/                e.g. L15-0331E-1257N_1327_3160_13/
        images/
          global_monthly_<date>_mosaic_<tile_id>.tif
        images_masked/
          global_monthly_<date>_mosaic_<tile_id>.tif
        labels/
          global_monthly_<date>_mosaic_<tile_id>_Buildings.geojson
          global_monthly_<date>_mosaic_<tile_id>_UDM.geojson
        labels_match/
          global_monthly_<date>_mosaic_<tile_id>_Buildings.geojson
        labels_match_pix/
          global_monthly_<date>_mosaic_<tile_id>_Buildings.geojson

Coordinate handling
-------------------
Buildings GeoJSON coordinates are WGS84 (lon/lat, EPSG:4326 / CRS84).

To project them onto image pixels the script uses one of three strategies,
in order of preference:

  1. rasterio  – reads the GeoTIFF's embedded geotransform.  Most accurate.
  2. Tile-name heuristic  – decodes the Web-Mercator tile indices embedded in
     the tile ID (e.g. L15-0331E-1257N_1327_3160_13 → zoom=13, tx=1327, ty=3160)
     and computes the tile's exact lon/lat bounding box.
  3. GeoJSON extent  – derives the bounding box from the union of all feature
     coordinates in the Buildings file.

Usage
-----
  # Auto-pick the first image in the dataset:
  python visualize_geojson.py --dataset /path/to/dataset

  # Pick a specific tile and image stem:
  python visualize_geojson.py --dataset /path/to/dataset \\
      --tile L15-0331E-1257N_1327_3160_13 \\
      --image global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif

  # Use images_masked/ instead of images/:
  python visualize_geojson.py --dataset /path/to/dataset --masked

  # Use labels_match_pix/ (pixel-aligned polygons) instead of labels/:
  python visualize_geojson.py --dataset /path/to/dataset --pix-labels

  # Limit polygons drawn (useful for very dense tiles):
  python visualize_geojson.py --dataset /path/to/dataset --max-features 300

  # Draw full polygon outlines instead of bounding boxes:
  python visualize_geojson.py --dataset /path/to/dataset --draw-polygons

  # Save to file (no interactive window):
  python visualize_geojson.py --dataset /path/to/dataset --save out.png --no-gui

  # Point directly at files (no dataset layout required):
  python visualize_geojson.py \\
      --image-path /abs/path/to/image.tif \\
      --geojson-path /abs/path/to/label_Buildings.geojson
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ── Optional heavy dependencies ────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from matplotlib.path import Path as MplPath
    import matplotlib.patheffects as pe
    from PIL import Image
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n"
             "Install with: pip install numpy matplotlib Pillow")


# ══════════════════════════════════════════════════════════════════════════════
# Geo utilities
# ══════════════════════════════════════════════════════════════════════════════

def tile_bounds_from_name(tile_id: str) -> Optional[tuple[float, float, float, float]]:
    """
    Parse a Planet/SpaceNet tile ID of the form:
        L15-0331E-1257N_1327_3160_13
    and return (lon_min, lat_min, lon_max, lat_max) using Web-Mercator maths.

    The _TX_TY_ZOOM suffix encodes the Slippy-map tile indices.
    """
    m = re.search(r"_(\d+)_(\d+)_(\d+)$", tile_id)
    if not m:
        return None
    tx, ty, zoom = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n = 2 ** zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


def geojson_extent(geojson_path: Path) -> Optional[tuple[float, float, float, float]]:
    """Return (lon_min, lat_min, lon_max, lat_max) from all feature coordinates."""
    with open(geojson_path) as f:
        fc = json.load(f)
    lons, lats = [], []
    for feat in fc.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        def collect(coords):
            if isinstance(coords[0], (int, float)):
                lons.append(coords[0]); lats.append(coords[1])
            else:
                for c in coords: collect(c)
        collect(geom["coordinates"])
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def rasterio_geo_to_pixel(transform, lon: float, lat: float) -> tuple[float, float]:
    """Convert lon/lat → fractional pixel (col, row) using an Affine transform."""
    # transform: rasterio Affine (col_offset, x_res, 0, row_offset, 0, y_res)
    col = (lon - transform.c) / transform.a
    row = (lat - transform.f) / transform.e
    return col, row


def linear_geo_to_pixel(lon: float, lat: float,
                         bbox: tuple[float, float, float, float],
                         img_w: int, img_h: int) -> tuple[float, float]:
    """
    Convert lon/lat → pixel (col, row) using a linear map onto the image.
    bbox = (lon_min, lat_min, lon_max, lat_max).
    lat increases upward, row increases downward → flip.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    col = (lon - lon_min) / (lon_max - lon_min) * img_w
    row = (lat_max - lat) / (lat_max - lat_min) * img_h
    return col, row


# ══════════════════════════════════════════════════════════════════════════════
# Image loading
# ══════════════════════════════════════════════════════════════════════════════

def load_image(image_path: Path):
    """
    Load a (possibly multi-band) GeoTIFF as an HxWx3 uint8 RGB array.
    Returns (array, transform_or_None).
    """
    if HAS_RASTERIO:
        with rasterio.open(image_path) as src:
            n = src.count
            bands = list(range(1, min(n, 3) + 1))
            data = src.read(bands).astype(np.float32)   # (C, H, W)
            data = np.moveaxis(data, 0, -1)              # (H, W, C)
            # 2–98 percentile stretch per channel
            for c in range(data.shape[2]):
                p2, p98 = np.percentile(data[..., c], (2, 98))
                if p98 > p2:
                    data[..., c] = np.clip((data[..., c] - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    data[..., c] = np.clip(data[..., c], 0, 255)
            data = data.astype(np.uint8)
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            return data, src.transform
    else:
        img = Image.open(image_path).convert("RGB")
        return np.array(img), None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset path resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_paths(args) -> tuple[Path, Path, str]:
    """Return (image_path, geojson_path, tile_id)."""

    if args.image_path and args.geojson_path:
        return Path(args.image_path), Path(args.geojson_path), ""

    dataset  = Path(args.dataset)
    split    = args.split
    img_dir_name = "images_masked" if args.masked else "images"
    lbl_dir_name = ("labels_match_pix" if args.pix_labels
                    else "labels_match" if args.match_labels
                    else "labels")

    split_dir = dataset / split
    if not split_dir.exists():
        sys.exit(f"[ERROR] Split directory not found: {split_dir}")

    # Enumerate tile directories
    tile_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not tile_dirs:
        sys.exit(f"[ERROR] No tile directories found in {split_dir}")

    tile_dir = None
    if args.tile:
        tile_dir = split_dir / args.tile
        if not tile_dir.exists():
            sys.exit(f"[ERROR] Tile not found: {tile_dir}")
    else:
        tile_dir = tile_dirs[0]
        print(f"[INFO]  No --tile specified. Using: {tile_dir.name}")

    tile_id   = tile_dir.name
    img_dir   = tile_dir / img_dir_name
    label_dir = tile_dir / lbl_dir_name

    if not img_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {img_dir}")
    if not label_dir.exists():
        # Try falling back to labels/
        fallback = tile_dir / "labels"
        if fallback.exists():
            print(f"[WARN]  '{lbl_dir_name}' not found; falling back to 'labels/'")
            label_dir = fallback
        else:
            sys.exit(f"[ERROR] Label directory not found: {label_dir}")

    # Pick image
    if args.image:
        image_path = img_dir / args.image
        if not image_path.exists():
            sys.exit(f"[ERROR] Image not found: {image_path}")
    else:
        tifs = sorted(img_dir.glob("*.tif"))
        if not tifs:
            sys.exit(f"[ERROR] No .tif files in {img_dir}")
        image_path = tifs[0]
        print(f"[INFO]  No --image specified. Using: {image_path.name}")

    # Derive matching GeoJSON
    stem       = image_path.stem   # e.g. global_monthly_2018_01_mosaic_L15-…
    geojson_path = label_dir / f"{stem}_Buildings.geojson"
    if not geojson_path.exists():
        # Search for any Buildings file in that directory
        candidates = list(label_dir.glob("*_Buildings.geojson"))
        if not candidates:
            sys.exit(f"[ERROR] No Buildings GeoJSON found in {label_dir}")
        geojson_path = candidates[0]
        print(f"[WARN]  Exact match not found; using: {geojson_path.name}")

    return image_path, geojson_path, tile_id


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

BUILDING_COLOR   = "#00FFAA"
BBOX_COLOR       = "#FF4455"
BG_COLOR         = "#0d0d0d"


def build_geo_to_px(transform, bbox, img_w, img_h):
    """Return a callable (lon, lat) → (col, row)."""
    if transform is not None:
        return lambda lon, lat: rasterio_geo_to_pixel(transform, lon, lat)
    else:
        return lambda lon, lat: linear_geo_to_pixel(lon, lat, bbox, img_w, img_h)


def polygon_to_pixel_ring(ring, geo_to_px):
    """Convert a GeoJSON coordinate ring to a numpy (N, 2) pixel array."""
    return np.array([geo_to_px(c[0], c[1]) for c in ring])


def draw(image: np.ndarray,
         geojson_path: Path,
         geo_to_px,
         draw_polygons: bool = False,
         max_features: Optional[int] = None,
         save_path: Optional[str] = None,
         no_gui: bool = False,
         title_extra: str = ""):

    with open(geojson_path) as f:
        fc = json.load(f)
    features = fc.get("features", [])
    n_total  = len(features)
    if max_features:
        features = features[:max_features]
    n_shown = len(features)

    img_h, img_w = image.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.imshow(image, origin="upper")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)   # image coords: row 0 at top

    # Collect patches for batch rendering (faster than adding one by one)
    bbox_patches   = []
    poly_paths     = []

    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        polys = []
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]

        for poly in polys:
            exterior = poly[0]
            px_ring  = polygon_to_pixel_ring(exterior, geo_to_px)

            if draw_polygons:
                # Full polygon outline via matplotlib Path
                verts = np.vstack([px_ring, px_ring[0]])   # close ring
                codes = ([MplPath.MOVETO]
                         + [MplPath.LINETO] * (len(px_ring) - 1)
                         + [MplPath.CLOSEPOLY])
                poly_paths.append(MplPath(verts, codes))
            else:
                # Axis-aligned bounding box
                x0, y0 = px_ring[:, 0].min(), px_ring[:, 1].min()
                bw = px_ring[:, 0].max() - x0
                bh = px_ring[:, 1].max() - y0
                bbox_patches.append(
                    mpatches.Rectangle((x0, y0), bw, bh)
                )

    # Batch-add bounding boxes
    if bbox_patches:
        pc = PatchCollection(bbox_patches,
                             edgecolors=BBOX_COLOR,
                             facecolors=(*plt.matplotlib.colors.to_rgb(BBOX_COLOR), 0.07),
                             linewidths=0.8)
        ax.add_collection(pc)

    # Batch-add polygon outlines
    if poly_paths:
        pc = PatchCollection([mpatches.PathPatch(p) for p in poly_paths],
                             edgecolors=BUILDING_COLOR,
                             facecolors=(*plt.matplotlib.colors.to_rgb(BUILDING_COLOR), 0.10),
                             linewidths=0.8)
        ax.add_collection(pc)

    # Legend
    mode_label = "polygon outlines" if draw_polygons else "bounding boxes"
    legend_color = BUILDING_COLOR if draw_polygons else BBOX_COLOR
    legend_patch = mpatches.Patch(edgecolor=legend_color,
                                  facecolor=(*plt.matplotlib.colors.to_rgb(legend_color), 0.15),
                                  linewidth=1.2, label=f"Building ({mode_label})")
    ax.legend(handles=[legend_patch], loc="lower right",
              facecolor="#1a1a1a", edgecolor="#444444",
              labelcolor="white", fontsize=9)

    # Annotation count
    count_str = f"{n_shown} / {n_total} features"
    if max_features and n_shown < n_total:
        count_str += f"  (limited to {max_features})"
    ax.set_title(f"{geojson_path.name}\n{count_str}{title_extra}",
                 color="white", fontsize=10, pad=8)

    ax.tick_params(colors="#555555", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_xlabel("pixel column", color="#666666", fontsize=8)
    ax.set_ylabel("pixel row",    color="#666666", fontsize=8)

    plt.tight_layout(pad=0.4)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[INFO]  Saved → {save_path}")

    if not no_gui:
        plt.show()
    else:
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize GeoJSON building labels on GeoTIFF satellite images "
                    "(original pre-YOLO dataset layout).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dataset layout
    p.add_argument("--dataset",      default=".", help="Dataset root directory")
    p.add_argument("--split",        default="train")
    p.add_argument("--tile",         default=None, help="Tile ID subdirectory name")
    p.add_argument("--image",        default=None, help="Image filename inside images/<tile>/")
    p.add_argument("--masked",       action="store_true",
                   help="Use images_masked/ instead of images/")
    p.add_argument("--pix-labels",   action="store_true",
                   help="Use labels_match_pix/ (pixel-aligned) instead of labels/")
    p.add_argument("--match-labels", action="store_true",
                   help="Use labels_match/ instead of labels/")
    # Direct file paths
    p.add_argument("--image-path",   default=None, help="Absolute path to a .tif image")
    p.add_argument("--geojson-path", default=None, help="Absolute path to a _Buildings.geojson")
    # Display options
    p.add_argument("--draw-polygons", action="store_true",
                   help="Draw full polygon outlines instead of bounding boxes")
    p.add_argument("--max-features",  type=int, default=None,
                   help="Max number of features to draw (default: all)")
    p.add_argument("--save",          default=None, help="Save figure to this path")
    p.add_argument("--no-gui",        action="store_true",
                   help="Skip interactive display (headless mode)")
    return p.parse_args()


def main():
    args = parse_args()

    image_path, geojson_path, tile_id = resolve_paths(args)
    print(f"[INFO]  Image  : {image_path}")
    print(f"[INFO]  GeoJSON: {geojson_path}")

    if not HAS_RASTERIO:
        print("[WARN]  rasterio not installed — using tile-name / GeoJSON extent for "
              "coordinate mapping.\n"
              "        Install with: pip install rasterio  (for exact geotransform)")

    # ── Load image ──────────────────────────────────────────────────────────────
    print("[INFO]  Loading image …")
    image, transform = load_image(image_path)
    img_h, img_w = image.shape[:2]
    print(f"[INFO]  Image shape: {img_h} x {img_w} px")

    # ── Determine geo → pixel mapping ──────────────────────────────────────────
    bbox = None
    title_extra = ""

    if transform is not None:
        print("[INFO]  Using embedded GeoTIFF geotransform (rasterio).")
        title_extra = "  [geotransform: rasterio]"
    else:
        # Try tile-name heuristic
        tid = tile_id or image_path.stem
        bbox = tile_bounds_from_name(tid)
        if bbox:
            print(f"[INFO]  Derived bbox from tile name: {bbox}")
            title_extra = "  [geotransform: tile-name heuristic]"
        else:
            print("[INFO]  Falling back to GeoJSON coordinate extent.")
            bbox = geojson_extent(geojson_path)
            if bbox is None:
                sys.exit("[ERROR] Cannot determine tile bounding box — GeoJSON has no features.")
            title_extra = "  [geotransform: GeoJSON extent]"

    geo_to_px = build_geo_to_px(transform, bbox, img_w, img_h)

    # ── Count features ──────────────────────────────────────────────────────────
    with open(geojson_path) as f:
        fc = json.load(f)
    n = len(fc.get("features", []))
    print(f"[INFO]  Building features: {n}")
    if n == 0:
        print("[WARN]  No features in GeoJSON — displaying image without annotations.")

    # ── Draw ────────────────────────────────────────────────────────────────────
    draw(
        image        = image,
        geojson_path = geojson_path,
        geo_to_px    = geo_to_px,
        draw_polygons= args.draw_polygons,
        max_features = args.max_features,
        save_path    = args.save,
        no_gui       = args.no_gui,
        title_extra  = title_extra,
    )


if __name__ == "__main__":
    main()
