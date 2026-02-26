"""
visualize_yolo.py
-----------------
Display bounding box annotations from a YOLO-format dataset on a satellite image.

Usage
-----
  # Auto-pick the first image in the dataset:
  python visualize_yolo.py --dataset /path/to/yolo_dataset

  # Visualize a specific image:
  python visualize_yolo.py --dataset /path/to/yolo_dataset \
                           --image global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif

  # Point directly at an image + label pair (no dataset layout needed):
  python visualize_yolo.py --image-path /abs/path/to/image.tif \
                           --label-path /abs/path/to/label.txt

  # Save instead of showing interactively:
  python visualize_yolo.py --dataset /path/to/yolo_dataset --save output.png

Options
-------
  --dataset   PATH   Root of the YOLO dataset (must contain images/ and labels/).
  --split     NAME   Dataset split to use (default: train).
  --image     NAME   Image filename to visualize (default: first image found).
  --image-path PATH  Absolute path to an image file (overrides --dataset/--image).
  --label-path PATH  Absolute path to the YOLO .txt label file.
  --class-names STR  Comma-separated class names (default: "building").
  --max-boxes INT    Maximum number of boxes to draw (default: all).
  --save      PATH   Save the figure to this path instead of displaying it.
  --no-gui           Skip interactive display (useful in headless environments).
"""

import argparse
import os
import sys
from pathlib import Path


# ── Dependency check ───────────────────────────────────────────────────────────
def require(pkg, install_hint=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        hint = install_hint or f"pip install {pkg}"
        sys.exit(f"[ERROR] Missing dependency '{pkg}'. Install with: {hint}")


np        = require("numpy")
plt_mod   = require("matplotlib", "pip install matplotlib")
Image_mod = require("PIL", "pip install Pillow")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image

# Try rasterio for GeoTIFF support; fall back to PIL
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image as an HxWx3 uint8 RGB array.
    Handles multi-band GeoTIFFs (picks first 3 bands) via rasterio when available.
    Falls back to PIL for standard formats.
    """
    suffix = image_path.suffix.lower()

    if suffix in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(image_path) as src:
            n_bands = src.count
            # Read up to 3 bands
            band_idx = list(range(1, min(n_bands, 3) + 1))
            data = src.read(band_idx)          # (bands, H, W)
            data = np.moveaxis(data, 0, -1)    # (H, W, bands)

            # Normalise to uint8
            data = data.astype(np.float32)
            p2, p98 = np.percentile(data, (2, 98))
            if p98 > p2:
                data = np.clip((data - p2) / (p98 - p2) * 255, 0, 255)
            else:
                data = np.clip(data, 0, 255)
            data = data.astype(np.uint8)

            # If only 1 band, replicate to 3 channels
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            return data

    # PIL fallback
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_labels(label_path: Path) -> list[dict]:
    """
    Parse a YOLO label file.
    Returns a list of dicts: {class_id, x_center, y_center, width, height}
    """
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            annotations.append({
                "class_id": int(parts[0]),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width":    float(parts[3]),
                "height":   float(parts[4]),
            })
    return annotations


# ── Dataset resolution ─────────────────────────────────────────────────────────

def resolve_paths(args) -> tuple[Path, Path]:
    """Return (image_path, label_path) based on CLI args."""

    # Direct paths provided
    if args.image_path:
        image_path = Path(args.image_path)
        if args.label_path:
            label_path = Path(args.label_path)
        else:
            # Guess label path next to image
            label_path = image_path.with_suffix(".txt")
        return image_path, label_path

    # Dataset layout
    dataset = Path(args.dataset)
    split   = args.split
    img_dir = dataset / "images" / split
    lbl_dir = dataset / "labels" / split

    if not img_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {img_dir}")
    if not lbl_dir.exists():
        sys.exit(f"[ERROR] Label directory not found: {lbl_dir}")

    image_files = sorted(img_dir.iterdir())
    image_files = [p for p in image_files if p.is_file()]
    if not image_files:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    if args.image:
        image_path = img_dir / args.image
        if not image_path.exists():
            sys.exit(f"[ERROR] Image not found: {image_path}")
    else:
        image_path = image_files[0]
        print(f"[INFO]  No --image specified. Using: {image_path.name}")

    label_path = lbl_dir / (image_path.stem + ".txt")
    return image_path, label_path


# ── Visualisation ──────────────────────────────────────────────────────────────

# Colour palette (one per class, cycles if needed)
PALETTE = [
    "#FF4444",  # red
    "#44AAFF",  # blue
    "#44FF88",  # green
    "#FFAA00",  # orange
    "#DD44FF",  # purple
    "#00DDCC",  # teal
]


def draw_boxes(image: np.ndarray,
               annotations: list[dict],
               class_names: list[str],
               max_boxes: int | None = None,
               save_path: str | None = None,
               no_gui: bool = False):

    h, w = image.shape[:2]
    n_total = len(annotations)
    if max_boxes is not None:
        annotations = annotations[:max_boxes]
    n_drawn = len(annotations)

    fig, ax = plt.subplots(1, figsize=(14, 14))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.imshow(image)

    for ann in annotations:
        cid  = ann["class_id"]
        xc   = ann["x_center"] * w
        yc   = ann["y_center"] * h
        bw   = ann["width"]    * w
        bh   = ann["height"]   * h
        x0   = xc - bw / 2
        y0   = yc - bh / 2

        color = PALETTE[cid % len(PALETTE)]
        rect  = patches.Rectangle(
            (x0, y0), bw, bh,
            linewidth=1.2,
            edgecolor=color,
            facecolor=(*plt.matplotlib.colors.to_rgb(color), 0.08),  # subtle fill
        )
        ax.add_patch(rect)

        # Label badge
        label = class_names[cid] if cid < len(class_names) else str(cid)
        ax.text(
            x0 + 2, y0 - 3,
            label,
            fontsize=6,
            color="white",
            bbox=dict(facecolor=color, edgecolor="none", pad=1, alpha=0.75),
        )

    # Title bar
    title = f"{n_drawn} / {n_total} boxes shown"
    if max_boxes and n_drawn < n_total:
        title += f"  (limited to {max_boxes})"
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xlabel(f"Width: {w}px", color="#888888", fontsize=9)
    ax.set_ylabel(f"Height: {h}px", color="#888888", fontsize=9)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout(pad=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[INFO]  Saved to: {save_path}")

    if not no_gui:
        plt.show()
    else:
        plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes on a satellite image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",     default=".",  help="YOLO dataset root directory")
    p.add_argument("--split",       default="train")
    p.add_argument("--image",       default=None, help="Image filename inside images/<split>/")
    p.add_argument("--image-path",  default=None, help="Absolute path to an image file")
    p.add_argument("--label-path",  default=None, help="Absolute path to a YOLO label .txt file")
    p.add_argument("--class-names", default="building",
                   help="Comma-separated class names (default: 'building')")
    p.add_argument("--max-boxes",   type=int, default=None)
    p.add_argument("--save",        default=None, help="Save figure to this path")
    p.add_argument("--no-gui",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    class_names = [c.strip() for c in args.class_names.split(",")]

    image_path, label_path = resolve_paths(args)

    print(f"[INFO]  Image : {image_path}")
    print(f"[INFO]  Labels: {label_path}")

    if not HAS_RASTERIO and image_path.suffix.lower() in (".tif", ".tiff"):
        print("[WARN]  rasterio not found — falling back to PIL for GeoTIFF loading.")
        print("        Install with: pip install rasterio")

    print("[INFO]  Loading image …")
    image = load_image(image_path)
    print(f"[INFO]  Image shape: {image.shape}")

    annotations = load_labels(label_path)
    print(f"[INFO]  Annotations loaded: {len(annotations)}")
    if not annotations:
        print("[WARN]  Label file is empty or missing — displaying image with no boxes.")

    draw_boxes(
        image,
        annotations,
        class_names,
        max_boxes=args.max_boxes,
        save_path=args.save,
        no_gui=args.no_gui,
    )


if __name__ == "__main__":
    main()
