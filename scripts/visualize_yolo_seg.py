#!/usr/bin/env python3
"""
visualize_yolo_labels.py

Draws YOLO segmentation polygon labels on a given image.

Usage:
    python visualize_yolo_labels.py <image_path> <label_path> [options]

Options:
    --classes FILE     Path to classes.txt  (default: classes.txt next to label)
    --output FILE      Where to save the result (default: <image_stem>_viz.png)
    --alpha FLOAT      Polygon fill opacity, 0.0–1.0 (default: 0.3)
    --thickness INT    Outline thickness in pixels  (default: 2)
    --no-fill          Draw outlines only, no filled polygons
    --no-labels        Don't draw class name text
    --show             Open the result in a window after saving (requires a display)

Example:
    python visualize_yolo_labels.py \\
        yolo_dataset/images/train/guatemala-volcano_00000000_post_disaster.png \\
        yolo_dataset/labels/train/guatemala-volcano_00000000_post_disaster.txt \\
        --classes yolo_dataset/classes.txt \\
        --output post_disaster_viz.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
PALETTE = [
    (57,  255, 20),   # 0 no-damage      → neon green
    (0,   165, 255),  # 1 minor-damage   → orange
    (0,   0,   255),  # 2 major-damage   → red
    (128, 0,   128),  # 3 destroyed      → purple
    (200, 200, 200),  # 4 un-classified  → light grey
]

def load_classes(label_path: Path, classes_arg: str | None) -> list[str]:
    """Find and load classes.txt, falling back to numeric ids."""
    candidates = []
    if classes_arg:
        candidates.append(Path(classes_arg))
    # Look next to the label file and its parents
    candidates += [
        label_path.parent / "classes.txt",
        label_path.parent.parent / "classes.txt",
        label_path.parent.parent.parent / "classes.txt",
        Path("classes.txt"),
    ]
    for p in candidates:
        if p.is_file():
            names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            print(f"  Classes loaded from: {p}")
            return names
    print("  Warning: classes.txt not found — class ids will be used as labels.")
    return []


def parse_label_file(label_path: Path) -> list[tuple[int, np.ndarray]]:
    """Return list of (class_id, points_array) from a YOLO segmentation .txt."""
    objects = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:          # need class_id + at least 3 x,y pairs
            continue
        class_id = int(parts[0])
        coords   = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            coords = coords[:-1]    # drop trailing orphan value
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        objects.append((class_id, pts))
    return objects


def draw_labels(
    img: np.ndarray,
    objects: list[tuple[int, np.ndarray]],
    class_names: list[str],
    alpha: float,
    thickness: int,
    fill: bool,
    show_labels: bool,
) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    result  = img.copy()

    for class_id, pts_norm in objects:
        colour = PALETTE[class_id % len(PALETTE)]
        # De-normalise
        pts_px = (pts_norm * np.array([w, h])).astype(np.int32)
        poly   = pts_px.reshape((-1, 1, 2))

        # Filled polygon on overlay
        if fill:
            cv2.fillPoly(overlay, [poly], colour)

        # Outline on result
        cv2.polylines(result, [poly], isClosed=True, color=colour, thickness=thickness)

        # Class label at centroid
        if show_labels:
            cx, cy = pts_px.mean(axis=0).astype(int)
            label  = class_names[class_id] if class_id < len(class_names) else str(class_id)
            font, scale, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, ft)
            # small background pill
            cv2.rectangle(
                result,
                (cx - 2, cy - th - 2),
                (cx + tw + 2, cy + baseline),
                colour, -1
            )
            # white text
            cv2.putText(result, label, (cx, cy), font, scale, (255, 255, 255), ft, cv2.LINE_AA)

    # Blend fill overlay
    if fill:
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    return result


def main():
    ap = argparse.ArgumentParser(
        description="Visualize YOLO segmentation polygon labels on an image."
    )
    ap.add_argument("image",  help="Path to the source image (.png / .jpg)")
    ap.add_argument("label",  help="Path to the YOLO label file (.txt)")
    ap.add_argument("--classes",   default=None,  help="Path to classes.txt")
    ap.add_argument("--output",    default=None,  help="Output image path (default: <stem>_viz.png)")
    ap.add_argument("--alpha",     type=float, default=0.30, help="Fill opacity (default: 0.3)")
    ap.add_argument("--thickness", type=int,   default=2,    help="Outline thickness (default: 2)")
    ap.add_argument("--no-fill",   action="store_true", help="Outlines only, no fill")
    ap.add_argument("--no-labels", action="store_true", help="Skip class name text")
    ap.add_argument("--show",      action="store_true", help="Open result in a window")
    args = ap.parse_args()

    image_path = Path(args.image)
    label_path = Path(args.label)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not image_path.is_file():
        sys.exit(f"Error: image not found: {image_path}")
    if not label_path.is_file():
        sys.exit(f"Error: label file not found: {label_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nImage : {image_path}")
    print(f"Labels: {label_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Error: could not read image: {image_path}")

    class_names = load_classes(label_path, args.classes)
    objects     = parse_label_file(label_path)
    print(f"  Found {len(objects)} object(s) in label file.")

    if not objects:
        print("  Nothing to draw — exiting.")
        sys.exit(0)

    # ── Draw ──────────────────────────────────────────────────────────────────
    result = draw_labels(
        img, objects, class_names,
        alpha=args.alpha,
        thickness=args.thickness,
        fill=not args.no_fill,
        show_labels=not args.no_labels,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else image_path.parent / f"{image_path.stem}_viz.png"
    cv2.imwrite(str(out_path), result)
    print(f"  Saved → {out_path}")

    # ── Legend ────────────────────────────────────────────────────────────────
    print("\n  Legend:")
    seen_ids = sorted({cid for cid, _ in objects})
    for cid in seen_ids:
        name = class_names[cid] if cid < len(class_names) else str(cid)
        b, g, r = PALETTE[cid % len(PALETTE)]
        print(f"    [{cid}] {name}  (RGB #{r:02X}{g:02X}{b:02X})")

    if args.show:
        cv2.imshow("YOLO Label Visualisation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
