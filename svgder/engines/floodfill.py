"""Floodfill Vectorizer: Seeded Lab flood-mask -> Inflate & block-out -> Layered magic-lasso Beziers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from svgder.curves import mask_to_svg_paths, rgb_to_hex


@dataclass
class FloodFillOptions:
    color_tolerance: float = 18.0
    edge_sensitivity: float = 28.0
    tolerance: float = 0.85
    min_area: float = 20.0
    max_stroke_width: float = 12.0
    detect_strokes: bool = False
    tuck_overlap: bool = True
    corner_angle_deg: float = 42.0
    smooth_sigma: float = 0.65


def _absorb_unclaimed_pixels(labels: np.ndarray, rgb: np.ndarray, max_passes: int = 24) -> np.ndarray:
    """Assign unclaimed fringe/noise pixels (label == 0) to nearest color-matching region."""
    if np.all(labels > 0) or not np.any(labels > 0):
        return labels

    h, w = labels.shape
    out = labels.copy()
    rgb_f = rgb.astype(np.float32)

    for _ in range(max_passes):
        zero_mask = out == 0
        if not np.any(zero_mask):
            break
        best_lbl = np.zeros((h, w), dtype=np.int32)
        best_diff = np.full((h, w), 1e9, dtype=np.float32)

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y_src = slice(max(0, -dy), h - max(0, dy))
            x_src = slice(max(0, -dx), w - max(0, dx))
            y_dst = slice(max(0, dy), h - max(0, -dy))
            x_dst = slice(max(0, dx), w - max(0, -dx))

            nbr_lbl = np.zeros((h, w), dtype=np.int32)
            nbr_lbl[y_dst, x_dst] = out[y_src, x_src]

            nbr_rgb = np.zeros((h, w, 3), dtype=np.float32)
            nbr_rgb[y_dst, x_dst] = rgb_f[y_src, x_src]

            diff = np.linalg.norm(rgb_f - nbr_rgb, axis=2)
            valid = zero_mask & (nbr_lbl > 0) & (diff < best_diff)
            best_diff[valid] = diff[valid]
            best_lbl[valid] = nbr_lbl[valid]

        changed = zero_mask & (best_lbl > 0)
        if not np.any(changed):
            break
        out[changed] = best_lbl[changed]

    return out


def _block_out_covered_holes(mask_u8: np.ndarray, later_patch: np.ndarray) -> np.ndarray:
    """Fill interior holes of mask_u8 when those holes will be covered by later high-detail shapes."""
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return mask_u8
    hier = hierarchy[0]
    out = mask_u8.copy()
    for idx, c in enumerate(contours):
        parent = int(hier[idx][3])
        if parent == -1:
            continue
        hole_mask = np.zeros_like(mask_u8)
        cv2.drawContours(hole_mask, [c], -1, 255, -1)
        hole_px = hole_mask > 0
        if np.any(hole_px) and float(np.mean(later_patch[hole_px] > 0)) >= 0.85:
            out[hole_px] = 255
    return out


def _offset_svg_path(d: str, dx: int, dy: int) -> str:
    """Translate all numeric coordinate pairs in an SVG path string by (dx, dy)."""
    tokens = d.replace(",", " ").split()
    out: list[str] = []
    coord_idx = 0
    for tok in tokens:
        if tok in ("M", "L", "C", "Z"):
            out.append(tok)
            coord_idx = 0
        else:
            val = float(tok) + (dx if coord_idx % 2 == 0 else dy)
            s = f"{val:.2f}".rstrip("0").rstrip(".")
            out.append("0" if s in ("-0", "") else s)
            coord_idx += 1
    return " ".join(out)


def _compute_floodfill_labels(
    bgr: np.ndarray,
    opts: FloodFillOptions,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run bilateral smoothing, edge-barrier seeded flood-fill, and fringe absorption."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    smooth_bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=22, sigmaSpace=5)
    lab = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2LAB)

    lab_f = lab.astype(np.float32)
    gx = cv2.Scharr(lab_f[:, :, 0], cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_f[:, :, 0], cv2.CV_32F, 0, 1)
    edge_mag = np.hypot(gx, gy) * 0.25

    flat_order = np.argsort(edge_mag.ravel(), kind="mergesort")

    work_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if opts.edge_sensitivity > 0:
        barriers = edge_mag > opts.edge_sensitivity
        work_mask[1:-1, 1:-1][barriers] = 1

    tol = max(2, int(round(opts.color_tolerance)))
    lo_up = (tol, tol, tol)
    flags = 4 | (2 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

    label_map = np.zeros((h, w), dtype=np.int32)
    region_id = 0

    for flat_idx in flat_order:
        sy = int(flat_idx // w)
        sx = int(flat_idx % w)
        if work_mask[sy + 1, sx + 1] != 0:
            continue

        retval, _, _, rect = cv2.floodFill(lab, work_mask, (sx, sy), (0, 0, 0), lo_up, lo_up, flags)
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0:
            work_mask[sy + 1, sx + 1] = 1
            continue

        roi_mask = work_mask[1 + ry : 1 + ry + rh, 1 + rx : 1 + rx + rw]
        filled = roi_mask == 2
        roi_mask[filled] = 1

        if retval >= opts.min_area:
            region_id += 1
            label_map[ry : ry + rh, rx : rx + rw][filled] = region_id

    label_map = _absorb_unclaimed_pixels(label_map, rgb)
    return label_map, rgb, region_id


def preview_floodfill(
    bgr: np.ndarray,
    opts: FloodFillOptions | None = None,
) -> np.ndarray:
    """Fast raster preview of Floodfill region segmentation without Bezier curve fitting."""
    if opts is None:
        opts = FloodFillOptions()
    label_map, rgb, region_id = _compute_floodfill_labels(bgr, opts)
    if region_id == 0:
        med = np.clip(np.round(np.median(rgb.reshape(-1, 3), axis=0)), 0, 255).astype(np.uint8)
        out_rgb = np.full_like(rgb, med)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    out_rgb = rgb.copy()
    for lbl in np.unique(label_map):
        if lbl <= 0:
            continue
        mask = label_map == lbl
        med = np.clip(np.round(np.median(rgb[mask], axis=0)), 0, 255).astype(np.uint8)
        out_rgb[mask] = med
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def vectorize_floodfill(
    bgr: np.ndarray,
    opts: FloodFillOptions | None = None,
) -> tuple[str, dict]:
    """Vectorize an image by flood-masking, inflating/blocking out base shapes, and layering high-detail lassos on top."""
    if opts is None:
        opts = FloodFillOptions()

    h, w = bgr.shape[:2]
    label_map, rgb, region_id = _compute_floodfill_labels(bgr, opts)

    if region_id == 0:
        bg_hex = rgb_to_hex(np.median(rgb.reshape(-1, 3), axis=0))
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" shape-rendering="geometricPrecision">\n'
            f'  <rect width="{w}" height="{h}" fill="{bg_hex}"/>\n'
            f"</svg>"
        )
        return svg, {"engine": "floodfill", "paths": 1, "fills": 1, "strokes": 0, "regions": 1}

    # Collect all regions as continuous shapes and sort largest -> smallest (high-detail on top)
    regions: list[tuple[int, str, tuple[int, int, int, int], np.ndarray]] = []
    for lbl in np.unique(label_map):
        if lbl <= 0:
            continue
        ys, xs = np.where(label_map == lbl)
        area = len(ys)
        if area < opts.min_area:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_mask = (label_map[y0:y1, x0:x1] == lbl).astype(np.uint8) * 255
        hex_col = rgb_to_hex(np.median(rgb[ys, xs], axis=0))
        regions.append((area, hex_col, (x0, y0, x1 - x0, y1 - y0), sub_mask))

    regions.sort(key=lambda item: item[0], reverse=True)

    # Track all pixels belonging to higher-detail shapes not yet drawn
    later_mask = np.zeros((h, w), dtype=np.uint8)
    for _, _, (x, y, bw, bh), sub_mask in regions:
        later_mask[y : y + bh, x : x + bw][sub_mask > 0] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fill_paths: list[tuple[str, str]] = []

    for _, hex_col, (x, y, bw, bh), sub_mask in regions:
        later_mask[y : y + bh, x : x + bw][sub_mask > 0] = 0

        pad = 2
        px0 = max(0, x - pad)
        py0 = max(0, y - pad)
        px1 = min(w, x + bw + pad)
        py1 = min(h, y + bh + pad)
        pw, ph = px1 - px0, py1 - py0

        padded_local = np.zeros((ph, pw), dtype=np.uint8)
        padded_local[y - py0 : y - py0 + bh, x - px0 : x - px0 + bw] = sub_mask
        later_patch = later_mask[py0:py1, px0:px1]

        # 1. Inflate slightly into upcoming shapes so seams never gap
        if opts.tuck_overlap:
            dilated = cv2.dilate(padded_local, kernel, iterations=1)
            traced_mask = cv2.bitwise_or(padded_local, cv2.bitwise_and(dilated, later_patch))
        else:
            traced_mask = padded_local

        # 2. Block out interior holes that will be covered by high-detail shapes layered on top
        traced_mask = _block_out_covered_holes(traced_mask, later_patch)

        # 3. Solve continuous magic-lasso cubic Beziers on the inflated/blocked-out shape
        paths = mask_to_svg_paths(
            traced_mask,
            tolerance=opts.tolerance,
            corner_angle_deg=opts.corner_angle_deg,
            min_area=max(4.0, opts.min_area * 0.5),
            upscale=2,
            smooth_sigma=opts.smooth_sigma,
        )
        for d in paths:
            d_shifted = _offset_svg_path(d, px0, py0) if (px0 or py0) else d
            fill_paths.append((hex_col, d_shifted))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" shape-rendering="geometricPrecision">',
    ]
    if fill_paths:
        lines.append('  <g id="floodfill-lassos">')
        for col, d in fill_paths:
            lines.append(f'    <path d="{d}" fill="{col}" fill-rule="evenodd"/>')
        lines.append("  </g>")
    lines.append("</svg>")

    svg = "\n".join(lines)
    stats = {
        "engine": "floodfill",
        "paths": len(fill_paths),
        "fills": len(fill_paths),
        "strokes": 0,
        "regions": int(region_id),
    }
    return svg, stats
