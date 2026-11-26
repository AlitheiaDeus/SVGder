"""Flood-Search Vectorizer: Seeded Lab region growing -> AA fringe absorption -> Stroke-vs-Fill classification -> Cubic Beziers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from svgder.curves import fit_closed_contour_to_commands, mask_to_svg_paths, rgb_to_hex


@dataclass
class FloodSearchOptions:
    color_tolerance: float = 18.0
    edge_sensitivity: float = 28.0
    tolerance: float = 0.9
    min_area: float = 20.0
    max_stroke_width: float = 12.0
    detect_strokes: bool = True
    corner_angle_deg: float = 42.0


_NEIGHBORS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def _thin_skeleton(binary: np.ndarray, max_iters: int = 16) -> np.ndarray:
    """Morphological thinning to a 1px-wide skeleton."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iters):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return cv2.bitwise_or(skel, img)


def _trace_skeleton_chains(skel: np.ndarray, ox: int = 0, oy: int = 0) -> list[tuple[np.ndarray, bool]]:
    """Trace a binary skeleton mask into ordered (x, y) polylines and closed flags."""
    h, w = skel.shape
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    active = skel > 0
    visited = np.zeros_like(active, dtype=bool)

    deg = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in _NEIGHBORS_8:
        shifted = np.zeros_like(active)
        y_src = slice(max(0, -dy), h - max(0, dy))
        x_src = slice(max(0, -dx), w - max(0, dx))
        y_dst = slice(max(0, dy), h - max(0, -dy))
        x_dst = slice(max(0, dx), w - max(0, -dx))
        shifted[y_dst, x_dst] = active[y_src, x_src]
        deg += shifted.astype(np.uint8)

    endpoints = [(int(y), int(x)) for y, x in zip(ys, xs) if deg[y, x] != 2]
    all_pts = [(int(y), int(x)) for y, x in zip(ys, xs)]
    chains: list[tuple[np.ndarray, bool]] = []

    def walk(start_y: int, start_x: int) -> None:
        if visited[start_y, start_x]:
            return
        pts = [(start_x + ox, start_y + oy)]
        visited[start_y, start_x] = True
        cy, cx = start_y, start_x
        prev_dy, prev_dx = 0, 0

        while True:
            best = None
            best_score = -999.0
            for dy, dx in _NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and active[ny, nx] and not visited[ny, nx]:
                    dot = dy * prev_dy + dx * prev_dx
                    cardinal = 0.5 if (dy == 0 or dx == 0) else 0.0
                    score = dot + cardinal
                    if score > best_score:
                        best_score = score
                        best = (ny, nx, dy, dx)
            if best is None:
                break
            ny, nx, dy, dx = best
            visited[ny, nx] = True
            pts.append((nx + ox, ny + oy))
            prev_dy, prev_dx = dy, dx
            cy, cx = ny, nx

        if len(pts) < 3:
            return
        dx_end = abs(pts[-1][0] - pts[0][0])
        dy_end = abs(pts[-1][1] - pts[0][1])
        is_closed = len(pts) >= 6 and dx_end <= 1 and dy_end <= 1
        arr = np.array(pts, dtype=np.float64) + 0.5
        chains.append((arr, is_closed))

    for y, x in endpoints:
        walk(y, x)
    for y, x in all_pts:
        walk(y, x)

    return chains


def _classify_stroke_vs_fill(
    mask_u8: np.ndarray,
    max_stroke_width: float,
) -> tuple[bool, float, np.ndarray | None]:
    """Estimate whether a binary region mask is an outline/stroke or a filled shape.

    Returns (is_stroke, estimated_stroke_width, skeleton_mask_or_None).
    """
    area = float(np.count_nonzero(mask_u8))
    if area < 6.0:
        return False, 0.0, None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False, 0.0, None
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    if perimeter < 1e-3:
        return False, 0.0, None

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_r = float(np.max(dist))
    if max_r * 2.0 > max_stroke_width:
        return False, 0.0, None

    # Isoperimetric elongation: circle = 1.0, thin ribbon/stroke >> 2.0
    elongation = perimeter / (2.0 * np.sqrt(np.pi * area))
    half_thickness = area / perimeter
    if elongation < 2.0 and half_thickness > max_stroke_width * 0.45:
        return False, 0.0, None

    skel = _thin_skeleton(mask_u8, max_iters=max(6, int(max_stroke_width)))
    skel_radii = dist[skel > 0]
    if len(skel_radii) < 4:
        return False, 0.0, None

    mean_r = float(np.mean(skel_radii))
    median_r = float(np.median(skel_radii))
    cv_r = float(np.std(skel_radii)) / (mean_r + 1e-6)
    est_width = max(1.0, min(max_stroke_width, median_r * 2.0))

    # Uniform-width thin ribbons -> strokes; variable-width blobs -> fills
    if cv_r <= 0.72 and est_width <= max_stroke_width:
        return True, est_width, skel
    return False, 0.0, None


def _absorb_unclaimed_pixels(labels: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Assign unclaimed fringe/noise pixels (label == 0) to nearest valid neighbor region."""
    if np.all(labels > 0):
        return labels

    h, w = labels.shape
    out = labels.copy()
    # Fast multi-pass morphological dilation of labels into 0-pixels
    # Preferring 4-connected neighbors, up to 6px fringe width
    for _ in range(6):
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
            nbr_rgb[y_dst, x_dst] = rgb[y_src, x_src].astype(np.float32)

            diff = np.linalg.norm(rgb.astype(np.float32) - nbr_rgb, axis=2)
            valid = zero_mask & (nbr_lbl > 0) & (diff < best_diff)
            best_diff[valid] = diff[valid]
            best_lbl[valid] = nbr_lbl[valid]

        changed = zero_mask & (best_lbl > 0)
        if not np.any(changed):
            break
        out[changed] = best_lbl[changed]

    return out


def vectorize_flood_search(
    bgr: np.ndarray,
    opts: FloodSearchOptions | None = None,
) -> tuple[str, dict]:
    """Vectorize an image by seeded flood-fill region growing and stroke-vs-fill estimation."""
    if opts is None:
        opts = FloodSearchOptions()

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    smooth_bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=24, sigmaSpace=5)
    lab = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2LAB)

    # Edge magnitude on L*a*b* to stop flood-fill from chaining across crisp edges
    lab_f = lab.astype(np.float32)
    gx = cv2.Scharr(lab_f[:, :, 0], cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_f[:, :, 0], cv2.CV_32F, 0, 1)
    edge_mag = np.hypot(gx, gy) * 0.25

    # Seed order: flattest interior pixels first (lowest edge magnitude)
    flat_order = np.argsort(edge_mag.ravel(), kind="mergesort")

    # work_mask: 0 = unexplored, 1 = claimed/explored, 2 = active flood-fill region
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

    # Absorb edge barriers & tiny < min_area speckles into adjacent valid regions
    label_map = _absorb_unclaimed_pixels(label_map, rgb)

    # Fallback if image was extremely noisy and produced 0 valid regions
    if region_id == 0:
        bg_hex = rgb_to_hex(np.median(rgb.reshape(-1, 3), axis=0))
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" shape-rendering="geometricPrecision">\n'
            f'  <rect width="{w}" height="{h}" fill="{bg_hex}"/>\n'
            f"</svg>"
        )
        return svg, {"engine": "flood-search", "paths": 1, "fills": 1, "strokes": 0, "regions": 1}

    # Inspect each region via bounding-box slices (using connectedComponents on label_map?
    # Label IDs are already connected, though absorption might create tiny disconnected 1px islands,
    # which min_area in mask_to_svg_paths naturally filters).
    fill_items: list[tuple[int, str, str]] = []  # (area, hex_color, d)
    stroke_items: list[tuple[int, str, float, str]] = []  # (area, hex_color, stroke_w, d)

    # Compute bounding boxes for all labels in one pass
    unique_labels = np.unique(label_map)
    for lbl in unique_labels:
        if lbl <= 0:
            continue
        ys, xs = np.where(label_map == lbl)
        area = len(ys)
        if area < opts.min_area:
            continue

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_mask = (label_map[y0:y1, x0:x1] == lbl).astype(np.uint8) * 255
        med_rgb = np.median(rgb[ys, xs], axis=0)
        hex_col = rgb_to_hex(med_rgb)

        is_stroke = False
        est_w = 0.0
        skel = None
        if opts.detect_strokes:
            is_stroke, est_w, skel = _classify_stroke_vs_fill(sub_mask, opts.max_stroke_width)

        if is_stroke and skel is not None:
            chains = _trace_skeleton_chains(skel, ox=x0, oy=y0)
            emitted_any = False
            for pts, is_closed in chains:
                d = fit_closed_contour_to_commands(
                    pts,
                    tolerance=opts.tolerance,
                    corner_angle_deg=opts.corner_angle_deg,
                    closed=is_closed,
                )
                if d:
                    stroke_items.append((area, hex_col, est_w, d))
                    emitted_any = True
            if emitted_any:
                continue

        # Vectorize as filled shape (pad and offset coordinates back to full canvas)
        paths = mask_to_svg_paths(
            sub_mask,
            tolerance=opts.tolerance,
            corner_angle_deg=opts.corner_angle_deg,
            min_area=max(4.0, opts.min_area * 0.5),
            upscale=2,
        )
        for d in paths:
            if x0 != 0 or y0 != 0:
                d = _offset_svg_path(d, x0, y0)
            fill_items.append((area, hex_col, d))

    # Painter's order: largest fills in back, smaller fills forward, detected strokes on top
    fill_items.sort(key=lambda item: item[0], reverse=True)
    stroke_items.sort(key=lambda item: item[0], reverse=True)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" shape-rendering="geometricPrecision">',
    ]
    if fill_items:
        lines.append('  <g id="flood-fills">')
        for _, col, d in fill_items:
            lines.append(
                f'    <path d="{d}" fill="{col}" fill-rule="evenodd" '
                f'stroke="{col}" stroke-width="0.35" stroke-linejoin="round"/>'
            )
        lines.append("  </g>")

    if stroke_items:
        lines.append('  <g id="flood-strokes" fill="none" stroke-linecap="round" stroke-linejoin="round">')
        for _, col, sw, d in stroke_items:
            lines.append(f'    <path d="{d}" stroke="{col}" stroke-width="{sw:.2f}"/>')
        lines.append("  </g>")

    lines.append("</svg>")
    svg = "\n".join(lines)
    total_paths = len(fill_items) + len(stroke_items)
    stats = {
        "engine": "flood-search",
        "paths": total_paths,
        "fills": len(fill_items),
        "strokes": len(stroke_items),
        "regions": int(region_id),
    }
    return svg, stats


def _offset_svg_path(d: str, dx: int, dy: int) -> str:
    """Translate all numeric coordinate pairs in an SVG path string by (dx, dy)."""
    tokens = d.split()
    out: list[str] = []
    coord_idx = 0
    for tok in tokens:
        if tok in ("M", "L", "C", "Z"):
            out.append(tok)
            coord_idx = 0
        else:
            has_comma = tok.endswith(",")
            num_str = tok[:-1] if has_comma else tok
            val = float(num_str) + (dx if coord_idx % 2 == 0 else dy)
            s = f"{val:.2f}".rstrip("0").rstrip(".")
            if s in ("", "-0"):
                s = "0"
            out.append(f"{s}," if has_comma else s)
            coord_idx += 1
    return " ".join(out)
