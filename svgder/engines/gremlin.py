"""Mythical Gremlin Vectorizer: Denoise -> Lab Quantize -> Watertight Painter's Order -> Smooth Cubic Beziers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from svgder.curves import mask_to_svg_paths, rgb_to_hex


@dataclass
class GremlinOptions:
    colors: int = 24
    min_delta_e: float = 6.5
    tolerance: float = 0.85
    corner_angle_deg: float = 42.0
    min_area: float = 14.0
    smooth_passes: int = 2
    mean_shift: bool = True
    background: Literal["auto", "transparent", "white"] = "auto"
    ink_on_top: bool = True
    tuck_overlap: bool = True


def _detect_background_mask(lab: np.ndarray, tol: float = 11.0) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """Flood-fill uniform border background in CIE Lab space."""
    h, w = lab.shape[:2]
    border = np.vstack([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]).astype(np.float32)
    # Check if border pixels are mostly consistent
    med = np.median(border, axis=0)
    dists_border = np.linalg.norm(border - med, axis=1)
    if float(np.mean(dists_border < tol)) < 0.55:
        return np.zeros((h, w), dtype=bool), None

    lab_f = lab.astype(np.float32)
    dist_map = np.linalg.norm(lab_f - med[None, None, :], axis=2)
    candidate = (dist_map <= tol).astype(np.uint8) * 255

    # Connected components touching the image border
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=4)
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.discard(0)

    if not border_labels:
        return np.zeros((h, w), dtype=bool), None

    bg_mask = np.isin(labels, list(border_labels))
    # Morphological close on small 1px AA fringe next to background? Keep exact for crisp edge.
    return bg_mask, tuple(int(v) for v in med)


def _quantize_lab_foreground(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    n_colors: int,
    min_delta_e: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize foreground pixels in CIE Lab space, fitting centers on low-gradient interiors."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab_f = lab.astype(np.float32)
    # OpenCV Lab scaling: L in [0,255] -> [0,100], a/b shifted by +128
    lab_std = np.empty_like(lab_f)
    lab_std[:, :, 0] = lab_f[:, :, 0] * (100.0 / 255.0)
    lab_std[:, :, 1] = lab_f[:, :, 1] - 128.0
    lab_std[:, :, 2] = lab_f[:, :, 2] - 128.0

    # Compute gradient magnitude on L* to avoid fitting cluster centers on AA transition pixels
    gx = cv2.Scharr(lab_std[:, :, 0], cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab_std[:, :, 0], cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)

    fg_pts = lab_std[fg_mask]
    if len(fg_pts) == 0:
        return np.full((h, w), -1, dtype=np.int32), np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32)

    fg_grad = grad[fg_mask]
    thresh = float(np.percentile(fg_grad, 65.0))
    interior_pts = fg_pts[fg_grad <= max(thresh, 1e-3)]
    if len(interior_pts) < max(64, n_colors * 4):
        interior_pts = fg_pts

    # Subsample if very large for fast K-means
    if len(interior_pts) > 60000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(interior_pts), size=60000, replace=False)
        sample_pts = interior_pts[idx]
    else:
        sample_pts = interior_pts

    k = max(2, min(n_colors, len(sample_pts)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, _, centers = cv2.kmeans(
        sample_pts.astype(np.float32),
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    # Assign all foreground pixels to nearest center
    fg_flat = fg_pts.astype(np.float32)
    dists = np.linalg.norm(fg_flat[:, None, :] - centers[None, :, :], axis=2)
    fg_labels = np.argmin(dists, axis=1)

    # Merge palette centers closer than min_delta_e
    if min_delta_e > 0 and len(centers) > 2:
        counts = np.bincount(fg_labels, minlength=len(centers)).astype(np.float32)
        active = counts > 0
        centers_list = [centers[i].copy() for i in range(len(centers)) if active[i]]
        counts_list = [counts[i] for i in range(len(centers)) if active[i]]

        while len(centers_list) > 2:
            c_arr = np.array(centers_list, dtype=np.float32)
            diff = c_arr[:, None, :] - c_arr[None, :, :]
            pdist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(pdist, np.inf)
            i, j = np.unravel_index(int(np.argmin(pdist)), pdist.shape)
            if pdist[i, j] >= min_delta_e:
                break
            total_w = counts_list[i] + counts_list[j]
            merged = (centers_list[i] * counts_list[i] + centers_list[j] * counts_list[j]) / max(1.0, total_w)
            keep, drop = min(i, j), max(i, j)
            centers_list[keep] = merged
            counts_list[keep] = total_w
            centers_list.pop(drop)
            counts_list.pop(drop)

        centers = np.array(centers_list, dtype=np.float32)
        dists = np.linalg.norm(fg_flat[:, None, :] - centers[None, :, :], axis=2)
        fg_labels = np.argmin(dists, axis=1)

    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[fg_mask] = fg_labels

    # Compute crisp representative RGB for each cluster using median of interior pixels
    palette_rgb = np.zeros((len(centers), 3), dtype=np.uint8)
    palette_l = np.zeros(len(centers), dtype=np.float32)
    rgb_fg = rgb[fg_mask]
    is_interior_fg = fg_grad <= max(thresh, 1e-3)

    for cid in range(len(centers)):
        mask_c = fg_labels == cid
        if not np.any(mask_c):
            continue
        mask_int = mask_c & is_interior_fg
        pts_rgb = rgb_fg[mask_int] if np.any(mask_int) else rgb_fg[mask_c]
        med_rgb = np.median(pts_rgb, axis=0)
        palette_rgb[cid] = np.clip(np.round(med_rgb), 0, 255).astype(np.uint8)
        palette_l[cid] = float(centers[cid, 0])

    return label_map, palette_rgb, palette_l


def _absorb_tiny_specks(label_map: np.ndarray, min_area: float) -> np.ndarray:
    """Reassign tiny connected label islands (< min_area) to their dominant neighbor label."""
    out = label_map.copy()
    unique_labels = [int(v) for v in np.unique(out) if v >= 0]
    for lbl in unique_labels:
        mask = (out == lbl).astype(np.uint8)
        num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= min_area:
                continue
            x, y, bw, bh, _ = stats[cid]
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(out.shape[1], x + bw + 2), min(out.shape[0], y + bh + 2)
            sub_comp = comp[y0:y1, x0:x1] == cid
            sub_lbl = out[y0:y1, x0:x1]
            dilated = cv2.dilate(sub_comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ring = dilated & (~sub_comp) & (sub_lbl >= 0)
            if np.any(ring):
                vals, cnts = np.unique(sub_lbl[ring], return_counts=True)
                best = int(vals[np.argmax(cnts)])
                out[y0:y1, x0:x1][sub_comp] = best
            else:
                out[y0:y1, x0:x1][sub_comp] = -1
    return out


def vectorize_gremlin(
    image_bgr: np.ndarray,
    options: GremlinOptions | None = None,
) -> tuple[str, dict]:
    """Convert an image into a clean, seam-free cubic Bezier SVG using the Gremlin pipeline."""
    opts = options or GremlinOptions()
    h, w = image_bgr.shape[:2]

    # 1. Edge-preserving smoothing to flatten JPEG artifacts & noisy shading while preserving ink
    smoothed = image_bgr.copy()
    if opts.mean_shift:
        smoothed = cv2.pyrMeanShiftFiltering(smoothed, sp=8, sr=18)
    for _ in range(max(0, opts.smooth_passes)):
        smoothed = cv2.bilateralFilter(smoothed, d=7, sigmaColor=24, sigmaSpace=24)

    rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

    # 2. Background separation
    bg_mask, bg_lab = _detect_background_mask(lab)
    if opts.background == "transparent":
        fg_mask = ~bg_mask
        bg_hex = None
    elif opts.background == "white":
        fg_mask = ~bg_mask
        bg_hex = "#ffffff"
    else:  # auto
        fg_mask = ~bg_mask
        if bg_lab is not None and np.any(bg_mask):
            orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            med_bg = np.median(orig_rgb[bg_mask], axis=0)
            bg_hex = rgb_to_hex(med_bg)
        else:
            bg_hex = None

    if not np.any(fg_mask):
        fg_mask = np.ones((h, w), dtype=bool)
        bg_hex = None

    # 3. Gradient-weighted CIE Lab quantization + palette merging
    label_map, palette_rgb, palette_l = _quantize_lab_foreground(
        rgb, fg_mask, n_colors=opts.colors, min_delta_e=opts.min_delta_e
    )

    # 4. Clean up sub-min-area AA specks
    label_map = _absorb_tiny_specks(label_map, min_area=max(4.0, opts.min_area * 0.8))

    # 5. Identify ink clusters (noticeably darker than median lightness)
    valid_l = palette_l[palette_l >= 0]
    median_l = float(np.median(valid_l)) if len(valid_l) else 50.0
    is_ink_label = {
        i: bool(opts.ink_on_top and palette_l[i] < 32.0 and (median_l - palette_l[i]) > 22.0)
        for i in range(len(palette_rgb))
    }

    # 6. Collect connected components across all labels
    # Each entry: (layer_tier, -area, label_id, bbox, local_mask)
    # layer_tier: 0 = color fills (largest to smallest), 1 = dark ink strokes on top
    components: list[tuple[int, int, int, tuple[int, int, int, int], np.ndarray]] = []
    for lbl in range(len(palette_rgb)):
        mask = (label_map == lbl).astype(np.uint8)
        if np.count_nonzero(mask) < opts.min_area:
            continue
        num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        tier = 1 if is_ink_label[lbl] else 0
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < opts.min_area:
                continue
            x, y, bw, bh, _ = (int(v) for v in stats[cid])
            local = (comp[y : y + bh, x : x + bw] == cid).astype(np.uint8) * 255
            components.append((tier, -area, lbl, (x, y, bw, bh), local))

    # Sort: fills (tier 0) largest->smallest, then ink (tier 1) largest->smallest
    components.sort(key=lambda item: (item[0], item[1]))

    # 7. Watertight Painter's Order with bounded 1px tuck into later_mask
    later_mask = (label_map >= 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    svg_elements: list[str] = []
    path_count = 0
    colors_used: set[str] = set()

    for tier, _, lbl, (x, y, bw, bh), local in components:
        # Remove current component from later_mask
        sub_later = later_mask[y : y + bh, x : x + bw]
        sub_later[local > 0] = 0

        pad = 2
        px0 = max(0, x - pad)
        py0 = max(0, y - pad)
        px1 = min(w, x + bw + pad)
        py1 = min(h, y + bh + pad)

        pw, ph = px1 - px0, py1 - py0
        padded_local = np.zeros((ph, pw), dtype=np.uint8)
        padded_local[y - py0 : y - py0 + bh, x - px0 : x - px0 + bw] = local

        if opts.tuck_overlap and tier == 0:
            dilated = cv2.dilate(padded_local, kernel, iterations=1)
            later_patch = later_mask[py0:py1, px0:px1]
            traced_mask = cv2.bitwise_or(padded_local, cv2.bitwise_and(dilated, later_patch))
        else:
            traced_mask = padded_local

        paths = mask_to_svg_paths(
            traced_mask,
            tolerance=opts.tolerance,
            corner_angle_deg=opts.corner_angle_deg,
            min_area=opts.min_area * 0.5,
            upscale=2,
            smooth_sigma=0.65,
        )
        if not paths:
            continue

        hex_color = rgb_to_hex(palette_rgb[lbl])
        colors_used.add(hex_color)
        # Shift paths from local patch (px0, py0) into full canvas coordinates
        # Better: adjust path coordinates directly? No, easier: shift only if px0/py0 != 0 via wrapper or transform,
        # OR pass offset into mask_to_svg_paths? Actually SVG transform="translate(x,y)" is messy for editing —
        # let's trace cleanly with zero-offset or offset points! Wait: let's add a lightweight offset helper!
        for d in paths:
            d_shifted = _offset_svg_path(d, px0, py0) if (px0 or py0) else d
            svg_elements.append(f'  <path d="{d_shifted}" fill="{hex_color}" fill-rule="evenodd"/>')
            path_count += 1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <!-- Generated by SVGder (Gremlin engine: {len(colors_used)} colors, {path_count} paths) -->',
    ]
    if bg_hex:
        lines.append(f'  <rect width="100%" height="100%" fill="{bg_hex}"/>')
    lines.extend(svg_elements)
    lines.append("</svg>")

    stats = {
        "engine": "gremlin",
        "width": w,
        "height": h,
        "colors": len(colors_used),
        "paths": path_count,
        "background": bg_hex or "transparent",
    }
    return "\n".join(lines), stats


def _offset_svg_path(d: str, dx: int, dy: int) -> str:
    """Shift all numeric coordinate pairs in an M/L/C/Z SVG path string by (dx, dy)."""
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
