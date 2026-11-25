"""Canny Edge & Ink Lineart Vectorizer: Auto-Canny -> 8-connected stroke chaining -> Cubic Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from svgder.curves import fit_closed_contour_to_commands, rgb_to_hex


@dataclass
class CannyOptions:
    sigma: float = 0.33
    low_thresh: int | None = None
    high_thresh: int | None = None
    blur_ksize: int = 5
    min_length: int = 12
    tolerance: float = 1.15
    stroke_width: float = 2.0
    style: Literal["ink", "colored", "neon", "blueprint"] = "ink"
    dilate_close: bool = True


_NEIGHBORS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def _trace_edge_chains(edges: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Trace 1px-thin binary edge image into ordered (x, y) polylines and closed flags."""
    h, w = edges.shape
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return []

    active = edges > 0
    visited = np.zeros_like(active, dtype=bool)

    # Compute 8-connected degree for each active pixel
    deg = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in _NEIGHBORS_8:
        shifted = np.zeros_like(active)
        y_src = slice(max(0, -dy), h - max(0, dy))
        x_src = slice(max(0, -dx), w - max(0, dx))
        y_dst = slice(max(0, dy), h - max(0, -dy))
        x_dst = slice(max(0, dx), w - max(0, -dx))
        shifted[y_dst, x_dst] = active[y_src, x_src]
        deg += shifted.astype(np.uint8)

    # Trace starting from endpoints/junctions (deg != 2) first, then remaining closed loops
    endpoints = [(int(y), int(x)) for y, x in zip(ys, xs) if deg[y, x] != 2]
    all_pts = [(int(y), int(x)) for y, x in zip(ys, xs)]

    chains: list[tuple[np.ndarray, bool]] = []

    def walk(start_y: int, start_x: int) -> None:
        if visited[start_y, start_x]:
            return
        pts = [(start_x, start_y)]
        visited[start_y, start_x] = True
        cy, cx = start_y, start_x
        prev_dy, prev_dx = 0, 0

        while True:
            best = None
            best_score = -999.0
            for dy, dx in _NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and active[ny, nx] and not visited[ny, nx]:
                    # Prefer continuing straight + cardinal steps before diagonals
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
            pts.append((nx, ny))
            prev_dy, prev_dx = dy, dx
            cy, cx = ny, nx

        if len(pts) < 3:
            return
        # Check if tail connects back to head
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


def _thin_zhang_suen_approx(binary: np.ndarray) -> np.ndarray:
    """Fast morphological thinning to 1px skeleton when closing dilated edges."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(6):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return cv2.bitwise_or(skel, img)


def vectorize_canny(
    image_bgr: np.ndarray,
    options: CannyOptions | None = None,
) -> tuple[str, dict]:
    """Extract clean Bezier edge strokes using auto-threshold Canny edge detection."""
    opts = options or CannyOptions()
    h, w = image_bgr.shape[:2]

    denoised = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=30, sigmaSpace=30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    k = opts.blur_ksize if opts.blur_ksize % 2 == 1 else opts.blur_ksize + 1

    # Multi-channel Lab Canny so pastel colors (yellow/pink on white) are captured alongside ink
    edge_channels = []
    l_low, l_high = 50, 150
    for ch_idx, scale_thresh in ((0, 1.0), (1, 0.55), (2, 0.55)):
        ch = lab[:, :, ch_idx]
        if k >= 3:
            ch = cv2.GaussianBlur(ch, (k, k), 0)
        if opts.low_thresh is not None and opts.high_thresh is not None:
            ch_low = int(opts.low_thresh * scale_thresh)
            ch_high = int(opts.high_thresh * scale_thresh)
        elif ch_idx == 0:
            v = float(np.median(ch))
            ch_low = int(max(10, (1.0 - opts.sigma) * v * 0.45))
            ch_high = int(min(250, max(ch_low + 25, (1.0 + opts.sigma) * v * 0.85)))
        else:
            ch_low, ch_high = 18, 52
        if ch_idx == 0:
            l_low, l_high = ch_low, ch_high
        edge_channels.append(cv2.Canny(ch, ch_low, ch_high, L2gradient=True))
    low, high = l_low, l_high

    edges = np.maximum.reduce(edge_channels)
    if opts.dilate_close:
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        edges = _thin_zhang_suen_approx(closed)
    else:
        edges = _thin_zhang_suen_approx(edges)

    chains = _trace_edge_chains(edges)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    if opts.style == "neon":
        bg_fill = "#0c0a18"
        default_stroke = "#ff5cc8"
    elif opts.style == "blueprint":
        bg_fill = "#0e2a47"
        default_stroke = "#d8f0ff"
    else:
        bg_fill = "#ffffff"
        default_stroke = "#2b1622"

    defs = ""
    extra_attr = ""
    if opts.style == "neon":
        defs = (
            "  <defs>\n"
            '    <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">\n'
            '      <feGaussianBlur stdDeviation="2.5" result="blur"/>\n'
            '      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>\n'
            "    </filter>\n"
            "  </defs>"
        )
        extra_attr = ' filter="url(#neon-glow)"'

    path_lines: list[str] = []
    for pts, is_closed in chains:
        if len(pts) < opts.min_length:
            continue
        d = fit_closed_contour_to_commands(
            pts,
            tolerance=opts.tolerance,
            corner_angle_deg=48.0,
            closed=is_closed,
        )
        if not d:
            continue

        if opts.style == "colored":
            # Sample average color around the chain (offset 2px inward/local)
            ix = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            med_c = np.median(rgb[iy, ix], axis=0)
            stroke = rgb_to_hex(med_c)
        else:
            stroke = default_stroke

        path_lines.append(
            f'  <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{opts.stroke_width:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra_attr}/>'
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f"  <!-- Generated by SVGder (Canny engine: {len(path_lines)} strokes, thresholds={low}/{high}) -->",
    ]
    if defs:
        lines.append(defs)
    lines.append(f'  <rect width="100%" height="100%" fill="{bg_fill}"/>')
    lines.extend(path_lines)
    lines.append("</svg>")

    stats = {
        "engine": "canny",
        "width": w,
        "height": h,
        "strokes": len(path_lines),
        "thresholds": [low, high],
        "style": opts.style,
    }
    return "\n".join(lines), stats
