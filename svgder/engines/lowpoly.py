"""Low-Poly Delaunay Triangulation Vectorizer: Edge-aware adaptive mesh."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from svgder.curves import rgb_to_hex


@dataclass
class LowPolyOptions:
    points: int = 800
    edge_bias: float = 0.72
    wireframe: bool = False
    seed: int = 42


def _sample_points_with_spacing(
    n_target: int,
    probs: np.ndarray,
    w: int,
    h: int,
    min_dist: float,
    occupied: list[tuple[float, float]],
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Sample points from a 2D probability map while enforcing minimum spacing."""
    if n_target <= 0:
        return []
    cell = max(2.0, min_dist)
    gw = int(math.ceil(w / cell)) + 1
    gh = int(math.ceil(h / cell)) + 1
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for px, py in occupied:
        grid.setdefault((int(px / cell), int(py / cell)), []).append((px, py))

    min_dist2 = min_dist * min_dist
    added: list[tuple[float, float]] = []
    flat_probs = probs.ravel()
    total = float(np.sum(flat_probs))
    if total <= 0:
        flat_probs = np.full(w * h, 1.0 / (w * h), dtype=np.float64)
    else:
        flat_probs = flat_probs / total

    # Draw in batches to keep overhead low
    max_attempts = max(n_target * 8, 500)
    batch_size = min(max_attempts, max(256, n_target * 4))
    attempts = 0
    while len(added) < n_target and attempts < max_attempts:
        batch = rng.choice(w * h, size=batch_size, p=flat_probs)
        for flat_idx in batch:
            attempts += 1
            y = float(flat_idx // w) + float(rng.uniform(-0.35, 0.35))
            x = float(flat_idx % w) + float(rng.uniform(-0.35, 0.35))
            x = max(1.0, min(w - 2.0, x))
            y = max(1.0, min(h - 2.0, y))
            gx, gy = int(x / cell), int(y / cell)
            too_close = False
            for nx in (gx - 1, gx, gx + 1):
                for ny in (gy - 1, gy, gy + 1):
                    for ox, oy in grid.get((nx, ny), ()):
                        dx = x - ox
                        dy = y - oy
                        if dx * dx + dy * dy < min_dist2:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if not too_close:
                pt = (x, y)
                added.append(pt)
                grid.setdefault((gx, gy), []).append(pt)
                if len(added) >= n_target:
                    break
    return added


def vectorize_lowpoly(
    image_bgr: np.ndarray,
    options: LowPolyOptions | None = None,
) -> tuple[str, dict]:
    """Convert an image into a crisp edge-aware Delaunay low-poly SVG."""
    opts = options or LowPolyOptions()
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(opts.seed)

    # Multi-channel gradient magnitude in Lab space so pastel/chroma edges are preserved
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    grad_sq = np.zeros((h, w), dtype=np.float32)
    for ch in range(3):
        gx = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 0, 1, ksize=3)
        grad_sq += gx * gx + gy * gy
    grad = np.sqrt(grad_sq)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.0)

    # Border anchors ensure full canvas coverage without missing corner triangles
    border_step = max(16.0, math.sqrt((w * h) / max(32, opts.points)) * 1.4)
    n_bx = max(2, int(round(w / border_step)))
    n_by = max(2, int(round(h / border_step)))
    pts: list[tuple[float, float]] = [
        (0.0, 0.0),
        (float(w - 1), 0.0),
        (0.0, float(h - 1)),
        (float(w - 1), float(h - 1)),
    ]
    for i in range(1, n_bx):
        x = float(i) * (w - 1) / n_bx
        pts.append((x, 0.0))
        pts.append((x, float(h - 1)))
    for j in range(1, n_by):
        y = float(j) * (h - 1) / n_by
        pts.append((0.0, y))
        pts.append((float(w - 1), y))

    remaining = max(12, opts.points - len(pts))
    n_edge = int(round(remaining * max(0.0, min(1.0, opts.edge_bias))))
    n_fill = remaining - n_edge

    # Sharpen gradient weights so vertices cluster cleanly along real edges
    edge_probs = np.power(grad, 1.5)
    min_spacing = max(2.5, math.sqrt((w * h) / max(32, opts.points)) * 0.38)
    edge_pts = _sample_points_with_spacing(n_edge, edge_probs, w, h, min_spacing, pts, rng)
    pts.extend(edge_pts)

    fill_probs = np.ones((h, w), dtype=np.float32)
    fill_pts = _sample_points_with_spacing(n_fill, fill_probs, w, h, min_spacing * 1.35, pts, rng)
    pts.extend(fill_pts)

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for px, py in pts:
        subdiv.insert((float(px), float(py)))

    raw_tris = subdiv.getTriangleList()
    polygons: list[str] = []
    recon = np.zeros_like(rgb)
    tri_count = 0

    for t in raw_tris:
        x1, y1, x2, y2, x3, y3 = (float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5]))
        if (
            min(x1, x2, x3) < -0.5
            or min(y1, y2, y3) < -0.5
            or max(x1, x2, x3) > w - 0.5
            or max(y1, y2, y3) > h - 0.5
        ):
            continue

        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
        if area < 0.5:
            continue

        bx0 = max(0, int(math.floor(min(x1, x2, x3))))
        by0 = max(0, int(math.floor(min(y1, y2, y3))))
        bx1 = min(w, int(math.ceil(max(x1, x2, x3))) + 1)
        by1 = min(h, int(math.ceil(max(y1, y2, y3))) + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue

        local_poly = np.array(
            [[x1 - bx0, y1 - by0], [x2 - bx0, y2 - by0], [x3 - bx0, y3 - by0]],
            dtype=np.int32,
        )
        mask = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_poly, 255)
        ys, xs = np.nonzero(mask)
        if len(xs) > 0:
            mean_rgb = rgb[by0 + ys, bx0 + xs].mean(axis=0)
            recon[by0 + ys, bx0 + xs] = mean_rgb.astype(np.uint8)
        else:
            cx = int(np.clip(round((x1 + x2 + x3) / 3.0), 0, w - 1))
            cy = int(np.clip(round((y1 + y2 + y3) / 3.0), 0, h - 1))
            mean_rgb = rgb[cy, cx].astype(np.float32)

        hex_c = rgb_to_hex(mean_rgb.astype(int))
        pts_str = f"{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} {x3:.1f},{y3:.1f}"
        if opts.wireframe:
            polygons.append(
                f'  <polygon points="{pts_str}" fill="{hex_c}" '
                f'stroke="#1a1826" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>'
            )
        else:
            polygons.append(
                f'  <polygon points="{pts_str}" fill="{hex_c}" '
                f'stroke="{hex_c}" stroke-width="0.6" stroke-linejoin="round"/>'
            )
        tri_count += 1

    rmse = float(np.sqrt(np.mean((rgb.astype(np.float32) - recon.astype(np.float32)) ** 2)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f"  <!-- Generated by SVGder (Low-Poly engine: {tri_count} triangles, {len(pts)} vertices, RMSE={rmse:.2f}) -->",
        *polygons,
        "</svg>",
    ]
    stats = {
        "engine": "lowpoly",
        "width": w,
        "height": h,
        "triangles": tri_count,
        "vertices": len(pts),
        "rmse": round(rmse, 2),
        "wireframe": opts.wireframe,
    }
    return "\n".join(lines), stats
