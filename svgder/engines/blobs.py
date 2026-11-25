"""Gaussian Blob Splatting Vectorizer: Greedy error-guided 2D Gaussian/ellipse optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from svgder.curves import rgb_to_hex


@dataclass
class BlobOptions:
    n_blobs: int = 240
    work_size: int = 220
    candidates_per_step: int = 28
    soft_gaussian: bool = True
    seed: int = 42


def _make_gaussian_patch(
    bw: int,
    bh: int,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
    soft: bool,
) -> np.ndarray:
    """Render an oriented 2D Gaussian or solid ellipse mask in [0, 1] over a local grid."""
    xs = np.arange(bw, dtype=np.float32) - cx
    ys = np.arange(bh, dtype=np.float32) - cy
    gx, gy = np.meshgrid(xs, ys)
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    u = gx * ca + gy * sa
    v = -gx * sa + gy * ca
    dist2 = (u / max(1.0, rx)) ** 2 + (v / max(1.0, ry)) ** 2
    if soft:
        # Smooth truncated Gaussian falloff reaching ~0 at dist2 >= 1.0
        mask = np.exp(-2.8 * dist2)
        mask[dist2 > 1.25] = 0.0
        return mask.astype(np.float32)
    return (dist2 <= 1.0).astype(np.float32)


def _optimal_color_and_gain(
    target_patch: np.ndarray,
    canvas_patch: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Closed-form MSE-optimal RGB color and MSE reduction for a given spatial mask."""
    w = (alpha * mask)[:, :, None]
    w2 = float(np.sum(w * w))
    if w2 < 1e-6:
        return np.zeros(3, dtype=np.float32), -1e18

    residual = target_patch - (1.0 - w) * canvas_patch
    color = np.sum(w * residual, axis=(0, 1)) / w2
    color = np.clip(color, 0.0, 255.0)

    before_err = float(np.sum((target_patch - canvas_patch) ** 2))
    new_patch = (1.0 - w) * canvas_patch + w * color[None, None, :]
    after_err = float(np.sum((target_patch - new_patch) ** 2))
    return color, before_err - after_err


def _eval_blob(
    target: np.ndarray,
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float,
    alpha: float,
    soft: bool,
) -> tuple[float, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    h, w = target.shape[:2]
    extent = int(math.ceil(max(rx, ry) * (1.15 if soft else 1.05))) + 1
    x0 = max(0, int(math.floor(cx - extent)))
    y0 = max(0, int(math.floor(cy - extent)))
    x1 = min(w, int(math.ceil(cx + extent)))
    y1 = min(h, int(math.ceil(cy + extent)))
    if x1 <= x0 or y1 <= y0:
        return -1e18, np.zeros(3, dtype=np.float32), (0, 0, 0, 0), np.zeros((0, 0), dtype=np.float32)

    mask = _make_gaussian_patch(x1 - x0, y1 - y0, cx - x0, cy - y0, rx, ry, angle, soft)
    color, gain = _optimal_color_and_gain(target[y0:y1, x0:x1], canvas[y0:y1, x0:x1], mask, alpha)
    return gain, color, (x0, y0, x1, y1), mask


def vectorize_blobs(
    image_bgr: np.ndarray,
    options: BlobOptions | None = None,
) -> tuple[str, dict]:
    """Chunk an image into optimal 2D Gaussian / ellipse splats."""
    opts = options or BlobOptions()
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(1.0, opts.work_size / float(max(orig_w, orig_h)))
    ww = max(32, int(round(orig_w * scale)))
    wh = max(32, int(round(orig_h * scale)))

    small_bgr = cv2.resize(image_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
    target = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Background color = median of image borders
    border = np.vstack([target[0, :], target[-1, :], target[:, 0], target[:, -1]])
    bg_rgb = np.median(border, axis=0)
    bg_hex = rgb_to_hex(bg_rgb.astype(int))
    canvas = np.broadcast_to(bg_rgb[None, None, :], target.shape).copy()

    rng = np.random.default_rng(opts.seed)
    inv_scale_x = orig_w / float(ww)
    inv_scale_y = orig_h / float(wh)

    defs: list[str] = []
    blobs_svg: list[str] = []

    for i in range(opts.n_blobs):
        err_map = np.sum((target - canvas) ** 2, axis=2)
        total_err = float(np.sum(err_map))
        if total_err < 1e-3:
            break

        probs = err_map.ravel()
        probs = probs / float(np.sum(probs))

        best_gain = -1e18
        best_params = None
        best_color = None
        best_box = None
        best_mask = None

        # Early blobs are larger; later blobs refine finer details
        progress = i / max(1, opts.n_blobs)
        max_r = max(5.0, min(ww, wh) * (0.34 * (1.0 - 0.72 * progress)))
        min_r = max(2.2, max_r * 0.16)

        sampled_flat = rng.choice(ww * wh, size=opts.candidates_per_step, p=probs)
        for flat_idx in sampled_flat:
            cy = float(flat_idx // ww)
            cx = float(flat_idx % ww)
            rx = float(rng.uniform(min_r, max_r))
            ry = float(rng.uniform(min_r, max_r))
            angle = float(rng.uniform(0.0, math.pi))
            alpha = float(rng.uniform(0.45, 0.92))
            gain, color, box, mask = _eval_blob(target, canvas, cx, cy, rx, ry, angle, alpha, opts.soft_gaussian)
            if gain > best_gain:
                best_gain = gain
                best_params = [cx, cy, rx, ry, angle, alpha]
                best_color = color
                best_box = box
                best_mask = mask

        if best_params is None or best_gain <= 0:
            continue

        # Fast coordinate hill-climbing refinement
        steps = [3.0, 3.0, max(1.5, best_params[2] * 0.25), max(1.5, best_params[3] * 0.25), 0.35, 0.12]
        for round_scale in (1.0, 0.45):
            for p_idx in range(6):
                for sign in (-1.0, 1.0):
                    cand = list(best_params)
                    cand[p_idx] += sign * steps[p_idx] * round_scale
                    cand[0] = float(np.clip(cand[0], 0.0, ww - 1.0))
                    cand[1] = float(np.clip(cand[1], 0.0, wh - 1.0))
                    cand[2] = float(np.clip(cand[2], 1.8, ww * 0.55))
                    cand[3] = float(np.clip(cand[3], 1.8, wh * 0.55))
                    cand[5] = float(np.clip(cand[5], 0.2, 0.96))
                    gain, color, box, mask = _eval_blob(
                        target, canvas, cand[0], cand[1], cand[2], cand[3], cand[4], cand[5], opts.soft_gaussian
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_params = cand
                        best_color = color
                        best_box = box
                        best_mask = mask

        x0, y0, x1, y1 = best_box
        w_patch = (best_params[5] * best_mask)[:, :, None]
        canvas[y0:y1, x0:x1] = (1.0 - w_patch) * canvas[y0:y1, x0:x1] + w_patch * best_color[None, None, :]

        ocx = best_params[0] * inv_scale_x
        ocy = best_params[1] * inv_scale_y
        orx = best_params[2] * inv_scale_x
        ory = best_params[3] * inv_scale_y
        deg = math.degrees(best_params[4])
        alpha_val = best_params[5]
        hex_c = rgb_to_hex(best_color.astype(int))

        if opts.soft_gaussian:
            gid = f"g{i}"
            defs.append(
                f'    <radialGradient id="{gid}" cx="50%" cy="50%" r="50%">\n'
                f'      <stop offset="0%" stop-color="{hex_c}" stop-opacity="{alpha_val:.3f}"/>\n'
                f'      <stop offset="55%" stop-color="{hex_c}" stop-opacity="{alpha_val * 0.45:.3f}"/>\n'
                f'      <stop offset="100%" stop-color="{hex_c}" stop-opacity="0"/>\n'
                f"    </radialGradient>"
            )
            blobs_svg.append(
                f'  <ellipse cx="{ocx:.1f}" cy="{ocy:.1f}" rx="{orx * 1.12:.1f}" ry="{ory * 1.12:.1f}" '
                f'transform="rotate({deg:.1f} {ocx:.1f} {ocy:.1f})" fill="url(#{gid})"/>'
            )
        else:
            blobs_svg.append(
                f'  <ellipse cx="{ocx:.1f}" cy="{ocy:.1f}" rx="{orx:.1f}" ry="{ory:.1f}" '
                f'transform="rotate({deg:.1f} {ocx:.1f} {ocy:.1f})" fill="{hex_c}" fill-opacity="{alpha_val:.3f}"/>'
            )

    rmse = float(np.sqrt(np.mean((target - canvas) ** 2)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" width="{orig_w}" height="{orig_h}">',
        f"  <!-- Generated by SVGder (Gaussian Blobs engine: {len(blobs_svg)} splats, RMSE={rmse:.2f}) -->",
    ]
    if defs:
        lines.append("  <defs>")
        lines.extend(defs)
        lines.append("  </defs>")
    lines.append(f'  <rect width="100%" height="100%" fill="{bg_hex}"/>')
    lines.extend(blobs_svg)
    lines.append("</svg>")

    stats = {
        "engine": "blobs",
        "width": orig_w,
        "height": orig_h,
        "blobs": len(blobs_svg),
        "rmse": round(rmse, 2),
        "soft_gaussian": opts.soft_gaussian,
    }
    return "\n".join(lines), stats
