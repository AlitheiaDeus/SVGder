"""Sub-pixel contour extraction and corner-preserving cubic Bezier curve fitting."""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


def _contour_perimeter(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0, append=pts[:1])
    return float(np.linalg.norm(diffs, axis=1).sum())


def _point_at_arc_offset(pts: np.ndarray, cum_len: np.ndarray, total_len: float, idx: int, offset: float) -> np.ndarray:
    """Sample point along closed polyline at signed arc-length offset from pts[idx]."""
    n = len(pts)
    target = (cum_len[idx] + offset) % total_len
    pos = int(np.searchsorted(cum_len, target, side="right")) - 1
    pos = pos % n
    next_pos = (pos + 1) % n
    seg_len = cum_len[next_pos] - cum_len[pos]
    if next_pos == 0:
        seg_len = total_len - cum_len[pos]
    if seg_len <= 1e-6:
        return pts[pos]
    t = (target - cum_len[pos]) / seg_len
    if target < cum_len[pos]:
        t = (target + total_len - cum_len[pos]) / seg_len
    t = max(0.0, min(1.0, float(t)))
    return pts[pos] * (1.0 - t) + pts[next_pos] * t


def detect_corners(
    pts: np.ndarray,
    window_px: float = 4.5,
    angle_threshold_deg: float = 42.0,
    min_corner_sep_px: float = 6.0,
) -> list[int]:
    """Find crisp corner indices on a closed contour using arc-length chord turning angles."""
    n = len(pts)
    if n < 6:
        return list(range(n))

    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len < window_px * 2.5:
        return []

    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens[:-1])))
    w = min(window_px, total_len * 0.18)

    angles = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_prev = _point_at_arc_offset(pts, cum_len, total_len, i, -w)
        p_next = _point_at_arc_offset(pts, cum_len, total_len, i, w)
        v1 = pts[i] - p_prev
        v2 = p_next - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-5 or n2 < 1e-5:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles[i] = math.degrees(math.acos(cos_a))

    # Local maxima over a small neighborhood
    candidates: list[tuple[float, int]] = []
    nbr = max(2, min(8, n // 20))
    for i in range(n):
        if angles[i] < angle_threshold_deg:
            continue
        is_max = True
        for d in range(-nbr, nbr + 1):
            if d == 0:
                continue
            j = (i + d) % n
            if angles[j] > angles[i] or (angles[j] == angles[i] and j < i):
                is_max = False
                break
        if is_max:
            candidates.append((angles[i], i))

    # Non-max suppression by arc-length separation
    candidates.sort(reverse=True)
    kept: list[int] = []
    for _, idx in candidates:
        too_close = False
        for k in kept:
            d = abs(cum_len[idx] - cum_len[k])
            d = min(d, total_len - d)
            if d < min_corner_sep_px:
                too_close = True
                break
        if not too_close:
            kept.append(idx)

    kept.sort()
    return kept


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / norm


def _chord_params(seg: np.ndarray) -> np.ndarray:
    diffs = np.diff(seg, axis=0)
    lens = np.linalg.norm(diffs, axis=1)
    u = np.concatenate(([0.0], np.cumsum(lens)))
    total = float(u[-1])
    if total < 1e-8:
        return np.linspace(0.0, 1.0, len(seg))
    return u / total


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
    om = 1.0 - u
    b0 = (om ** 3)[:, None]
    b1 = (3.0 * (om ** 2) * u)[:, None]
    b2 = (3.0 * om * (u ** 2))[:, None]
    b3 = (u ** 3)[:, None]
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _generate_bezier(seg: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit cubic Bezier control points (p0, p1, p2, p3) using Schneider's least-squares formulation."""
    p0 = seg[0]
    p3 = seg[-1]
    chord = float(np.linalg.norm(p3 - p0))

    if len(seg) <= 2 or chord < 1e-6:
        dist = chord / 3.0
        return p0, p0 + t1 * dist, p3 + t2 * dist, p3

    om = 1.0 - u
    a1 = (3.0 * (om ** 2) * u)[:, None] * t1[None, :]
    a2 = (3.0 * om * (u ** 2))[:, None] * t2[None, :]

    c00 = float(np.sum(a1 * a1))
    c01 = float(np.sum(a1 * a2))
    c11 = float(np.sum(a2 * a2))

    b0 = ((om ** 3) + 3.0 * (om ** 2) * u)[:, None] * p0[None, :]
    b3 = ((u ** 3) + 3.0 * om * (u ** 2))[:, None] * p3[None, :]
    rhs = seg - (b0 + b3)

    x0 = float(np.sum(rhs * a1))
    x1 = float(np.sum(rhs * a2))

    det = c00 * c11 - c01 * c01
    if abs(det) > 1e-9:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det
    else:
        alpha1 = -1.0
        alpha2 = -1.0

    # Wu/Barsky heuristic fallback if handles invert or explode
    eps = 1e-6 * chord
    if alpha1 < eps or alpha2 < eps or alpha1 > chord * 2.5 or alpha2 > chord * 2.5:
        dist = chord / 3.0
        return p0, p0 + t1 * dist, p3 + t2 * dist, p3

    return p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3


def _reparameterize(seg: np.ndarray, u: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step to refine parameter values u toward closest points on the Bezier."""
    q = _eval_cubic(p0, p1, p2, p3, u)
    # First derivative Q'(u)
    om = 1.0 - u
    d1 = (
        (3.0 * (om ** 2))[:, None] * (p1 - p0)
        + (6.0 * om * u)[:, None] * (p2 - p1)
        + (3.0 * (u ** 2))[:, None] * (p3 - p2)
    )
    # Second derivative Q''(u)
    d2 = (6.0 * om)[:, None] * (p2 - 2.0 * p1 + p0) + (6.0 * u)[:, None] * (p3 - 2.0 * p2 + p1)

    diff = q - seg
    num = np.sum(diff * d1, axis=1)
    den = np.sum(d1 * d1 + diff * d2, axis=1)
    safe = np.abs(den) > 1e-8
    u_new = u.copy()
    u_new[safe] = np.clip(u[safe] - num[safe] / den[safe], 0.0, 1.0)
    u_new[0] = 0.0
    u_new[-1] = 1.0
    return u_new


def _max_error(seg: np.ndarray, u: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[float, int]:
    pts = _eval_cubic(p0, p1, p2, p3, u)
    dists = np.linalg.norm(pts - seg, axis=1)
    idx = int(np.argmax(dists))
    return float(dists[idx]), idx


def _endpoint_tangent(seg: np.ndarray, at_start: bool, span: int = 4) -> np.ndarray:
    k = min(span, len(seg) - 1)
    if k <= 0:
        return np.array([1.0, 0.0], dtype=np.float64)
    if at_start:
        return _unit(seg[k] - seg[0])
    return _unit(seg[-1 - k] - seg[-1])


def _fit_cubic_recursive(
    seg: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tolerance: float,
    depth: int = 0,
) -> list[tuple[str, tuple[np.ndarray, ...]]]:
    """Fit a polyline segment into a list of ('L', (p3,)) or ('C', (p1, p2, p3)) commands."""
    n = len(seg)
    if n <= 2:
        return [("L", (seg[-1],))]

    # Check if the segment is essentially a straight line
    p0, p3 = seg[0], seg[-1]
    line_vec = p3 - p0
    line_len = float(np.linalg.norm(line_vec))
    if line_len > 1e-6:
        unit_line = line_vec / line_len
        proj = np.dot(seg - p0, unit_line)
        perp = np.linalg.norm((seg - p0) - proj[:, None] * unit_line[None, :], axis=1)
        if float(np.max(perp)) <= tolerance * 0.45 and np.all(proj >= -tolerance) and np.all(proj <= line_len + tolerance):
            return [("L", (p3,))]

    u = _chord_params(seg)
    p0, p1, p2, p3 = _generate_bezier(seg, u, t1, t2)
    err, split_idx = _max_error(seg, u, p0, p1, p2, p3)

    if err <= tolerance or depth >= 8 or n <= 4:
        return [("C", (p1, p2, p3))]

    # Refine parameterization up to 2 iterations
    for _ in range(2):
        u_ref = _reparameterize(seg, u, p0, p1, p2, p3)
        p0, p1, p2, p3 = _generate_bezier(seg, u_ref, t1, t2)
        err, split_idx = _max_error(seg, u_ref, p0, p1, p2, p3)
        u = u_ref
        if err <= tolerance:
            return [("C", (p1, p2, p3))]

    split_idx = max(1, min(n - 2, split_idx))
    t_center = _unit(seg[split_idx + 1] - seg[split_idx - 1])
    left = _fit_cubic_recursive(seg[: split_idx + 1], t1, -t_center, tolerance, depth + 1)
    right = _fit_cubic_recursive(seg[split_idx:], t_center, t2, tolerance, depth + 1)
    return left + right


def _fmt(val: float, decimals: int = 2) -> str:
    s = f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" or s == "" else s


def fit_closed_contour_to_commands(
    pts: np.ndarray,
    tolerance: float = 1.0,
    corner_angle_deg: float = 42.0,
    closed: bool = True,
) -> str:
    """Convert a dense contour Nx2 array into an SVG path string (M ... C/L ... Z)."""
    n = len(pts)
    if n < 2:
        return ""
    if not closed:
        t1 = _endpoint_tangent(pts, at_start=True)
        t2 = _endpoint_tangent(pts, at_start=False)
        cmds = _fit_cubic_recursive(pts, t1, t2, tolerance)
        parts = [f"M {_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
        for kind, coords in cmds:
            if kind == "L":
                p = coords[0]
                parts.append(f"L {_fmt(p[0])} {_fmt(p[1])}")
            else:
                p1, p2, p3 = coords
                parts.append(
                    f"C {_fmt(p1[0])} {_fmt(p1[1])}, {_fmt(p2[0])} {_fmt(p2[1])}, {_fmt(p3[0])} {_fmt(p3[1])}"
                )
        return " ".join(parts)

    if n < 3:
        return ""

    corners = detect_corners(pts, angle_threshold_deg=corner_angle_deg)
    segments: list[tuple[np.ndarray, bool, bool]] = []  # (seg_pts, smooth_start, smooth_end)

    if not corners:
        # Smooth closed loop: split into 2 halves with shared periodic tangents
        mid = n // 2
        seg1 = np.vstack([pts[: mid + 1]])
        seg2 = np.vstack([pts[mid:], pts[:1]])
        segments = [(seg1, True, True), (seg2, True, True)]
        start_pt = pts[0]
    else:
        for idx, c0 in enumerate(corners):
            c1 = corners[(idx + 1) % len(corners)]
            if c1 > c0:
                seg = pts[c0 : c1 + 1]
            else:
                seg = np.vstack([pts[c0:], pts[: c1 + 1]])
            if len(seg) >= 2:
                segments.append((seg, False, False))
        start_pt = segments[0][0][0] if segments else pts[0]

    parts = [f"M {_fmt(start_pt[0])} {_fmt(start_pt[1])}"]
    for seg_idx, (seg, smooth_start, smooth_end) in enumerate(segments):
        if smooth_start and not corners:
            # Periodic tangent at start/end of halves
            i0 = 0 if seg_idx == 0 else n // 2
            i1 = n // 2 if seg_idx == 0 else 0
            t1 = _unit(pts[(i0 + 3) % n] - pts[(i0 - 3) % n])
            t2 = -_unit(pts[(i1 + 3) % n] - pts[(i1 - 3) % n])
        else:
            t1 = _endpoint_tangent(seg, at_start=True)
            t2 = _endpoint_tangent(seg, at_start=False)
        cmds = _fit_cubic_recursive(seg, t1, t2, tolerance)
        for kind, coords in cmds:
            if kind == "L":
                p = coords[0]
                parts.append(f"L {_fmt(p[0])} {_fmt(p[1])}")
            else:
                p1, p2, p3 = coords
                parts.append(
                    f"C {_fmt(p1[0])} {_fmt(p1[1])}, {_fmt(p2[0])} {_fmt(p2[1])}, {_fmt(p3[0])} {_fmt(p3[1])}"
                )
    parts.append("Z")
    return " ".join(parts)


def mask_to_svg_paths(
    mask: np.ndarray,
    tolerance: float = 0.9,
    corner_angle_deg: float = 42.0,
    min_area: float = 6.0,
    upscale: int = 2,
    smooth_sigma: float = 0.65,
) -> list[str]:
    """Extract sub-pixel smooth outer+hole compound SVG path strings from a uint8 binary mask."""
    if np.count_nonzero(mask) == 0:
        return []

    h, w = mask.shape[:2]
    if upscale > 1:
        # Pad 2px so contours touching image boundaries close cleanly
        padded = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        up = cv2.resize(
            padded,
            ((w + 4) * upscale, (h + 4) * upscale),
            interpolation=cv2.INTER_LINEAR,
        )
        if smooth_sigma > 0:
            up = cv2.GaussianBlur(up, (0, 0), smooth_sigma * upscale)
        _, binary = cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)
        offset = 2.0
    else:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        binary = padded
        offset = 1.0

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return []

    hier = hierarchy[0]
    scale_f = float(upscale)
    min_area_scaled = min_area * (scale_f * scale_f)

    # Map each top-level outer contour index -> list of child hole contour indices
    outer_groups: dict[int, list[int]] = {}
    for idx, c in enumerate(contours):
        parent = int(hier[idx][3])
        area = cv2.contourArea(c)
        if parent == -1:
            if area >= min_area_scaled:
                outer_groups[idx] = []
        else:
            if parent in outer_groups and area >= max(4.0, min_area_scaled * 0.35):
                outer_groups[parent].append(idx)

    paths: list[str] = []
    for outer_idx, hole_indices in outer_groups.items():
        subpaths: list[str] = []
        for c_idx in [outer_idx, *hole_indices]:
            raw = contours[c_idx].reshape(-1, 2).astype(np.float64)
            # Map back from upscaled padded coordinates to [0..w, 0..h]
            pts = (raw + 0.5) / scale_f - offset
            pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w))
            pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h))
            d = fit_closed_contour_to_commands(
                pts,
                tolerance=tolerance,
                corner_angle_deg=corner_angle_deg,
                closed=True,
            )
            if d:
                subpaths.append(d)
        if subpaths:
            paths.append(" ".join(subpaths))
    return paths


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
