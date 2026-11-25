"""Smoke tests for SVGder curve fitting and all three vectorization engines."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from svgder.curves import fit_closed_contour_to_commands, mask_to_svg_paths
from svgder.engines.blobs import BlobOptions, vectorize_blobs
from svgder.engines.canny import CannyOptions, vectorize_canny
from svgder.engines.gremlin import GremlinOptions, vectorize_gremlin
from svgder.engines.lowpoly import LowPolyOptions, vectorize_lowpoly


def _make_synthetic_image() -> np.ndarray:
    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    # Circle + star-like diamond in BGR
    import cv2

    cv2.circle(img, (80, 80), 48, (220, 180, 240), -1)
    cv2.circle(img, (80, 80), 48, (50, 25, 70), 4)
    pts = np.array([[80, 50], [95, 80], [80, 110], [65, 80]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (120, 220, 255))
    return img


def test_curve_fitting_valid_path() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    import cv2

    cv2.circle(mask, (50, 50), 25, 255, -1)
    cv2.circle(mask, (50, 50), 10, 0, -1)
    paths = mask_to_svg_paths(mask)
    assert len(paths) == 1
    assert paths[0].startswith("M ")
    assert paths[0].endswith("Z")
    assert "C " in paths[0]

    open_line = np.array([[10.0, 10.0], [50.0, 10.0], [90.0, 10.0]])
    d = fit_closed_contour_to_commands(open_line, closed=False)
    assert d.startswith("M 10 10 L 90 10")


def test_gremlin_engine_valid_svg() -> None:
    img = _make_synthetic_image()
    svg, stats = vectorize_gremlin(img, GremlinOptions(colors=8))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert stats["paths"] > 0
    assert stats["colors"] >= 2


def test_canny_engine_valid_svg() -> None:
    img = _make_synthetic_image()
    svg, stats = vectorize_canny(img, CannyOptions(style="neon"))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert stats["strokes"] > 0


def test_blobs_engine_valid_svg() -> None:
    img = _make_synthetic_image()
    svg, stats = vectorize_blobs(img, BlobOptions(n_blobs=15, work_size=64, candidates_per_step=8))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert stats["blobs"] > 0


def test_lowpoly_engine_valid_svg() -> None:
    img = _make_synthetic_image()
    svg, stats = vectorize_lowpoly(img, LowPolyOptions(points=120))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert stats["triangles"] > 0


if __name__ == "__main__":
    test_curve_fitting_valid_path()
    test_gremlin_engine_valid_svg()
    test_canny_engine_valid_svg()
    test_blobs_engine_valid_svg()
    test_lowpoly_engine_valid_svg()
    print("All SVGder tests passed!")
