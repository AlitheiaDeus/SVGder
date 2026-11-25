"""CLI entrypoint for SVGder."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from svgder.engines.blobs import BlobOptions, vectorize_blobs
from svgder.engines.canny import CannyOptions, vectorize_canny
from svgder.engines.gremlin import GremlinOptions, vectorize_gremlin


def convert_file(
    input_path: Path,
    output_path: Path,
    mode: str = "gremlin",
    colors: int = 24,
    tolerance: float = 0.85,
    background: str = "auto",
    canny_style: str = "ink",
    stroke_width: float = 2.0,
    n_blobs: int = 240,
    soft_gaussian: bool = True,
) -> dict:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    t0 = time.perf_counter()
    if mode == "gremlin":
        svg, stats = vectorize_gremlin(
            img,
            GremlinOptions(
                colors=colors,
                tolerance=tolerance,
                background=background,  # type: ignore[arg-type]
            ),
        )
    elif mode == "canny":
        svg, stats = vectorize_canny(
            img,
            CannyOptions(
                tolerance=tolerance,
                stroke_width=stroke_width,
                style=canny_style,  # type: ignore[arg-type]
            ),
        )
    elif mode == "blobs":
        svg, stats = vectorize_blobs(
            img,
            BlobOptions(
                n_blobs=n_blobs,
                soft_gaussian=soft_gaussian,
            ),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed = time.perf_counter() - t0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    stats["elapsed_s"] = round(elapsed, 2)
    stats["bytes"] = len(svg.encode("utf-8"))
    stats["output"] = str(output_path)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="svgder",
        description="Convert raster images into crisp SVGs using Gremlin, Canny, or Gaussian Blob engines.",
    )
    parser.add_argument("input", nargs="?", help="Input raster image path (or omit with --serve).")
    parser.add_argument("-o", "--output", help="Output .svg path (defaults to <input>.svg).")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["gremlin", "canny", "blobs", "all"],
        default="gremlin",
        help="Vectorization engine (default: gremlin).",
    )
    parser.add_argument("--colors", type=int, default=24, help="Palette size for gremlin mode (default: 24).")
    parser.add_argument("--tolerance", type=float, default=0.85, help="Bezier curve fit tolerance in px (default: 0.85).")
    parser.add_argument("--background", choices=["auto", "transparent", "white"], default="auto")
    parser.add_argument("--canny-style", choices=["ink", "colored", "neon", "blueprint"], default="ink")
    parser.add_argument("--stroke-width", type=float, default=2.0)
    parser.add_argument("--blobs", type=int, default=240, help="Number of Gaussian splats in blobs mode.")
    parser.add_argument("--hard-blobs", action="store_true", help="Use crisp ellipses instead of Gaussian gradients.")
    parser.add_argument("--serve", action="store_true", help="Launch the interactive SVGder Studio web UI.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve (default: 8765).")

    args = parser.parse_args(argv)

    if args.serve:
        from svgder.server import run_server

        run_server(port=args.port)
        return 0

    if not args.input:
        parser.error("Provide an input image path or pass --serve to launch SVGder Studio.")

    inp = Path(args.input)
    if args.mode == "all":
        stem = inp.with_suffix("")
        for m in ("gremlin", "canny", "blobs"):
            out = Path(f"{stem}_{m}.svg")
            stats = convert_file(
                inp,
                out,
                mode=m,
                colors=args.colors,
                tolerance=args.tolerance,
                background=args.background,
                canny_style=args.canny_style,
                stroke_width=args.stroke_width,
                n_blobs=args.blobs,
                soft_gaussian=not args.hard_blobs,
            )
            print(f"[{m}] -> {out} | {stats}")
        return 0

    out = Path(args.output) if args.output else inp.with_suffix(".svg")
    stats = convert_file(
        inp,
        out,
        mode=args.mode,
        colors=args.colors,
        tolerance=args.tolerance,
        background=args.background,
        canny_style=args.canny_style,
        stroke_width=args.stroke_width,
        n_blobs=args.blobs,
        soft_gaussian=not args.hard_blobs,
    )
    print(f"[{args.mode}] -> {out} | {stats}")
    return 0
