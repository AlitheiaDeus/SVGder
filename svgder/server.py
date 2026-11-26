"""FastAPI server for the interactive SVGder Studio."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn

from svgder.engines.blobs import BlobOptions, vectorize_blobs
from svgder.engines.canny import CannyOptions, vectorize_canny
from svgder.engines.floodfill import FloodFillOptions, vectorize_floodfill
from svgder.engines.gremlin import GremlinOptions, vectorize_gremlin
from svgder.engines.lowpoly import LowPolyOptions, vectorize_lowpoly

STATIC_DIR = Path(__file__).parent / "static"
WORKSPACE_ROOT = Path.cwd()

app = FastAPI(title="SVGder Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")
    # Guard max dimension at 1600px for responsive web preview
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ff6fb5"/>
      <stop offset="100%" stop-color="#8be9fd"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="#130f1a"/>
  <rect x="12" y="14" width="8" height="8" rx="1.5" fill="#ff6fb5" opacity="0.9"/>
  <rect x="22" y="14" width="8" height="8" rx="1.5" fill="#ff6fb5" opacity="0.55"/>
  <rect x="12" y="24" width="8" height="8" rx="1.5" fill="#ff6fb5" opacity="0.55"/>
  <path d="M 16 48 C 20 22, 44 42, 48 16" fill="none" stroke="url(#g)" stroke-width="4.5" stroke-linecap="round"/>
  <circle cx="16" cy="48" r="4" fill="#ff6fb5"/>
  <circle cx="48" cy="16" r="4" fill="#8be9fd"/>
  <rect x="44" y="42" width="7" height="7" rx="1.5" fill="#8be9fd" opacity="0.85"/>
</svg>"""


@app.get("/favicon.ico")
def favicon() -> HTMLResponse:
    return HTMLResponse(_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/api/sample")
def get_sample() -> FileResponse:
    sample = WORKSPACE_ROOT / "Ramapillar.jpeg"
    if not sample.exists():
        raise HTTPException(status_code=404, detail="Ramapillar.jpeg not found in workspace.")
    return FileResponse(sample, media_type="image/jpeg")


@app.post("/api/vectorize")
async def api_vectorize(
    file: UploadFile | None = File(default=None),
    use_sample: bool = Form(default=False),
    mode: str = Form(default="gremlin"),
    colors: int = Form(default=24),
    tolerance: float = Form(default=0.85),
    background: str = Form(default="auto"),
    ink_on_top: bool = Form(default=True),
    canny_style: str = Form(default="ink"),
    stroke_width: float = Form(default=2.0),
    sigma: float = Form(default=0.33),
    n_blobs: int = Form(default=220),
    soft_gaussian: bool = Form(default=True),
    poly_points: int = Form(default=800),
    wireframe: bool = Form(default=False),
    flood_color_tol: float = Form(default=18.0),
    flood_edge_sens: float = Form(default=28.0),
    flood_min_area: float = Form(default=20.0),
    flood_max_stroke: float = Form(default=12.0),
    flood_detect_strokes: bool = Form(default=True),
) -> JSONResponse:
    if file is not None:
        raw = await file.read()
        img = _decode_image_bytes(raw)
    elif use_sample:
        sample = WORKSPACE_ROOT / "Ramapillar.jpeg"
        if not sample.exists():
            raise HTTPException(status_code=404, detail="Ramapillar.jpeg not found.")
        img = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or set use_sample=true.")

    t0 = time.perf_counter()
    if mode == "gremlin":
        svg, stats = vectorize_gremlin(
            img,
            GremlinOptions(
                colors=max(2, min(48, colors)),
                tolerance=max(0.2, min(5.0, tolerance)),
                background=background,  # type: ignore[arg-type]
                ink_on_top=ink_on_top,
            ),
        )
    elif mode == "canny":
        svg, stats = vectorize_canny(
            img,
            CannyOptions(
                sigma=max(0.05, min(0.9, sigma)),
                tolerance=max(0.2, min(5.0, tolerance)),
                stroke_width=max(0.5, min(10.0, stroke_width)),
                style=canny_style,  # type: ignore[arg-type]
            ),
        )
    elif mode == "blobs":
        svg, stats = vectorize_blobs(
            img,
            BlobOptions(
                n_blobs=max(20, min(600, n_blobs)),
                soft_gaussian=soft_gaussian,
            ),
        )
    elif mode == "lowpoly":
        svg, stats = vectorize_lowpoly(
            img,
            LowPolyOptions(
                points=max(80, min(3000, poly_points)),
                wireframe=wireframe,
            ),
        )
    elif mode == "floodfill":
        svg, stats = vectorize_floodfill(
            img,
            FloodFillOptions(
                color_tolerance=max(3.0, min(60.0, flood_color_tol)),
                edge_sensitivity=max(0.0, min(80.0, flood_edge_sens)),
                tolerance=max(0.2, min(5.0, tolerance)),
                min_area=max(4.0, min(500.0, flood_min_area)),
                max_stroke_width=max(2.0, min(30.0, flood_max_stroke)),
                detect_strokes=flood_detect_strokes,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    stats["bytes"] = len(svg.encode("utf-8"))
    return JSONResponse({"svg": svg, "stats": stats})


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
