"""FastAPI server for the interactive SVGder Studio."""

from __future__ import annotations

import base64
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
from svgder.engines.floodfill import FloodFillOptions, preview_floodfill, vectorize_floodfill
from svgder.engines.gremlin import GremlinOptions, preview_gremlin, vectorize_gremlin
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


def _wrap_raster_preview_svg(bgr: np.ndarray, orig_w: int, orig_h: int) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode preview PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" '
        f'width="{orig_w}" height="{orig_h}">'
        f'<image href="data:image/png;base64,{b64}" width="{orig_w}" height="{orig_h}" '
        f'preserveAspectRatio="none"/></svg>'
    )


@app.post("/api/preview")
async def api_preview(
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

    orig_h, orig_w = img.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 560:
        scale = 560.0 / max_dim
        small = cv2.resize(img, (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = img

    t0 = time.perf_counter()
    if mode == "gremlin":
        prev_bgr = preview_gremlin(
            small,
            GremlinOptions(
                colors=max(2, min(48, colors)),
                background=background,  # type: ignore[arg-type]
                ink_on_top=ink_on_top,
            ),
        )
        svg = _wrap_raster_preview_svg(prev_bgr, orig_w, orig_h)
    elif mode == "floodfill":
        scaled_area = max(4.0, flood_min_area * (scale * scale))
        prev_bgr = preview_floodfill(
            small,
            FloodFillOptions(
                color_tolerance=max(3.0, min(60.0, flood_color_tol)),
                edge_sensitivity=max(0.0, min(80.0, flood_edge_sens)),
                min_area=scaled_area,
            ),
        )
        svg = _wrap_raster_preview_svg(prev_bgr, orig_w, orig_h)
    elif mode == "canny":
        svg, _ = vectorize_canny(
            small,
            CannyOptions(
                sigma=max(0.05, min(0.9, sigma)),
                tolerance=max(0.2, min(5.0, tolerance)),
                stroke_width=max(0.5, min(10.0, stroke_width)),
                style=canny_style,  # type: ignore[arg-type]
            ),
        )
    elif mode == "blobs":
        svg, _ = vectorize_blobs(
            small,
            BlobOptions(
                n_blobs=max(20, min(160, n_blobs)),
                soft_gaussian=soft_gaussian,
            ),
        )
    elif mode == "lowpoly":
        svg, _ = vectorize_lowpoly(
            small,
            LowPolyOptions(
                points=max(80, min(1200, poly_points)),
                wireframe=wireframe,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return JSONResponse({"svg": svg, "elapsed_ms": elapsed_ms, "mode": mode})


def _norm_mae_pct(orig_bgr: np.ndarray, approx_bgr: np.ndarray) -> float:
    o = orig_bgr.astype(np.float32)
    a = approx_bgr.astype(np.float32)
    mean_c = np.mean(o, axis=(0, 1), keepdims=True)
    base = float(np.sum(np.abs(o - mean_c)))
    err = float(np.sum(np.abs(o - a)))
    if base < 1.0:
        return 0.0 if err < 1.0 else 100.0
    return (err / base) * 100.0


@app.post("/api/autotune")
async def api_autotune(
    file: UploadFile | None = File(default=None),
    use_sample: bool = Form(default=False),
    mode: str = Form(default="gremlin"),
    background: str = Form(default="auto"),
    ink_on_top: bool = Form(default=True),
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

    orig_h, orig_w = img.shape[:2]
    max_dim = max(orig_h, orig_w)
    scale = min(1.0, 320.0 / max_dim)
    small = (
        cv2.resize(img, (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img
    )

    t0 = time.perf_counter()
    params: dict = {}

    if mode == "gremlin":
        best_loss = 1e9
        best_c = 24
        for c in (8, 12, 16, 20, 26, 32):
            prev = preview_gremlin(
                small,
                GremlinOptions(colors=c, background=background, ink_on_top=ink_on_top),  # type: ignore[arg-type]
            )
            diff_pct = _norm_mae_pct(small, prev)
            loss = diff_pct + 0.32 * c
            if loss < best_loss:
                best_loss = loss
                best_c = c
        params = {"colors": best_c}
    elif mode == "floodfill":
        best_loss = 1e9
        best_pair = (18, 28, 20)
        for tol_v, edge_v in ((10, 22), (14, 26), (18, 28), (24, 34), (32, 42)):
            scaled_area = max(3.0, 20.0 * (scale * scale))
            prev = preview_floodfill(
                small,
                FloodFillOptions(color_tolerance=float(tol_v), edge_sensitivity=float(edge_v), min_area=scaled_area),
            )
            diff_pct = _norm_mae_pct(small, prev)
            # Penalize overly tight tolerance that creates too many micro-regions
            loss = diff_pct + max(0.0, (24.0 - tol_v) * 0.25)
            if loss < best_loss:
                best_loss = loss
                best_pair = (tol_v, edge_v, 20)
        params = {
            "flood_color_tol": best_pair[0],
            "flood_edge_sens": best_pair[1],
            "flood_min_area": best_pair[2],
        }
    elif mode == "canny":
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        best_sig = 0.33
        best_dist = 1e9
        for sig in (0.18, 0.27, 0.33, 0.45, 0.58):
            lo = int(max(0, (1.0 - sig) * med))
            hi = int(min(255, (1.0 + sig) * med))
            edges = cv2.Canny(gray, lo, hi)
            density = float(np.mean(edges > 0))
            dist = abs(density - 0.065)
            if dist < best_dist:
                best_dist = dist
                best_sig = sig
        params = {"sigma": best_sig}
    elif mode == "lowpoly":
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        pts = int(np.clip(350 + lap_var * 0.65, 350, 1800))
        pts = int(round(pts / 50.0) * 50)
        params = {"poly_points": pts}
    elif mode == "blobs":
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_std = float(np.mean(np.std(lab, axis=(0, 1))))
        nb = int(np.clip(80 + color_std * 6.0, 100, 380))
        nb = int(round(nb / 20.0) * 20)
        params = {"n_blobs": nb}
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return JSONResponse({"mode": mode, "params": params, "elapsed_ms": elapsed_ms})


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
