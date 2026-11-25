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
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    stats["bytes"] = len(svg.encode("utf-8"))
    return JSONResponse({"svg": svg, "stats": stats})


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
