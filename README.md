# SVGder

> **Raster-to-SVG Lab** — three distinct vectorization engines in one CLI & interactive Studio: a **Mythical Gremlin** Bezier tracer, a **Canny** multi-channel edge vectorizer, and a **Gaussian Blob** splatting optimizer.

---

## Ramapillar Showcase

Converted from the `1024×1024` raster [Ramapillar.jpeg](Ramapillar.jpeg):

| Original Raster | **Gremlin** (Flagship) | **Canny** (`colored`) | **Gaussian Blobs** (Soft) |
| :---: | :---: | :---: | :---: |
| <img src="Ramapillar.jpeg" width="210" alt="Ramapillar Original"/> | <img src="Ramapillar_gremlin.svg" width="210" alt="Ramapillar Gremlin SVG"/> | <img src="Ramapillar_canny.svg" width="210" alt="Ramapillar Canny SVG"/> | <img src="Ramapillar_blobs.svg" width="210" alt="Ramapillar Blobs SVG"/> |
| `1024×1024` JPEG (`169.5 KB`) | **18 colors · 298 paths** (`160.8 KB`) | **119 Bezier strokes** (`69.7 KB`) | **218 Gaussian splats** (`85.7 KB`) |

---

## Engines

### 1. Gremlin (`--mode gremlin`, default)
Designed for crisp illustrations, stickers, logos, and cel-shaded art:
- **Edge-preserving denoising**: `pyrMeanShiftFiltering` + bilateral filtering scrubs JPEG compression artifacts without blurring ink lines.
- **Interior-weighted CIE Lab K-Means**: Fits cluster centers on low-gradient interiors (ignoring anti-aliased transition fringes) and merges perceptually close shades ($\Delta E$).
- **Watertight Painter's Order**: Separates dark ink outlines onto the top layer and tucks earlier fill regions 1 px underneath later layers (`later_mask`), eliminating hairline SVG seams.
- **Corner-preserving Cubic Beziers**: Upscales masks 2× with Gaussian smoothing, extracts outer + hole contours (`fill-rule="evenodd"`), preserves sharp corners (sparkles, kanji tips), and fits smooth curves via recursive Schneider least-squares cubic Beziers.

### 2. Canny (`--mode canny`)
Extracts clean vector lineart and sketch strokes:
- **Multi-channel Lab Canny**: Fuses luminance (`L*`) and chroma (`a*`, `b*`) edges so pastel details (like yellow/pink sparkles on white) aren't lost.
- **8-connected Stroke Chaining**: Walks thinned edges from endpoints/junctions into open and closed Bezier paths.
- **Styles**: `ink`, `colored`, `neon` (glow filter on dark canvas), and `blueprint`.

### 3. Gaussian Blobs (`--mode blobs`)
Painterly 2D Gaussian / ellipse splatting:
- Samples error-weighted regions, computes the **closed-form MSE-optimal RGB color** for each candidate splat, and refines `(cx, cy, rx, ry, angle, alpha)` via coordinate hill-climbing.
- Uses soft `<radialGradient>` falloff by default, or pass `--hard-blobs` for crisp geometric ellipses.

---

## Quickstart

### Install

```bash
pip install -e .
```

*(Requires Python 3.11+ with `numpy`, `opencv-python`, `pillow`, `fastapi`, `uvicorn`, `python-multipart`.)*

### CLI Usage

```bash
# Flagship Gremlin conversion
python -m svgder Ramapillar.jpeg -o Ramapillar.svg --mode gremlin

# Generate all three engines at once
python -m svgder Ramapillar.jpeg --mode all

# Canny lineart variants
python -m svgder Ramapillar.jpeg -o Ramapillar_neon.svg --mode canny --canny-style neon --stroke-width 2.2

# Geometric (hard) ellipse splats instead of soft Gaussian blur
python -m svgder Ramapillar.jpeg -o Ramapillar_geo.svg --mode blobs --blobs 320 --hard-blobs
```

### Interactive SVGder Studio

Launch the local web UI with drag-and-drop upload, live parameter sliders, zoom, and a split-screen **Raster vs. SVG** comparison slider:

```bash
python -m svgder --serve --port 8765
```

Then open `http://127.0.0.1:8765`.

---

## Tests

```bash
python -m tests.test_svgder
```
