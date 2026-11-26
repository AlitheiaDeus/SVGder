"""Vectorization engines."""

from svgder.engines.blobs import vectorize_blobs
from svgder.engines.canny import vectorize_canny
from svgder.engines.flood_search import vectorize_flood_search
from svgder.engines.gremlin import vectorize_gremlin
from svgder.engines.lowpoly import vectorize_lowpoly

__all__ = [
    "vectorize_gremlin",
    "vectorize_canny",
    "vectorize_blobs",
    "vectorize_lowpoly",
    "vectorize_flood_search",
]
