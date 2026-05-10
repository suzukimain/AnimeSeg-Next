"""anime_seg_next public API."""

from .mask2former.mask2former_pipeline import AnimeSegNextPipeline, AnimeSegPipeline
from .types import AnimeSegOutput
from .core import SERIES_CLASS_MAP, build_semantic_colors

__all__ = [
    "AnimeSegNextPipeline",
    "AnimeSegPipeline",
    "AnimeSegOutput",
    "SERIES_CLASS_MAP",
    "build_semantic_colors",
]
