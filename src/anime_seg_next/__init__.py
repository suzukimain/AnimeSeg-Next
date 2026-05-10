"""AnimeSeg-Next: Anime character segmentation and depth estimation."""
from .types import AnimeSegOutput
from .core import (
    NEXT_V2_CLASSES,
    PALETTE_37,
    SERIES_CLASS_MAP,
    resolve_class_names,
    resolve_class_colors,
    resolve_series,
)
from .mask2former.model import Mask2FormerModel, DepthHead
from .mask2former.pipeline import AnimeSegNextPipeline

__all__ = [
    "AnimeSegOutput",
    "AnimeSegNextPipeline",
    "Mask2FormerModel",
    "DepthHead",
    "NEXT_V2_CLASSES",
    "PALETTE_37",
    "SERIES_CLASS_MAP",
    "resolve_class_names",
    "resolve_class_colors",
    "resolve_series",
]
