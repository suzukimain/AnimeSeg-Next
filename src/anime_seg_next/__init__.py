"""anime_seg_next public API."""

from .mask2former.mask2former_pipeline import AnimeSegNextPipeline

AnimeSegPipeline = AnimeSegNextPipeline

__all__ = ["AnimeSegNextPipeline", "AnimeSegPipeline"]
