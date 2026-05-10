"""Output types for AnimeSeg-Next."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class AnimeSegOutput:
    """Segmentation (+ optional depth) result from AnimeSegNextPipeline.

    Attributes:
        segmentation_map: HxW int32 array of class indices (at model resolution).
        color_map: PIL Image with RGB class colors (at requested output resolution).
        class_names: Ordered list of class label strings.
        id_to_color: Mapping from class index → (R, G, B) tuple.
        depth: Optional HxW float32 depth map in [0, 1] (at output resolution).
        overlay_map: Lazy-computed 60/40 blend of source image and color_map.
    """

    segmentation_map: np.ndarray          # HxW, int32
    color_map: Image.Image                 # RGB, output resolution
    class_names: List[str]
    id_to_color: Dict[int, Tuple[int, int, int]]
    depth: Optional[np.ndarray] = None    # HxW, float32, [0,1]
    _source_image: Optional[Image.Image] = field(default=None, repr=False)
    _overlay_map: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def overlay_map(self) -> Optional[Image.Image]:
        """60/40 blend of source image over color_map (lazy, auto-resized)."""
        if self._source_image is None:
            return None
        if self._overlay_map is None:
            src = self._source_image.convert("RGB")
            cm = self.color_map.resize(src.size, Image.NEAREST).convert("RGB")
            self._overlay_map = Image.blend(src, cm, alpha=0.4)
        return self._overlay_map

    def __repr__(self) -> str:
        h, w = self.segmentation_map.shape
        has_depth = self.depth is not None
        return (
            f"AnimeSegOutput(seg={h}x{w}, classes={len(self.class_names)}, "
            f"depth={has_depth})"
        )
