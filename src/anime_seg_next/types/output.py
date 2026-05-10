"""AnimeSegOutput — rich result object returned by AnimeSegNextPipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class AnimeSegOutput:
    """Segmentation result with lazy overlay generation and pixel-level queries.

    Attributes:
        segmentation_map: H×W int32 array of class IDs.
        color_map: Colored segmentation mask as a PIL Image.
        class_names: Ordered list mapping class ID → human-readable name.
        id_to_color: Mapping class ID → (R, G, B) tuple.
    """

    segmentation_map: np.ndarray
    color_map: Image.Image
    class_names: List[str]
    id_to_color: Dict[int, Tuple[int, int, int]]
    depth: Optional[np.ndarray] = None
    _source_image: Optional[Image.Image] = field(default=None, repr=False)
    _overlay: Optional[Image.Image] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def overlay_map(self) -> Image.Image:
        """60 / 40 blend of color_map over the source image (lazy).

        The source image is stored automatically when keep_source=True
        (the default) is passed to __call__.
        """
        if self._overlay is not None:
            return self._overlay
        if self._source_image is None:
            raise RuntimeError(
                "source image not available — pass keep_source=True to __call__"
            )
        src = self._source_image.resize(self.color_map.size, Image.BILINEAR).convert("RGB")
        self._overlay = Image.blend(src, self.color_map.convert("RGB"), 0.6)
        return self._overlay

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def class_name_at(self, row: int, col: int) -> str:
        """Return the class name for the pixel at (row, col)."""
        class_id = int(self.segmentation_map[row, col])
        if class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def class_mask(self, class_name_or_id: "str | int") -> np.ndarray:
        """Return a boolean H×W mask for a single class."""
        if isinstance(class_name_or_id, str):
            if class_name_or_id not in self.class_names:
                raise KeyError(f"Unknown class: {class_name_or_id!r}")
            class_id = self.class_names.index(class_name_or_id)
        else:
            class_id = int(class_name_or_id)
        return self.segmentation_map == class_id

    def present_classes(self) -> List[str]:
        """Return names of classes actually present in the segmentation map."""
        ids = np.unique(self.segmentation_map).tolist()
        return [
            self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            for i in ids
        ]

    def __repr__(self) -> str:
        h, w = self.segmentation_map.shape
        return (
            f"AnimeSegOutput(size={w}×{h}, "
            f"num_classes={self.num_classes}, "
            f"present={self.present_classes()})"
        )

    @property
    def images(self) -> list:
        """Return image list for compatibility with callers expecting `.images[0]`.

        By convention the first image is the color mask. Additional images
        (overlay, etc.) may be provided in future.
        """
        return [self.color_map]
