"""Class metadata, palette definitions, and series resolution for AnimeSeg-Next."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 37-class label set (Next-v2 series)
# ---------------------------------------------------------------------------

NEXT_V2_CLASSES: List[str] = [
    "background",
    "back_hair",
    "bottomwear",
    "ears_left",
    "ears_right",
    "earwear_left",
    "earwear_right",
    "eyebrow_left",
    "eyebrow_right",
    "eyelash_left",
    "eyelash_right",
    "eyewear_left",
    "eyewear_right",
    "eyewhite_left",
    "eyewhite_right",
    "face",
    "footwear",
    "front_hair",
    "handwear",
    "headwear",
    "irides_left",
    "irides_right",
    "legwear",
    "mouth",
    "neck",
    "neckwear",
    "nose",
    "objects",
    "tail",
    "topwear",
    "wings",
    "handwear_L",
    "handwear_R",
    "legwear_L",
    "legwear_R",
    "footwear_L",
    "footwear_R",
]

# Series registry: series key → class list
SERIES_CLASS_MAP: Dict[str, List[str]] = {
    "next-v2": NEXT_V2_CLASSES,
}

# Fixed 37-class palette (hardcoded for training consistency)
PALETTE_37: Dict[int, Tuple[int, int, int]] = {
    0:  (0, 0, 0),
    1:  (75, 0, 130),
    2:  (0, 102, 204),
    3:  (100, 180, 220),
    4:  (220, 120, 80),
    5:  (70, 120, 200),
    6:  (220, 100, 40),
    7:  (50, 100, 200),
    8:  (200, 80, 30),
    9:  (40, 80, 180),
    10: (180, 60, 20),
    11: (100, 160, 240),
    12: (240, 140, 60),
    13: (200, 240, 255),
    14: (255, 240, 200),
    15: (100, 150, 255),
    16: (32, 64, 96),
    17: (255, 128, 0),
    18: (192, 192, 192),
    19: (200, 100, 50),
    20: (80, 140, 220),
    21: (220, 180, 80),
    22: (204, 51, 102),
    23: (255, 0, 150),
    24: (210, 170, 140),
    25: (100, 100, 100),
    26: (255, 140, 0),
    27: (128, 128, 128),
    28: (200, 50, 50),
    29: (0, 128, 0),
    30: (255, 255, 0),
    31: (180, 100, 255),
    32: (255, 140, 100),
    33: (100, 200, 255),
    34: (255, 200, 100),
    35: (100, 255, 180),
    36: (255, 100, 160),
}


def resolve_series(config_obj: Optional[Dict], num_classes: int) -> Optional[str]:
    """Return the series key from config_obj if it maps to the right class count."""
    if not config_obj:
        return None
    series = config_obj.get("series") or config_obj.get("Series")
    if series and series in SERIES_CLASS_MAP:
        if len(SERIES_CLASS_MAP[series]) >= num_classes:
            return series
    # Auto-detect by count
    for key, names in SERIES_CLASS_MAP.items():
        if len(names) == num_classes:
            return key
    return None


def resolve_class_names(config_obj: Optional[Dict], num_classes: int) -> List[str]:
    series = resolve_series(config_obj, num_classes)
    if series:
        names = list(SERIES_CLASS_MAP[series])
        if len(names) >= num_classes:
            return names[:num_classes]
        return names + [f"class_{i}" for i in range(len(names), num_classes)]
    raw = (config_obj or {}).get("class_names") or (config_obj or {}).get("ClassNames")
    if isinstance(raw, list) and raw:
        names = [str(n).strip() for n in raw if str(n).strip()]
        if len(names) < num_classes:
            names += [f"class_{i}" for i in range(len(names), num_classes)]
        return names[:num_classes]
    return [f"class_{i}" for i in range(num_classes)]


def resolve_class_colors(
    config_obj: Optional[Dict], num_classes: int
) -> Dict[int, Tuple[int, int, int]]:
    if num_classes == 37:
        return dict(PALETTE_37)
    names = resolve_class_names(config_obj, num_classes)
    return _build_semantic_colors(names)


def _build_semantic_colors(names: List[str]) -> Dict[int, Tuple[int, int, int]]:
    """Generate visually distinct colors from class name hashes."""
    import hashlib
    result: Dict[int, Tuple[int, int, int]] = {}
    for i, name in enumerate(names):
        h = hashlib.md5(name.encode()).digest()
        result[i] = (h[0], h[1], h[2])
    return result
