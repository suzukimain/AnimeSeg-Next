"""Series-based class metadata: name lists and semantic color generation."""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Class name lists per series                                                   #
# --------------------------------------------------------------------------- #

SERIES_CLASS_MAP: Dict[str, List[str]] = {
    # 31 classes — next-generation anime segmentation v1
    "next-v1": [
        "background",    # 0
        "back_hair",     # 1
        "bottomwear",    # 2
        "ears_left",     # 3
        "ears_right",    # 4
        "earwear_left",  # 5
        "earwear_right", # 6
        "eyebrow_left",  # 7
        "eyebrow_right", # 8
        "eyelash_left",  # 9
        "eyelash_right", # 10
        "eyewear_left",  # 11
        "eyewear_right", # 12
        "eyewhite_left", # 13
        "eyewhite_right",# 14
        "face",          # 15
        "footwear",      # 16
        "front_hair",    # 17
        "handwear",      # 18
        "headwear",      # 19
        "irides_left",   # 20
        "irides_right",  # 21
        "legwear",       # 22
        "mouth",         # 23
        "neck",          # 24
        "neckwear",      # 25
        "nose",          # 26
        "objects",       # 27
        "tail",          # 28
        "topwear",       # 29
        "wings",         # 30
    ],
    # 37 classes — adds fine-grained mouth/eye/body detail
    "next-v2": [
        "background",    # 0
        "back_hair",     # 1
        "bottomwear",    # 2
        "ears_left",     # 3
        "ears_right",    # 4
        "earwear_left",  # 5
        "earwear_right", # 6
        "eyebrow_left",  # 7
        "eyebrow_right", # 8
        "eyelash_left",  # 9
        "eyelash_right", # 10
        "eyewear_left",  # 11
        "eyewear_right", # 12
        "eyewhite_left", # 13
        "eyewhite_right",# 14
        "face",          # 15
        "footwear",      # 16
        "front_hair",    # 17
        "handwear",      # 18
        "headwear",      # 19
        "irides_left",   # 20
        "irides_right",  # 21
        "legwear",       # 22
        "mouth",         # 23
        "neck",          # 24
        "neckwear",      # 25
        "nose",          # 26
        "objects",       # 27
        "tail",          # 28
        "topwear",       # 29
        "wings",         # 30
        "handwear_L",    # 31
        "handwear_R",    # 32
        "legwear_L",     # 33
        "legwear_R",     # 34
        "footwear_L",    # 35
        "footwear_R",    # 36
    ],
    # 12 classes — legacy early-training schema
    "legacy-v1": [
        "background",    # 0
        "skin",          # 1
        "clothes_top",   # 2
        "clothes_bottom",# 3
        "hair_front",    # 4
        "hair_back",     # 5
        "face",          # 6
        "eyes",          # 7
        "mouth",         # 8
        "accessories",   # 9
        "other_clothes", # 10
        "accessory",     # 11
    ],
}

# Reverse lookup: class count → series key.
# If two series share the same count the later one wins; be explicit if needed.
_COUNT_TO_SERIES: Dict[int, str] = {
    len(names): key for key, names in SERIES_CLASS_MAP.items()
}


def resolve_series(config_obj: Dict, num_classes: int) -> Optional[str]:
    """Return series name from config or by matching class count."""
    series = (config_obj or {}).get("series")
    if series and series in SERIES_CLASS_MAP:
        return series
    return _COUNT_TO_SERIES.get(num_classes)


# --------------------------------------------------------------------------- #
# Semantic color generation                                                     #
# --------------------------------------------------------------------------- #

# Hard-coded overrides for classes where hue is semantically meaningful
_SEMANTIC_HUE_OVERRIDES: Dict[str, Tuple[float, float, float]] = {
    # (hue, sat, val)
    "background":    (0.0,  0.0,  0.0),   # black — handled separately
    "blush":         (0.95, 0.55, 0.92),
    "lips":          (0.93, 0.72, 0.78),
    "tongue":        (0.94, 0.60, 0.72),
    "teeth":         (0.0,  0.06, 0.97),
    "eyewhite":      (0.0,  0.08, 0.97),
    "face":          (0.07, 0.25, 0.95),  # warm pale skin
    "neck":          (0.07, 0.22, 0.90),
    "body":          (0.07, 0.20, 0.88),
    "nose":          (0.06, 0.28, 0.88),
    "irides":        (0.58, 0.90, 0.92),  # vivid blue default; right becomes amber
}

_GOLDEN = 0.6180339887498949


def build_semantic_colors(
    class_names: List[str],
) -> Dict[int, Tuple[int, int, int]]:
    """Assign perceptually distinct RGB colors driven by class name semantics.

    Rules applied in order:
    - background → (0, 0, 0)
    - hard override table (_SEMANTIC_HUE_OVERRIDES) for specific bases
    - _left/_right pairs: same base hue; right is complementary (+180°)
    - back_ prefix → darkened (val × 0.55)
    - front_ prefix → brightened (val × 1.0, sat slightly boosted)
    - remaining classes: golden-angle HSV spacing
    """
    base_hue: Dict[str, float] = {}
    hue_counter = 0
    colors: Dict[int, Tuple[int, int, int]] = {}

    for idx, name in enumerate(class_names):
        if name == "background":
            colors[idx] = (0, 0, 0)
            continue

        # ── Decompose side suffix ──────────────────────────────────────────
        if name.endswith("_left"):
            base, side = name[:-5], "left"
        elif name.endswith("_right"):
            base, side = name[:-6], "right"
        else:
            base, side = name, None

        # ── Hard override for specific base concepts ───────────────────────
        if base in _SEMANTIC_HUE_OVERRIDES:
            h, s, v = _SEMANTIC_HUE_OVERRIDES[base]
            if side == "right":
                h = (h + 0.5) % 1.0
        else:
            # ── Auto golden-angle hue ──────────────────────────────────────
            if base not in base_hue:
                base_hue[base] = (hue_counter * _GOLDEN) % 1.0
                hue_counter += 1
            h = base_hue[base]
            if side == "right":
                h = (h + 0.5) % 1.0
            s, v = 0.65, 0.88

        # ── Semantic brightness modifiers ─────────────────────────────────
        if base.startswith("back"):
            v = min(v, 0.48)
        elif base.startswith("front"):
            v = max(v, 0.92)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[idx] = (round(r * 255), round(g * 255), round(b * 255))

    return colors
