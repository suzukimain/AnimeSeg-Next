"""AnimeSegNextPipeline — Mask2Former pipeline with series-based class resolution."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anime_seg.mask2former.mask2former_pipeline import Mask2FormerAnimeSegPipeline

from ..types import AnimeSegOutput
from ..core import SERIES_CLASS_MAP, build_semantic_colors, resolve_series


DEFAULT_REPO_ID = "suzukimain/AnimeSeg-Next"


class AnimeSegNextPipeline(Mask2FormerAnimeSegPipeline):
    """AnimeSeg-Next Mask2Former pipeline.

    Extends the base pipeline with:
    - Series-based class resolution (weights → series key → code constants)
    - Semantic color generation driven by class name patterns
    - Rich AnimeSegOutput return type with lazy overlay and pixel queries
    - Fluent .to() for method chaining
    """

    # ------------------------------------------------------------------ #
    # Class-resolution overrides (called by parent __init__)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_class_names(config_obj: Dict, num_classes: int) -> List[str]:
        """Resolve class names: series key > auto-detect by count > legacy array > generic."""
        series = resolve_series(config_obj, num_classes)
        if series:
            names = list(SERIES_CLASS_MAP[series])
            if len(names) >= num_classes:
                return names[:num_classes]
            return names + [f"class_{i}" for i in range(len(names), num_classes)]

        # Legacy: explicit array in config
        raw = (config_obj or {}).get("class_names") or (config_obj or {}).get("ClassNames")
        if isinstance(raw, list) and raw:
            names = [str(n).strip() for n in raw if str(n).strip()]
            if len(names) < num_classes:
                names += [f"class_{i}" for i in range(len(names), num_classes)]
            return names[:num_classes]

        return [f"class_{i}" for i in range(num_classes)]

    @staticmethod
    def _resolve_class_colors(
        config_obj: Dict, num_classes: int
    ) -> Dict[int, Tuple[int, int, int]]:
        """Generate semantic colors: fixed 37-class palette or dynamic semantic colors."""
        if num_classes == 37:
            return AnimeSegNextPipeline.get_fixed_37_palette()
        
        names = AnimeSegNextPipeline._resolve_class_names(config_obj, num_classes)
        return build_semantic_colors(names)

    @staticmethod
    def get_fixed_37_palette() -> Dict[int, Tuple[int, int, int]]:
        """Definitive 37-class palette matching the training ground truth."""
        return {
            0: (0, 0, 0),       # background
            1: (75, 0, 130),    # back_hair
            2: (0, 102, 204),   # bottomwear
            3: (100, 180, 220), # ears_left
            4: (220, 120, 80),  # ears_right
            5: (70, 120, 200),  # earwear_left
            6: (220, 100, 40),  # earwear_right
            7: (50, 100, 200),  # eyebrow_left
            8: (200, 80, 30),   # eyebrow_right
            9: (40, 80, 180),   # eyelash_left
            10: (180, 60, 20),  # eyelash_right
            11: (100, 160, 240),# eyewear_left
            12: (240, 140, 60), # eyewear_right
            13: (200, 240, 255),# eyewhite_left
            14: (255, 240, 200),# eyewhite_right
            15: (100, 150, 255),# face
            16: (32, 64, 96),   # footwear
            17: (255, 128, 0),  # front_hair
            18: (192, 192, 192),# handwear
            19: (200, 100, 50), # headwear
            20: (80, 140, 220), # irides_left
            21: (220, 180, 80), # irides_right
            22: (204, 51, 102), # legwear
            23: (255, 0, 150),  # mouth
            24: (210, 170, 140),# neck
            25: (100, 100, 100),# neckwear
            26: (255, 140, 0),  # nose
            27: (128, 128, 128),# objects
            28: (200, 50, 50),  # tail
            29: (0, 128, 0),    # topwear
            30: (255, 255, 0),  # wings
            31: (180, 100, 255),# handwear_L
            32: (255, 140, 100),# handwear_R
            33: (100, 200, 255),# legwear_L
            34: (255, 200, 100),# legwear_R
            35: (100, 255, 180),# footwear_L
            36: (255, 100, 160),# footwear_R
        }

    # ------------------------------------------------------------------ #
    # Constructor / factory                                                #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=device,
            base_model=base_model,
            config_name=config_name,
            remove_bg=remove_bg,
        )

    @classmethod
    def from_mask2former(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> "AnimeSegNextPipeline":
        """Load an AnimeSeg-Next Mask2Former pipeline.

        Priority for class metadata:
          1. Weight shape  → physical num_classes (always wins on conflict)
          2. config.json ``series`` key  → code-defined name list
          3. config.json ``class_names`` array (legacy fallback)
          4. Generic ``class_N`` labels

        Args:
            repo_id: HuggingFace repo or local directory.
            filename: Specific checkpoint filename; empty = auto-select latest.
            token / hf_token: HF access token (either name accepted).
            device: Target device, e.g. ``"cuda"`` or ``"cpu"``.
            base_model: Swin backbone variant to initialise.
            config_name: Metadata JSON filename in the repo.
            remove_bg: Apply background removal before segmentation.

        Returns:
            Loaded and ready-to-use pipeline.
        """
        final_token = hf_token if hf_token is not None else token
        pipe = cls(
            repo_id=repo_id,
            filename=filename,
            token=final_token,
            device=None,
            base_model=base_model,
            config_name=config_name,
            remove_bg=remove_bg,
        )
        if device is not None:
            pipe.to(device)
        return pipe

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        image: Union[str, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_source: bool = True,
        output_overlay: bool = False,
    ) -> AnimeSegOutput:
        """Run segmentation and return a rich AnimeSegOutput.

        Args:
            image: Path string or PIL Image (RGB or RGBA accepted).
            width: Output width in pixels (defaults to source width).
            height: Output height in pixels (defaults to source height).
            keep_source: Store source image so overlay_map can be computed
                lazily later. Disable to save memory.
            output_overlay: Immediately compute the overlay blend and store
                it in the returned output (implies keep_source=True).

        Returns:
            AnimeSegOutput with segmentation_map, color_map, and optionally
            overlay_map.
        """
        if isinstance(image, str):
            source_img = Image.open(image).convert("RGB")
        else:
            source_img = image.convert("RGB")

        working_img = source_img
        if getattr(self, "remove_bg", False) and hasattr(self, "bg_remover"):
            import numpy as _np
            mask = self.bg_remover(working_img, return_mask=True, return_type="numpy")
            img_np = _np.array(working_img)
            bg = _np.array([255, 255, 255], dtype=_np.uint8)
            working_img = Image.fromarray(
                (mask * img_np + (1 - mask) * bg).astype(_np.uint8)
            )

        original_size = working_img.size
        # If width/height not provided, keep original image size
        target_w = int(width) if width is not None else original_size[0]
        target_h = int(height) if height is not None else original_size[1]
        if target_w <= 0 or target_h <= 0:
            raise ValueError("output size must be positive")

        input_tensor = self._preprocess(working_img)

        h_in, w_in = input_tensor.shape[-2:]

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw = self.model.model(pixel_values=input_tensor)
            else:
                raw = self.model.model(pixel_values=input_tensor)

            # Match training _semantic_from_queries exactly:
            # sigmoid BEFORE interpolate, drop no-object (last) class slot
            cls_probs  = raw.class_queries_logits.softmax(dim=-1)[..., :-1]
            mask_probs = F.interpolate(
                raw.masks_queries_logits.sigmoid(),
                size=(h_in, w_in), mode="bilinear", align_corners=False,
            )
            sem = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)
            preds = sem.argmax(dim=1).cpu().numpy()[0]

        # Build coloured mask
        h, w = preds.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.id_to_color.items():
            colored[preds == class_id] = color

        color_map = Image.fromarray(colored).resize((target_w, target_h), Image.NEAREST)

        depth_np = None
        depth_head = getattr(self.model, "depth_head", None)
        if depth_head is not None:
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        depth_logits = depth_head(raw.pixel_decoder_last_hidden_state)
                else:
                    depth_logits = depth_head(raw.pixel_decoder_last_hidden_state)
            depth_up = F.interpolate(
                depth_logits, size=(h_in, w_in), mode="bilinear", align_corners=False
            ).sigmoid()
            depth_np = depth_up.cpu().numpy()[0, 0]
            if depth_np.shape != (target_h, target_w):
                import cv2 as _cv2
                depth_np = _cv2.resize(depth_np, (target_w, target_h), interpolation=_cv2.INTER_LINEAR)

        stored_source = source_img if (keep_source or output_overlay) else None
        class_names = getattr(self, "class_names", None)
        if not class_names:
            num_classes = int(getattr(self, "num_classes", len(self.id_to_color)))
            class_names = [f"class_{i}" for i in range(num_classes)]
        result = AnimeSegOutput(
            segmentation_map=preds.astype(np.int32),
            color_map=color_map,
            class_names=list(class_names),
            id_to_color=dict(self.id_to_color),
            depth=depth_np,
            _source_image=stored_source,
        )

        if output_overlay:
            _ = result.overlay_map  # eagerly populate

        return result

    # ------------------------------------------------------------------ #
    # Device management — fluent API                                      #
    # ------------------------------------------------------------------ #

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "AnimeSegNextPipeline":
        """Move pipeline to device. Supports method chaining.

        Example::

            pipe = AnimeSegNextPipeline.from_mask2former().to("cuda")
        """
        self.device = str(device)
        self.use_amp = self.device.startswith("cuda")
        self.model.to(device)
        if getattr(self, "remove_bg", False) and hasattr(self, "bg_remover"):
            self.bg_remover.to(device)
        return self


# Backward-compatibility alias
AnimeSegPipeline = AnimeSegNextPipeline
