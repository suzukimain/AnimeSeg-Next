"""AnimeSegNextPipeline — self-contained inference pipeline for AnimeSeg-Next."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import Mask2FormerModel
from ..core import resolve_class_names, resolve_class_colors
from ..types import AnimeSegOutput

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_REPO_ID = "suzukimain/AnimeSeg-Next"
DEFAULT_BASE_MODEL = "facebook/mask2former-swin-large-ade-semantic"
DEFAULT_INPUT_SIZE = 768


class AnimeSegNextPipeline:
    """Segmentation pipeline for AnimeSeg-Next models.

    Load with :meth:`from_pretrained` (HuggingFace Hub or local dir) or
    :meth:`from_checkpoint` (direct path).

    Example::

        pipe = AnimeSegNextPipeline.from_pretrained().to("cuda")
        result = pipe("image.png")
        result.color_map.save("seg.png")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: Mask2FormerModel,
        class_names: List[str],
        id_to_color: Dict[int, Tuple[int, int, int]],
        input_size: int = DEFAULT_INPUT_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.id_to_color = id_to_color
        self.input_size = input_size
        self.device = device
        self.use_amp = device.startswith("cuda")

    # ------------------------------------------------------------------
    # Factory: HuggingFace Hub
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = DEFAULT_BASE_MODEL,
        config_name: str = "config.json",
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> "AnimeSegNextPipeline":
        """Load from a HuggingFace Hub repo.

        Args:
            repo_id: HF repo ID or local directory.
            filename: Checkpoint filename; empty = auto-select latest *.safetensors.
            token: HF access token.
            device: Target device (``"cuda"``, ``"cpu"``). Defaults to auto-detect.
            base_model: Swin backbone HF ID for architecture initialisation.
            config_name: Metadata JSON filename inside the repo.
            input_size: Square input resolution fed to the model.

        Returns:
            Loaded pipeline on the requested device.
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        # Resolve checkpoint filename
        if not filename:
            try:
                files = list(list_repo_files(repo_id, token=token))
                safetensors = [
                    f for f in files if f.endswith(".safetensors")
                ]
                if not safetensors:
                    raise RuntimeError(
                        f"No .safetensors files found in repo {repo_id!r}"
                    )
                filename = safetensors[-1]  # latest by listing order
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to list files in repo {repo_id!r}: {exc}"
                ) from exc

        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=filename, token=token
        )

        # Config (best-effort)
        config_obj: Optional[Dict] = None
        try:
            cfg_path = hf_hub_download(
                repo_id=repo_id, filename=config_name, token=token
            )
            with open(cfg_path, encoding="utf-8") as f:
                config_obj = json.load(f)
        except Exception:
            pass

        return cls._build(
            ckpt_path=ckpt_path,
            config_obj=config_obj,
            base_model=base_model,
            input_size=input_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Factory: local checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        base_model: str = DEFAULT_BASE_MODEL,
        input_size: int = DEFAULT_INPUT_SIZE,
        device: Optional[str] = None,
    ) -> "AnimeSegNextPipeline":
        """Load from a local .safetensors or .pt file.

        Args:
            checkpoint_path: Path to the model weights file.
            config_path: Optional path to config.json with class metadata.
            base_model: Swin backbone HF ID for architecture initialisation.
            input_size: Square input resolution fed to the model.
            device: Target device.
        """
        config_obj: Optional[Dict] = None
        if config_path and os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                config_obj = json.load(f)

        return cls._build(
            ckpt_path=checkpoint_path,
            config_obj=config_obj,
            base_model=base_model,
            input_size=input_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        ckpt_path: str,
        config_obj: Optional[Dict],
        base_model: str,
        input_size: int,
        device: Optional[str],
    ) -> "AnimeSegNextPipeline":
        # Determine num_classes from checkpoint shape
        num_classes = _infer_num_classes(ckpt_path)

        model = Mask2FormerModel(
            base_model=base_model,
            num_classes=num_classes,
            load_base_pretrained=False,
        )
        model.load_checkpoint(ckpt_path)
        model.eval()

        class_names = resolve_class_names(config_obj, num_classes)
        id_to_color = resolve_class_colors(config_obj, num_classes)

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipe = cls(
            model=model,
            class_names=class_names,
            id_to_color=id_to_color,
            input_size=input_size,
            device=target_device,
        )
        pipe.to(target_device)
        return pipe

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: Union[str, torch.device]) -> "AnimeSegNextPipeline":
        """Move pipeline to device. Supports method chaining."""
        self.device = str(device)
        self.use_amp = self.device.startswith("cuda")
        self.model.to(device)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Union[str, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_source: bool = True,
        output_overlay: bool = False,
    ) -> AnimeSegOutput:
        """Run segmentation on a single image.

        Args:
            image: File path string or PIL Image (RGB/RGBA).
            width: Output width in pixels (defaults to source width).
            height: Output height in pixels (defaults to source height).
            keep_source: Store source image for lazy overlay_map computation.
            output_overlay: Eagerly compute overlay_map.

        Returns:
            :class:`AnimeSegOutput` with segmentation_map, color_map, and
            optionally depth and overlay_map.
        """
        if isinstance(image, str):
            source_img = Image.open(image).convert("RGB")
        else:
            source_img = image.convert("RGB")

        target_w = int(width) if width is not None else source_img.width
        target_h = int(height) if height is not None else source_img.height
        if target_w <= 0 or target_h <= 0:
            raise ValueError("Output dimensions must be positive integers.")

        input_tensor = self._preprocess(source_img)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)

        preds = outputs["semantic_logits"].argmax(dim=1).cpu().numpy()[0]  # HxW

        # Build color mask
        h, w = preds.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.id_to_color.items():
            colored[preds == class_id] = color
        color_map = Image.fromarray(colored).resize(
            (target_w, target_h), Image.NEAREST
        )

        # Depth (optional)
        depth_np: Optional[np.ndarray] = None
        if "depth" in outputs:
            depth_np = outputs["depth"].cpu().numpy()[0, 0]  # HxW float32
            if depth_np.shape != (target_h, target_w):
                import cv2 as _cv2
                depth_np = _cv2.resize(
                    depth_np, (target_w, target_h), interpolation=_cv2.INTER_LINEAR
                )

        stored_source = source_img if (keep_source or output_overlay) else None
        result = AnimeSegOutput(
            segmentation_map=preds.astype(np.int32),
            color_map=color_map,
            class_names=list(self.class_names),
            id_to_color=dict(self.id_to_color),
            depth=depth_np,
            _source_image=stored_source,
        )
        if output_overlay:
            _ = result.overlay_map
        return result

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """Resize to input_size, normalise with ImageNet stats, return BCHW tensor."""
        sz = self.input_size
        arr = np.array(
            img.resize((sz, sz), Image.BILINEAR), dtype=np.float32
        ) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return t.to(self.device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_num_classes(ckpt_path: str) -> int:
    """Read class count from checkpoint's class_predictor weight shape."""
    try:
        from safetensors import safe_open
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "class_predictor" in k and k.endswith(".weight"):
                    return f.get_tensor(k).shape[0] - 1  # minus no-object
    except Exception:
        pass
    # pt/pth fallback
    try:
        import torch as _t
        sd = _t.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                if isinstance(sd.get(key), dict):
                    sd = sd[key]
                    break
        for k, v in sd.items():
            if "class_predictor" in k and k.endswith(".weight"):
                return v.shape[0] - 1
    except Exception:
        pass
    return 37  # sensible default
