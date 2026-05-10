"""AnimeSegNextPipeline — self-contained inference pipeline for AnimeSeg-Next."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
    """Segmentation (+ depth) pipeline for AnimeSeg-Next.

    Factories
    ---------
    - :meth:`from_pretrained` — HuggingFace Hub (auto-selects latest checkpoint)
    - :meth:`from_mask2former` — alias of ``from_pretrained`` (legacy name)
    - :meth:`from_checkpoint` — local file path

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
        remove_bg: bool = True,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.id_to_color = id_to_color
        self.input_size = input_size
        self.device = device
        self.use_amp = device.startswith("cuda")
        self._bg_remover: Optional[Any] = None
        self.remove_bg = remove_bg
        if remove_bg:
            self._init_bg_remover()

    # ------------------------------------------------------------------
    # Background remover
    # ------------------------------------------------------------------

    def _init_bg_remover(self) -> None:
        """Lazy-load BgRemover (ISNet). Skips silently if unavailable."""
        try:
            from anime_seg.remove_bg.bg_remover_pipeline import BgRemover
            self._bg_remover = BgRemover.from_single_file(device=self.device)
        except Exception:
            self._bg_remover = None

    def _apply_bg_removal(self, img: Image.Image) -> Image.Image:
        """Apply ISNet background removal; returns white-bg image."""
        if self._bg_remover is None:
            return img
        img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        mask = self._bg_remover(img, use_amp=self.use_amp, return_mask=True, return_type="numpy")
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        bg = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        out = (mask * img_np + (1 - mask) * bg).clip(0, 1)
        return Image.fromarray((out * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Factory: HuggingFace Hub
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = DEFAULT_BASE_MODEL,
        config_name: str = "config.json",
        input_size: int = DEFAULT_INPUT_SIZE,
        remove_bg: bool = True,
    ) -> "AnimeSegNextPipeline":
        """Load from a HuggingFace Hub repo.

        Args:
            repo_id: HF repo ID or local directory.
            filename: Checkpoint filename; empty = auto-select latest *.safetensors.
            token / hf_token: HF access token (either accepted).
            device: Target device. Defaults to CUDA if available.
            base_model: Swin backbone HF ID for architecture initialisation.
            config_name: Metadata JSON filename inside the repo.
            input_size: Square input resolution fed to the model.
            remove_bg: Apply ISNet background removal before segmentation (default: True).
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        final_token = hf_token or token

        # 1. Download config first to potentially guide model selection
        config_obj: Optional[Dict] = None
        try:
            cfg_path = hf_hub_download(
                repo_id=repo_id, filename=config_name, token=final_token
            )
            with open(cfg_path, encoding="utf-8") as f:
                config_obj = json.load(f)
        except Exception:
            pass

        # 2. Determine filename if not provided
        if not filename:
            # Try to get from config_obj["models"] list
            if config_obj and "models" in config_obj and isinstance(config_obj["models"], list):
                # Use the first one (usually latest if updated via update_hf_config.py)
                models = config_obj["models"]
                if models and isinstance(models[0], dict):
                    filename = models[0].get("FilePath")

            # Fallback: scan repo files for latest version based on naming convention
            if not filename:
                try:
                    files = list(list_repo_files(repo_id, token=final_token))
                    safetensors = [f for f in files if f.endswith(".safetensors")]
                    if not safetensors:
                        raise RuntimeError(
                            f"No .safetensors files found in repo {repo_id!r}"
                        )
                    filename = _pick_latest_filename(safetensors)
                except Exception as exc:
                    if not filename:
                        raise RuntimeError(
                            f"Failed to determine model file in repo {repo_id!r}: {exc}"
                        ) from exc

        # 3. Download the actual checkpoint
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=filename, token=final_token
        )

        return cls._build(
            ckpt_path=ckpt_path,
            config_obj=config_obj,
            base_model=base_model,
            input_size=input_size,
            device=device,
            remove_bg=remove_bg,
        )

    # Alias used by the merge script and old code
    from_mask2former = from_pretrained

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
        remove_bg: bool = True,
    ) -> "AnimeSegNextPipeline":
        """Load from a local .safetensors or .pt file.

        Args:
            checkpoint_path: Path to the model weights file or directory.
            config_path: Optional path to config.json with class metadata.
            base_model: Swin backbone HF ID for architecture initialisation.
            input_size: Square input resolution fed to the model.
            device: Target device.
            remove_bg: Apply ISNet background removal before segmentation (default: True).
        """
        # 1. Handle directory input (auto-resolve latest)
        if os.path.isdir(checkpoint_path):
            dir_path = checkpoint_path
            if not config_path:
                potential_cfg = os.path.join(dir_path, "config.json")
                if os.path.isfile(potential_cfg):
                    config_path = potential_cfg
            
            # Resolve latest checkpoint from directory
            checkpoint_path = _resolve_latest_local(dir_path, config_path)

        # 2. Load config
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
            remove_bg=remove_bg,
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
        remove_bg: bool = True,
    ) -> "AnimeSegNextPipeline":
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
            remove_bg=remove_bg,
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
        if self._bg_remover is not None:
            try:
                self._bg_remover.to(device)
            except Exception:
                pass
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
        remove_bg: Optional[bool] = None,
    ) -> AnimeSegOutput:
        """Run segmentation on a single image.

        Args:
            image: File path string or PIL Image (RGB/RGBA).
            width: Output width in pixels (defaults to source width).
            height: Output height in pixels (defaults to source height).
            keep_source: Store source image for lazy overlay_map computation.
            output_overlay: Eagerly compute overlay_map.
            remove_bg: Override the pipeline-level remove_bg setting for this call.
                       ``None`` (default) uses the pipeline setting.

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

        # Background removal (before segmentation)
        do_remove_bg = self.remove_bg if remove_bg is None else remove_bg
        working_img = self._apply_bg_removal(source_img) if do_remove_bg else source_img

        input_tensor = self._preprocess(working_img)

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
            depth_np = outputs["depth"].cpu().numpy()[0, 0]
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
                    return f.get_tensor(k).shape[0] - 1
    except Exception:
        pass
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
    return 37


def _pick_latest_filename(filenames: List[str]) -> str:
    """Pick the filename with the highest _v{version} suffix."""
    import re
    ver_pattern = re.compile(r"_v(\d+)\.safetensors$")

    def get_version(path: str) -> int:
        match = ver_pattern.search(path)
        return int(match.group(1)) if match else -1

    # Sort descending by version
    sorted_files = sorted(filenames, key=get_version, reverse=True)
    return sorted_files[0]


def _resolve_latest_local(dir_path: str, config_path: Optional[str]) -> str:
    """Resolve the latest local checkpoint from a directory, using config if available."""
    # 1. Try to use config.json metadata
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
                if "models" in data and isinstance(data["models"], list) and data["models"]:
                    rel_path = data["models"][0].get("FilePath")
                    if rel_path:
                        # Check various potential locations for the file
                        candidates = [
                            os.path.join(dir_path, os.path.basename(rel_path)),
                            os.path.join(os.path.dirname(config_path), os.path.basename(rel_path)),
                            os.path.join(dir_path, rel_path),
                        ]
                        for c in candidates:
                            if os.path.isfile(c):
                                return c
        except Exception:
            pass

    # 2. Fallback: scan directory for .safetensors and pick highest version
    files = [
        f for f in os.listdir(dir_path) 
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(dir_path, f))
    ]
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in directory: {dir_path}")

    latest_name = _pick_latest_filename(files)
    return os.path.join(dir_path, latest_name)
