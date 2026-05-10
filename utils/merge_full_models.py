"""Merge a training checkpoint (delta weights) with the HF base model.

Usage
-----
    python utils/merge_full_models.py \\
        --checkpoint models/base/depth/step_0022000.safetensors \\
        --output     models/base/depth/anime_seg-next_mask2former_v3.safetensors \\
        --base-model facebook/mask2former-swin-large-ade-semantic \\
        --validate   image/test-input/2.png

The merged .safetensors file is self-contained (all weights included) and can be
loaded with AnimeSegNextPipeline.from_checkpoint() without any HF download.

Validation compares source-model output against merged-model output and checks
pixel accuracy / mIoU to confirm no degradation occurred during the merge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Also try to locate anime_seg for BgRemover (optional, non-fatal if missing)
for _candidate in [
    ROOT.parent / "AnimeSeg" / "src",
    ROOT.parent.parent / "AnimeSeg" / "src",
]:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_checkpoint(
    checkpoint_path: str,
    base_model: str,
    output_path: str,
    num_classes: int = 37,
) -> None:
    """Load HF base model + training checkpoint and save merged safetensors.

    The training checkpoint may contain:
    - Only the fine-tuned keys (delta) in various prefix forms
      (e.g. ``model.*``, no prefix, ``_orig_mod.*``)
    - Additional heads (e.g. ``depth_head.*``)

    All are resolved and merged into a single flat file where:
    - HF model weights are stored **without** ``model.`` prefix
    - Extra heads (depth_head, etc.) are stored as-is
    """
    try:
        from transformers import (
            Mask2FormerConfig,
            Mask2FormerForUniversalSegmentation,
        )
    except ImportError as exc:
        raise ImportError("pip install transformers") from exc

    print(f"[1/4] Loading HF base model: {base_model}", flush=True)
    config = Mask2FormerConfig.from_pretrained(base_model)
    hf_model = Mask2FormerForUniversalSegmentation(config)

    # Resize classifier if num_classes differs
    if getattr(hf_model.config, "num_labels", None) != num_classes:
        hidden_dim = int(getattr(hf_model.config, "hidden_dim", 256))
        hf_model.config.num_labels = num_classes
        hf_model.class_predictor = torch.nn.Linear(hidden_dim, num_classes + 1)

    print(f"[2/4] Loading training checkpoint: {checkpoint_path}", flush=True)
    ckpt_sd = _load_raw_state_dict(checkpoint_path)
    ckpt_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt_sd.items()}

    # Separate depth_head (and other extra heads) from HF weights
    extra_keys = {k for k in ckpt_sd if not _is_hf_key(k)}
    extra_sd: Dict[str, torch.Tensor] = {k: ckpt_sd[k] for k in extra_keys}

    hf_ckpt = {k: v for k, v in ckpt_sd.items() if k not in extra_keys}

    target_keys = set(hf_model.state_dict().keys())
    hf_ckpt = _best_prefix_mapping(hf_ckpt, target_keys)

    # Filter out shape-mismatched keys before loading (e.g. criterion.empty_weight
    # changes shape between training config and base HF model — not used in inference)
    hf_ref = hf_model.state_dict()
    hf_ckpt_filtered = {
        k: v for k, v in hf_ckpt.items()
        if k in hf_ref and hf_ref[k].shape == v.shape
    }
    skipped = set(hf_ckpt.keys()) - set(hf_ckpt_filtered.keys())
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched keys: {sorted(skipped)[:5]}", flush=True)
    hf_ckpt = hf_ckpt_filtered

    # Load delta into base model
    result = hf_model.load_state_dict(hf_ckpt, strict=False)
    print(
        f"  HF load: missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}",
        flush=True,
    )
    if result.missing_keys[:5]:
        print(f"  missing sample: {result.missing_keys[:5]}", flush=True)

    print("[3/4] Building merged state dict...", flush=True)
    # Store HF weights WITHOUT model. prefix for maximum compatibility
    merged: Dict[str, torch.Tensor] = {}
    for k, v in hf_model.state_dict().items():
        merged[k] = v.detach().cpu().contiguous()
    # Append extra heads as-is
    for k, v in extra_sd.items():
        merged[k] = v.detach().cpu().contiguous()

    print(f"[4/4] Saving → {output_path}", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, output_path)

    # Also save HF config alongside output for easy pipeline loading
    config_dir = Path(output_path).parent / "mask2former_merged_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    hf_model.config.save_pretrained(str(config_dir))
    print(f"  Config saved → {config_dir}", flush=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_merge(
    checkpoint_path: str,
    merged_path: str,
    val_image_path: str,
    base_model: str = "facebook/mask2former-swin-large-ade-semantic",
    min_pixel_acc: float = 0.99,
    min_miou: float = 0.99,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Compare source-model vs merged-model output on a validation image.

    Returns (pixel_acc, mIoU).  Raises RuntimeError if thresholds not met.
    """
    # Import here so merge_checkpoint() can be used standalone
    from anime_seg_next import AnimeSegNextPipeline

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(val_image_path).convert("RGB")

    print("  Loading source model (training ckpt)...", flush=True)
    src_pipe = AnimeSegNextPipeline.from_checkpoint(
        checkpoint_path, device=dev, remove_bg=False
    )
    src_out = src_pipe(img, remove_bg=False)
    src_mask = src_out.color_map

    del src_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  Loading merged model...", flush=True)
    cfg_dir = str(Path(merged_path).parent / "mask2former_merged_config" / "config.json")
    cfg_path = cfg_dir if Path(cfg_dir).exists() else None
    merged_pipe = AnimeSegNextPipeline.from_checkpoint(
        merged_path, config_path=cfg_path, device=dev, remove_bg=False
    )
    merged_out = merged_pipe(img, remove_bg=False)
    merged_mask = merged_out.color_map

    pixel_acc, miou = _mask_similarity(merged_mask, src_mask)
    print(f"  pixel_acc={pixel_acc:.6f}  mIoU={miou:.6f}", flush=True)

    if pixel_acc < min_pixel_acc or miou < min_miou:
        raise RuntimeError(
            f"Merged model quality below threshold: "
            f"pixel_acc={pixel_acc:.4f} (min={min_pixel_acc}), "
            f"mIoU={miou:.4f} (min={min_miou})"
        )
    return pixel_acc, miou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw_state_dict(path: str) -> Dict[str, torch.Tensor]:
    pl = path.lower()
    if pl.endswith(".safetensors"):
        return dict(load_file(path))
    if pl.endswith((".pt", ".pth")):
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                if isinstance(raw.get(key), dict):
                    return dict(raw[key])
            return dict(raw)
        raise RuntimeError("Unsupported .pt/.pth checkpoint format")
    raise RuntimeError(f"Unsupported checkpoint extension: {path}")


def _is_hf_key(k: str) -> bool:
    """True if key belongs to the HF Mask2Former model (not an extra head)."""
    EXTRA_PREFIXES = ("depth_head.",)
    return not any(k.startswith(p) for p in EXTRA_PREFIXES)


def _best_prefix_mapping(
    sd: Dict[str, torch.Tensor],
    target_keys: set,
) -> Dict[str, torch.Tensor]:
    def _strip(d: Dict, n: int) -> Dict:
        out = {}
        for k, v in d.items():
            for _ in range(n):
                if k.startswith("model."):
                    k = k[len("model."):]
            out[k] = v
        return out

    def _add(d: Dict, prefix: str) -> Dict:
        return {prefix + k: v for k, v in d.items()}

    candidates = {
        "identity": sd,
        "strip1": _strip(sd, 1),
        "strip2": _strip(sd, 2),
        "add_model": _add(sd, "model."),
    }
    best_name, best_n = "identity", -1
    for name, cand in candidates.items():
        n = len(target_keys.intersection(cand.keys()))
        if n > best_n:
            best_n, best_name = n, name
    return candidates[best_name]


def _mask_similarity(
    pred: Image.Image, ref: Image.Image
) -> Tuple[float, float]:
    pred_np = np.array(pred.convert("RGB"), dtype=np.uint8)
    ref_np = np.array(ref.convert("RGB"), dtype=np.uint8)

    if pred_np.shape != ref_np.shape:
        pred_np = np.array(
            Image.fromarray(pred_np).resize(
                (ref_np.shape[1], ref_np.shape[0]), Image.NEAREST
            ),
            dtype=np.uint8,
        )

    flat_p = pred_np.reshape(-1, 3)
    flat_r = ref_np.reshape(-1, 3)
    pixel_acc = float(np.all(flat_p == flat_r, axis=1).mean())

    colors = np.unique(np.vstack([flat_p, flat_r]), axis=0)
    ious: List[float] = []
    for color in colors:
        pm = np.all(flat_p == color, axis=1)
        rm = np.all(flat_r == color, axis=1)
        union = np.logical_or(pm, rm).sum()
        if union == 0:
            continue
        ious.append(float(np.logical_and(pm, rm).sum() / union))
    miou = float(np.mean(ious)) if ious else 0.0
    return pixel_acc, miou


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge training checkpoint + HF base → single safetensors"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Training checkpoint path (.safetensors / .pt / .pth)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .safetensors path"
    )
    parser.add_argument(
        "--base-model",
        default="facebook/mask2former-swin-large-ade-semantic",
        help="HF base model ID for architecture",
    )
    parser.add_argument(
        "--num-classes", type=int, default=37,
        help="Number of segmentation classes (default: 37)"
    )
    parser.add_argument(
        "--validate", metavar="IMAGE",
        help="Path to a validation image. If given, runs pixel_acc/mIoU check."
    )
    parser.add_argument(
        "--min-pixel-acc", type=float, default=0.99,
        help="Minimum acceptable pixel accuracy (default: 0.99)"
    )
    parser.add_argument(
        "--min-miou", type=float, default=0.99,
        help="Minimum acceptable mIoU (default: 0.99)"
    )
    args = parser.parse_args()

    merge_checkpoint(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        output_path=args.output,
        num_classes=args.num_classes,
    )
    print("[merge] Done.", flush=True)

    if args.validate:
        print(f"[validate] Running on {args.validate}...", flush=True)
        pixel_acc, miou = validate_merge(
            checkpoint_path=args.checkpoint,
            merged_path=args.output,
            val_image_path=args.validate,
            base_model=args.base_model,
            min_pixel_acc=args.min_pixel_acc,
            min_miou=args.min_miou,
        )
        print(
            f"[validate] PASS  pixel_acc={pixel_acc:.6f}  mIoU={miou:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
