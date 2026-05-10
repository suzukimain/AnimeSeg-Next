from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
# Try to find AnimeSeg sibling to add its src to path (required for base classes)
ANIME_SEG_ROOT = ROOT.parent / "AnimeSeg"
if not (ANIME_SEG_ROOT / "src").exists():
    ANIME_SEG_ROOT = ROOT.parent.parent / "AnimeSeg"

if (ANIME_SEG_ROOT / "src").exists():
    sys.path.insert(0, str(ANIME_SEG_ROOT / "src"))

# Add AnimeSeg-Next src to path (SHOULD BE FIRST to override any duplicates in AnimeSeg)
sys.path.insert(0, str(ROOT / "src"))

try:
    from anime_seg_next import AnimeSegNextPipeline  # noqa: E402
except ImportError:
    print("[error] Failed to import anime_seg_next. Make sure AnimeSeg and AnimeSeg-Next are in the same parent directory or installed.")
    sys.exit(1)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_temp_config(data: dict) -> Path:
    # Use repo root for tmp if possible, otherwise ROOT
    repo_root = ROOT.parents[2] if len(ROOT.parents) > 2 else ROOT
    tmp_dir = repo_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="merge_cfg_next_", suffix=".json", dir=str(tmp_dir))
    Path(tmp_name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(tmp_name).chmod(0o666)
    Path(tmp_name).touch()
    return Path(tmp_name)


def _mask_similarity(pred: Image.Image, ref: Image.Image) -> Tuple[float, float]:
    pred_np = np.array(pred.convert("RGB"), dtype=np.uint8)
    ref_np = np.array(ref.convert("RGB"), dtype=np.uint8)

    if pred_np.shape != ref_np.shape:
        pred_np = np.array(
            Image.fromarray(pred_np).resize((ref_np.shape[1], ref_np.shape[0]), Image.Resampling.NEAREST),
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
        inter = np.logical_and(pm, rm).sum()
        ious.append(float(inter / union))

    miou = float(np.mean(ious)) if ious else 0.0
    return pixel_acc, miou


def _sanitize_for_source(config_data: dict, idx: int) -> dict:
    data = copy.deepcopy(config_data)
    item = data["models"][idx]
    cfg = item.get("Config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["merged_full"] = False
    item["Config"] = cfg

    arch = str(item.get("Architecture", "")).lower()
    if arch == "mask2former":
        base = str(item.get("BaseModel", ""))
        # If it's a local path or custom, reset to default for source loading if we want to merge from HF base
        if "/" in base and not base.startswith("facebook/"):
             # Keep it if it's already a known HF model, otherwise fallback
             pass
    return data


def _set_merged_flags(config_data: dict, idx: int, mask2former_config_relpath: str | None) -> dict:
    data = copy.deepcopy(config_data)
    item = data["models"][idx]
    cfg = item.get("Config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["merged_full"] = True
    item["Config"] = cfg

    if mask2former_config_relpath is not None:
        item["BaseModel"] = mask2former_config_relpath

    return data


def _create_source_pipe(arch: str, filename: str, config_path: Path, device: str):
    if arch == "mask2former":
        # AnimeSegNextPipeline.from_mask2former uses config_name (path or filename in repo)
        return AnimeSegNextPipeline.from_mask2former(
            repo_id="suzukimain/AnimeSeg-Next", # Dummy valid repo id
            filename=filename, 
            config_name=str(config_path)
        ).to(device)
    raise RuntimeError(f"Unsupported architecture: {arch}")


def _merge_one(entry: Dict, source_pipe, model_path: Path) -> str | None:
    arch = str(entry.get("Architecture", "")).lower()

    if arch == "mask2former":
        # extract state dict from our wrapper model
        # it contains 'model.*' (HF) and possibly 'depth_head.*' (Multitask)
        full_sd = source_pipe.model.state_dict()
        
        state_dict = {}
        for k, v in full_sd.items():
            if k.startswith("model."):
                # Remove 'model.' prefix for standard HF compatibility
                state_dict[k[len("model."):]] = v.detach().cpu().contiguous()
            else:
                # Keep other keys (like depth_head.*) as is
                state_dict[k] = v.detach().cpu().contiguous()

        save_file(state_dict, str(model_path))

        config_dir = model_path.parent / "mask2former_merged_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        source_pipe.model.model.config.save_pretrained(str(config_dir))
        
        try:
            return str(config_dir.relative_to(ROOT).as_posix())
        except ValueError:
            # Fallback to repo root or absolute
            repo_root = ROOT.parents[2] if len(ROOT.parents) > 2 else ROOT
            try:
                return str(config_dir.relative_to(repo_root).as_posix())
            except ValueError:
                return str(config_dir.resolve().as_posix())

    raise RuntimeError(f"Unsupported architecture: {arch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AnimeSeg-Next checkpoints into full single-file safetensors and validate similarity.")
    parser.add_argument("--config", default="config.json", help="Path to model config JSON")
    parser.add_argument("--image", default="../../../image/test-input/sample.png", help="Validation image path")
    parser.add_argument("--min-miou", type=float, default=0.995, help="Minimum mIoU between source and merged outputs")
    parser.add_argument("--min-acc", type=float, default=0.995, help="Minimum pixel accuracy between source and merged outputs")
    args = parser.parse_args()

    config_path = ROOT / args.config
    val_image_path = ROOT / args.image
    if not val_image_path.exists():
        # Try absolute or relative from CWD
        val_image_path = Path(args.image).resolve()
        if not val_image_path.exists():
            raise FileNotFoundError(f"Validation image not found: {args.image}")

    config_data = _load_json(config_path)
    models = config_data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Invalid model config format: 'models' must be a list")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_image = Image.open(val_image_path).convert("RGB")

    print(f"device={device}")
    print(f"validation_image={val_image_path}")

    report: List[Tuple[str, float, float]] = []

    for index, item in enumerate(models):
        if not isinstance(item, dict):
            continue

        arch = str(item.get("Architecture", "")).lower()
        file_path = str(item.get("FilePath", "")).strip()
        if arch not in {"mask2former"} or not file_path:
            continue

        cfg = item.get("Config", {})
        if cfg.get("merged_full"):
            print(f"Skipping already merged model {index}: {file_path}")
            continue

        # Output paths: we always want to save to models/base/ for distribution
        version = item.get("Version", index)
        if version == 3:
            model_path = ROOT / "models/base/depth/anime_seg-next_mask2former_v3.safetensors"
        else:
            model_path = ROOT / f"models/base/anime_seg-next_mask2former_v{version}.safetensors"
        
        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing model {index}: {file_path}")

        source_cfg = _sanitize_for_source(config_data, index)
        source_cfg_path = _write_temp_config(source_cfg)
        
        # Load source (which pulls base weights from HF if merged_full=False)
        source_pipe = _create_source_pipe(arch, file_path, source_cfg_path, device)
        source_output = source_pipe(val_image)
        # Handle both AnimeSegOutput (with .color_map) and legacy Image return
        source_mask = source_output.color_map if hasattr(source_output, "color_map") else source_output

        # Merge and save to a temporary file first to avoid locking issues
        temp_model_path = model_path.with_suffix(".tmp_safetensors")
        merged_base = _merge_one(item, source_pipe, temp_model_path)
        
        # Close pipe/model if possible to release file handles
        del source_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reload as merged and validate
        merged_cfg = _set_merged_flags(config_data, index, merged_base if arch == "mask2former" else None)
        merged_cfg_path = _write_temp_config(merged_cfg)
        
        # We need to replace the original with the temp before loading the 'merged' pipe
        # because the merged pipe will look for the file at model_path
        if model_path.exists():
             # We might need to wait or use a different approach if it's still locked
             import os
             import time
             for _ in range(5):
                 try:
                     if model_path.exists():
                         model_path.unlink()
                     break
                 except PermissionError:
                     time.sleep(1)
        temp_model_path.rename(model_path)

        merged_pipe = _create_source_pipe(arch, file_path, merged_cfg_path, device)
        merged_output = merged_pipe(val_image)
        merged_mask = merged_output.color_map if hasattr(merged_output, "color_map") else merged_output

        pixel_acc, miou = _mask_similarity(merged_mask, source_mask)
        report.append((f"{arch}_v{item.get('Version', index)}", pixel_acc, miou))

        # Save check image
        repo_root = ROOT.parents[2] if len(ROOT.parents) > 2 else ROOT
        out_mask_dir = repo_root / "tmp" / "img"
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        out_mask = out_mask_dir / f"merged_check_next_{arch}_v{item.get('Version', index)}_mask.png"
        merged_mask.save(out_mask)

        print(f"[ok] {arch} merged -> {model_path}")
        print(f"[ok] {arch} similarity pixel_acc={pixel_acc:.6f} mIoU={miou:.6f}")

        if pixel_acc < args.min_acc or miou < args.min_miou:
            print(f"[warning] Merged model quality check failed for {arch}: "
                  f"pixel_acc={pixel_acc:.6f} (min={args.min_acc}), "
                  f"mIoU={miou:.6f} (min={args.min_miou})")

        config_data = merged_cfg

    _save_json(config_path, config_data)
    
    # Also update FilePath to the new merged locations in the final config
    final_config = _load_json(config_path)
    for index, item in enumerate(final_config["models"]):
        version = item.get("Version", index)
        if version == 3:
            item["FilePath"] = "models/base/depth/anime_seg-next_mask2former_v3.safetensors"
        else:
            item["FilePath"] = f"models/base/anime_seg-next_mask2former_v{version}.safetensors"
    _save_json(config_path, final_config)

    print(f"[ok] updated config: {config_path}")

    for name, pixel_acc, miou in report:
        print(f"[report] {name}: pixel_acc={pixel_acc:.6f}, mIoU={miou:.6f}")


if __name__ == "__main__":
    main()
