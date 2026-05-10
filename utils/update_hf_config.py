#!/usr/bin/env python3
"""Analyse .safetensors checkpoints and push an updated config.json to HF Hub.

Usage
-----
# Update config.json locally only
python scripts/update_hf_config.py models/anime_seg_mask2former_v2.safetensors --version 2

# Update and push to HF
python scripts/update_hf_config.py models/anime_seg_mask2former_v2.safetensors \\
    --version 2 --push --token $HF_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safetensors.torch import load_file as safetensors_load


# --------------------------------------------------------------------------- #
# Checkpoint analysis                                                           #
# --------------------------------------------------------------------------- #

def detect_num_classes(ckpt_path: str) -> Optional[int]:
    """Infer num_classes from the class_predictor weight shape.

    Returns None if the key is absent (merged_full or unknown architecture).
    """
    try:
        state_dict = safetensors_load(ckpt_path, device="cpu")
    except Exception as exc:
        print(f"[warn] Could not load {ckpt_path}: {exc}", file=sys.stderr)
        return None

    normalized = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    for key in ("class_predictor.weight", "model.class_predictor.weight"):
        w = normalized.get(key)
        if w is not None and len(w.shape) >= 1:
            # Shape is (num_classes + 1, hidden_dim) — subtract no-object class
            return int(w.shape[0]) - 1
    return None


def detect_merged_full(ckpt_path: str) -> bool:
    """Return True if the checkpoint contains the encoder embedding weights."""
    try:
        keys = list(safetensors_load(ckpt_path, device="cpu").keys())
    except Exception:
        return False
    norm = [k.replace("_orig_mod.", "") for k in keys]
    return any(
        k.startswith("model.pixel_level_module.encoder.embeddings")
        or k.startswith("pixel_level_module.encoder.embeddings")
        for k in norm
    )


# --------------------------------------------------------------------------- #
# Config update                                                                 #
# --------------------------------------------------------------------------- #

def _load_config(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


def _save_config(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _resolve_series(num_classes: int) -> str:
    from anime_seg_next.series import _COUNT_TO_SERIES
    return _COUNT_TO_SERIES.get(num_classes, f"custom-{num_classes}")


def build_entry(
    ckpt_path: str,
    base_model: str,
    train_image_size: int,
    version: int,
) -> Dict:
    num_classes = detect_num_classes(ckpt_path)
    if num_classes is None:
        raise RuntimeError(
            f"Could not detect num_classes from {ckpt_path}. "
            "Ensure the checkpoint contains a class_predictor weight."
        )
    merged_full = detect_merged_full(ckpt_path)
    series = _resolve_series(num_classes)
    return {
        "FilePath": f"models/{Path(ckpt_path).name}",
        "BaseModel": base_model,
        "TrainImageSize": train_image_size,
        "Architecture": "mask2former",
        "Version": version,
        "series": series,
        "num_classes": num_classes,
        "merged_full": merged_full,
    }


def upsert_entry(data: Dict, entry: Dict) -> None:
    """Insert or replace the config entry matching FilePath."""
    models: list = data.setdefault("models", [])
    for i, m in enumerate(models):
        if m.get("FilePath") == entry["FilePath"]:
            models[i] = entry
            return
    models.append(entry)
    # Keep highest version first
    models.sort(key=lambda m: int(m.get("Version", 0)), reverse=True)


# --------------------------------------------------------------------------- #
# HF Hub upload                                                                 #
# --------------------------------------------------------------------------- #

def push_config(config_path: str, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=repo_id,
        token=token,
        commit_message="chore: auto-update config.json [checkpoint analysis]",
    )
    print(f"✓ Pushed config.json → {repo_id}")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse checkpoints and update HF config.json"
    )
    p.add_argument("checkpoint", nargs="+", help=".safetensors checkpoint path(s)")
    p.add_argument("--config", default="config.json", help="Local config.json path")
    p.add_argument("--repo-id", default="suzukimain/AnimeSeg-Next")
    p.add_argument(
        "--base-model",
        default="facebook/mask2former-swin-large-ade-semantic",
    )
    p.add_argument("--train-image-size", type=int, default=768)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--push", action="store_true", help="Push to HF Hub after update")
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF access token (default: $HF_TOKEN)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    data = _load_config(config_path)

    for ckpt in args.checkpoint:
        print(f"Analysing {ckpt} …")
        entry = build_entry(
            ckpt_path=ckpt,
            base_model=args.base_model,
            train_image_size=args.train_image_size,
            version=args.version,
        )
        upsert_entry(data, entry)
        print(
            f"  num_classes={entry['num_classes']}  "
            f"series={entry['series']}  "
            f"merged_full={entry['merged_full']}"
        )

    _save_config(config_path, data)
    print(f"✓ Wrote {config_path}")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if args.push:
        if not args.token:
            sys.exit("--push requires --token or $HF_TOKEN")
        push_config(str(config_path), args.repo_id, args.token)


if __name__ == "__main__":
    main()
