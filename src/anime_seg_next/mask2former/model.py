"""Self-contained Mask2Former model wrapper with optional depth head."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class DepthHead(nn.Module):
    """Depth head that adapts to the checkpoint architecture.

    Supports two variants detected from checkpoint keys:
    - ``"conv"``: Conv2d-based head (net.0.weight …). Input: BxCxH'xW' pixel-decoder feats.
    - ``"linear"``: Linear projection head (0.weight …). Input: BxQxC query features
      → weighted by mask → upsample.
    """

    def __init__(self, variant: str = "conv", in_channels: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.variant = variant
        if variant == "conv":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.GroupNorm(32, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(32, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
        else:  # linear: query_dim → 1
            self.net = nn.Sequential(nn.Linear(in_channels, 1))

    def forward_conv(self, pixel_decoder_feats: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_decoder_feats)

    def forward_linear(
        self,
        query_feats: torch.Tensor,      # BxQxC
        mask_probs: torch.Tensor,        # BxQxHxW (already sigmoid+upsampled)
    ) -> torch.Tensor:
        # per-query depth scalar: BxQ
        depth_per_query = self.net(query_feats).squeeze(-1)  # BxQ
        # weighted average across queries: Bx1xHxW
        weights = mask_probs / (mask_probs.sum(dim=1, keepdim=True) + 1e-6)
        depth = torch.einsum("bq,bqhw->bhw", depth_per_query, weights).unsqueeze(1)
        return depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Mask2FormerModel(nn.Module):
    """Mask2Former wrapper with lazy depth head and flexible checkpoint loading.

    Args:
        base_model: HF model ID or local directory for config/weights.
        num_classes: Number of segmentation classes (excluding no-object).
        load_base_pretrained: If True, load ImageNet weights from HF.
    """

    def __init__(
        self,
        base_model: str,
        num_classes: int = 37,
        load_base_pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import (
                Mask2FormerConfig,
                Mask2FormerForUniversalSegmentation,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required: pip install transformers"
            ) from exc

        self.num_classes = num_classes
        if load_base_pretrained:
            self.hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                base_model
            )
        else:
            config = Mask2FormerConfig.from_pretrained(base_model)
            self.hf_model = Mask2FormerForUniversalSegmentation(config)

        # Replace classifier head if class count differs
        if getattr(self.hf_model.config, "num_labels", None) != num_classes:
            hidden_dim = int(getattr(self.hf_model.config, "hidden_dim", 256))
            self.hf_model.config.num_labels = num_classes
            self.hf_model.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

        self.depth_head: Optional[DepthHead] = None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[List[str], List[str]]:
        """Load weights from a .safetensors or .pt/.pth file.

        Handles key-prefix variants automatically (identity / strip model. /
        add model.) and initialises the depth head on demand.

        Returns:
            (missing_keys, unexpected_keys) from the HF model load.
        """
        sd = _load_state_dict(checkpoint_path)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

        target_keys = set(self.hf_model.state_dict().keys())
        best_sd = _best_prefix_mapping(sd, target_keys)

        # Split into hf_model weights and depth_head weights
        hf_sd: Dict[str, torch.Tensor] = {}
        depth_sd: Dict[str, torch.Tensor] = {}
        hf_ref = self.hf_model.state_dict()
        for k, v in best_sd.items():
            if k.startswith("depth_head."):
                depth_sd[k[len("depth_head."):]] = v
            elif k in hf_ref and hf_ref[k].shape == v.shape:
                hf_sd[k] = v

        # Initialise depth head lazily — detect variant from checkpoint key shapes
        if depth_sd and self.depth_head is None:
            feat_dim = int(getattr(self.hf_model.config, "feature_size", 256))
            variant = _detect_depth_head_variant(depth_sd, feat_dim)
            if variant is not None:
                self.depth_head = DepthHead(variant=variant, in_channels=feat_dim)
                self.depth_head.to(next(self.hf_model.parameters()).device)

        missing, unexpected = self.hf_model.load_state_dict(hf_sd, strict=False)
        if self.depth_head is not None and depth_sd:
            self.depth_head.load_state_dict(depth_sd, strict=False)

        max_allowed_missing = max(32, int(0.2 * len(hf_ref)))
        if len(missing) > max_allowed_missing:
            raise RuntimeError(
                f"Too many missing keys ({len(missing)}) after checkpoint load. "
                "Check that base_model matches the checkpoint architecture."
            )
        return missing, unexpected

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run segmentation (and optionally depth) inference.

        Formula matches training _semantic_from_queries exactly:
            sigmoid → interpolate → einsum
        (sigmoid is applied BEFORE upsampling to preserve boundary sharpness)

        Args:
            pixel_values: BCHW float tensor, ImageNet-normalised.

        Returns:
            Dict with keys:
              - ``semantic_logits``: BxCxHxW float (C = num_classes)
              - ``query_mask_logits``: BxQxHxW float (sigmoid + upsampled)
              - ``query_part_logits``: BxQx(C+1) float (raw class logits)
              - ``depth`` (optional): Bx1xHxW float in [0, 1]
        """
        h, w = pixel_values.shape[-2:]
        raw = self.hf_model(pixel_values=pixel_values)

        cls_logits = raw.class_queries_logits   # BxQx(C+1)
        mask_logits = raw.masks_queries_logits  # BxQxH'xW'

        cls_probs = F.softmax(cls_logits, dim=-1)[..., :-1]  # drop no-object
        mask_probs = F.interpolate(
            mask_logits.sigmoid(), size=(h, w), mode="bilinear", align_corners=False
        )
        sem = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)

        out: Dict[str, torch.Tensor] = {
            "semantic_logits": sem,
            "query_mask_logits": mask_probs,
            "query_part_logits": cls_logits,
        }

        if self.depth_head is not None:
            if self.depth_head.variant == "conv":
                feats = raw.pixel_decoder_last_hidden_state
                depth_logits = self.depth_head.forward_conv(feats)
            else:
                # linear: use transformer decoder query states
                query_feats = raw.transformer_decoder_last_hidden_state  # BxQxC
                depth_logits = self.depth_head.forward_linear(query_feats, mask_probs)
            depth = F.interpolate(
                depth_logits.sigmoid(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            out["depth"] = depth

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    pl = path.lower()
    if pl.endswith(".safetensors"):
        return load_file(path)
    if pl.endswith((".pt", ".pth")):
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                if isinstance(raw.get(key), dict):
                    return raw[key]
            return raw
        raise RuntimeError("Unsupported .pt/.pth checkpoint format")
    raise RuntimeError(f"Unsupported checkpoint extension: {path}")


def _detect_depth_head_variant(
    depth_sd: Dict[str, torch.Tensor], feat_dim: int
) -> Optional[str]:
    """Detect depth head variant from its state dict keys/shapes.

    Returns ``"conv"``, ``"linear"``, or ``None`` if unrecognised.
    """
    keys = set(depth_sd.keys())
    # Conv variant: keys start with "net."
    if any(k.startswith("net.") for k in keys):
        return "conv"
    # Linear variant: weight shape is [1, feat_dim] (nn.Linear)
    for k, v in depth_sd.items():
        if k.endswith(".weight") or k == "0.weight":
            if v.ndim == 2 and v.shape == (1, feat_dim):
                return "linear"
    return None


def _best_prefix_mapping(
    sd: Dict[str, torch.Tensor],
    target_keys: set,
) -> Dict[str, torch.Tensor]:
    """Return the key-remapping of sd that maximises overlap with target_keys."""

    def _strip(d: Dict, n: int) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            for _ in range(n):
                if k.startswith("model."):
                    k = k[len("model."):]
            out[k] = v
        return out

    def _add(d: Dict, prefix: str) -> Dict[str, torch.Tensor]:
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
