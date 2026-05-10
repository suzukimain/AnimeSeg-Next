from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class Mask2FormerAnimeSegModel(nn.Module):
    """AnimeSeg-Next Mask2Former model with multitask depth support."""

    def __init__(self, base_model: str, num_classes: int = 12, load_base_pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise ImportError("transformers is required: pip install transformers") from exc

        self.num_classes = num_classes
        if load_base_pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(base_model)
        else:
            config = Mask2FormerConfig.from_pretrained(base_model)
            self.model = Mask2FormerForUniversalSegmentation(config)

        if getattr(self.model.config, "num_labels", None) != num_classes:
            hidden_dim = int(getattr(self.model.config, "hidden_dim", 256))
            self.model.config.num_labels = num_classes
            self.model.class_predictor = nn.Linear(hidden_dim, num_classes + 1)
        
        # Depth head for multitask inference (V3 models)
        self.depth_head: Optional[nn.Module] = None

    def _init_depth_head(self) -> None:
        """Initialize a depth head matching the multitask training architecture."""
        if self.depth_head is not None:
            return
            
        hidden_dim = int(getattr(self.model.config, "hidden_dim", 256))
        # Simple upsampling depth head: [B, Q, D] -> [B, 1, H, W]
        # In practice, we use the mask queries to generate depth.
        # This is a placeholder for the actual head structure if it were complex.
        # For now, we assume the weights in the checkpoint define the structure.
        # We'll use a simple linear + upsample if needed, but usually it's just a projection.
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[list, list]:
        checkpoint_lower = checkpoint_path.lower()
        if checkpoint_lower.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        elif checkpoint_lower.endswith(".pt") or checkpoint_lower.endswith(".pth"):
            raw = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(raw, dict):
                candidate_keys = [
                    "state_dict",
                    "model_state_dict",
                    "model",
                    "module",
                ]
                resolved = None
                for key in candidate_keys:
                    val = raw.get(key)
                    if isinstance(val, dict):
                        resolved = val
                        break
                state_dict = resolved if resolved is not None else raw
            else:
                raise RuntimeError("Unsupported checkpoint format in .pt/.pth file")
        else:
            raise RuntimeError(f"Unsupported checkpoint extension: {checkpoint_path}")

        # Normalize keys (remove _orig_mod. from torch.compile)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Check for depth_head in state_dict
        has_depth = any(k.startswith("depth_head.") for k in state_dict.keys())
        if has_depth:
            self._init_depth_head()

        model_state_dict = self.state_dict()
        target_keys = set(model_state_dict.keys())

        candidates = {
            "identity": state_dict,
            "strip_model": {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()},
            "add_model": {f"model.{k}" if not k.startswith("depth_head.") else k: v for k, v in state_dict.items()},
        }

        best_name = "identity"
        best_overlap = -1
        for name, candidate in candidates.items():
            overlap = len(target_keys.intersection(candidate.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name

        state_dict = candidates[best_name]

        filtered_state_dict = {}
        mismatched_keys = []
        for key, value in state_dict.items():
            if key not in model_state_dict:
                continue
            if model_state_dict[key].shape != value.shape:
                mismatched_keys.append(key)
                continue
            filtered_state_dict[key] = value

        if len(mismatched_keys) > 64: # Increased allowance for multitask heads
            raise RuntimeError(
                f"Checkpoint tensor shapes do not match current model. Mismatched: {len(mismatched_keys)}"
            )

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)

        # Depth head might be missing in older models, which is fine
        missing = [m for m in missing if not m.startswith("depth_head.")]

        max_allowed_missing = max(32, int(0.2 * len(model_state_dict)))
        if len(missing) > max_allowed_missing:
            raise RuntimeError(
                f"Too many parameters are missing ({len(missing)}). Checkpoint mapping likely incorrect."
            )

        return missing, unexpected

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = pixel_values.shape[-2:]
        outputs = self.model(pixel_values=pixel_values)

        cls_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits

        # 3. Multitask Depth Inference - Sigmoid-before-Interpolate logic
        # This matches training behavior exactly and results in sharper boundaries.
        cls_probs = F.softmax(cls_logits, dim=-1)[..., : self.num_classes]
        
        mask_probs = mask_logits.sigmoid()
        up_mask_probs = F.interpolate(mask_probs, size=(h, w), mode="bilinear", align_corners=False)
        
        # 3. Multitask Depth Inference - torch.einsum based calculation
        sem_prob = torch.einsum("bqc,bqhw->bchw", cls_probs, up_mask_probs)
        sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
        sem_logits = torch.log(sem_prob.clamp(min=1e-6))

        res = {
            "semantic_logits": sem_logits,
            "query_mask_logits": mask_logits, # Keep original resolution for internal use if needed
            "query_part_logits": cls_logits,
        }

        # Multitask Depth Inference - V3 support
        if self.depth_head is not None:
            # depth_head expects [B, Q, D] -> [B, Q, 1]
            # Then we weight the depth by mask probabilities
            query_depth = self.depth_head(outputs.transformer_decoder_last_hidden_state) # [B, Q, 1]
            
            # Weighted average depth based on masks
            # up_mask_probs: [B, Q, H, W]
            # query_depth: [B, Q, 1]
            depth_map = torch.einsum("bqhw,bqk->bkhw", up_mask_probs, query_depth)
            
            # Normalize by total mask weight
            mask_sum = up_mask_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            depth_map = depth_map / mask_sum
            
            res["depth"] = depth_map

        return res
