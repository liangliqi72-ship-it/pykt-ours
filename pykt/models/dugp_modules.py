"""DUGP-KT modules.

This file contains the distance- and uncertainty-aware residual shrinkage
adapter used by AKT-DUGP.  It is intentionally separated from the AKT backbone
so that the same adapter can be reused for future backbones.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class DUGPResidualAdapter(nn.Module):
    """Distance- and uncertainty-aware group-prior residual adapter.

    Given an individual hidden state h_t from a KT backbone and a dynamic group
    id z_t generated from causal prefix features, this module constructs a
    learnable group prior g_t and shrinks the individual residual h_t - g_t by
    a sample-level, timestep-level gate alpha_t.

    Main mode:
        h*_t = g_t + alpha_t * (h_t - g_t)

    The same module also supports official ablation modes used in the paper.
    """

    VALID_MODES = {
        "full",
        "fixed_fusion",
        "group_add",
        "group_only",
        "alpha_only",
        "no_distance",
        "no_uncertainty",
        "no_behavior",
        "none",
    }

    def __init__(
        self,
        hidden_dim: int,
        num_groups: int = 9,
        alpha_feat_dim: int = 5,
        dropout: float = 0.1,
        mode: str = "full",
        fixed_alpha: float = 0.5,
        alpha_hidden_dim: int = 64,
        alpha_init_bias: float = 2.94443897917,  # sigmoid ~= 0.95
        use_layer_norm: bool = False,
        detach_distance: bool = False,
        dugp_residual_scale: float = 0.1,
        learnable_residual_scale: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown DUGP mode: {mode}. Valid modes: {sorted(self.VALID_MODES)}")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if alpha_feat_dim <= 0:
            raise ValueError("alpha_feat_dim must be positive")

        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.unknown_group = num_groups - 1
        self.alpha_feat_dim = alpha_feat_dim
        self.mode = mode
        self.fixed_alpha = float(fixed_alpha)
        self.detach_distance = detach_distance
        self.dugp_residual_scale = float(dugp_residual_scale)
        self.learnable_residual_scale = bool(learnable_residual_scale)

        self.group_emb = nn.Embedding(num_groups, hidden_dim)
        self.group_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Start the group-add branch as an identity-safe perturbation.
        # This prevents a randomly initialized group prior from damaging the AKT hidden state
        # during the first epochs.
        nn.init.zeros_(self.group_proj[-1].weight)
        nn.init.zeros_(self.group_proj[-1].bias)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(alpha_feat_dim + 1, alpha_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(alpha_hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.alpha_mlp[-2].bias, alpha_init_bias)

        self.out_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        init_logit = torch.logit(torch.tensor(self.dugp_residual_scale, dtype=torch.float32).clamp(1e-4, 1 - 1e-4))
        if self.learnable_residual_scale:
            self.residual_scale_logit = nn.Parameter(init_logit.clone())
        else:
            self.register_buffer("residual_scale_logit", init_logit.clone())

        self.last_alpha: Optional[torch.Tensor] = None
        self.last_distance: Optional[torch.Tensor] = None
        self.last_group_id: Optional[torch.Tensor] = None
        self.last_residual_norm: Optional[torch.Tensor] = None

    def _prepare_group_id(self, group_id: Optional[torch.Tensor], batch: int, length: int, device) -> torch.Tensor:
        if group_id is None:
            group_id = torch.full((batch, length), self.unknown_group, dtype=torch.long, device=device)
        else:
            group_id = group_id.to(device=device, dtype=torch.long)
            if group_id.dim() == 1:
                group_id = group_id.unsqueeze(1).expand(-1, length)
            elif group_id.dim() == 2 and group_id.size(1) == 1:
                group_id = group_id.expand(-1, length)
            elif group_id.dim() != 2:
                raise ValueError(f"group_id must have shape [B], [B,1], or [B,L], got {tuple(group_id.shape)}")
            if group_id.size(0) != batch or group_id.size(1) != length:
                raise ValueError(
                    f"group_id must match hidden shape [B,L]. group_id={tuple(group_id.shape)}, hidden={(batch, length)}"
                )
        return group_id.clamp_(min=0, max=self.num_groups - 1)

    def _prepare_alpha_feat(self, alpha_feat: Optional[torch.Tensor], batch: int, length: int, dtype, device) -> torch.Tensor:
        if alpha_feat is None:
            return torch.zeros(batch, length, self.alpha_feat_dim, dtype=dtype, device=device)
        alpha_feat = alpha_feat.to(device=device, dtype=dtype)
        if alpha_feat.dim() == 2:
            alpha_feat = alpha_feat.unsqueeze(1).expand(-1, length, -1)
        if alpha_feat.dim() != 3:
            raise ValueError(f"alpha_feat must have shape [B,F] or [B,L,F], got {tuple(alpha_feat.shape)}")
        if alpha_feat.size(0) != batch or alpha_feat.size(1) != length:
            raise ValueError(
                f"alpha_feat must match hidden shape [B,L]. alpha_feat={tuple(alpha_feat.shape)}, hidden={(batch, length)}"
            )
        if alpha_feat.size(-1) != self.alpha_feat_dim:
            raise ValueError(f"alpha_feat last dim should be {self.alpha_feat_dim}, got {alpha_feat.size(-1)}")
        return torch.nan_to_num(alpha_feat, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(
        self,
        hidden: torch.Tensor,
        group_id: Optional[torch.Tensor] = None,
        alpha_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must have shape [B,L,D], got {tuple(hidden.shape)}")
        batch, length, dim = hidden.shape
        if dim != self.hidden_dim:
            raise ValueError(f"hidden dim mismatch: expected {self.hidden_dim}, got {dim}")

        group_id = self._prepare_group_id(group_id, batch, length, hidden.device)
        alpha_feat = self._prepare_alpha_feat(alpha_feat, batch, length, hidden.dtype, hidden.device)

        group_prior = self.group_emb(group_id)
        residual = hidden - group_prior
        distance = torch.linalg.vector_norm(residual, ord=2, dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        distance = torch.log1p(distance)
        if self.detach_distance:
            distance_for_gate = distance.detach()
        else:
            distance_for_gate = distance

        gate_feat = alpha_feat
        gate_dist = distance_for_gate

        if self.mode == "no_distance":
            gate_dist = torch.zeros_like(gate_dist)
        elif self.mode == "no_uncertainty":
            gate_feat = torch.zeros_like(gate_feat)
        elif self.mode == "no_behavior":
            gate_feat = gate_feat.clone()
            if gate_feat.size(-1) >= 5:
                gate_feat[..., 4] = 0.0

        gate_input = torch.cat([gate_dist, gate_feat], dim=-1)
        alpha = self.alpha_mlp(gate_input)

        # `residual_scale` makes the full DUGP branch identity-safe:
        # corrected = hidden + s * (candidate - hidden), where s starts small.
        # This keeps AKT performance stable while still allowing the model to learn
        # when the group prior should influence the state.
        residual_scale = torch.sigmoid(self.residual_scale_logit).to(dtype=hidden.dtype, device=hidden.device)

        if self.mode == "none":
            corrected = hidden
        elif self.mode == "group_only":
            candidate = group_prior
            corrected = hidden + residual_scale * (candidate - hidden)
        elif self.mode == "group_add":
            corrected = hidden + self.dropout(self.group_proj(group_prior))
        elif self.mode == "fixed_fusion":
            fixed = torch.full_like(alpha, self.fixed_alpha)
            candidate = group_prior + fixed * residual
            corrected = hidden + residual_scale * (candidate - hidden)
            alpha = fixed
        elif self.mode == "alpha_only":
            # Use alpha as a small reliability modulation instead of replacing hidden.
            corrected = hidden * (1.0 + residual_scale * (alpha - alpha.detach().mean().clamp(1e-4, 1.0)))
        else:
            # full, no_distance, no_uncertainty, no_behavior
            candidate = group_prior + alpha * residual
            corrected = hidden + residual_scale * (candidate - hidden)

        if self.mode != "none":
            corrected = self.out_norm(corrected)

        if mask is not None:
            mask = mask.to(device=hidden.device).bool()
            if mask.dim() != 2 or mask.size(0) != batch or mask.size(1) != length:
                raise ValueError(f"mask must have shape [B,L], got {tuple(mask.shape)}")
            corrected = corrected * mask.unsqueeze(-1).to(hidden.dtype)
            alpha = alpha * mask.unsqueeze(-1).to(hidden.dtype)
            distance = distance * mask.unsqueeze(-1).to(hidden.dtype)

        self.last_alpha = alpha.detach()
        self.last_distance = distance.detach()
        self.last_group_id = group_id.detach()
        self.last_residual_norm = torch.linalg.vector_norm(residual.detach(), ord=2, dim=-1, keepdim=True)

        aux = {
            "alpha": alpha,
            "distance": distance,
            "group_id": group_id,
            "residual_norm": self.last_residual_norm,
            "residual_scale": residual_scale.detach(),
        }
        return corrected, aux

    def last_stats(self) -> Dict[str, float]:
        """Return scalar diagnostics from the last forward pass."""
        out: Dict[str, float] = {}
        if self.last_alpha is not None:
            a = self.last_alpha.float().detach().cpu()
            out.update({
                "alpha_mean": float(a.mean()),
                "alpha_std": float(a.std(unbiased=False)),
                "alpha_min": float(a.min()),
                "alpha_max": float(a.max()),
            })
        if self.last_distance is not None:
            d = self.last_distance.float().detach().cpu()
            out.update({
                "distance_mean": float(d.mean()),
                "distance_std": float(d.std(unbiased=False)),
                "distance_min": float(d.min()),
                "distance_max": float(d.max()),
            })
        if hasattr(self, "residual_scale_logit"):
            out["residual_scale"] = float(torch.sigmoid(self.residual_scale_logit.detach()).cpu())
        return out
