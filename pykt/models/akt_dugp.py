"""AKT backbone with DUGP residual shrinkage.

This is the formal experiment model for DUGP-KT.  It keeps the original AKT
encoder unchanged and injects DUGP after the AKT hidden state is produced.
"""

from __future__ import annotations

import torch
from torch import nn

from .akt import AKT
from .dugp_modules import DUGPResidualAdapter


class AKTDUGP(AKT):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_groups=9,
        alpha_feat_dim=5,
        dugp_mode="full",
        fixed_alpha=0.5,
        alpha_hidden_dim=64,
        alpha_init_bias=2.94443897917,
        dugp_layer_norm=False,
        detach_distance=False,
        dugp_residual_scale=0.1,
        learnable_residual_scale=True,
        model_name="akt_dugp",
    ):
        super().__init__(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
        )
        self.model_name = model_name
        self.model_type = "akt"
        self.dugp_mode = dugp_mode
        self.alpha_feat_dim = alpha_feat_dim
        self.num_groups = num_groups

        self.dugp_adapter = DUGPResidualAdapter(
            hidden_dim=d_model,
            num_groups=num_groups,
            alpha_feat_dim=alpha_feat_dim,
            dropout=dropout,
            mode=dugp_mode,
            fixed_alpha=fixed_alpha,
            alpha_hidden_dim=alpha_hidden_dim,
            alpha_init_bias=alpha_init_bias,
            use_layer_norm=dugp_layer_norm,
            detach_distance=detach_distance,
            dugp_residual_scale=dugp_residual_scale,
            learnable_residual_scale=learnable_residual_scale,
        )

        self.last_alpha = None
        self.last_distance = None
        self.last_group_id = None

    def forward(self, q_data, target, pid_data=None, group_id=None, alpha_feat=None, qtest=False):
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        else:
            raise NotImplementedError(f"AKTDUGP currently supports qid-style embeddings, got emb_type={emb_type}")

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = 0.0

        hidden = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        hidden, aux = self.dugp_adapter(hidden, group_id=group_id, alpha_feat=alpha_feat)

        self.last_alpha = aux["alpha"].detach()
        self.last_distance = aux["distance"].detach()
        self.last_group_id = aux["group_id"].detach()

        concat_q = torch.cat([hidden, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)
        if not qtest:
            return preds, c_reg_loss
        return preds, c_reg_loss, concat_q

    def dugp_stats(self):
        return self.dugp_adapter.last_stats()


# Backward-compatible name used by your current scripts/configs.
class AKTOurs(AKTDUGP):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model_name", "akt_ours")
        super().__init__(*args, **kwargs)
