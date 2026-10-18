#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.modeling_utils as hf_modeling_utils
from transformers.modeling_outputs import SequenceClassifierOutput

NUM_LABELS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--eval_splits", type=str, default="test")
    parser.add_argument("--raw_ts_shape", type=str, default="8,13")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    # ---- ablation-specific flags ----
    parser.add_argument("--ablation_mask", type=str, default=None,
                        help="Comma-separated column indices to KEEP in raw_ts.")
    parser.add_argument("--no_lora", action="store_true", default=False,
                        help="Model was trained without LoRA — skip adapter loading.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_shape(text: str) -> Tuple[int, int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(text)
    return vals[0], vals[1]


def normalize_raw_ts(raw_ts: Any, expect_shape: Tuple[int, int]) -> np.ndarray:
    h, w = expect_shape
    arr = np.asarray(raw_ts, dtype=np.float32)
    if arr.ndim == 1 and arr.size == h * w:
        arr = arr.reshape(h, w)
    if arr.shape != (h, w):
        raise ValueError(f"raw_ts shape 异常，期望 {(h, w)}，实际 {arr.shape}")
    return arr.astype(np.float32)


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _sanitize_parallel_plan(obj: Any) -> None:
    for attr in ["tp_plan", "pp_plan", "base_model_tp_plan", "base_model_pp_plan", "_tp_plan", "_pp_plan"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is None:
                setattr(obj, attr, {})
            elif isinstance(val, dict):
                setattr(obj, attr, {k: v for k, v in val.items() if v is not None})


def _normalize_parallel_styles() -> None:
    fallback = {None, "colwise", "rowwise", "colwise_rep", "rowwise_rep", "gather", "replicate", "local", "sequence_parallel"}
    styles = getattr(hf_modeling_utils, "ALL_PARALLEL_STYLES", None)
    if styles is None:
        hf_modeling_utils.ALL_PARALLEL_STYLES = fallback
        return
    if isinstance(styles, tuple):
        hf_modeling_utils.ALL_PARALLEL_STYLES = tuple(list(styles) + [x for x in fallback if x not in styles])
        return
    if isinstance(styles, list):
        for x in fallback:
            if x not in styles:
                styles.append(x)
        return
    if isinstance(styles, set):
        styles.update(fallback)
        return
    try:
        hf_modeling_utils.ALL_PARALLEL_STYLES = set(styles) | fallback
    except Exception:
        hf_modeling_utils.ALL_PARALLEL_STYLES = fallback


class RevINRawTs(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x = (x - mean) / std
        return x * self.affine_weight + self.affine_bias


class MultiScaleRawTsEncoder(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)

    def forward(self, raw_ts: torch.Tensor) -> torch.Tensor:
        coarse = raw_ts.unfold(dimension=1, size=2, step=2).mean(dim=-1)
        if coarse.shape[1] * 2 != raw_ts.shape[1]:
            return raw_ts
        coarse_up = coarse.repeat_interleave(2, dim=1)
        residual = raw_ts - coarse_up
        return torch.cat([raw_ts, coarse_up, residual], dim=1)


class DagResidualBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = 256,
        alpha_init: float = -3.0,
        dropout: float = 0.0,
        norm_mode: str = "indegree",
        num_layers: int = 1,
        use_attention: bool = False,
        use_edge_gate: bool = False,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)
        self.dropout_p = float(dropout)
        self.norm_mode = str(norm_mode)
        self.num_layers = int(num_layers)
        self.use_attention = bool(use_attention)
        self.use_edge_gate = bool(use_edge_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.ModuleDict({
                "msg_proj": nn.Linear(self.model_dim, self.model_dim, bias=False),
                "pre_norm": nn.LayerNorm(self.model_dim),
                "out_proj": nn.Sequential(
                    nn.Linear(self.model_dim, self.hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_dim, self.model_dim, bias=False),
                ),
            })
            self.layers.append(layer)

        if self.use_attention:
            self.attn_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
            self.attn_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self._ensure_edge_gate_module()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    @staticmethod
    def remap_old_state_dict(sd: dict) -> dict:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith(("msg_proj.", "pre_norm.", "out_proj.")):
                new_sd[f"layers.0.{k}"] = v
            else:
                new_sd[k] = v
        return new_sd

    def _ensure_edge_gate_module(self) -> None:
        if not self.use_edge_gate:
            return
        if hasattr(self, "edge_gate"):
            return
        self.edge_gate = nn.Sequential(
            nn.Linear(self.model_dim * 2 + 2, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.gate_hidden_dim, 1),
        )

    def _compute_edge_gate(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        dag_level: Optional[torch.Tensor],
        is_root: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n = h.shape[0]
        parent_feat = h.unsqueeze(1).expand(n, n, -1)
        child_feat = h.unsqueeze(0).expand(n, n, -1)
        if dag_level is None:
            level_delta = torch.zeros((n, n, 1), device=h.device, dtype=h.dtype)
        else:
            level = dag_level.to(device=h.device, dtype=h.dtype).view(n, 1)
            level_delta = (level - level.T).unsqueeze(-1)
        if is_root is None:
            child_root = torch.zeros((n, n, 1), device=h.device, dtype=h.dtype)
        else:
            child_root = is_root.to(device=h.device, dtype=h.dtype).view(1, n, 1).expand(n, n, 1)
        gate_in = torch.cat([parent_feat, child_feat, level_delta, child_root], dim=-1)
        gate = torch.sigmoid(self.edge_gate(gate_in)).squeeze(-1)
        return gate * (A > 0).to(dtype=gate.dtype)

    def _compute_adj_norm(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        dag_level: Optional[torch.Tensor],
        is_root: Optional[torch.Tensor],
    ) -> torch.Tensor:
        A_eff = A
        if self.use_edge_gate:
            A_eff = A_eff * self._compute_edge_gate(h, A, dag_level=dag_level, is_root=is_root)
        if self.use_attention:
            Q = self.attn_q(h)
            K = self.attn_k(h)
            scale = float(self.model_dim) ** 0.5
            scores = (Q @ K.T) / scale
            scores = scores.masked_fill(A_eff <= 0, float("-inf"))
            A_norm = torch.softmax(scores, dim=-1)
            A_norm = A_norm.nan_to_num(0.0)
            if self.use_edge_gate:
                A_norm = A_norm * (A_eff > 0).to(dtype=A_norm.dtype)
                A_norm = A_norm / A_norm.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return A_norm
        indeg = A_eff.sum(dim=1, keepdim=True)
        if self.norm_mode == "indegree":
            return A_eff / indeg.clamp(min=1.0)
        elif self.norm_mode == "none":
            return A_eff
        raise ValueError(f"不支持的 dag_residual_norm: {self.norm_mode}")

    def forward(self, pooled: torch.Tensor, dag_adj: torch.Tensor, dag_level=None, is_root=None) -> torch.Tensor:
        if dag_adj is None:
            return pooled
        base_dtype = pooled.dtype
        A = dag_adj.to(device=pooled.device, dtype=torch.float32).clamp(min=0.0)
        h = pooled.float()
        for layer in self.layers:
            A_norm = self._compute_adj_norm(h, A, dag_level=dag_level, is_root=is_root)
            has_parent = (A_norm.sum(dim=1, keepdim=True) > 0).float()
            m = A_norm @ layer["msg_proj"](h)
            u = layer["pre_norm"](m)
            u = self.act(u)
            u = layer["out_proj"](u)
            u = self.dropout(u)
            u = u * has_parent
            alpha = torch.sigmoid(self.alpha_logit)
            h = h + alpha * u

        return h.to(dtype=base_dtype)


class LocalInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, raw_ts: torch.Tensor) -> torch.Tensor:
        return self.net(raw_ts)


class SpatialTransportBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                "msg_proj": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "pre_norm": nn.LayerNorm(self.hidden_dim),
                "out_proj": nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                ),
            }))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

    def forward(self, states: torch.Tensor, dag_adj: torch.Tensor):
        if dag_adj is None:
            zero_delta = torch.zeros_like(states)
            return states, zero_delta
        base_dtype = states.dtype
        A = dag_adj.to(device=states.device, dtype=torch.float32).clamp(min=0.0)
        indeg = A.sum(dim=1, keepdim=True)
        A_norm = A / indeg.clamp(min=1.0)
        has_parent = (indeg.squeeze(1) > 0).float().view(-1, 1, 1)

        h = states.float()
        delta_total = torch.zeros_like(h)
        alpha = torch.sigmoid(self.alpha_logit)
        for layer in self.layers:
            msg = torch.einsum("nm,mtd->ntd", A_norm, layer["msg_proj"](h))
            delta = layer["pre_norm"](msg)
            delta = self.act(delta)
            delta = layer["out_proj"](delta)
            delta = self.dropout(delta)
            delta = delta * has_parent
            h = h + alpha * delta
            delta_total = delta_total + alpha * delta
        return h.to(dtype=base_dtype), delta_total.to(dtype=base_dtype)


class TemporalMixBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                "time_mix_k": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "time_mix_v": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "time_mix_r": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "key": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "value": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "receptance": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "output": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "ln": nn.LayerNorm(self.hidden_dim),
            }))
        self.dropout = nn.Dropout(self.dropout_p)

    @staticmethod
    def _shift(states: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(states[:, :1, :])
        return torch.cat([zero, states[:, :-1, :]], dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        h = states
        for layer in self.layers:
            prev = self._shift(h)
            mix_k = h * layer["time_mix_k"] + prev * (1.0 - layer["time_mix_k"])
            mix_v = h * layer["time_mix_v"] + prev * (1.0 - layer["time_mix_v"])
            mix_r = h * layer["time_mix_r"] + prev * (1.0 - layer["time_mix_r"])
            k = layer["key"](mix_k)
            v = layer["value"](mix_v)
            r = torch.sigmoid(layer["receptance"](mix_r))
            mixed = layer["output"](r * torch.tanh(k) * v)
            mixed = self.dropout(mixed)
            h = layer["ln"](h + mixed)
        return h


class RawTsPrefixSequenceClassifier(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        raw_ts_shape: Tuple[int, int] = (8, 13),
        prefix_tokens: int = 4,
        raw_ts_hidden_dim: int = 512,
        num_labels: int = NUM_LABELS,
        classifier_dropout: float = 0.1,
        pooling_mode: str = "mean",
        root_aware_loss: bool = False,
        root_weight: float = 1.5,
        root_label_extra=None,
        max_root_sample_weight: float = 4.0,
        dag_residual_enable: bool = False,
        dag_residual_hidden_dim: int = 256,
        dag_residual_alpha_init: float = -3.0,
        dag_residual_dropout: float = 0.0,
        dag_residual_norm: str = "indegree",
        dag_residual_layers: int = 1,
        dag_residual_attention: bool = False,
        dag_residual_edge_gate: bool = False,
        dag_residual_gate_hidden_dim: int = 128,
        raw_ts_use_revin: bool = False,
        raw_ts_use_multiscale: bool = False,
        raw_ts_preprefix_graph_enable: bool = False,
        raw_ts_preprefix_graph_hidden_dim: int = 256,
        raw_ts_preprefix_graph_alpha_init: float = -3.0,
        raw_ts_preprefix_graph_dropout: float = 0.0,
        raw_ts_preprefix_graph_norm: str = "indegree",
        raw_ts_preprefix_graph_layers: int = 1,
        raw_ts_preprefix_graph_attention: bool = False,
        pcd_graph_enable: bool = False,
        pcd_hidden_dim: int = 128,
        pcd_transport_layers: int = 2,
        pcd_transport_dropout: float = 0.0,
        pcd_temporal_num_layers: int = 1,
    ):
        super().__init__()
        self.lm = lm
        self.raw_ts_shape = tuple(raw_ts_shape)
        self.prefix_tokens = int(prefix_tokens)
        self.num_labels = int(num_labels)
        self.pooling_mode = str(pooling_mode)
        self.root_aware_loss = bool(root_aware_loss)
        self.root_weight = float(root_weight)
        self.root_label_extra = dict(root_label_extra or {})
        self.max_root_sample_weight = float(max_root_sample_weight)
        self.dag_residual_enable = bool(dag_residual_enable)
        self.dag_residual_hidden_dim = int(dag_residual_hidden_dim)
        self.dag_residual_alpha_init = float(dag_residual_alpha_init)
        self.dag_residual_dropout = float(dag_residual_dropout)
        self.dag_residual_norm = str(dag_residual_norm)
        self.dag_residual_attention = bool(dag_residual_attention)
        self.dag_residual_edge_gate = bool(dag_residual_edge_gate)
        self.dag_residual_gate_hidden_dim = int(dag_residual_gate_hidden_dim)
        self.raw_ts_use_revin = bool(raw_ts_use_revin)
        self.raw_ts_use_multiscale = bool(raw_ts_use_multiscale)
        self.raw_ts_preprefix_graph_enable = bool(raw_ts_preprefix_graph_enable)
        self.raw_ts_preprefix_graph_hidden_dim = int(raw_ts_preprefix_graph_hidden_dim)
        self.raw_ts_preprefix_graph_alpha_init = float(raw_ts_preprefix_graph_alpha_init)
        self.raw_ts_preprefix_graph_dropout = float(raw_ts_preprefix_graph_dropout)
        self.raw_ts_preprefix_graph_norm = str(raw_ts_preprefix_graph_norm)
        self.raw_ts_preprefix_graph_layers = int(raw_ts_preprefix_graph_layers)
        self.raw_ts_preprefix_graph_attention = bool(raw_ts_preprefix_graph_attention)
        self.pcd_graph_enable = bool(pcd_graph_enable)
        self.pcd_hidden_dim = int(pcd_hidden_dim)
        self.pcd_transport_layers = int(pcd_transport_layers)
        self.pcd_transport_dropout = float(pcd_transport_dropout)
        self.pcd_temporal_num_layers = int(pcd_temporal_num_layers)

        embed_layer = self.lm.get_input_embeddings()
        hidden_size = int(embed_layer.embedding_dim)
        effective_shape = (self.raw_ts_shape[0] * (3 if self.raw_ts_use_multiscale else 1), self.raw_ts_shape[1])
        raw_dim = int(np.prod(effective_shape))
        mid_dim = int(max(raw_ts_hidden_dim, hidden_size))
        effective_seq_len = effective_shape[0]
        effective_feat_dim = effective_shape[1]
        pcd_output_dim = self.pcd_hidden_dim * effective_seq_len
        self.raw_ts_revin = RevINRawTs(self.raw_ts_shape[1]) if self.raw_ts_use_revin else None
        self.raw_ts_multiscale = MultiScaleRawTsEncoder(*self.raw_ts_shape) if self.raw_ts_use_multiscale else None
        self.local_interaction_encoder = (
            LocalInteractionEncoder(input_dim=effective_feat_dim, hidden_dim=self.pcd_hidden_dim)
            if self.pcd_graph_enable
            else None
        )
        self.spatial_transport_block = (
            SpatialTransportBlock(
                hidden_dim=self.pcd_hidden_dim,
                num_layers=self.pcd_transport_layers,
                dropout=self.pcd_transport_dropout,
            )
            if self.pcd_graph_enable
            else None
        )
        self.temporal_mix_block = (
            TemporalMixBlock(
                hidden_dim=self.pcd_hidden_dim,
                num_layers=self.pcd_temporal_num_layers,
                dropout=self.pcd_transport_dropout,
            )
            if self.pcd_graph_enable
            else None
        )
        self.raw_ts_projector = nn.Sequential(
            nn.LayerNorm(pcd_output_dim if self.pcd_graph_enable else raw_dim),
            nn.Linear(pcd_output_dim if self.pcd_graph_enable else raw_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, self.prefix_tokens * hidden_size),
        )
        self.raw_ts_preprefix_graph = (
            DagResidualBlock(
                model_dim=effective_shape[0] * effective_shape[1],
                hidden_dim=self.raw_ts_preprefix_graph_hidden_dim,
                alpha_init=self.raw_ts_preprefix_graph_alpha_init,
                dropout=self.raw_ts_preprefix_graph_dropout,
                norm_mode=self.raw_ts_preprefix_graph_norm,
                num_layers=self.raw_ts_preprefix_graph_layers,
                use_attention=self.raw_ts_preprefix_graph_attention,
            )
            if self.raw_ts_preprefix_graph_enable
            else None
        )
        self.dropout = nn.Dropout(float(classifier_dropout))
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.dag_residual_block = (
            DagResidualBlock(
                model_dim=hidden_size,
                hidden_dim=self.dag_residual_hidden_dim,
                alpha_init=self.dag_residual_alpha_init,
                dropout=self.dag_residual_dropout,
                norm_mode=self.dag_residual_norm,
                num_layers=dag_residual_layers,
                use_attention=dag_residual_attention,
                use_edge_gate=self.dag_residual_edge_gate,
                gate_hidden_dim=self.dag_residual_gate_hidden_dim,
            )
            if self.dag_residual_enable
            else None
        )

    @property
    def config(self):
        return self.lm.config

    def _build_prefixed_inputs(self, input_ids, attention_mask, raw_ts, dag_adj=None):
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        bs = input_embeds.size(0)
        hidden = input_embeds.size(-1)
        raw_ts_in = raw_ts.to(device=input_embeds.device, dtype=torch.float32)
        if self.raw_ts_revin is not None:
            raw_ts_in = self.raw_ts_revin(raw_ts_in)
        if self.raw_ts_multiscale is not None:
            raw_ts_in = self.raw_ts_multiscale(raw_ts_in)
        if self.pcd_graph_enable and self.local_interaction_encoder is not None:
            transport_states = self.local_interaction_encoder(raw_ts_in)
            if self.spatial_transport_block is not None:
                transport_states, _ = self.spatial_transport_block(transport_states, dag_adj)
            if self.temporal_mix_block is not None:
                transport_states = self.temporal_mix_block(transport_states)
            raw_flat = transport_states.reshape(bs, -1)
        else:
            raw_flat = raw_ts_in.reshape(bs, -1)
            if self.raw_ts_preprefix_graph is not None and dag_adj is not None:
                raw_flat = self.raw_ts_preprefix_graph(raw_flat, dag_adj)
        prefix = self.raw_ts_projector(raw_flat).view(bs, self.prefix_tokens, hidden)
        prefix = prefix.to(dtype=input_embeds.dtype)
        prefix_mask = torch.ones((bs, self.prefix_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_embeds = torch.cat([prefix, input_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return full_embeds, full_mask

    def _pool_hidden(self, hidden: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "last":
            last_index = full_mask.long().sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        mask = full_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids=None, attention_mask=None, raw_ts=None, dag_adj=None, dag_level=None, labels=None, is_root=None, **kwargs):
        if raw_ts is None:
            raise ValueError("forward 缺少 raw_ts")
        if input_ids is None or attention_mask is None:
            raise ValueError("forward 缺少 input_ids/attention_mask")
        if self.dag_residual_enable and dag_adj is None:
            raise ValueError("dag_residual_enable=True 时 forward 缺少 dag_adj")

        inputs_embeds, full_mask = self._build_prefixed_inputs(input_ids, attention_mask, raw_ts, dag_adj=dag_adj)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("labels", None)
        kwargs.pop("is_root", None)
        kwargs.pop("dag_adj", None)
        kwargs.pop("dag_level", None)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        pooled = self._pool_hidden(hidden, full_mask)
        if self.dag_residual_enable and self.dag_residual_block is not None:
            pooled = self.dag_residual_block(pooled, dag_adj, dag_level=dag_level, is_root=is_root)
        dropped = self.dropout(pooled)
        self.classifier = self.classifier.to(device=dropped.device, dtype=dropped.dtype)
        logits = self.classifier(dropped)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_tokenizer_and_model(args: argparse.Namespace):
    _normalize_parallel_styles()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    _sanitize_parallel_plan(cfg)
    base_lm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    _sanitize_parallel_plan(base_lm.config)
    if args.no_lora:
        print("[ABLATION] no_lora=True: skipping adapter loading")
        base_lm.eval()
    else:
        base_lm = PeftModel.from_pretrained(base_lm, args.adapter_dir, is_trainable=False)
        base_lm.eval()

    raw_cfg = _load_json(
        Path(args.adapter_dir) / "raw_ts_config.json",
        {"raw_ts_shape": list(parse_shape(args.raw_ts_shape)), "prefix_tokens": 4, "raw_ts_hidden_dim": 512},
    )
    cls_cfg = _load_json(
        Path(args.adapter_dir) / "cls_config.json",
        {"num_labels": NUM_LABELS, "classifier_dropout": 0.1, "pooling_mode": "mean"},
    )
    dag_residual_cfg = _load_json(
        Path(args.adapter_dir) / "dag_residual_config.json",
        {
            "dag_residual_enable": True,
            "dag_residual_hidden_dim": 256,
            "dag_residual_alpha_init": -3.0,
            "dag_residual_dropout": 0.0,
            "dag_residual_norm": "indegree",
        },
    )
    root_region_cfg = _load_json(
        Path(args.adapter_dir) / "root_region_config.json",
        {
            "root_region_env_fusion": False,
            "root_region_env_weight": 0.5,
            "root_region_exclude_self": False,
        },
    )

    raw_ts_shape = tuple(raw_cfg.get("raw_ts_shape", list(parse_shape(args.raw_ts_shape))))
    prefix_tokens = int(raw_cfg.get("prefix_tokens", 4))
    raw_ts_hidden_dim = int(raw_cfg.get("raw_ts_hidden_dim", 512))
    num_labels = int(cls_cfg.get("num_labels", NUM_LABELS))
    classifier_dropout = float(cls_cfg.get("classifier_dropout", 0.1))
    pooling_mode = str(cls_cfg.get("pooling_mode", "mean"))
    root_aware_loss = bool(cls_cfg.get("root_aware_loss", False))
    root_weight = float(cls_cfg.get("root_weight", 1.5))
    root_label_extra_raw = cls_cfg.get("root_label_extra", None)
    root_label_extra = {int(k): float(v) for k, v in root_label_extra_raw.items()} if root_label_extra_raw else None
    max_root_sample_weight = float(cls_cfg.get("max_root_sample_weight", 4.0))
    dag_residual_enable = bool(dag_residual_cfg.get("dag_residual_enable", True))
    dag_residual_hidden_dim = int(dag_residual_cfg.get("dag_residual_hidden_dim", 256))
    dag_residual_alpha_init = float(dag_residual_cfg.get("dag_residual_alpha_init", -3.0))
    dag_residual_dropout = float(dag_residual_cfg.get("dag_residual_dropout", 0.0))
    dag_residual_norm = str(dag_residual_cfg.get("dag_residual_norm", "indegree"))
    dag_residual_layers = int(dag_residual_cfg.get("dag_residual_layers", 1))
    dag_residual_attention = bool(dag_residual_cfg.get("dag_residual_attention", False))
    dag_residual_edge_gate = bool(dag_residual_cfg.get("dag_residual_edge_gate", False))
    dag_residual_gate_hidden_dim = int(dag_residual_cfg.get("dag_residual_gate_hidden_dim", 128))
    raw_ts_use_revin = bool(raw_cfg.get("raw_ts_use_revin", False))
    raw_ts_use_multiscale = bool(raw_cfg.get("raw_ts_use_multiscale", False))
    raw_ts_preprefix_graph_enable = bool(raw_cfg.get("raw_ts_preprefix_graph_enable", False))
    raw_ts_preprefix_graph_hidden_dim = int(raw_cfg.get("raw_ts_preprefix_graph_hidden_dim", 256))
    raw_ts_preprefix_graph_alpha_init = float(raw_cfg.get("raw_ts_preprefix_graph_alpha_init", -3.0))
    raw_ts_preprefix_graph_dropout = float(raw_cfg.get("raw_ts_preprefix_graph_dropout", 0.0))
    raw_ts_preprefix_graph_norm = str(raw_cfg.get("raw_ts_preprefix_graph_norm", "indegree"))
    raw_ts_preprefix_graph_layers = int(raw_cfg.get("raw_ts_preprefix_graph_layers", 1))
    raw_ts_preprefix_graph_attention = bool(raw_cfg.get("raw_ts_preprefix_graph_attention", False))
    pcd_graph_enable = bool(raw_cfg.get("pcd_graph_enable", False))
    pcd_hidden_dim = int(raw_cfg.get("pcd_hidden_dim", 128))
    pcd_transport_layers = int(raw_cfg.get("pcd_transport_layers", 2))
    pcd_transport_dropout = float(raw_cfg.get("pcd_transport_dropout", 0.0))
    pcd_temporal_num_layers = int(raw_cfg.get("pcd_temporal_num_layers", 1))

    model = RawTsPrefixSequenceClassifier(
        lm=base_lm,
        raw_ts_shape=raw_ts_shape,
        prefix_tokens=prefix_tokens,
        raw_ts_hidden_dim=raw_ts_hidden_dim,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        pooling_mode=pooling_mode,
        root_aware_loss=root_aware_loss,
        root_weight=root_weight,
        root_label_extra=root_label_extra,
        max_root_sample_weight=max_root_sample_weight,
        dag_residual_enable=dag_residual_enable,
        dag_residual_hidden_dim=dag_residual_hidden_dim,
        dag_residual_alpha_init=dag_residual_alpha_init,
        dag_residual_dropout=dag_residual_dropout,
        dag_residual_norm=dag_residual_norm,
        dag_residual_layers=dag_residual_layers,
        dag_residual_attention=dag_residual_attention,
        dag_residual_edge_gate=dag_residual_edge_gate,
        dag_residual_gate_hidden_dim=dag_residual_gate_hidden_dim,
        raw_ts_use_revin=raw_ts_use_revin,
        raw_ts_use_multiscale=raw_ts_use_multiscale,
        raw_ts_preprefix_graph_enable=raw_ts_preprefix_graph_enable,
        raw_ts_preprefix_graph_hidden_dim=raw_ts_preprefix_graph_hidden_dim,
        raw_ts_preprefix_graph_alpha_init=raw_ts_preprefix_graph_alpha_init,
        raw_ts_preprefix_graph_dropout=raw_ts_preprefix_graph_dropout,
        raw_ts_preprefix_graph_norm=raw_ts_preprefix_graph_norm,
        raw_ts_preprefix_graph_layers=raw_ts_preprefix_graph_layers,
        raw_ts_preprefix_graph_attention=raw_ts_preprefix_graph_attention,
        pcd_graph_enable=pcd_graph_enable,
        pcd_hidden_dim=pcd_hidden_dim,
        pcd_transport_layers=pcd_transport_layers,
        pcd_transport_dropout=pcd_transport_dropout,
        pcd_temporal_num_layers=pcd_temporal_num_layers,
    )

    proj_path = Path(args.adapter_dir) / "raw_ts_projector.pt"
    if not proj_path.exists():
        raise FileNotFoundError(proj_path)
    model.raw_ts_projector.load_state_dict(torch.load(proj_path, map_location="cpu"), strict=True)
    revin_path = Path(args.adapter_dir) / "raw_ts_revin.pt"
    if raw_ts_use_revin and model.raw_ts_revin is not None and revin_path.exists():
        model.raw_ts_revin.load_state_dict(torch.load(revin_path, map_location="cpu"), strict=True)
    local_interaction_path = Path(args.adapter_dir) / "local_interaction_encoder.pt"
    if pcd_graph_enable and model.local_interaction_encoder is not None and local_interaction_path.exists():
        model.local_interaction_encoder.load_state_dict(torch.load(local_interaction_path, map_location="cpu"), strict=True)
    spatial_transport_path = Path(args.adapter_dir) / "spatial_transport_block.pt"
    if pcd_graph_enable and model.spatial_transport_block is not None and spatial_transport_path.exists():
        model.spatial_transport_block.load_state_dict(torch.load(spatial_transport_path, map_location="cpu"), strict=True)
    temporal_mix_path = Path(args.adapter_dir) / "temporal_mix_block.pt"
    if pcd_graph_enable and model.temporal_mix_block is not None and temporal_mix_path.exists():
        model.temporal_mix_block.load_state_dict(torch.load(temporal_mix_path, map_location="cpu"), strict=True)
    preprefix_path = Path(args.adapter_dir) / "raw_ts_preprefix_graph.pt"
    if raw_ts_preprefix_graph_enable and model.raw_ts_preprefix_graph is not None and preprefix_path.exists():
        model.raw_ts_preprefix_graph.load_state_dict(torch.load(preprefix_path, map_location="cpu"), strict=True)

    cls_path = Path(args.adapter_dir) / "classifier_head.pt"
    if not cls_path.exists():
        raise FileNotFoundError(cls_path)
    model.classifier.load_state_dict(torch.load(cls_path, map_location="cpu"), strict=True)

    if dag_residual_enable and model.dag_residual_block is not None:
        block_path = Path(args.adapter_dir) / "dag_residual_block.pt"
        if not block_path.exists():
            raise FileNotFoundError(block_path)
        block_sd = torch.load(block_path, map_location="cpu")
        if any(k.startswith("msg_proj.") for k in block_sd):
            block_sd = DagResidualBlock.remap_old_state_dict(block_sd)
        model.dag_residual_block.load_state_dict(block_sd, strict=True)

    model.eval()
    if (not args.load_in_4bit) and torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model, raw_ts_shape, root_region_cfg, dag_residual_cfg


def render_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def maybe_fuse_root_raw_ts(
    raw_ts: np.ndarray,
    is_root: np.ndarray,
    enable: bool,
    weight: float,
    exclude_self: bool,
) -> np.ndarray:
    if not enable:
        return raw_ts
    total = raw_ts.sum(axis=0, keepdims=False).astype(np.float32)
    count = int(raw_ts.shape[0])
    out = raw_ts.astype(np.float32).copy()
    for i in range(count):
        if int(is_root[i]) != 1:
            continue
        if exclude_self and count > 1:
            region_mean = (total - raw_ts[i]) / float(count - 1)
        else:
            region_mean = total / float(max(count, 1))
        out[i] = (raw_ts[i] + float(weight) * region_mean).astype(np.float32)
    return out


def build_group_inputs(tokenizer, prompts: List[str], max_input_length: int, use_chat_template: bool):
    token_ids: List[List[int]] = []
    prompt_lens: List[int] = []
    for prompt in prompts:
        text = render_prompt(tokenizer, prompt, use_chat_template)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_input_length:
            ids = ids[-max_input_length:]
        token_ids.append(ids)
        prompt_lens.append(len(ids))

    pad_id = tokenizer.pad_token_id
    max_len = max(len(x) for x in token_ids)
    input_ids = []
    attention_mask = []
    for ids in token_ids:
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        prompt_lens,
    )


@torch.no_grad()
def predict_group(
    model,
    tokenizer,
    group_row: Dict[str, Any],
    raw_ts_shape: Tuple[int, int],
    max_input_length: int,
    use_chat_template: bool,
    root_region_cfg: Dict[str, Any],
    ablation_keep_cols=None,
) -> List[Dict[str, Any]]:
    city_names = _load_json(Path(group_row["city_names_path"]), [])
    prompt_entries = _load_json(Path(group_row["prompts_path"]), [])
    if prompt_entries and isinstance(prompt_entries[0], dict):
        prompts = [str(x.get("prompt", "")) for x in prompt_entries]
    else:
        prompts = [str(x) for x in prompt_entries]

    raw_ts = np.load(group_row["raw_ts_path"]).astype(np.float32)
    labels = np.load(group_row["labels_path"]).astype(np.int64)
    dag_adj = np.load(group_row["dag_adj_path"]).astype(np.float32)
    dag_level = np.load(group_row["dag_level_path"]).astype(np.int64)
    is_root = np.load(group_row["is_root_path"]).astype(np.int64)
    parent_count = np.load(group_row["parent_count_path"]).astype(np.int64)

    if raw_ts.ndim != 3 or raw_ts.shape[1:] != raw_ts_shape:
        raise ValueError(f"raw_ts shape 异常: {raw_ts.shape}, expect=(*,{raw_ts_shape[0]},{raw_ts_shape[1]})")
    if dag_adj.ndim != 2 or dag_adj.shape[0] != dag_adj.shape[1] or dag_adj.shape[0] != raw_ts.shape[0]:
        raise ValueError(f"dag_adj shape 异常: {dag_adj.shape}, raw_ts={raw_ts.shape}")

    raw_ts = maybe_fuse_root_raw_ts(
        raw_ts=raw_ts,
        is_root=is_root,
        enable=bool(root_region_cfg.get("root_region_env_fusion", False)),
        weight=float(root_region_cfg.get("root_region_env_weight", 0.5)),
        exclude_self=bool(root_region_cfg.get("root_region_exclude_self", False)),
    )

    # Ablation: zero out columns not in keep list
    if ablation_keep_cols is not None:
        col_mask = np.zeros(raw_ts.shape[-1], dtype=np.float32)
        for c in ablation_keep_cols:
            if 0 <= c < raw_ts.shape[-1]:
                col_mask[c] = 1.0
        raw_ts = raw_ts * col_mask[np.newaxis, np.newaxis, :]

    input_ids, attention_mask, prompt_lens = build_group_inputs(
        tokenizer=tokenizer,
        prompts=prompts,
        max_input_length=max_input_length,
        use_chat_template=use_chat_template,
    )

    device = next(model.parameters()).device
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        raw_ts=torch.tensor(raw_ts, dtype=torch.float32, device=device),
        dag_adj=torch.tensor(dag_adj, dtype=torch.float32, device=device),
        is_root=torch.tensor(is_root.astype(np.float32), dtype=torch.float32, device=device),
    )
    logits = outputs.logits.detach().float().cpu().numpy()
    preds = np.argmax(logits, axis=-1).astype(np.int64)

    rows: List[Dict[str, Any]] = []
    for i in range(raw_ts.shape[0]):
        row = {
            "group_id": str(group_row["group_id"]),
            "target_date": str(group_row.get("target_date", group_row["group_id"])),
            "city_name": str(city_names[i]),
            "dag_level": int(dag_level[i]),
            "parent_count": int(parent_count[i]),
            "label": int(labels[i]),
            "pred": int(preds[i]),
            "correct": int(int(preds[i]) == int(labels[i])),
            "prompt_token_len": int(prompt_lens[i]),
            "is_root": int(is_root[i]),
        }
        for k in range(NUM_LABELS):
            row[f"logit_{k}"] = float(logits[i, k])
        rows.append(row)
    return rows


def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[str, Any]:
    n = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / n if n > 0 else 0.0
    per_label = []
    tp_global = 0
    fp_global = 0
    fn_global = 0
    for lab in labels:
        tp = sum(int((yt == lab) and (yp == lab)) for yt, yp in zip(y_true, y_pred))
        fp = sum(int((yt != lab) and (yp == lab)) for yt, yp in zip(y_true, y_pred))
        fn = sum(int((yt == lab) and (yp != lab)) for yt, yp in zip(y_true, y_pred))
        support = sum(int(yt == lab) for yt in y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_label.append({"label": int(lab), "support": int(support), "precision": precision, "recall": recall, "f1": f1})
        tp_global += tp
        fp_global += fp
        fn_global += fn
    micro_precision = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0.0
    micro_recall = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
    macro_f1 = float(np.mean([x["f1"] for x in per_label])) if per_label else 0.0
    return {"num_samples": n, "accuracy": accuracy, "micro_f1": micro_f1, "macro_f1": macro_f1, "per_label": per_label}


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted(set(int(x["label"]) for x in rows) | set(int(x["pred"]) for x in rows))
    y_true = [int(x["label"]) for x in rows]
    y_pred = [int(x["pred"]) for x in rows]
    summary = compute_metrics(y_true, y_pred, labels)
    summary["true_hist"] = {str(k): int(v) for k, v in sorted(Counter(y_true).items())}
    summary["pred_hist"] = {str(k): int(v) for k, v in sorted(Counter(y_pred).items())}
    return summary


def evaluate_split(
    split_name: str,
    data_dir: Path,
    model,
    tokenizer,
    raw_ts_shape: Tuple[int, int],
    max_input_length: int,
    use_chat_template: bool,
    root_region_cfg: Dict[str, Any],
    out_dir: Path,
    ablation_keep_cols=None,
) -> Dict[str, Any]:
    ensure_dir(str(out_dir))
    idx_path = data_dir / "assets" / split_name / "group_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    group_rows = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                group_rows.append(json.loads(line))

    pred_rows: List[Dict[str, Any]] = []
    for group_row in tqdm(group_rows, desc=f"eval[{split_name}]"):
        pred_rows.extend(
            predict_group(
                model=model,
                tokenizer=tokenizer,
                group_row=group_row,
                raw_ts_shape=raw_ts_shape,
                max_input_length=max_input_length,
                use_chat_template=use_chat_template,
                root_region_cfg=root_region_cfg,
                ablation_keep_cols=ablation_keep_cols,
            )
        )

    summary = summarize_rows(pred_rows)
    slice_summaries: Dict[str, Any] = {}
    root_rows = [x for x in pred_rows if int(x["is_root"]) == 1]
    nonroot_rows = [x for x in pred_rows if int(x["is_root"]) == 0]
    if root_rows:
        slice_summaries["root"] = summarize_rows(root_rows)
    if nonroot_rows:
        slice_summaries["nonroot"] = summarize_rows(nonroot_rows)

    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_parent_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        by_level[str(int(row["dag_level"]))].append(row)
        p = int(row["parent_count"])
        if p == 0:
            key = "p0"
        elif p <= 2:
            key = "p1_2"
        elif p <= 4:
            key = "p3_4"
        else:
            key = "p5_plus"
        by_parent_bin[key].append(row)
    slice_summaries["by_dag_level"] = {k: summarize_rows(v) for k, v in sorted(by_level.items()) if v}
    slice_summaries["by_parent_bin"] = {k: summarize_rows(v) for k, v in sorted(by_parent_bin.items()) if v}

    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id", "target_date", "city_name", "dag_level", "parent_count",
            "label", "pred", "correct", "prompt_token_len", "is_root",
        ] + [f"logit_{i}" for i in range(NUM_LABELS)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pred_rows:
            writer.writerow(row)

    payload = {"split": split_name, **summary, "slices": slice_summaries}
    save_json(str(out_dir / "summary.json"), payload)

    per_label_sorted = sorted(payload.get("per_label", []), key=lambda x: int(x["label"]))
    precision_arr = np.array([float(x["precision"]) for x in per_label_sorted], dtype=np.float64)
    recall_arr = np.array([float(x["recall"]) for x in per_label_sorted], dtype=np.float64)
    f1_arr = np.array([float(x["f1"]) for x in per_label_sorted], dtype=np.float64)
    np.save(str(out_dir / "per_label_precision.npy"), precision_arr)
    np.save(str(out_dir / "per_label_recall.npy"), recall_arr)
    np.save(str(out_dir / "per_label_f1.npy"), f1_arr)

    return payload


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    tokenizer, model, raw_ts_shape, root_region_cfg, dag_residual_cfg = load_tokenizer_and_model(args)

    # Parse ablation_mask
    ablation_keep_cols = None
    if args.ablation_mask is not None:
        ablation_keep_cols = [int(x.strip()) for x in args.ablation_mask.split(",") if x.strip()]
        print(f"[ABLATION] ablation_mask: keep columns {ablation_keep_cols}")

    split_names = [x.strip() for x in args.eval_splits.split(",") if x.strip()]
    all_summaries = []
    for split_name in split_names:
        out_dir = Path(args.out_dir) / split_name
        summary = evaluate_split(
            split_name=split_name,
            data_dir=Path(args.data_dir),
            model=model,
            tokenizer=tokenizer,
            raw_ts_shape=raw_ts_shape,
            max_input_length=args.max_input_length,
            use_chat_template=args.use_chat_template,
            root_region_cfg=root_region_cfg,
            out_dir=out_dir,
            ablation_keep_cols=ablation_keep_cols,
        )
        all_summaries.append(summary)
        print("=" * 80)
        print(f"[{split_name}] num_samples = {summary['num_samples']}")
        print(f"[{split_name}] accuracy   = {summary['accuracy']:.6f}")
        print(f"[{split_name}] micro_f1   = {summary['micro_f1']:.6f}")
        print(f"[{split_name}] macro_f1   = {summary['macro_f1']:.6f}")
        root = summary.get("slices", {}).get("root")
        nonroot = summary.get("slices", {}).get("nonroot")
        if root:
            print(f"[{split_name}] root_macro_f1    = {root['macro_f1']:.6f}")
        if nonroot:
            print(f"[{split_name}] nonroot_macro_f1 = {nonroot['macro_f1']:.6f}")
    save_json(
        str(Path(args.out_dir) / "summary_all.json"),
        {
            "summaries": all_summaries,
            "dag_residual_config": dag_residual_cfg,
            "root_region_config": root_region_cfg,
        },
    )


if __name__ == "__main__":
    main()
