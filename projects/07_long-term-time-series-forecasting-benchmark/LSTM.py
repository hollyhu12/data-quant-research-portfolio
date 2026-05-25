import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "PatchTST"
AUTO_SPLIT_STRATEGIES = {
    "ETTm1": "custom",
    "weather": "custom",
    "electricity": "custom",
}


def str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(device_arg)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device string: {device_arg}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this environment.")
    return device


def mse_loss(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae_loss(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse_loss(pred: np.ndarray, true: np.ndarray) -> float:
    return float(math.sqrt(mse_loss(pred, true)))


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        seq_len: int,
        pred_len: int,
        split_name: str,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"{split_name} features must be 2D, got shape {features.shape}.")
        if target.ndim != 1:
            raise ValueError(f"{split_name} target must be 1D, got shape {target.shape}.")
        if len(features) != len(target):
            raise ValueError(
                f"{split_name} features/target length mismatch: {len(features)} vs {len(target)}."
            )
        if len(features) < seq_len + pred_len:
            raise ValueError(
                f"{split_name} split is too short for seq_len={seq_len} and pred_len={pred_len}. "
                f"Need at least {seq_len + pred_len} rows, got {len(features)}."
            )

        self.features = features.astype(np.float32, copy=False)
        self.target = target.astype(np.float32, copy=False)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(features) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.seq_len
        pred_end = end + self.pred_len
        x = self.features[start:end]
        y = self.target[end:pred_end]
        return torch.from_numpy(x), torch.from_numpy(y)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)


def get_activation_fn(name: str) -> nn.Module:
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def positional_encoding(q_len: int, d_model: int, learn_pe: bool = True) -> nn.Parameter:
    pe = torch.zeros(q_len, d_model)
    nn.init.uniform_(pe, a=-0.02, b=0.02)
    return nn.Parameter(pe, requires_grad=learn_pe)


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        self.mean: Optional[Tensor] = None
        self.stdev: Optional[Tensor] = None
        self.last: Optional[Tensor] = None

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x: Tensor) -> None:
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
            self.mean = None
        else:
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.last = None
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("RevIN last value was not initialized before normalization.")
            x = x - self.last
        else:
            if self.mean is None:
                raise RuntimeError("RevIN mean was not initialized before normalization.")
            x = x - self.mean

        if self.stdev is None:
            raise RuntimeError("RevIN stdev was not initialized before normalization.")
        x = x / self.stdev

        if self.affine:
            x = (x * self.affine_weight) + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.stdev is None:
            raise RuntimeError("RevIN stdev was not initialized before denormalization.")

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        x = x * self.stdev
        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("RevIN last value was not initialized before denormalization.")
            x = x + self.last
        else:
            if self.mean is None:
                raise RuntimeError("RevIN mean was not initialized before denormalization.")
            x = x + self.mean
        return x

    def denormalize_target(self, x: Tensor, target_idx: int) -> Tensor:
        if self.stdev is None:
            raise RuntimeError("RevIN stdev was not initialized before target denormalization.")

        if self.affine:
            target_bias = self.affine_bias[:, :, target_idx]
            target_weight = self.affine_weight[:, :, target_idx]
            x = (x - target_bias) / (target_weight + self.eps)

        target_std = self.stdev[:, :, target_idx]
        x = x * target_std
        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("RevIN last value was not initialized before target denormalization.")
            x = x + self.last[:, :, target_idx]
        else:
            if self.mean is None:
                raise RuntimeError("RevIN mean was not initialized before target denormalization.")
            x = x + self.mean[:, :, target_idx]
        return x


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        left_pad = (self.kernel_size - 1) // 2
        right_pad = self.kernel_size - 1 - left_pad
        front = x[:, :1, :].repeat(1, left_pad, 1)
        end = x[:, -1:, :].repeat(1, right_pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size=kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        trend = self.moving_average(x)
        residual = x - trend
        return residual, trend


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
    ) -> None:
        super().__init__()
        self.res_attention = res_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights, None


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.res_attention = res_attention

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.sdp_attn = ScaledDotProductAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout),
        )

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        batch_size = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        q_proj = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_proj = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_output, attn_weights, attn_scores = self.sdp_attn(
            q=q_proj,
            k=k_proj,
            v=v_proj,
            prev=prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return output, attn_weights, attn_scores


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.self_attn = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            res_attention=res_attention,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = self._build_norm(norm, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = self._build_norm(norm, d_model)

    @staticmethod
    def _build_norm(norm: str, d_model: int) -> nn.Module:
        if "batch" in norm.lower():
            return nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        return nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.pre_norm:
            src = self.norm_attn(src)

        src2, _, scores = self.self_attn(
            q=src,
            k=src,
            v=src,
            prev=prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src, scores if self.res_attention else None


class TSTEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        scores = None
        for layer in self.layers:
            output, scores = layer(
                output,
                prev=scores if self.res_attention else None,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        return output


class TSTiEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(q_len=patch_num, d_model=d_model, learn_pe=True)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=activation,
            res_attention=res_attention,
            pre_norm=pre_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        num_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)
        z = torch.reshape(z, (-1, num_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        return z


class FlattenHead(nn.Module):
    def __init__(
        self,
        individual: bool,
        n_vars: int,
        nf: int,
        target_window: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.flattens = nn.ModuleList(nn.Flatten(start_dim=-2) for _ in range(n_vars))
            self.linears = nn.ModuleList(nn.Linear(nf, target_window) for _ in range(n_vars))
            self.dropouts = nn.ModuleList(nn.Dropout(head_dropout) for _ in range(n_vars))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.individual:
            outputs = []
            for idx in range(self.n_vars):
                z = self.flattens[idx](x[:, idx, :, :])
                z = self.linears[idx](z)
                z = self.dropouts[idx](z)
                outputs.append(z)
            return torch.stack(outputs, dim=1)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TargetAwareHead(nn.Module):
    def __init__(
        self,
        n_vars: int,
        target_idx: int,
        d_model: int,
        patch_num: int,
        target_window: int,
        head_dropout: float = 0.0,
        fc_dropout: float = 0.2,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.target_idx = target_idx
        self.head_nf = d_model * patch_num

        self.target_flatten = nn.Flatten(start_dim=-2)
        self.base_head = nn.Sequential(
            nn.LayerNorm(self.head_nf),
            nn.Dropout(head_dropout),
            nn.Linear(self.head_nf, target_window),
        )

        self.channel_token_proj = nn.Sequential(
            nn.LayerNorm(self.head_nf),
            nn.Linear(self.head_nf, d_model),
            nn.GELU(),
            nn.Dropout(fc_dropout),
        )
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.context_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fusion_hidden_dim, target_window),
        )

    def forward(self, x: Tensor) -> Tensor:
        target_encoded = x[:, self.target_idx, :, :]
        target_flat = self.target_flatten(target_encoded)
        base_pred = self.base_head(target_flat)

        flat_features = x.flatten(start_dim=-2)
        channel_tokens = self.channel_token_proj(flat_features)
        target_summary = channel_tokens[:, self.target_idx, :]

        query = self.query(target_summary).unsqueeze(1)
        keys = self.key(channel_tokens)
        values = self.value(channel_tokens)

        attn_scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_summary = torch.matmul(attn_weights, values).squeeze(1)

        context_pred = self.context_head(torch.cat([target_summary, context_summary], dim=-1))
        return base_pred + context_pred


class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        target_idx: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        padding_patch: Optional[str] = "end",
        revin: bool = True,
        affine: bool = False,
        subtract_last: bool = False,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.target_idx = target_idx
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(
            c_in=c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=activation,
            res_attention=res_attention,
            pre_norm=pre_norm,
        )

        head_nf = d_model * patch_num
        self.head = TargetAwareHead(
            n_vars=c_in,
            target_idx=target_idx,
            d_model=d_model,
            patch_num=patch_num,
            target_window=target_window,
            head_dropout=head_dropout,
            fc_dropout=fc_dropout,
            fusion_hidden_dim=fusion_hidden_dim,
        )

    def forward(self, z: Tensor) -> Tensor:
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)

        if self.revin:
            z = self.revin_layer.denormalize_target(z, self.target_idx)
        return z


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        target_idx: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        padding_patch: Optional[str] = "end",
        revin: bool = True,
        affine: bool = False,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 25,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.decomposition = decomposition
        backbone_kwargs = dict(
            c_in=c_in,
            target_idx=target_idx,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            n_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm="BatchNorm",
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=activation,
            res_attention=True,
            pre_norm=False,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            fusion_hidden_dim=fusion_hidden_dim,
        )

        if self.decomposition:
            self.decomp_module = SeriesDecomposition(kernel_size=kernel_size)
            self.model_res = PatchTSTBackbone(**backbone_kwargs)
            self.model_trend = PatchTSTBackbone(**backbone_kwargs)
        else:
            self.model = PatchTSTBackbone(**backbone_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res = self.model_res(res_init.permute(0, 2, 1))
            trend = self.model_trend(trend_init.permute(0, 2, 1))
            return res + trend

        x = x.permute(0, 2, 1)
        return self.model(x)


@dataclass
class RunConfig:
    data_path: str
    dataset_name: str
    target_column: str = "OT"
    split_strategy: str = "custom"
    seq_len: int = 336
    pred_len: int = 96
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 20
    pct_start: float = 0.3
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 16
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    fc_dropout: float = 0.2
    head_dropout: float = 0.0
    attn_dropout: float = 0.0
    fusion_hidden_dim: int = 256
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    padding_patch: str = "end"
    early_stop_min_delta: float = 0.0
    grad_clip: float = 0.0
    num_workers: int = 0
    use_amp: bool = False
    seed: int = 42
    device: str = "auto"


def validate_args(config: RunConfig) -> None:
    data_path = Path(config.data_path)
    if not config.dataset_name:
        raise ValueError("dataset_name must be a non-empty string.")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"data_path must point to a CSV file, got: {data_path}")
    if config.seq_len <= 0 or config.pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive.")
    if config.batch_size <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive.")
    if config.patch_len <= 0 or config.stride <= 0:
        raise ValueError("patch_len and stride must be positive.")
    if config.patch_len > config.seq_len:
        raise ValueError("patch_len cannot be larger than seq_len.")
    if config.d_model <= 0 or config.n_heads <= 0 or config.e_layers <= 0 or config.d_ff <= 0:
        raise ValueError("d_model, n_heads, e_layers, and d_ff must be positive.")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if config.lr <= 0 or config.weight_decay < 0 or config.pct_start <= 0 or config.pct_start >= 1:
        raise ValueError("lr must be positive, weight_decay non-negative, and pct_start in (0, 1).")
    if config.dropout < 0 or config.head_dropout < 0 or config.attn_dropout < 0:
        raise ValueError("dropout values must be non-negative.")
    if config.fc_dropout < 0 or config.fusion_hidden_dim <= 0:
        raise ValueError("fc_dropout must be non-negative and fusion_hidden_dim positive.")
    if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if config.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if config.early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be non-negative.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.padding_patch not in {"end", "none", "None", ""}:
        raise ValueError("padding_patch must be one of: end, none.")
    if config.split_strategy not in {"auto", "custom", "ettm1"}:
        raise ValueError("split_strategy must be one of: auto, custom, ettm1.")


def resolve_padding_patch(padding_patch: str) -> Optional[str]:
    if padding_patch in {"none", "None", ""}:
        return None
    return padding_patch


def resolve_split_strategy(dataset_name: str, split_strategy: str) -> str:
    if split_strategy != "auto":
        return split_strategy
    return AUTO_SPLIT_STRATEGIES.get(dataset_name, "custom")


def load_and_prepare_dataframe(data_path: Path, target_column: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise ValueError(f"Required column 'date' is missing in {data_path}.")
    if target_column not in df.columns:
        raise ValueError(f"Required target column '{target_column}' is missing in {data_path}.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"Column 'date' contains invalid timestamps in {data_path}.")

    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = [col for col in df.columns if col != "date" and pd.api.types.is_numeric_dtype(df[col])]
    if target_column not in numeric_cols:
        raise ValueError(f"Target column '{target_column}' must be numeric in {data_path}.")

    ordered_feature_cols = [col for col in numeric_cols if col != target_column] + [target_column]
    selected = df[["date"] + ordered_feature_cols].copy()
    if selected[ordered_feature_cols].isna().any().any():
        raise ValueError(
            f"Numeric columns contain missing values in {data_path}. Please clean the CSV before training."
        )
    return selected


def build_split_slices(
    dataset_name: str,
    num_rows: int,
    seq_len: int,
    split_strategy: str,
) -> Tuple[Dict[str, slice], slice, str]:
    resolved_strategy = resolve_split_strategy(dataset_name, split_strategy)

    if resolved_strategy == "ettm1":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
        if num_rows < test_end:
            raise ValueError(
                f"ETTm1 official split requires at least {test_end} rows, but got {num_rows}."
            )
        train_slice = slice(0, train_end)
        splits = {
            "train": train_slice,
            "val": slice(train_end - seq_len, val_end),
            "test": slice(val_end - seq_len, test_end),
        }
        return splits, train_slice, resolved_strategy

    if resolved_strategy == "custom":
        num_train = int(num_rows * 0.7)
        num_test = int(num_rows * 0.2)
        num_val = num_rows - num_train - num_test
        if min(num_train, num_val, num_test) <= seq_len:
            raise ValueError(
                "Dataset is too short for the requested seq_len under the custom 70/10/20 split."
            )
        border1s = [0, num_train - seq_len, num_rows - num_test - seq_len]
        border2s = [num_train, num_train + num_val, num_rows]
        train_slice = slice(border1s[0], border2s[0])
        splits = {
            "train": train_slice,
            "val": slice(border1s[1], border2s[1]),
            "test": slice(border1s[2], border2s[2]),
        }
        return splits, train_slice, resolved_strategy

    raise ValueError(f"Unsupported split strategy: {resolved_strategy}")


def compute_train_stats(train_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_array.mean(axis=0)
    std = train_array.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_with_stats(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((array - mean) / std).astype(np.float32)


def build_datasets(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    split_strategy: str,
    seq_len: int,
    pred_len: int,
) -> Tuple[Dict[str, TimeSeriesWindowDataset], Dict[str, np.ndarray], List[str], int, str]:
    feature_cols = [col for col in df.columns if col != "date"]
    target_idx = feature_cols.index(target_column)
    values = df[feature_cols].to_numpy(dtype=np.float32)

    splits, train_slice, resolved_split_strategy = build_split_slices(
        dataset_name=dataset_name,
        num_rows=len(df),
        seq_len=seq_len,
        split_strategy=split_strategy,
    )
    train_values = values[train_slice]
    mean, std = compute_train_stats(train_values)

    datasets: Dict[str, TimeSeriesWindowDataset] = {}
    split_arrays: Dict[str, np.ndarray] = {"mean": mean, "std": std}

    for split_name, split_slice in splits.items():
        split_values = values[split_slice]
        normalized = normalize_with_stats(split_values, mean, std)
        target = normalized[:, target_idx]
        datasets[split_name] = TimeSeriesWindowDataset(
            features=normalized,
            target=target,
            seq_len=seq_len,
            pred_len=pred_len,
            split_name=split_name,
        )

    return datasets, split_arrays, feature_cols, target_idx, resolved_split_strategy


def create_dataloaders(
    datasets: Dict[str, TimeSeriesWindowDataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    start_time = time.perf_counter()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch_x)
        preds.append(outputs.float().cpu().numpy())
        trues.append(batch_y.numpy())
    inference_time = time.perf_counter() - start_time

    pred_array = np.concatenate(preds, axis=0)
    true_array = np.concatenate(trues, axis=0)
    return pred_array, true_array, inference_time


def inverse_target_scale(array: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return (array * target_std) + target_mean


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_summary(summary_path: Path, row: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_history(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)


def train_and_evaluate(config: RunConfig) -> Dict[str, object]:
    validate_args(config)
    set_random_seed(config.seed)

    script_dir = Path(__file__).resolve().parent
    data_path = Path(config.data_path)
    device = resolve_device(config.device)

    df = load_and_prepare_dataframe(data_path, target_column=config.target_column)
    datasets, split_arrays, feature_cols, target_idx, resolved_split_strategy = build_datasets(
        df=df,
        dataset_name=config.dataset_name,
        target_column=config.target_column,
        split_strategy=config.split_strategy,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
    )
    dataloaders = create_dataloaders(
        datasets=datasets,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PatchTSTModel(
        c_in=len(feature_cols),
        target_idx=target_idx,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        fc_dropout=config.fc_dropout,
        head_dropout=config.head_dropout,
        attn_dropout=config.attn_dropout,
        activation="gelu",
        padding_patch=resolve_padding_patch(config.padding_patch),
        revin=config.revin,
        affine=config.affine,
        subtract_last=config.subtract_last,
        decomposition=config.decomposition,
        kernel_size=config.kernel_size,
        fusion_hidden_dim=config.fusion_hidden_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        steps_per_epoch=max(len(dataloaders["train"]), 1),
        epochs=config.max_epochs,
        pct_start=config.pct_start,
    )
    use_amp = config.use_amp and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=use_amp)

    output_dir = script_dir / "outputs" / "patchtst" / config.dataset_name / f"pred_{config.pred_len}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_trained = 0
    patience_counter = 0
    history_rows: List[Dict[str, float]] = []

    train_start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=config.grad_clip,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        epochs_trained = epoch
        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | lr={current_lr:.6g} | elapsed={epoch_time:.2f}s"
        )
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(current_lr),
            }
        )

        if val_loss < (best_val_loss - config.early_stop_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    training_time = time.perf_counter() - train_start

    if not best_model_path.exists():
        raise RuntimeError("Best model checkpoint was not saved. Training did not complete correctly.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    pred_norm_all, true_norm_all, inference_time = predict(
        model=model,
        loader=dataloaders["test"],
        device=device,
        use_amp=use_amp,
    )
    pred_norm = pred_norm_all
    true_norm = true_norm_all

    target_mean = float(split_arrays["mean"][target_idx])
    target_std = float(split_arrays["std"][target_idx])
    pred = inverse_target_scale(pred_norm, target_mean, target_std)
    true = inverse_target_scale(true_norm, target_mean, target_std)

    metrics = {
        "dataset_name": config.dataset_name,
        "model_name": MODEL_NAME,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "epochs_trained": epochs_trained,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_mse": mse_loss(pred, true),
        "test_mae": mae_loss(pred, true),
        "test_rmse": rmse_loss(pred, true),
        "training_wall_clock_time_seconds": training_time,
        "test_inference_time_seconds": inference_time,
        "trainable_parameter_count": count_trainable_parameters(model),
        "device_used": str(device),
        "seed": config.seed,
    }

    config_payload = asdict(config)
    config_payload["resolved_device"] = str(device)
    config_payload["feature_columns"] = feature_cols
    config_payload["target_column"] = config.target_column
    config_payload["target_index"] = target_idx
    config_payload["task_mode"] = "MS"
    config_payload["resolved_split_strategy"] = resolved_split_strategy

    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions=pred.astype(np.float32),
        targets=true.astype(np.float32),
        predictions_normalized=pred_norm.astype(np.float32),
        targets_normalized=true_norm.astype(np.float32),
    )
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "config.json", config_payload)
    save_training_history(output_dir / "training_history.csv", history_rows)

    summary_row = {
        "dataset_name": config.dataset_name,
        "model_name": MODEL_NAME,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "epochs_trained": epochs_trained,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_mse": metrics["test_mse"],
        "test_mae": metrics["test_mae"],
        "test_rmse": metrics["test_rmse"],
        "training_wall_clock_time_seconds": training_time,
        "test_inference_time_seconds": inference_time,
        "trainable_parameter_count": metrics["trainable_parameter_count"],
        "device_used": str(device),
        "seed": config.seed,
        "output_dir": str(output_dir),
    }
    append_summary(script_dir / "outputs" / "patchtst" / "summary.csv", summary_row)

    print("-" * 80)
    print(
        f"Summary | dataset={config.dataset_name} | pred_len={config.pred_len} | "
        f"test_mse={metrics['test_mse']:.6f} | test_mae={metrics['test_mae']:.6f} | "
        f"test_rmse={metrics['test_rmse']:.6f}"
    )
    print(
        f"Training time={training_time:.2f}s | Inference time={inference_time:.2f}s | "
        f"Parameters={metrics['trainable_parameter_count']} | output_dir={output_dir}"
    )

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Target-aware PatchTST runner for multivariate-input single-target forecasting."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for logging and split presets.")
    parser.add_argument("--target_column", type=str, default="OT", help="Target column name.")
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="custom",
        choices=["auto", "custom", "ettm1"],
        help="Dataset split strategy.",
    )
    parser.add_argument("--seq_len", type=int, default=336, help="Input sequence length.")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction horizon.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--pct_start", type=float, default=0.3, help="OneCycleLR pct_start.")
    parser.add_argument("--patch_len", type=int, default=16, help="Patch length.")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride.")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension.")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--e_layers", type=int, default=3, help="Number of encoder layers.")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward dimension.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Encoder dropout.")
    parser.add_argument("--fc_dropout", type=float, default=0.2, help="Fusion head dropout.")
    parser.add_argument("--head_dropout", type=float, default=0.0, help="Forecast head dropout.")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument("--fusion_hidden_dim", type=int, default=256, help="Hidden size of target-aware fusion head.")
    parser.add_argument("--revin", type=str_to_bool, default=True, help="Enable RevIN normalization.")
    parser.add_argument("--affine", type=str_to_bool, default=False, help="Enable affine parameters in RevIN.")
    parser.add_argument("--subtract_last", type=str_to_bool, default=False, help="Use last-value centering in RevIN.")
    parser.add_argument("--decomposition", type=str_to_bool, default=False, help="Enable trend/residual decomposition.")
    parser.add_argument("--kernel_size", type=int, default=25, help="Moving-average kernel size.")
    parser.add_argument(
        "--padding_patch",
        type=str,
        default="end",
        choices=["end", "none"],
        help="Patch padding mode.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience.",
    )
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm. Set 0 to disable.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--use_amp", type=str_to_bool, default=False, help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, for example 'auto', 'cpu', or 'cuda'.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = RunConfig(**vars(args))

    try:
        train_and_evaluate(config)
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
