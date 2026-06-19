"""
ChanDistill —— 将缠论蒸馏进神经网络的端到端模型。

架构：1D-CNN 编码器 + 多任务输出头。
全部使用 ONNX 原生算子，导出零依赖。

推理时不需要 chan.py，一个 ONNX 全部搞定。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChanDistill(nn.Module):
    """
    参数
    ----
    seq_len    : 输入序列长度（默认 60）
    d_model    : 隐藏维度（默认 64）
    dropout    : Dropout 率（默认 0.1）
    """

    def __init__(self, seq_len: int = 60, d_model: int = 64, dropout: float = 0.1,
                 in_channels: int = 5):
        """
        参数
        ----
        seq_len     : 输入序列长度（默认 60）
        d_model     : 隐藏维度（默认 64）
        dropout     : Dropout 率（默认 0.1）
        in_channels : 输入通道数（默认 5 = OHLCV）
        """
        super().__init__()
        self.seq_len = seq_len

        # ── 时序金字塔 CNN ──────────────────────────
        # 短序列(≤120): 无步长 3 层（原行为）
        # 长序列: 逐层降采样的金字塔
        #   1m→5m(×5) →15m(×3) →30m(×2) →1h(×2) →4h(×4) →1d(×6) →1w(×7)
        def _build(in_ch, d_m, seq_len):
            # 短序列：原样保留
            if seq_len <= 120:
                return nn.Sequential(
                    nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(64, d_m, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

            # 长序列：构建金字塔
            strides_design = [5, 3, 2, 2, 4, 6, 7]  # 缩放到周级别
            used = []
            tmp = seq_len
            for s in strides_design:
                used.append(s)
                tmp = (tmp + s - 1) // s
                if tmp <= 1:
                    break
            n = len(used)

            # 通道渐进：从 in_ch 平稳增长到 d_m
            if in_ch >= d_m:
                chs = [in_ch] * (n + 1)
            else:
                def _rnd(v): return max(8, ((int(v) + 7) // 8) * 8)
                chs = [in_ch] + [_rnd(d_m * i / n) for i in range(1, n + 1)]
                chs[-1] = d_m

            layers = []
            for i in range(n):
                ci, co, s = chs[i], chs[i+1], used[i]
                k1 = 7 if i == 0 else 5          # 降采样核
                layers += [
                    nn.Conv1d(ci, co, k1, stride=s, padding=k1//2),
                    nn.ReLU(),
                    nn.Conv1d(co, co, 3, padding=1),  # 精炼
                    nn.ReLU(),
                ]
            return nn.Sequential(*layers)

        self.conv = _build(in_channels, d_model, seq_len)

        # ── 全局平均池化 ─────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── 输出头 ────────────────────────────────
        def head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, out_dim),
            )

        self.head_return = head(d_model, 3)          # 未来涨跌：跌/平/涨
        self.head_top_fractal = head(d_model, 1)     # Level-1 顶分型
        self.head_bottom_fractal = head(d_model, 1)  # Level-1 底分型
        self.head_bi_direction = head(d_model, 3)    # Level-1 笔方向
        self.head_bi_strength = head(d_model, 1)     # Level-1 笔强度（回归）
        self.head_is_bsp = head(d_model, 1)           # 是否 BSP
        self.head_bsp_direction = head(d_model, 3)    # BSP 方向
        # ── Level-2 区间套输出头 ──────────────────────
        self.head_top_fractal_L2 = head(d_model, 1)     # Level-2 顶分型
        self.head_bottom_fractal_L2 = head(d_model, 1)  # Level-2 底分型
        self.head_bi_direction_L2 = head(d_model, 3)    # Level-2 笔方向
        self.head_bi_strength_L2 = head(d_model, 1)     # Level-2 笔强度（回归）

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        参数
        ----
        x : (batch, seq_len, in_channels)
            in_channels=13 (OHLCV + 子级首 Chan + 子级末 Chan)

        返回
        ----
        {
            "return_logits": (B, 3),
            "top_fractal": (B, 1),
            "bottom_fractal": (B, 1),
            "bi_direction_logits": (B, 3),
            "bi_strength": (B, 1),
            "is_bsp": (B, 1),
            "bsp_direction_logits": (B, 3),
        }
        """
        x = x.permute(0, 2, 1)                       # (B, C, T) → Conv1D 格式
        x = self.conv(x)                             # (B, d_model, T)
        x = self.pool(x).squeeze(-1)                 # (B, d_model)

        return {
            "return_logits": self.head_return(x),
            "top_fractal": torch.sigmoid(self.head_top_fractal(x)),
            "bottom_fractal": torch.sigmoid(self.head_bottom_fractal(x)),
            "bi_direction_logits": self.head_bi_direction(x),
            "bi_strength": self.head_bi_strength(x).squeeze(-1),
            "is_bsp": torch.sigmoid(self.head_is_bsp(x)),
            "bsp_direction_logits": self.head_bsp_direction(x),
            "top_fractal_L2": torch.sigmoid(self.head_top_fractal_L2(x)),
            "bottom_fractal_L2": torch.sigmoid(self.head_bottom_fractal_L2(x)),
            "bi_direction_logits_L2": self.head_bi_direction_L2(x),
            "bi_strength_L2": self.head_bi_strength_L2(x).squeeze(-1),
        }

    def predict_signal(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """推理时调用：直接输出交易信号。纯张量运算，ONNX 兼容。"""
        outputs = self.forward(x)
        is_bsp = outputs["is_bsp"].squeeze(-1)
        bsp_dir_probs = F.softmax(outputs["bsp_direction_logits"], dim=-1)

        # 纯张量运算，不用布尔索引
        buy_score = is_bsp * bsp_dir_probs[:, 1]
        sell_score = is_bsp * bsp_dir_probs[:, 2]
        max_score, pred_dir = torch.max(
            torch.stack([torch.zeros_like(buy_score), buy_score, sell_score], dim=1),
            dim=1,
        )

        return {
            "signal_direction": pred_dir.float(),
            "signal_confidence": max_score,
            "is_bsp": is_bsp,
        }


def compute_loss(outputs, targets, weights=None):
    """多任务损失"""
    default_weights = {
        "return": 1.0, "top_fractal": 0.3, "bottom_fractal": 0.3,
        "bi_direction": 0.2, "bi_strength": 0.1,
        "is_bsp": 0.5, "bsp_direction": 0.3,
        "top_fractal_L2": 0.2, "bottom_fractal_L2": 0.2,
        "bi_direction_L2": 0.15, "bi_strength_L2": 0.05,
    }
    if weights:
        default_weights.update(weights)

    losses = {}
    losses["return"] = F.cross_entropy(outputs["return_logits"], targets["return_label"])
    losses["top_fractal"] = F.binary_cross_entropy(outputs["top_fractal"].squeeze(-1), targets["top_fractal"].float())
    losses["bottom_fractal"] = F.binary_cross_entropy(outputs["bottom_fractal"].squeeze(-1), targets["bottom_fractal"].float())
    losses["bi_direction"] = F.cross_entropy(outputs["bi_direction_logits"], targets["bi_direction_label"])
    losses["bi_strength"] = F.mse_loss(outputs["bi_strength"], targets["bi_strength"].float())
    losses["is_bsp"] = F.binary_cross_entropy(outputs["is_bsp"].squeeze(-1), targets["is_bsp"].float())
    # Level-2 损失
    losses["top_fractal_L2"] = F.binary_cross_entropy(outputs["top_fractal_L2"].squeeze(-1), targets["top_fractal_L2"].float())
    losses["bottom_fractal_L2"] = F.binary_cross_entropy(outputs["bottom_fractal_L2"].squeeze(-1), targets["bottom_fractal_L2"].float())
    losses["bi_direction_L2"] = F.cross_entropy(outputs["bi_direction_logits_L2"], targets["bi_direction_label_L2"])
    losses["bi_strength_L2"] = F.mse_loss(outputs["bi_strength_L2"], targets["bi_strength_L2"].float())

    bsp_mask = targets["is_bsp"] > 0.5
    if bsp_mask.sum() > 0:
        losses["bsp_direction"] = F.cross_entropy(
            outputs["bsp_direction_logits"][bsp_mask], targets["bsp_direction_label"][bsp_mask])
    else:
        losses["bsp_direction"] = torch.tensor(0.0, device=outputs["return_logits"].device)

    total = sum(losses[k] * default_weights.get(k, 1.0) for k in losses)
    return total, losses
