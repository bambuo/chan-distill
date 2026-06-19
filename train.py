#!/usr/bin/env python3
"""
多任务训练脚本。

用法：
    python train.py                                          # 默认训练
    python train.py --epochs 20 --batch-size 64              # 自定义参数
    python train.py --data-pattern data/labels/BTCUSDT_1h.parquet  # 指定数据
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# 加入项目路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from model.chandistill import ChanDistill, compute_loss


# ═══════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════
class ChanLabelDataset(Dataset):
    """
    加载 generate_labels.py 产出的 parquet 文件。

    每行包含：
        klines: bytes → (60, 5) OHLCV
        各种标签：fractal, bi, zs, bsp
    """

    def __init__(self, data_paths: List[str], seq_len: int = 60):
        import pandas as pd
        dfs = []
        for p in data_paths:
            df = pd.read_parquet(p)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        # 按时间排序，确保时间序列切分的正确性
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df.sort_values("timestamp", inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        self.seq_len = seq_len

        # 标签列
        self.return_bins = [-np.inf, -0.01, 0.01, np.inf]  # 跌/平/涨

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # K 线序列（兼容 5 通道单级别 / 15 通道多级别）
        klines = np.frombuffer(row["klines"], dtype=np.float32).reshape(row["klines_shape"])
        n_channels = klines.shape[-1]

        # 如果序列不足 seq_len，前补零（正常不会发生）
        if len(klines) < self.seq_len:
            pad = np.zeros((self.seq_len - len(klines), n_channels), dtype=np.float32)
            klines = np.vstack([pad, klines])

        # 主任务：未来收益 → 三分类
        ret = row["next_return"]
        ret_label = np.digitize(ret, self.return_bins) - 1
        ret_label = np.clip(ret_label, 0, 2)

        # 辅助任务：分型
        top_f = int(row.get("has_top_fractal", 0))
        bot_f = int(row.get("has_bottom_fractal", 0))

        # 辅助任务：笔
        bi_dir = int(row.get("bi_direction", 0)) + 1  # -1/0/1 → 0/1/2
        bi_str = float(row.get("bi_strength", 0.0))

        # 辅助任务：中枢
        zs_cnt = float(row.get("zs_count", 0))

        # 辅助任务：BSP
        is_bsp = int(row.get("is_bsp", 0))
        bsp_dir = int(row.get("bsp_direction", 0))

        # 辅助任务：Level-2（区间套）
        top_l2 = int(row.get("has_top_fractal_L2", 0))
        bot_l2 = int(row.get("has_bottom_fractal_L2", 0))
        bi_dir_l2 = int(row.get("bi_direction_L2", 0)) + 1  # -1/0/1 → 0/1/2
        bi_str_l2 = float(row.get("bi_strength_L2", 0.0))

        return {
            "klines": torch.from_numpy(klines).float(),
            "return_label": torch.tensor(ret_label, dtype=torch.long),
            "top_fractal": torch.tensor(top_f, dtype=torch.long),
            "bottom_fractal": torch.tensor(bot_f, dtype=torch.long),
            "bi_direction_label": torch.tensor(bi_dir, dtype=torch.long),
            "bi_strength": torch.tensor(bi_str, dtype=torch.float),
            "is_bsp": torch.tensor(is_bsp, dtype=torch.long),
            "bsp_direction_label": torch.tensor(bsp_dir, dtype=torch.long),
            "top_fractal_L2": torch.tensor(top_l2, dtype=torch.long),
            "bottom_fractal_L2": torch.tensor(bot_l2, dtype=torch.long),
            "bi_direction_label_L2": torch.tensor(bi_dir_l2, dtype=torch.long),
            "bi_strength_L2": torch.tensor(bi_str_l2, dtype=torch.float),
        }


# ═══════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    bsp_weight: float = 2.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    task_losses: Dict[str, float] = {}

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        klines = batch["klines"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "klines"}

        optimizer.zero_grad()
        outputs = model(klines)
        weights = {"is_bsp": bsp_weight, "bsp_direction": bsp_weight * 0.6}
        loss, losses = compute_loss(outputs, targets, weights=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            task_losses[k] = task_losses.get(k, 0) + v.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    result = {"total_loss": total_loss / n}
    for k, v in task_losses.items():
        result[k] = v / n
    return result


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_is_bsp = []
    all_is_bsp_pred = []

    with torch.no_grad():
        for batch in loader:
            klines = batch["klines"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "klines"}
            outputs = model(klines)
            loss, _ = compute_loss(outputs, targets)
            total_loss += loss.item()

            all_is_bsp.extend(targets["is_bsp"].cpu().numpy())
            all_is_bsp_pred.extend(outputs["is_bsp"].cpu().numpy())

    from sklearn.metrics import average_precision_score, roc_auc_score

    result = {"val_loss": total_loss / len(loader)}
    try:
        result["bsp_auprc"] = average_precision_score(all_is_bsp, all_is_bsp_pred)
        result["bsp_auroc"] = roc_auc_score(all_is_bsp, all_is_bsp_pred)
    except Exception:
        pass

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ChanDistill 多任务训练")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--data-pattern", nargs="+",
                        default=["data/labels/*.parquet"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bsp-weight", type=float, default=2.0,
                        help="BSP 检测任务权重（默认 2.0）")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"  🔧 设备: {device}")

    # 数据
    import glob
    data_paths = []
    for pattern in args.data_pattern:
        found = glob.glob(os.path.join(PROJECT_ROOT, pattern))
        data_paths.extend(found)
    if not data_paths:
        # 尝试默认路径
        data_paths = list(Path(PROJECT_ROOT / "data" / "labels").glob("*.parquet"))

    if not data_paths:
        print("❌ 未找到数据文件。请先运行: python scripts/generate_labels.py")
        return

    print(f"  📊 数据文件: {len(data_paths)} 个")
    for p in data_paths:
        size = Path(p).stat().st_size / 1024 / 1024
        print(f"      {Path(p).name} ({size:.1f} MB)")

    dataset = ChanLabelDataset([str(p) for p in data_paths], seq_len=args.seq_len)
    n = len(dataset)
    train_n = int(n * 0.8)
    val_n = n - train_n

    # 时间序列切分：前 80% 训练，后 20% 验证
    # （防止相邻样本重叠导致随机切分下 train/val 泄漏）
    all_indices = np.arange(n)
    train_set = torch.utils.data.Subset(dataset, all_indices[:train_n])
    val_set = torch.utils.data.Subset(dataset, all_indices[train_n:])

    # BSP 类别不均衡：加权采样，让 BSP 样本被抽到的概率 ≈ 非 BSP
    bsp_weights = []
    for i in train_set.indices:
        is_bsp = dataset.df.iloc[i]["is_bsp"]
        bsp_weights.append(5.0 if is_bsp else 1.0)  # BSP ~5% 正例，权重 5x
    sampler = WeightedRandomSampler(bsp_weights, num_samples=len(bsp_weights), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=sampler, num_workers=4,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_start = dataset.df.iloc[train_set.indices[0]]["timestamp"]
    train_end = dataset.df.iloc[train_set.indices[-1]]["timestamp"]
    val_start = dataset.df.iloc[val_set.indices[0]]["timestamp"]
    val_end = dataset.df.iloc[val_set.indices[-1]]["timestamp"]
    print(f"  📊 训练: {len(train_set)} 样本 ({train_start} ~ {train_end})")
    print(f"  📊 验证: {len(val_set)} 样本 ({val_start} ~ {val_end})")
    print(f"     序列长度: {args.seq_len}")

    # 模型（自动检测输入通道数：5=单级别, 15=多级别区间套）
    sample_klines = np.frombuffer(
        dataset.df.iloc[0]["klines"], dtype=np.float32
    ).reshape(dataset.df.iloc[0]["klines_shape"])
    in_channels = sample_klines.shape[-1]
    model = ChanDistill(
        seq_len=args.seq_len,
        d_model=args.d_model,
        in_channels=in_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  🧠 模型参数: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    best_bsp_auprc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch,
                                     bsp_weight=args.bsp_weight)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        bsp_auprc = val_metrics.get("bsp_auprc", 0)

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss: {train_metrics['total_loss']:.4f} | "
            f"val_loss: {val_metrics['val_loss']:.4f} | "
            f"BSP AUPRC: {bsp_auprc:.4f} | "
            f"lr: {lr:.2e}"
        )

        # 始终保存最后一个 epoch 的模型
        ckpt_path = checkpoint_dir / "latest.pt"
        torch.save(model.state_dict(), ckpt_path)

        # 如果 AUPRC 有提升则另存为 best
        if bsp_auprc > best_bsp_auprc:
            best_bsp_auprc = bsp_auprc
            best_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"    ✅ 新最佳模型: AUPRC={bsp_auprc:.4f}")

    print(f"\n🎉 训练完成！最佳 BSP AUPRC: {best_bsp_auprc:.4f}")
    print(f"   模型: {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
