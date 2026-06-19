#!/usr/bin/env python3
"""
推理演示 — 从订单到信号，不需要 chan.py / PyTorch。

用法：
    # 先运行 data 和 export 生成 chan_distill.onnx
    python infer.py                          # 模拟数据演示
    python infer.py --live                   # 读取真实 CSV 数据演示
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def normalize_klines(klines: np.ndarray) -> np.ndarray:
    """
    K 线归一化（与训练时一致）。

    输入:
        C=5:  (T, 5)  OHLCV
        C=13: (T, 13) OHLCV(5) + 子级别缠论特征(4+4)
    输出: (T, C) 价格以第一根 close 为基准，成交量以均值为基准
    """
    out = klines.copy().astype(np.float32)
    base_price = out[0, 3] if out[0, 3] > 0 else 1.0
    # 通道 0-4: 主级别 OHLCV
    out[:, :4] /= base_price
    vol_mean = np.mean(out[:, 4]) + 1e-10
    out[:, 4] /= vol_mean
    # 通道 5-12: 子级别缠论特征（分型0/1, 笔方向0/1/2, 笔强度float）— 不需要价格归一化
    return out


class ChanDistillInference:
    """
    推理引擎 — 一个 ONNX 搞定全部。

    用法：
        engine = ChanDistillInference("chan_distill.onnx")
        signal = engine.update(new_klines)  # 传入最新 K 线
    """

    def __init__(self, model_path: str, seq_len: int = 60, threshold: float = 0.5):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.seq_len = seq_len
        self.threshold = threshold
        self.buffer: List[np.ndarray] = []  # 滚动缓存

    def update(self, ohlcv: tuple) -> Optional[dict]:
        """
        喂一根新 K 线 (open, high, low, close, volume)。

        返回信号字典或 None。
        """
        self.buffer.append(np.array(ohlcv, dtype=np.float32))
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_len:
            return None  # 还没攒够

        klines = np.stack(self.buffer)                    # (T, 5)
        klines = normalize_klines(klines)                 # 归一化
        inp = klines[np.newaxis, :, :]                   # (1, T, 5)

        results = self.session.run(
            ["signal_direction", "signal_confidence", "is_bsp"], {"input": inp}
        )
        dir_out, conf_out, bsp_out = results

        direction = int(dir_out[0])
        confidence = float(conf_out[0])
        is_bsp = float(bsp_out[0])

        if is_bsp > self.threshold and direction > 0:
            return {
                "direction": "buy" if direction == 1 else "sell",
                "confidence": round(confidence, 4),
                "bsp_score": round(is_bsp, 4),
            }
        return None


def demo_mock_data(seq_len=60):
    """模拟数据演示 — 不需要任何外部依赖"""
    print("=" * 50)
    print(f"ChanDistill 推理演示（模拟数据, seq_len={seq_len}）")
    print("=" * 50)

    model_path = "chan_distill.onnx"
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行: python export_onnx.py")
        return

    engine = ChanDistillInference(model_path, seq_len=seq_len)

    # 模拟 seq_len + 40 根 K 线（让窗口能填满）
    np.random.seed(42)
    price = 50000.0
    total_bars = seq_len + 40
    print(f"  模拟 {total_bars} 根 K 线 ...")

    for i in range(total_bars):
        # 模拟价格波动
        ret = np.random.randn() * 0.002 + 0.0003
        price *= (1 + ret)
        ohlcv = (
            price * (1 - np.random.rand() * 0.001),
            price * (1 + np.random.rand() * 0.002),
            price * (1 - np.random.rand() * 0.002),
            price * (1 + np.random.rand() * 0.001 - 0.0005),
            np.random.rand() * 100 + 50,
        )
        signal = engine.update(ohlcv)

        if signal:
            print(f"  ⚡ {i:3d}: {signal['direction']:4s}  "
                  f"置信度 {signal['confidence']:.2f}  "
                  f"BSP {signal['bsp_score']:.2f}")

    print("\n✅ 演示完成（没有 chan.py，没有 PyTorch）")


def demo_live_data(csv_path: str, model_path: str = "chan_distill.onnx", seq_len: int = 60):
    """读取真实 CSV 演示"""
    import pandas as pd

    print(f"  读取 {csv_path} ...")
    df = pd.read_csv(csv_path, header=None,
                     names=["ts", "open", "high", "low", "close", "volume"])

    engine = ChanDistillInference(model_path, seq_len=seq_len)
    signals = 0

    for i, row in df.iterrows():
        signal = engine.update((row["open"], row["high"], row["low"],
                                row["close"], row["volume"]))
        if signal:
            signals += 1
            if signals <= 5:
                print(f"  ⚡ {row['ts']}: {signal['direction']}  "
                      f"置信度 {signal['confidence']:.2f}")

    print(f"\n  总计: {signals} 信号 / {len(df)} K 线")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ChanDistill 推理演示")
    parser.add_argument("--live", type=str, default=None,
                        help="CSV 文件路径（可选）")
    parser.add_argument("--model", default="chan_distill.onnx")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="K 线序列长度（与训练时一致）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="信号输出阈值")
    args = parser.parse_args()

    if args.live:
        demo_live_data(args.live, args.model, args.seq_len)
    else:
        demo_mock_data(args.seq_len)


if __name__ == "__main__":
    main()
