#!/usr/bin/env python3
"""
实时订单流 → K 线 → ONNX 推理 → 信号。

模拟用法：
    python live_pipeline.py

真实接入（用 WebSocket 替换模拟数据）：
    python live_pipeline.py --replay ../chan/data/processed/BTCUSDT_5m.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer import normalize_klines


class KlineAggregator:
    """订单流 → K 线"""

    def __init__(self, interval_sec=300):  # 默认 5 分钟
        self.interval_sec = interval_sec
        self.reset()

    def reset(self):
        self.open = 0.0
        self.high = 0.0
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0.0

    def on_trade(self, price: float, amount: float):
        if self.volume == 0:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += amount

    def kline(self):
        return [self.open, self.high, self.low, self.close, self.volume]


class LiveEngine:
    """订单入、信号出"""

    def __init__(self, model_path: str, seq_len: int = 60):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.seq_len = seq_len
        self.buffer: deque = deque(maxlen=seq_len)  # 滚动窗口
        self.kline_builder = KlineAggregator()

    def on_trade(self, price: float, amount: float):
        """来一笔订单"""
        self.kline_builder.on_trade(price, amount)

    def on_kline_complete(self):
        """一根 K 线走完（定时触发）"""
        k = self.kline_builder.kline()
        self.buffer.append(np.array(k, dtype=np.float32))
        self.kline_builder.reset()

        # 还没攒够 60 根，不出信号
        if len(self.buffer) < self.seq_len:
            return None

        # 够 60 根了 → 推理
        klines = np.stack(list(self.buffer))
        klines = normalize_klines(klines)
        inp = klines[np.newaxis, :, :]

        d, c, b = self.session.run(
            ["signal_direction", "signal_confidence", "is_bsp"],
            {"input": inp},
        )

        return {
            "direction": "buy" if d[0] == 1 else ("sell" if d[0] == 2 else None),
            "confidence": round(float(c[0]), 4),
            "bsp_score": round(float(b[0]), 4),
        }


# ── 演示 ──────────────────────────────────────────
def demo_replay(csv_path: str = None, seq_len: int = 60):
    """回放历史 CSV 模拟实时"""
    import pandas as pd

    model = "cs_distill.onnx" if Path("cs_distill.onnx").exists() else "chan_distill.onnx"
    if not Path(model).exists():
        print(f"❌ 模型文件不存在: {model}")
        return

    engine = LiveEngine(model, seq_len=seq_len)

    if csv_path:
        df = pd.read_csv(csv_path, header=None,
                         names=["ts", "open", "high", "low", "close", "volume"])
        print(f"回放 {len(df)} 根 K 线 ...\n")
        for _, row in df.iterrows():
            # 模拟一笔订单（用收盘价和成交量）
            engine.on_trade(float(row["close"]), float(row["volume"]))
            signal = engine.on_kline_complete()
            if signal and signal["direction"]:
                print(f"  {row['ts']}  {signal['direction']:4s}  "
                      f"置信度 {signal['confidence']:.2f}  "
                      f"BSP {signal['bsp_score']:.2f}")
    else:
        # 模拟订单流
        print("模拟实时订单流 ...\n")
        np.random.seed(42)
        price = 50000.0
        for i in range(200):
            price *= (1 + np.random.randn() * 0.002)
            vol = np.random.rand() * 100
            engine.on_trade(price, vol)
            signal = engine.on_kline_complete()
            if signal and signal["direction"]:
                print(f"  K线 {i:3d}  {signal['direction']:4s}  "
                      f"置信度 {signal['confidence']:.2f}  "
                      f"BSP {signal['bsp_score']:.2f}")

    print(f"\n✅ 完成（累积 {len(engine.buffer)} 根 K 线）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="回放 CSV 文件模拟实时")
    parser.add_argument("--seq-len", type=int, default=60, help="K 线序列长度")
    args = parser.parse_args()

    demo_replay(args.replay, args.seq_len)
