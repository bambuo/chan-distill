#!/usr/bin/env python3
"""
导出单 ONNX 模型。

用法：
    python export_onnx.py --checkpoint checkpoints/latest.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from model.chandistill import ChanDistill


def export(checkpoint_path: str, output_path: str = "chan_distill.onnx",
           seq_len: int = 60, d_model: int = 128, in_channels: int = 15):
    print(f"  📦 加载 checkpoint: {checkpoint_path}")

    model = ChanDistill(seq_len=seq_len, d_model=d_model, in_channels=in_channels)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 自动检测通道数：从 state_dict 的 conv.0.weight 推断
    first_weight_key = "conv.0.weight"
    if first_weight_key in state:
        detected_channels = state[first_weight_key].shape[1]
        if detected_channels != in_channels:
            print(f"  🔄 自动检测到 in_channels={detected_channels}，覆盖默认值")
            in_channels = detected_channels
            model = ChanDistill(seq_len=seq_len, d_model=d_model, in_channels=in_channels)
    model.load_state_dict(state)
    model.eval()

    class Wrapper(torch.nn.Module):
        def forward(self, x):
            r = model.predict_signal(x)
            return r["signal_direction"], r["signal_confidence"], r["is_bsp"]

    wrapper = Wrapper().eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            torch.randn(1, seq_len, in_channels),
            output_path,
            input_names=["input"],
            output_names=["signal_direction", "signal_confidence", "is_bsp"],
            opset_version=17,
        )

    # 验证
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(output_path)
    for _ in range(3):
        test = np.random.randn(1, seq_len, in_channels).astype(np.float32)
        d, c, b = session.run(None, {"input": test})
        assert d.shape == (1,) and c.shape == (1,) and b.shape == (1,)

    size = Path(output_path).stat().st_size / 1024
    print(f"  ✅ {output_path} ({size:.0f} KB)")
    print(f"  ✅ ONNX Runtime 验证通过")
    print(f"     输入: (1, {seq_len}, {in_channels})"
          f" {'OHLCV+子级首+子级末' if in_channels==15 else 'OHLCV'}")
    print(f"     输出: signal_direction(0=无,1=买,2=卖), confidence, is_bsp")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="导出 ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--output", default="chan_distill.onnx")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=13,
                        help="输入通道数（5=单级别, 13=多级别区间套）")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.seq_len, args.d_model, args.in_channels)
