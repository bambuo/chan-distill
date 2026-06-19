#!/usr/bin/env python3
"""
数据标注管线 —— 用 chan.py 给历史 K 线打上缠论标签。

每个样本 = 60 根连续 K 线 + 对应的缠论结构标签。

输出：data/labels/{symbol}_{timeframe}.parquet
  每行: klines(60×5) + has_top_fractal + has_bottom_fractal +
        bi_direction + bi_strength + zs_count + is_bsp +
        bsp_direction + bsp_types + next_return

依赖：需在 chan-xgb 项目同级目录下，复用其数据和 chan.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── 复用 chan-xgb 的库（仅需要 chan.py）─────────────
CHAN_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "chan"
sys.path.insert(0, str(CHAN_PROJECT_DIR / "lib" / "chan.py"))
sys.path.insert(0, str(CHAN_PROJECT_DIR))

from Chan import CChan, CKLine_Unit
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, FX_TYPE, KL_TYPE
from DataAPI import CSVTrainingAPI


# ── 时间周期映射 ────────────────────────────────────
TF_MAP = {
    "1m": KL_TYPE.K_1M, "5m": KL_TYPE.K_5M, "15m": KL_TYPE.K_15M,
    "1h": KL_TYPE.K_60M, "4h": KL_TYPE.K_4H, "1d": KL_TYPE.K_DAY,
}

STR_TO_TF = {v: k for k, v in TF_MAP.items()}

# ── 1m 数据配置（7 天窗口 = 10080 根）─────────────
DEFAULT_SEQ_LEN = 10080
DEFAULT_STRIDE = 1008


# ── V2 标注配置（与 chan-xgb config.yaml 对齐）─────────
V2_LABELING_CFG = {
    "min_window": 3,          # 最短窗口（K 线根数）
    "max_window": 24,         # 最长窗口
    "base_atr_multiple": 1.5, # 基准 ATR 倍数
    "high_vol_ratio": 1.5,    # 高波动判定：ATR > median × 此值
    "high_vol_multiple": 2.0, # 高波动时用此倍数
    "low_vol_ratio": 0.5,     # 低波动判定：ATR < median × 此值
    "low_vol_multiple": 1.2,  # 低波动时用此倍数
    "fee_rate": 0.001,        # 手续费 0.1%
    "slippage": 0.0005,       # 滑点 0.05%
}


def _calc_atr_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """计算完整的 ATR 序列"""
    n = len(highs)
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    atr[:period-1] = atr[period-1]
    return atr


def _compute_v2_return(idx: int, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                        atr_values: np.ndarray, atr_baseline: np.ndarray, cfg: dict) -> float:
    """
    V2 动态 ATR 窗口收益。

    对于每个位置 idx：
    1. 计算自适应 ATR 阈值（高/中/低波动三种倍数）
    2. 在 [min_w, max_w] 窗口内扫描
    3. 当价格首次达到阈值方向时返回该点的收益率
    4. 如果窗口结束还没突破阈值，返回窗口末尾的收益率

    这解决了固定 12-bar 窗口的两个问题：
    - BSP 样本不因趋势延续而被错误标记为负收益
    - 高波动期用更宽松的阈值，低波动期用更严格的阈值
    """
    n = len(closes)
    if idx >= n - cfg["min_window"]:
        return 0.0

    entry = closes[idx]
    current_atr = atr_values[idx]

    # 自适应 ATR 倍数（atr_baseline 是到当前位为止的滚动均值，无未来泄漏）
    if current_atr > atr_baseline[idx] * cfg["high_vol_ratio"]:
        effective_mult = cfg["high_vol_multiple"]
    elif current_atr < atr_baseline[idx] * cfg["low_vol_ratio"]:
        effective_mult = cfg["low_vol_multiple"]
    else:
        effective_mult = cfg["base_atr_multiple"]

    cost_factor = 1 + cfg["fee_rate"] + cfg["slippage"]
    threshold = effective_mult * current_atr * cost_factor

    end = min(idx + cfg["max_window"] + 1, n)

    # 扫描窗口：找第一次达到阈值的方向（上或下）
    for t in range(idx + 1, end):
        if highs[t] >= entry + threshold:
            return (closes[t] - entry) / entry   # 涨
        if lows[t] <= entry - threshold:
            return (closes[t] - entry) / entry   # 跌

    # 窗口内没突破 → 取窗口末尾的收益
    return (closes[end - 1] - entry) / entry


def _extract_chan_labels_at_position(snap, current_idx: int) -> Dict[str, Any]:
    """
    从 CChan 快照中提取当前 K 线位置的缠论结构标签。

    正确做法：遍历 chan 结构，找到包含 current_idx 的元素，
    而不是取最后一个。

    参数
    ----
    snap        : CChan 实例（step_load 的 yield）
    current_idx : 当前 raw K-line 索引（0-based）

    返回
    ----
    {
        "has_top_fractal": 0/1,
        "has_bottom_fractal": 0/1,
        "bi_direction": 1/-1/0,
        "bi_strength": float,
        "zs_count": int,
    }
    """
    labels: Dict[str, Any] = {
        "has_top_fractal": 0, "has_bottom_fractal": 0,
        "bi_direction": 0, "bi_strength": 0.0,
        "zs_count": 0,
    }

    kl_list = snap[0]  # CKLine_List
    if not kl_list:
        return labels

    # ── 分型 (Fractal) ────────────────────────────
    # 由于 update_fx 只在 lst[-2] 上被调用（滞后 1 步），
    # 此处不做实时匹配；分型标签由主循环中的回溯更新机制处理。

    # ── 笔 (Bi) ──────────────────────────────────
    # 找到 current_idx 落在哪一笔内（如果落在一笔内的话）
    for bi in kl_list.bi_list:
        begin_idx = bi.get_begin_klu().idx
        end_idx = bi.get_end_klu().idx
        if begin_idx <= current_idx <= end_idx:
            labels["bi_direction"] = 1 if bi.is_up() else -1
            try:
                labels["bi_strength"] = bi.amp() / max(bi.get_end_val(), 1e-10)
            except Exception:
                pass
            break

    # ── 中枢 (Zhongshu) 计数 ─────────────────────
    # 【已废弃】zs_count 是从数据起点累计的，模型只有 60 根视野，无法学习。
    # 保留此字段仅用于数据分析，不作为训练目标。

    return labels


def generate_dataset(
    symbol: str,
    timeframe: str,
    seq_len: int = 60,
    stride: int = 10,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    为单个（币种，周期）生成标注数据集。

    参数
    ----
    symbol      : "BTCUSDT"
    timeframe   : "1h"
    seq_len     : 每个样本包含的 K 线数（默认 60）
    stride      : 滑动步长
    max_samples : 最大样本数

    返回
    ----
    DataFrame 列：
        timestamp: 当前 K 线时间
        klines: 60×5 数组（OHLCV，归一化）
        has_top_fractal, has_bottom_fractal, ...
        next_return: 未来 N 根 K 线收益率
    """
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    csv_path = processed_dir / f"{symbol}_{timeframe}.csv"

    if not csv_path.exists():
        print(f"  ⏭️  跳过 {symbol} {timeframe}: 文件不存在")
        return pd.DataFrame()

    # 读取全量 K 线
    df = pd.read_csv(csv_path, header=None,
                     names=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    total = len(df)
    print(f"\n  📊 {symbol} {timeframe}: {total} 根 K 线 ({total/60/24:.1f} 天)")

    # 注册 CSV 路径
    CSVTrainingAPI.set_csv_override(symbol, str(csv_path))

    # 创建 CChan
    kl_type = TF_MAP[timeframe]
    config = CChanConfig({
        "trigger_step": True, "bi_strict": False, "bi_fx_check": "half",
        "seg_algo": "chan", "divergence_rate": 0.8, "min_zs_cnt": 1,
        "bs_type": "1,1p,2,2s,3a,3b", "print_warning": False,
    })
    chan = CChan(
        code=symbol, begin_time=None, end_time=None,
        data_src="custom:CSVTrainingAPI.CSVTrainingAPI",
        lv_list=[kl_type], config=config, autype=AUTYPE.QFQ,
    )

    # 逐步回放，采集标注
    # ── V2 标签预计算：ATR 序列 ──────────────────────
    closes_arr = df["close"].values.astype(np.float64)
    highs_arr = df["high"].values.astype(np.float64)
    lows_arr = df["low"].values.astype(np.float64)
    atr_values = _calc_atr_series(highs_arr, lows_arr, closes_arr, period=14)
    # 滚动均值作为波动率基线（无未来信息泄漏）
    # 每个位置 i 只用到 0..i-1 的数据
    cumsum = np.cumsum(atr_values)
    expanding_mean = cumsum / (np.arange(len(atr_values), dtype=np.float64) + 1.0)
    atr_baseline = np.empty_like(atr_values)
    atr_baseline[0] = atr_values[0]
    atr_baseline[1:] = expanding_mean[:-1]
    v2_cfg = V2_LABELING_CFG

    records = []
    idx_to_record: Dict[int, Dict] = {}  # raw K-line idx → record dict（用于回溯更新分型）
    seen_bsp_set = set()
    step = 0

    # ── Level-2 区间套：已完成的 Level-1 笔构成 Level-2 K 线 ──
    completed_l1_bi: List[Dict] = []  # {begin_idx, end_idx, open, high, low, close, dir, strength}
    prev_bi_count = 0
    # Level-2 标签回溯：bi_idx → {has_top, has_bottom}
    l2_fractal_updates: Dict[int, Dict] = {}

    for snap in chan.step_load():
        step += 1
        if step < seq_len:
            continue
        # 滑动步长采样：只取每 stride 根 K 线生成一条记录
        if (step - seq_len) % stride != 0:
            continue

        # 当前 K 线
        if step - 1 >= total:
            continue
        cur_row = df.iloc[step - 1]
        idx = step - 1

        # 检测当前 K 线位置是否本身就是 BSP
        is_bsp = 0
        bsp_direction = 0
        bsp_types = []
        try:
            latest = snap.get_latest_bsp(number=1)
            if latest and latest[0].klu and latest[0].klu.idx == idx:
                bsp = latest[0]
                bsp_key = (bsp.klu.idx, bsp.is_buy)
                if bsp_key not in seen_bsp_set:
                    seen_bsp_set.add(bsp_key)
                    is_bsp = 1
                    bsp_direction = 1 if bsp.is_buy else 2
                    bsp_types = [t.value for t in bsp.type]
        except Exception:
            pass

        # 分型 / 笔 / 中枢 标签（从 CChan 快照中按位置提取）
        chan_labels = _extract_chan_labels_at_position(snap, idx)
        has_top = chan_labels["has_top_fractal"]
        has_bottom = chan_labels["has_bottom_fractal"]
        bi_dir = chan_labels["bi_direction"]
        bi_str = chan_labels["bi_strength"]
        zs_cnt = chan_labels["zs_count"]

        # ── Level-2 区间套：检测新完成的 Level-1 笔 ────────────
        try:
            kl_list = snap[0]
            if kl_list:
                curr_bi_count = len(list(kl_list.bi_list))
                if curr_bi_count > prev_bi_count:
                    # 有新笔完成：收集上次到这次之间新增的笔
                    for bi_idx in range(prev_bi_count, curr_bi_count):
                        bi = list(kl_list.bi_list)[bi_idx]
                        # 每根笔映射为 Level-2 K 线
                        begin_klu = bi.get_begin_klu()
                        end_klu = bi.get_end_klu()
                        amp = bi.amp()
                        end_val = bi.get_end_val()
                        completed_l1_bi.append({
                            "begin_idx": begin_klu.idx,
                            "end_idx": end_klu.idx,
                            "open": bi.get_begin_val(),
                            "high": bi._high(),
                            "low": bi._low(),
                            "close": end_val,
                            "dir": 1 if bi.is_up() else -1,
                            "strength": amp / max(end_val, 1e-10),
                        })
                    prev_bi_count = curr_bi_count

                # Level-2 分型检测：3 笔一组，中间那笔是分型候选
                # 用 lst[-2] 一样的滞后逻辑：第 N 次检测确认第 N-1 根 L2-K 线
                if len(completed_l1_bi) >= 3:
                    center = completed_l1_bi[-2]
                    left = completed_l1_bi[-3]
                    right = completed_l1_bi[-1]
                    is_top = center["high"] > left["high"] and center["high"] > right["high"]
                    is_bot = center["low"] < left["low"] and center["low"] < right["low"]
                    l2_bi_idx = len(completed_l1_bi) - 2
                    l2_fractal_updates[l2_bi_idx] = {
                        "has_top": 1 if is_top else 0,
                        "has_bottom": 1 if is_bot else 0,
                    }
        except Exception:
            pass

        # 初始化 Level-2 标签（当前不确认，靠回溯更新）
        has_top_l2 = 0
        has_bottom_l2 = 0
        bi_dir_l2 = 0
        bi_str_l2 = 0.0

        # ── 分型回溯更新 ────────────────────────────────
        # update_fx 在 kl_list.lst[-2] 上被调用（滞后 1+ 步），
        # 所以当前步不能知道当前 idx 的分型状态。
        # 但我们可以检查 lst[-2]（刚被评估的那根合并 K 线），
        # 如果它有分型，回去更新对应 raw K-line 的记录。
        try:
            kl_list = snap[0]
            if kl_list and len(kl_list.lst) >= 3:
                center_klc = kl_list.lst[-2]
                if center_klc.fx == FX_TYPE.TOP:
                    for klu in center_klc.lst:
                        if klu.idx in idx_to_record:
                            idx_to_record[klu.idx]["has_top_fractal"] = 1
                elif center_klc.fx == FX_TYPE.BOTTOM:
                    for klu in center_klc.lst:
                        if klu.idx in idx_to_record:
                            idx_to_record[klu.idx]["has_bottom_fractal"] = 1
        except Exception:
            pass

        # ── Level-2 回溯更新 ────────────────────────────
        # 当有新 Level-2 分型确认时，回溯更新对应 raw K 线的记录
        for l2_bi_idx, fx_info in l2_fractal_updates.items():
            if l2_bi_idx < len(completed_l1_bi):
                bi = completed_l1_bi[l2_bi_idx]
                for rk_idx in range(bi["begin_idx"], bi["end_idx"] + 1):
                    if rk_idx in idx_to_record:
                        rec = idx_to_record[rk_idx]
                        if fx_info["has_top"]:
                            rec["has_top_fractal_L2"] = 1
                        if fx_info["has_bottom"]:
                            rec["has_bottom_fractal_L2"] = 1
        # 清理已处理的 Level-2 分型更新（只处理一次）
        if l2_fractal_updates:
            l2_fractal_updates.clear()

        # ── 当前 K 线的 Level-2 标签（找它落在哪根 Level-2 K 线内）──
        for l2_bi in reversed(completed_l1_bi):
            if l2_bi["begin_idx"] <= idx <= l2_bi["end_idx"]:
                has_top_l2 = 0
                has_bottom_l2 = 0
                bi_dir_l2 = l2_bi["dir"]
                bi_str_l2 = l2_bi["strength"]
                # 检查当前 Level-2 K 线是否已被确认为分型
                break

        # ── 多级别 K 线序列（主级别 OHLCV + 子级别缠论特征）───
        start = max(0, idx - seq_len + 1)
        klines_main = df.iloc[start:idx + 1][["open", "high", "low", "close", "volume"]].values.astype(np.float32)

        # 归一化基准（用主级别第一根 close）
        base_price = klines_main[0, 3] if klines_main[0, 3] > 0 else 1.0
        klines_main[:, :4] /= base_price
        klines_main[:, 4] /= np.mean(klines_main[:, 4]) + 1e-10

        # K 线序列：主级别 OHLCV → (seq_len, 5)
        # 金字塔 CNN 在模型内部处理多尺度，不需要外部的子级别特征
        klines = klines_main

        # 如果长度不足 seq_len，前补零（正常不会发生）
        if len(klines) < seq_len:
            pad = np.zeros((seq_len, 5), dtype=np.float32)
            pad[-len(klines):] = klines
            klines = pad

        # 未来收益率（V2 动态 ATR 窗口，替代固定 12-bar）
        future_return = _compute_v2_return(idx, closes_arr, highs_arr, lows_arr,
                                           atr_values, atr_baseline, v2_cfg)

        rec = {
            "timestamp": str(cur_row["timestamp"]),
            "klines": klines.tobytes(),
            "klines_shape": klines.shape,
            "has_top_fractal": has_top,
            "has_bottom_fractal": has_bottom,
            "bi_direction": bi_dir,
            "bi_strength": bi_str,
            "zs_count": zs_cnt,
            "is_bsp": is_bsp,
            "bsp_direction": bsp_direction,
            "bsp_types": json.dumps(bsp_types),
            "next_return": future_return,
            "has_top_fractal_L2": has_top_l2,
            "has_bottom_fractal_L2": has_bottom_l2,
            "bi_direction_L2": bi_dir_l2,
            "bi_strength_L2": bi_str_l2,
        }
        records.append(rec)
        idx_to_record[idx] = rec  # 注册，让分型回溯更新能找到这条记录

        if max_samples and len(records) >= max_samples:
            break

        if len(records) % 500 == 0:
            print(f"     ... {len(records)} 样本")

    result = pd.DataFrame(records)
    print(f"  ✅ {len(result)} 样本")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成缠论标注数据集")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1m"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-samples", type=int, default=None, help="每个文件最多样本数")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "labels"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("缠论标注数据生成")
    print(f"序列长度: {args.seq_len}, 步长: {args.stride}")
    print(f"币种: {', '.join(args.symbols)}")
    print(f"周期: {', '.join(args.timeframes)}")
    print("=" * 50)

    total = 0
    for sym in args.symbols:
        for tf in args.timeframes:
            df = generate_dataset(
                symbol=sym,
                timeframe=tf,
                seq_len=args.seq_len,
                stride=args.stride,
                max_samples=args.max_samples,
            )
            if df.empty:
                continue
            out_path = output_dir / f"{sym}_{tf}.parquet"
            df.to_parquet(out_path, index=False)
            total += len(df)
            print(f"  💾 保存: {out_path}")

    print(f"\n{'='*50}")
    print(f"总计: {total} 样本")
    print(f"输出: {output_dir}/")


if __name__ == "__main__":
    main()
