# Chan-Distill RS

**时序金字塔 + 缠论多周期标注 —— 纯 Rust 端到端缠论辅助学习系统**

用缠论标注提供结构化先验，1D-CNN 从原始 1m K 线学习价格模式，同时输出 8 个时间周期的缠论结构（分型、笔、中枢、买卖点）与涨跌预测信号。

> 这不是"蒸馏"（不追求模型替代缠论），而是"缠论辅助学习"——缠论标注提供结构先验，主任务（涨跌预测）提供密集训练信号。

---

## 目录

- [安装](#安装)
- [数据格式](#数据格式)
- [工作流程](#工作流程)
- [标注](#1-标注-label)
- [训练](#2-训练-train)
- [导出](#3-导出-export)
- [推理](#4-推理-infer)
- [模型架构](#模型架构)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 安装

需要 Rust 工具链 ≥ 1.80：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

克隆并编译：

```bash
git clone <repo-url> chan-distill-rs
cd chan-distill-rs

# 编译（--release 约 5 分钟）
cargo build --release
```

编译产物：`./target/release/chan_distill_rs`

---

## 数据格式

### 输入 CSV（1m K 线）

无表头，6 列，时间升序：

```
2023-01-01 00:00:00,16541.77,16544.76,16538.45,16543.67,83.08
2023-01-01 00:01:00,16543.04,16544.41,16538.48,16539.31,80.45
```

| 列 | 内容 | 示例 |
|----|------|------|
| 1 | 时间戳 | `2023-01-01 00:00:00` |
| 2 | 开盘价 Open | `16541.77` |
| 3 | 最高价 High | `16544.76` |
| 4 | 最低价 Low | `16538.45` |
| 5 | 收盘价 Close | `16543.67` |
| 6 | 成交量 Vol | `83.08` |

支持的时间格式：
- `%Y-%m-%d %H:%M:%S`
- ISO 8601 (`2024-01-01T00:00:00Z`)
- Unix 秒时间戳（10 位）
- Unix 毫秒时间戳（13 位）

有表头的 CSV 也可以，标注时不要加 `--no-header` 即可。

### 输出 Parquet（44 列）

标注后生成 44 列 Parquet 文件：

| 列范围 | 内容 | 例 |
|--------|------|----|
| `dt`–`amount` | OHLCV 原始行情（7 列: dt, open, close, high, low, vol, amount） | |
| `1m_has_top`–`1w_zs` | 8 周期 × 4 缠论字段 | `1m_has_top: 0/1` |
| `1m_bi_str` | 1m 笔强度 | |
| `1m_is_bsp` | 1m 买卖点标志 | |
| `1m_bsp_dir` | 1m 买卖方向 | 1=买, 2=卖 |
| `next_return` | 未来收益率 | |
| `return_label` | 涨跌分类 | 0=跌, 1=平, 2=涨 |

---

## 工作流程

```
1m CSV → label → 44 列 Parquet → train → safetensors → export → ONNX → infer → JSONL 信号
```

---

## 1. 标注 `label`

从 1m K 线生成 8 周期缠论标注。

```bash
# 基本用法（无表头 CSV，60 根 K 线序列）
./target/release/chan_distill_rs label \
    -d data/processed/BTCUSDT_1m.csv \
    -o data/labels/BTCUSDT_1m.parquet \
    --no-header \
    --seq-len 60

# 7 天序列（覆盖 1m→1w 全尺度）
./target/release/chan_distill_rs label \
    -d data/processed/BTCUSDT_1m.csv \
    -o data/labels/BTCUSDT_1m_full.parquet \
    --no-header \
    --seq-len 10080
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d, --data` | 必填 | 输入 CSV 路径 |
| `-o, --output` | 必填 | 输出 Parquet 路径 |
| `--no-header` | — | CSV 无表头（固定列序: dt,open,high,low,close,vol） |
| `--seq-len` | `60` | 序列长度（控制 lookahead = seq_len/5） |
| `--lookahead` | seq_len/5 | 向前看多少根 K 线算未来收益 |
| `--fx-threshold` | `50` | CZSC 最大笔数 |
| `--ret-threshold` | 自动（波动率自适应） | 涨跌分类阈值（百分比） |
| `--freq` | `1h` | K 线周期（仅用于日志，不影响标注逻辑） |

### 标注 179 万条 1m 数据的时间参考

| seq_len | 数据量 | 时间 |
|---------|--------|------|
| 60 | 5,000 行 | < 1s |
| 60 | 200,000 行 | ~10s |
| 10080 | 200,000 行 | ~30s |
| 10080 | 1,800,000 行 | ~5m |

---

## 2. 训练 `train`

使用时序金字塔模型训练。

```bash
# 训练（自动检测 44 列 → 金字塔模型）
./target/release/chan_distill_rs train \
    -d "data/labels/*.parquet" \
    --seq-len 60 \
    --epochs 50 \
    --batch-size 64
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d, --data` | 必填 | Parquet 文件 glob 模式 |
| `--seq-len` | `60` | 序列长度（须与 label 一致） |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `64` | 批次大小 |
| `--learning-rate` | `0.001` | 学习率 |
| `-o, --output-dir` | `checkpoints` | 模型保存目录 |
| `--val-split` | `0.1` | 验证集比例 |
| `--patience` | `10` | Early stopping 耐心值 |
| `--seed` | `42` | 随机种子 |

### 损失函数

所有输出头使用**不确定性加权**（Uncertainty Weighting）：

```
L_total = Σ (L_i × exp(-log_var_i) + 0.5 × log_var_i)
```

每个任务有一个可学习的 `log_var` 参数，训练过程中自动平衡各任务权重。噪声大的任务权重自动降低。

### 输出

训练完成后在 `--output-dir` 下生成：

| 文件 | 说明 |
|------|------|
| `best.safetensors` | 最佳验证损失 checkpoint |
| `epoch_0010.safetensors` | 每 10 epoch 保存 |

---

## 3. 导出 `export`

验证并导出模型权重（可选，用于备份或迁移）。

```bash
# 验证模型并导出
./target/release/chan_distill_rs export \
    -m checkpoints/best.safetensors \
    -o models/exported.safetensors \
    --seq-len 60
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m, --checkpoint` | 必填 | 输入的 safetensors checkpoint |
| `-o, --output` | — | 输出路径（不指定则只验证不保存） |
| `--seq-len` | `60` | 序列长度（须与训练一致） |

---

## 4. 推理 `infer`

加载训练好的模型，对 1m K 线输出交易信号。支持 Candle 原生（`.safetensors`）和 ONNX Runtime（`.onnx`）两种后端。

```bash
# Candle 原生推理
./target/release/chan_distill_rs infer \
    -m checkpoints/best.safetensors \
    -d data/processed/BTCUSDT_1m.csv \
    --seq-len 60

# 只输出 BSP（买卖点）信号，结果写入文件
./target/release/chan_distill_rs infer \
    -m checkpoints/best.safetensors \
    -d data/processed/BTCUSDT_1m.csv \
    --seq-len 60 \
    --bsp-only \
    -o signals.jsonl

# ONNX Runtime 推理
./target/release/chan_distill_rs infer \
    -m models/model.onnx \
    -d data/processed/BTCUSDT_1m.csv \
    --seq-len 60
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m, --model` | 必填 | 模型路径（.safetensors 或 .onnx） |
| `-d, --data` | 必填 | 输入 CSV 路径（需有表头） |
| `-o, --output` | stdout | 输出 JSONL 文件路径 |
| `--seq-len` | `60` | 序列长度（须与训练一致） |
| `--bsp-only` | — | 只输出 BSP 概率 > 0.5 的信号 |
| `--threshold` | `0.5` | 涨跌信号置信度阈值 |

### 输出格式（JSONL）

```jsonl
{"time":"2023-06-19 12:00:00","direction":1,"confidence":0.72,"is_bsp_prob":0.89}
{"time":"2023-06-19 12:01:00","direction":0,"confidence":0.45,"is_bsp_prob":0.12}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `time` | string | K 线时间戳 |
| `direction` | int | 0=无信号, 1=买入, 2=卖出 |
| `confidence` | float | 置信度 [0, 1] |
| `is_bsp_prob` | float | 买卖点概率 |

---

## 模型架构

### 时序金字塔

7 层 Conv1D，步长序列 `[5, 3, 2, 2, 4, 6, 7]`，每层两级卷积：

```
输入: (B, T, 5) 1m OHLCV
  │
  ├── Level 0: Conv(5→32, k=7, s=5)  → Conv(32→48, k=5)   → (B, 48,  T/5)
  ├── Level 1: Conv(48→48, k=7, s=3) → Conv(48→128, k=5)  → (B, 128, T/15)
  ├── Level 2: Conv(128→128, k=3,s=2)→ Conv(128→128, k=5) → (B, 128, T/30)
  ├── Level 3: Conv(128→128, k=3,s=2)→ Conv(128→128, k=5) → (B, 128, T/60)
  ├── Level 4: Conv(128→128, k=7,s=4)→ Conv(128→128, k=5) → (B, 128, T/240)
  ├── Level 5: Conv(128→128, k=7,s=6)→ Conv(128→128, k=5) → (B, 128, T/1440)
  └── Level 6: Conv(128→128, k=7,s=7)→ Conv(128→128, k=5) → (B, 128, T/10080)
  │
  GAP → (B, 128) → 36 个 Linear 头
```

> kernel 大小根据步长自适应：stride ≥ 5 → k=7，stride ≥ 3 → k=5，否则 k=3。

T=10080 时的降采样路径：

```
10080 → 2016 → 672 → 336 → 168 → 42 → 7 → 1
```

短序列（T < 7）自动截断多余层。

### 36 个输出头

| 索引 | 名称 | 维度 | 类型 | 周期 |
|------|------|------|------|------|
| 0 | return_logits | 3 | CE 分类 | 1m |
| 1 | 1m_has_top | 1 | BCE | 1m |
| 2 | 1m_has_bottom | 1 | BCE | 1m |
| 3 | 1m_bi_dir | 3 | CE | 1m |
| 4 | 1m_bi_str | 1 | MSE 回归 | 1m |
| 5 | 1m_zs | 1 | MSE 回归 | 1m |
| 6 | 1m_is_bsp | 1 | BCE | 1m |
| 7 | 1m_bsp_dir | 3 | CE | 1m |
| 8-11 | 5m_top/btm/bi_dir/zs | 1/1/3/1 | BCE/BCE/CE/MSE | 5m |
| 12-15 | 15m ... | 同上 | | 15m |
| 16-19 | 30m ... | | | 30m |
| 20-23 | 1h ... | | | 1h |
| 24-27 | 4h ... | | | 4h |
| 28-31 | 1d ... | | | 1d |
| 32-35 | 1w ... | | | 1w |

### 三类买卖点检测（标注阶段）

`BspState` 状态机，跨 bar 持久化追踪：

| 类型 | 条件 |
|------|------|
| **一买** | 下跌段背驰（新低 + 力度减弱）+ 反转向上 |
| **二买** | 一买后回拉不破前低 |
| **三买** | 突破中枢后回拉不进入中枢 |
| 卖点对称 | Up/Down 互换 |

---

## 快速示例

从原始数据到交易信号的全流程：

```bash
# 1. 标注
./target/release/chan_distill_rs label \
    -d data/processed/BTCUSDT_1m.csv \
    -o data/labels/BTCUSDT_1m.parquet \
    --no-header --seq-len 10080

# 2. 训练
./target/release/chan_distill_rs train \
    -d data/labels/BTCUSDT_1m.parquet \
    --seq-len 10080 --epochs 30 --batch-size 32

# 3. 导出（可选）
./target/release/chan_distill_rs export \
    -m checkpoints/best.safetensors \
    -o models/best.safetensors \
    --seq-len 10080

# 4. 推理
./target/release/chan_distill_rs infer \
    -m checkpoints/best.safetensors \
    -d data/processed/BTCUSDT_1m.csv \
    --seq-len 10080 -o signals.jsonl
```

---

## 目录结构

```
chan-distill-rs/
├── Cargo.toml
├── src/
│   ├── main.rs                 # CLI 入口（label / train / export / infer）
│   ├── label/
│   │   ├── mod.rs              # 标注主流程 + CLI 参数
│   │   ├── csv_reader.rs       # CSV 读取 + 时间解析 + OHLCV 校验
│   │   ├── czsc_label.rs       # CZSC 多周期管理 + 分型/笔/中枢提取
│   │   ├── bsp.rs              # 三类买卖点状态机 + 中枢分组
│   │   └── parquet_out.rs      # 44 列 Parquet 输出
│   ├── model/
│   │   ├── mod.rs              # pub mod multi_tf
│   │   └── multi_tf.rs         # 时序金字塔模型 + 36 头 + 不确定性加权损失
│   ├── train/
│   │   └── mod.rs              # 金字塔模型训练（AdamW + Early Stopping）
│   └── infer/
│       ├── mod.rs              # 导出 + 推理协调入口
│       ├── candle.rs           # Candle 原生推理（.safetensors）
│       └── onnx.rs             # ONNX Runtime 推理（.onnx）
├── data/
│   ├── processed/              # 原始 1m CSV
│   └── labels/                 # 标注后 Parquet
├── checkpoints/                # 训练 checkpoint
└── models/                     # 导出模型
```

---

## 常见问题

**Q: Metal GPU 报错 "Failed to create metal resource"**

训练默认用 CPU，Metal 对小张量支持不稳定。如需 Metal，设置环境变量：
```bash
RUST_LOG=info ./target/release/chan_distill_rs train ...
```

**Q: 训练很慢**

`--release` 模式是必须的。Debug 模式慢 10-50 倍。如果 CPU 训练太慢，可将 `batch-size` 调到 `16` 或 `8`。

**Q: 标注很慢**

`seq-len` 越大，`lookahead` 越大，标注时间线性增长。10080 的 seq-len 标注 200 万行约 5 分钟。初次测试可以用 `--seq-len 60` 快速验证。

**Q: 怎么跨币种训练**

将多个币种的 Parquet 文件放在同个目录，训练时用 glob 模式：
```bash
./target/release/chan_distill_rs train -d "data/labels/*.parquet" --seq-len 10080
```

所有 OHLCV 数据会在加载时做价格标准化（除以窗口最后一个 close），不同币种自动对齐。

**Q: 怎么导出 ONNX**

Candle 的 `candle-onnx` 不支持导出。如需 ONNX 格式，使用 Python 版加载 safetensors 后导出，或在 Candle 新版本中等待支持。

---

## 技术栈

| 组件 | 选型 |
|------|------|
| 深度学习 | Candle (HuggingFace) v0.10.2 |
| 缠论引擎 | czsc-core v1.0.0-rc.8 |
| 数据格式 | CSV → Parquet (Polars v0.45) |
| 推理 | Candle 原生 (.safetensors) / ONNX Runtime (ort v2.0) |
| GPU | 可选 Metal (default feature) / CUDA |
| Rust | Edition 2024, 工具链 ≥ 1.80 |
