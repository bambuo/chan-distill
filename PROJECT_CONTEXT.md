# Chan-Distill 项目上下文

> 本文档写给下一个接手这个项目的 AI 助手或开发者，
> 说明来龙去脉、当前状态、待解决的问题。

---

## 一、这个项目是干什么的

**一句话定位：**
把缠论（Chan Theory）蒸馏进一个神经网络，实现**单个 ONNX 模型、输入 K 线、输出交易信号**，推理时不需要依赖任何外部库（不需要 chan.py、不需要 XGBoost、不需要特征工程）。

**对比同目录上级项目 chan-xgb：**

| 维度 | chan-xgb（已上线） | chan-distill（本项目） |
|------|-------------------|----------------------|
| 方法 | chan.py + XGBoost | 1D-CNN 多任务学习 |
| 推理依赖 | 需要 chan.py 侧车算 19 个缠论特征 | 一个 ONNX 文件搞定 |
| AUC | 0.94 | ~0.16（开发中） |
| 状态 | ✅ 可上线 | 🔄 研发中 |

用户最终想要的是：**原始订单流进去，交易信号出来，中间什么都不需要管。** chan-xgb 做不到这个（必须有 Python 侧车跑缠论），chan-distill 的目标就是做到。

---

## 二、为什么会有这个项目

### 用户的需求演变

1. **最初**：用 chan.py + XGBoost 训练数字货币信号模型
2. **做出 chan-xgb**：61 维特征 + Walk-Forward + Ensemble，AUC 0.94
3. **发现问题**：这个模型推理时必须依赖 Python 侧车（chan.py）算 19 个缠论特征，外部服务没法直接拿 K 线推理
4. **提出新需求**：能不能做一个端到端模型，订单进去信号出来，不依赖任何外部库？
5. **成立 chan-distill**：把缠论蒸馏进神经网络

### 讨论过程中的关键认知

- 模型不能直接吃原始订单，必须先聚合为 K 线（这个前提讨论了好几轮才确认）
- 1D-CNN / LSTM / Transformer 都能学 K 线形态，但不是所有架构都能导出 ONNX
- 多任务学习（同时学分型、笔、中枢、BSP）理论上能让模型内部形成类缠论的表征
- ONNX 导出在 PyTorch 2.12 上坑很多（dynamo 导出器不兼容 LSTM/Transformer，最终用 1D-CNN + 固定 batch=1 才跑通）

---

## 三、项目结构

```text
chan-distill/
├── PROJECT_CONTEXT.md        # 本文档
├── README.md                 # 使用说明
├── requirements.txt          # torch, onnxruntime, numpy, pandas, scikit-learn, tqdm
│
├── scripts/
│   └── generate_labels.py    # ★ 数据标注管线（复用 chan-xgb 的 chan.py）
│
├── model/
│   └── chandistill.py        # ★ 模型定义（1D-CNN + 8 个多任务输出头）
│
├── train.py                  # ★ 多任务训练脚本（MPS 加速）
├── export_onnx.py            # ONNX 导出（固定 batch=1）
├── infer.py                  # 推理演示（零依赖）
├── live_pipeline.py          # ★ 实时订单流 → K 线 → ONNX → 信号 演示
│
├── data/labels/              # 标注好的训练数据（parquet）
│   ├── BTCUSDT_1h.parquet
│   └── ETHUSDT_1h.parquet
│
├── checkpoints/              # 训练权重
│   ├── latest.pt
│   └── best.pt
│
└── cs_distill.onnx           # 导出的 ONNX 模型
```

---

## 四、当前状态（2026-06-19）

### 数据

- 生成脚本：`scripts/generate_labels.py`
- 用 chan-xgb 已有的 `data/processed/` CSV 数据，通过 chan.py `step_load` 逐 K 线回放打标签
- 当前训练数据：**BTCUSDT + ETHUSDT 1h，各 ~30k 条，共 ~6 万条**
- BSP 占比：约 5.3%（与 chan-xgb 一致）

### 模型

- 架构：3 层 1D-CNN → 全局平均池化 → 8 个输出头
- 参数量：40,910（d_model=64） / 80,350（d_model=128）
- 输出头：
  - 主任务：未来涨跌方向（跌/平/涨，3 分类）
  - 辅助：顶分型检测（二分类）— **当前标注有问题，一直是 0**
  - 辅助：底分型检测（二分类）— **当前标注有问题，一直是 0**
  - 辅助：笔方向（3 分类）— **当前标注有问题，全是 1**
  - 辅助：笔强度（回归）
  - 辅助：中枢数量（回归）
  - 辅助：BSP 检测（二分类）
  - 辅助：BSP 方向（3 分类）

### 训练结果

| 配置 | BSP AUPRC | 说明 |
|------|----------|------|
| d_model=64, epoch=20, bsp_weight=0.5 | 0.095 | 基线 |
| d_model=64, epoch=30, bsp_weight=2.0 | 0.154 | BSP 加权后提升 |
| d_model=128, epoch=40, bsp_weight=2.0 | 0.158 | 模型翻倍但提升有限 |

**随机基线 AUPRC = 0.053**（正例率 5.3%），当前最好 0.158 是随机的 3 倍，远不够上线。

### ONNX 导出

- 成功导出 `cs_distill.onnx`（33 KB）
- 验证通过：ONNX Runtime 输入 `(1, 60, 5)` → 输出 `signal_direction`, `signal_confidence`, `is_bsp`
- 推理不需要 chan.py / PyTorch / XGBoost

### 已知问题

#### 问题 1：辅助标签质量差（最严重）

`generate_labels.py` 里的分型、笔方向、中枢数量标签提取逻辑有问题：

- `has_top_fractal` 和 `has_bottom_fractal` 一直是 0 → 多任务学习的两大辅助任务形同虚设
- `bi_direction` 一直是 1 → 笔方向没学到
- `zs_count` 虽然有值但区分度低（BSP 0.52 vs 非 BSP 0.30）

**修复方向：** 改写 `generate_labels.py` 中提取这些标签的逻辑，正确从 chan.py 的快照中读取分型、笔、中枢信息。

#### 问题 2：未来收益与 BSP 冲突

BSP 样本的平均 `next_return` = -0.06%，非 BSP 样本 = +0.07%。模型被要求同时做两件矛盾的事：
- "检测 BSP"
- "预测 BSP 后价格上涨"

**修复方向：** 把未来收益的标注逻辑与 chan-xgb 对齐（动态窗口 + ATR 自适应 + 手续费扣除），而不是用固定的 12 根 K 线。

#### 问题 3：模型架构偏简单

4 万参数的 1D-CNN 在 60 根 K 线上学缠论的层级结构能力有限。可以尝试：
- 增加层数和通道数（d_model=256）
- 换成时序融合网络（TCN）
- 或者回到 Transformer（但 ONNX 导出需要另外解决）

#### 问题 4：数据量不够

只用 2 币种 × 1 周期。可以扩充到 5 币种 × 5 周期，样本量从 6 万提升到 50 万+。

---

## 五、与 chan-xgb 的关系

```text
chan-xgb/ （同目录上级项目）
├── 缠论 + XGBoost → 信号评分
├── AUC 0.94 ✅ 可上线
├── 推理需要 Python 侧车跑 chan.py
└── 训练数据可被 chan-distill 复用

chan-distill/ （本项目）
├── 1D-CNN 多任务学习 → 端到端信号
├── AUPRC 0.158 🔄 研发中
├── 推理不需要任何外部依赖
└── generate_labels.py 依赖 chan-xgb 的数据和 chan.py
```

两个项目可以共存，用同一份数据，解决同一个问题，但路线不同。

---

## 六、迭代历史

| 步骤 | 做了什么 | 结果 |
|------|---------|------|
| 1 | 方案设计：Transformer + 8 头多任务 | 架构方案确认 |
| 2 | 创建项目骨架 + 数据标注管线 | BTC/ETH 1h 标注完成 |
| 3 | Transformer 实现 + MPS 训练 | 训练可跑通 |
| 4 | ONNX 导出（Transformer 炸了） | Transformer 内部 reshape ONNX 不兼容 |
| 5 | 换 LSTM（ONNX 又炸了） | LSTM 权重导出有问题 |
| 6 | 换 1D-CNN（ONNX 终于通了） | 33 KB ONNX，推理零依赖 |
| 7 | 发现 BSP AUPRC=0 → 诊断 | 标注数据里 BSP 全是 0 |
| 8 | 修复标注脚本 | BSP 率从 0% → 5.3% |
| 9 | 调参：增大 BSP 权重 | AUPRC 0.095 → 0.158 |
| 10 | 增大模型容量翻倍 | 0.158 几乎没提升 |
| 11 | 诊断 → 发现辅助标签全坏了 | 当前卡在这里 |

---

## 七、下一步建议

### 短期（能快速见效的）

1. **修复辅助标签**：改 `generate_labels.py` 的分型/笔/中枢提取逻辑，让多任务学习真正起作用
2. **扩充数据**：5 币种 × 5 周期都跑一遍标注，样本量提升到 50 万
3. **修复未来收益标注**：与 chan-xgb 对齐，用动态 ATR 窗口而不是固定 12 根

### 中期

4. **模型升级**：更大容量的 1D-CNN（d_model=256）或 TCN，配合更多数据
5. **对比 chan-xgb 的回测**：在同样的历史数据上跑两个模型的信号，对比 AUC

### 长期

6. **接入实时订单流测试**：用 `live_pipeline.py` 接 WebSocket 做实盘信号测试
7. **如果 AUPRC 到 0.6+**：可以考虑替代 chan-xgb 上线

---

## 八、给下一个 AI 助手的提示

- 与用户沟通时注意：用户不是 ML 专家，是交易策略开发者。用大白话解释技术概念。
- 用户有 M4 Pro Mac，支持 MPS 加速，GPU 训练可用。
- 两个项目在桌面 `/Users/johana/Desktop/chan/` 和 `/Users/johana/Desktop/chan-distill/`。
- 当前最大的瓶颈是辅助标签质量，不是模型架构。
- ONNX 导出只能用 1D-CNN（LSTM/Transformer 在 PyTorch 2.12 上导出有问题），除非用户愿意升级/降级 PyTorch 版本。
- 用户之前负责对接的是：自己训练模型，然后交由其他编程语言的技术团队把模型集成到他们的高并发系统中。
