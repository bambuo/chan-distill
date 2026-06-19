//! 金字塔模型训练
//!
//! 使用 Candle 训练时序金字塔模型。
//! 自动利用 Metal（macOS MPS）进行 GPU 加速。
//!
//! 自动检测 Parquet 列数，≥40 列使用多周期金字塔模型。

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use clap::Args;
use rand::Rng;

use crate::model::multi_tf::{MultiTfModel, loss_uncertainty, head_count, DEFAULT_STRIDES};

// ─── CLI ──────────────────────────────────────────────

/// 训练 CLI 参数
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// 标注数据路径（支持 glob 模式）
    #[arg(short, long, value_name = "GLOB")]
    pub data: String,

    /// 训练轮数
    #[arg(short, long, default_value_t = 50)]
    pub epochs: usize,

    /// 批次大小
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// 学习率
    #[arg(long, default_value_t = 1e-3)]
    pub learning_rate: f64,

    /// 序列长度
    #[arg(long, default_value_t = 60)]
    pub seq_len: usize,

    /// 输出目录
    #[arg(short, long, default_value = "checkpoints")]
    pub output_dir: PathBuf,

    /// 验证集比例
    #[arg(long, default_value_t = 0.1)]
    pub val_split: f64,

    /// 随机种子
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Early stopping 耐心值
    #[arg(long, default_value_t = 10)]
    pub patience: usize,
}

// ─── 数据加载 ─────────────────────────────────────────

use polars::prelude::SerReader;

/// 检测 Parquet 列数
fn detect_column_count(glob_pattern: &str) -> Result<usize> {
    let path = glob::glob(glob_pattern)?
        .next()
        .context("找不到 Parquet 文件")??;
    let file = std::fs::File::open(&path)?;
    let df = polars::prelude::ParquetReader::new(file).finish()?;
    Ok(df.width())
}

// ─── 训练循环 ─────────────────────────────────────────

/// 运行训练（自动检测列数选模式）
pub fn run(args: TrainArgs) -> Result<()> {
    let ncols = detect_column_count(&args.data)?;
    if ncols < 40 {
        anyhow::bail!("列数不足: 检测到 {} 列，需要 ≥40 列（多周期标注）", ncols);
    }
    log::info!("检测到 {} 列 → 使用时序金字塔模型", ncols);
    run_multi_tf(args)
}

// ─── 金字塔模型训练 ────────────────────────────────

fn run_multi_tf(args: TrainArgs) -> Result<()> {
    log::info!("开始训练 🚀");
    log::info!("配置: epochs={}, batch_size={}, lr={}, seq_len={}",
        args.epochs, args.batch_size, args.learning_rate, args.seq_len);

    std::fs::create_dir_all(&args.output_dir)?;

    // 1. 加载 44 列 → 平坦数组
    let paths: Vec<_> = glob::glob(&args.data)?.filter_map(|p| p.ok()).collect();
    if paths.is_empty() { anyhow::bail!("找不到 Parquet 文件: {}", args.data); }

    let mut all_data: Vec<[f64; 44]> = Vec::new();
    for path in &paths {
        log::info!("加载: {}", path.display());
        let file = std::fs::File::open(path)?;
        let df = polars::prelude::ParquetReader::new(file).finish()?;
        let n = df.height();
        let col_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        for row in 0..n {
            let mut row_data = [0.0_f64; 44];
            for (ci, name) in col_names.iter().enumerate() {
                if ci >= 44 { break; }
                if let Ok(ca) = df.column(name).and_then(|c| c.f64()) {
                    row_data[ci] = ca.get(row).unwrap_or(0.0);
                }
            }
            all_data.push(row_data);
        }
    }
    let n = all_data.len();
    if n < args.seq_len { anyhow::bail!("数据不足"); }

    // 2. 窗索引
    let indices: Vec<usize> = (args.seq_len - 1..n).collect();
    let n_val = (indices.len() as f64 * args.val_split).round() as usize;
    let n_val = n_val.max(1);
    let n_train = indices.len() - n_val;
    log::info!("{} 样本, {} 训练, {} 验证", indices.len(), n_train, n_val);

    // 3. 设备 + 模型
    let device = Device::Cpu;
    log::info!("设备: {:?}", device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MultiTfModel::new(&DEFAULT_STRIDES, &vb)?;
    let n_heads = head_count(&DEFAULT_STRIDES);
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), args.learning_rate)?;

    // 4. 训练循环
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;

    for epoch in 0..args.epochs {
        let mut train_loss = 0.0_f64;
        let mut n_batches = 0;

        let mut order: Vec<usize> = (0..n_train).collect();
        for i in (1..order.len()).rev() {
            let j = rand::thread_rng().gen_range(0..=i);
            order.swap(i, j);
        }

        for ch in order.chunks(args.batch_size) {
            let bs = ch.len();
            let seq = args.seq_len;

            let mut xs = Vec::with_capacity(bs * seq * 5);
            let mut target_f32: Vec<Vec<f32>> = (0..n_heads).map(|_| Vec::with_capacity(bs)).collect();
            let mut target_i64: Vec<Vec<i64>> = (0..n_heads).map(|_| Vec::with_capacity(bs)).collect();

            for &idx in ch {
                let end = indices[idx];
                let start = end + 1 - seq;
                let base = all_data[end];

                // 价格标准化
                let base_price = all_data[end][2] as f32;
                let base_price = if base_price <= 0.0 { 1.0 } else { base_price };
                let vol_sum: f32 = (start..=end).map(|i| all_data[i][5] as f32).sum();
                let vol_mean = vol_sum / seq as f32;
                let vol_mean = if vol_mean <= 0.0 { 1.0 } else { vol_mean };

                for i in start..=end {
                    let r = &all_data[i];
                    xs.push(r[1] as f32 / base_price);
                    xs.push(r[3] as f32 / base_price);
                    xs.push(r[4] as f32 / base_price);
                    xs.push(r[2] as f32 / base_price);
                    xs.push(r[5] as f32 / vol_mean);
                }

                for hi in 0..n_heads {
                    let val = get_mtf_target(&base, hi);
                    match hi {
                        0 | 3 | 7 | 10 | 14 | 18 | 22 | 26 | 30 | 34 => target_i64[hi].push(val as i64),
                        _ => target_f32[hi].push(val as f32),
                    }
                }
            }

            let x_t = Tensor::from_slice(&xs, (bs, seq, 5), &device)?;
            let preds = model.forward(&x_t)?;

            let tf32: Vec<Tensor> = target_f32.iter()
                .map(|v| Tensor::from_slice(v, bs, &device).unwrap())
                .collect();
            let ti64: Vec<Tensor> = target_i64.iter()
                .map(|v| Tensor::from_slice(v, bs, &device).unwrap())
                .collect();

            let tf32_refs: Vec<&Tensor> = tf32.iter().collect();
            let ti64_refs: Vec<&Tensor> = ti64.iter().collect();

            let loss = loss_uncertainty(&preds, &tf32_refs, &ti64_refs, &model.log_var_values()?)?;
            optimizer.backward_step(&loss)?;

            train_loss += loss.to_scalar::<f64>()?;
            n_batches += 1;
        }

        let avg_loss = train_loss / n_batches as f64;
        log::info!("Epoch {:3}/{} | loss={:.4}", epoch + 1, args.epochs, avg_loss);

        if avg_loss < best_loss {
            best_loss = avg_loss;
            best_epoch = epoch;
            varmap.save(&args.output_dir.join("best.safetensors"))?;
        }

        if epoch - best_epoch > args.patience {
            log::info!("Early stop at epoch {}", epoch + 1);
            break;
        }
    }

    log::info!("训练完成 ✅ 最佳 loss={:.4} epoch={}", best_loss, best_epoch + 1);
    Ok(())
}

/// 从 44 列中提取第 hi 个目标的值
fn get_mtf_target(row: &[f64; 44], hi: usize) -> f64 {
    match hi {
        0 => row[43],        // return_logits ← return_label
        1 => row[7],         // 1m_has_top
        2 => row[8],         // 1m_has_bottom
        3 => row[9],         // 1m_bi_dir
        4 => row[39],        // 1m_bi_str
        5 => row[10],        // 1m_zs
        6 => row[40],        // 1m_is_bsp
        7 => row[41],        // 1m_bsp_dir
        8..=35 => {
            let tf_offset = (hi - 8) / 4;
            let field = (hi - 8) % 4;
            let col = 11 + tf_offset * 4 + field;
            if col < 39 { row[col] } else { 0.0 }
        }
        _ => 0.0,
    }
}
