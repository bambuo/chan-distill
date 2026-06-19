//! 金字塔模型导出与推理
//!
//! 支持两种推理后端：
//! - **Candle 原生**：加载 `.safetensors` 权重
//! - **ONNX Runtime**：加载 `.onnx` 文件
//!
//! 价格标准化（除以窗口最后一根 close）与训练时一致。

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Args;

use crate::model::multi_tf::{MultiTfModel, head_count, DEFAULT_STRIDES};

// ─── CLI ──────────────────────────────────────────────

/// 导出 CLI 参数
#[derive(Args, Debug)]
pub struct ExportArgs {
    #[arg(short, long, value_name = "FILE")]
    pub checkpoint: PathBuf,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value_t = 60)]
    pub seq_len: usize,
}

/// 推理 CLI 参数
#[derive(Args, Debug)]
pub struct InferArgs {
    #[arg(short, long, value_name = "FILE")]
    pub model: PathBuf,
    #[arg(short, long, value_name = "FILE")]
    pub data: PathBuf,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub bsp_only: bool,
    #[arg(long, default_value_t = 60)]
    pub seq_len: usize,
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f64,
}

/// 推理输出
#[derive(Debug, serde::Serialize)]
pub struct SignalRecord {
    pub time: String,
    pub direction: i64,
    pub confidence: f64,
    pub is_bsp_prob: f64,
}

// ─── CSV 读取 ─────────────────────────────────────────

#[derive(Debug, Clone)]
struct OhlcvRecord {
    dt: String,
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    vol: f32,
}

fn read_ohlcv_csv(path: &std::path::Path) -> Result<Vec<OhlcvRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .with_context(|| format!("无法打开 CSV: {}", path.display()))?;

    let headers = reader.headers()?.clone();
    let cols: Vec<String> = headers.iter().map(|h| h.to_lowercase()).collect();

    let idx_dt = cols.iter().position(|c| c == "dt").unwrap_or(0);
    let idx_open = find_col(&cols, &["open", "开盘"]).unwrap_or(1);
    let idx_high = find_col(&cols, &["high", "最高"]).unwrap_or(2);
    let idx_low = find_col(&cols, &["low", "最低"]).unwrap_or(3);
    let idx_close = find_col(&cols, &["close", "收盘"]).unwrap_or(4);
    let idx_vol = find_col(&cols, &["vol", "volume", "成交量"]).unwrap_or(5);

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        records.push(OhlcvRecord {
            dt: record.get(idx_dt).unwrap_or("").to_string(),
            open: record.get(idx_open).unwrap_or("0").parse().unwrap_or(0.0),
            high: record.get(idx_high).unwrap_or("0").parse().unwrap_or(0.0),
            low: record.get(idx_low).unwrap_or("0").parse().unwrap_or(0.0),
            close: record.get(idx_close).unwrap_or("0").parse().unwrap_or(0.0),
            vol: record.get(idx_vol).unwrap_or("0").parse().unwrap_or(0.0),
        });
    }

    if records.is_empty() {
        anyhow::bail!("CSV 文件中没有数据");
    }
    Ok(records)
}

fn find_col(cols: &[String], names: &[&str]) -> Option<usize> {
    names.iter().find_map(|n| cols.iter().position(|c| c == *n))
}

// ─── 价格标准化 ─────────────────────────────────────

/// 与训练时 run_multi_tf 的预处理一致
fn normalize_window(records: &[OhlcvRecord], start: usize, end: usize, seq_len: usize) -> Vec<f32> {
    let base_price = records[end].close;
    let base_price = if base_price <= 0.0 { 1.0 } else { base_price };

    let vol_mean: f32 = {
        let sum: f32 = (start..=end).map(|i| records[i].vol).sum();
        sum / seq_len as f32
    };
    let vol_mean = if vol_mean <= 0.0 { 1.0 } else { vol_mean };

    let mut data = Vec::with_capacity(seq_len * 5);
    for i in start..=end {
        let r = &records[i];
        data.push(r.open / base_price);
        data.push(r.high / base_price);
        data.push(r.low / base_price);
        data.push(r.close / base_price);
        data.push(r.vol / vol_mean);
    }
    data
}

// ─── 头索引 ──────────────────────────────────────────

/// 从 36 头中取信号 (hi=0→return_logits, hi=6→is_bsp_raw)
fn extract_signal(outputs: &[Tensor], threshold: f64) -> Result<(i64, f64, f64)> {
    // hi=0: return_logits (B, 3) → softmax → direction + confidence
    let probs = candle_nn::ops::softmax(&outputs[0], 1)?;
    let p_drop: f64 = probs.get(0)?.to_scalar()?;
    let p_rise: f64 = probs.get(2)?.to_scalar()?;

    let (dir, conf) = if p_rise > p_drop && p_rise >= threshold {
        (1_i64, p_rise)
    } else if p_drop > p_rise && p_drop >= threshold {
        (2_i64, p_drop)
    } else {
        (0_i64, p_drop.max(p_rise))
    };

    // hi=6: is_bsp raw logits (B, 1) → sigmoid → prob
    let bsp_logit = &outputs[6];
    let bsp_prob = if bsp_logit.dim(1)? >= 1 {
        let bsp_prob_t = candle_nn::ops::sigmoid(&bsp_logit)?;
        bsp_prob_t.squeeze(1)?.to_vec1::<f32>()?[0] as f64
    } else {
        0.0
    };

    Ok((dir, conf, bsp_prob))
}

// ─── 导出 ─────────────────────────────────────────────

pub fn run_export(args: ExportArgs) -> Result<()> {
    log::info!("导出金字塔模型");
    log::info!("checkpoint: {:?}, seq_len: {}", args.checkpoint, args.seq_len);

    if !args.checkpoint.exists() {
        anyhow::bail!("checkpoint 文件不存在: {}", args.checkpoint.display());
    }

    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    varmap.load(&args.checkpoint)?;
    let model = MultiTfModel::new(&DEFAULT_STRIDES, &vb)?;

    let total_params: usize = varmap.all_vars().iter().map(|v| v.shape().elem_count()).sum();
    log::info!("参数量: {:.1}K", total_params as f64 / 1000.0);
    log::info!("架构: 时序金字塔 {} 层 + {} 头", DEFAULT_STRIDES.len(), head_count(&DEFAULT_STRIDES));
    log::info!("步长: {:?}", DEFAULT_STRIDES);

    // 前向验证
    let sample = Tensor::zeros((1, args.seq_len, 5), DType::F32, &device)?;
    let outputs = model.forward(&sample)?;
    log::info!("前向通过: {} 个输出, 首个 shape {:?}", outputs.len(), outputs[0].shape());

    if let Some(output_path) = &args.output {
        std::fs::create_dir_all(output_path.parent().unwrap_or(std::path::Path::new("")))?;
        varmap.save(output_path)?;
        log::info!("模型已保存到: {}", output_path.display());
    } else {
        log::info!("模型验证通过，跳过保存");
    }

    log::info!("导出完成 ✅");
    Ok(())
}

// ─── 推理入口 ─────────────────────────────────────────

pub fn run_infer(args: InferArgs) -> Result<()> {
    log::info!("加载模型: {:?}", args.model);
    log::info!("数据: {:?}, bsp_only: {}", args.data, args.bsp_only);

    let records = read_ohlcv_csv(&args.data)?;
    let n = records.len();
    if n < args.seq_len {
        anyhow::bail!("数据不足: 需要 >= {} 根 K 线，当前 {} 根", args.seq_len, n);
    }
    log::info!("读取了 {} 根 K 线", n);

    let ext = args.model.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    let signals = match ext.as_str() {
        "safetensors" => run_candle_infer(&args, &records)?,
        "onnx" => run_ort_infer(&args, &records)?,
        _ => anyhow::bail!("不支持: .{ext}，请用 .safetensors 或 .onnx"),
    };

    let filtered: Vec<&SignalRecord> = if args.bsp_only {
        signals.iter().filter(|s| s.is_bsp_prob > 0.5).collect()
    } else {
        signals.iter().collect()
    };

    if let Some(output_path) = &args.output {
        let file = std::fs::File::create(output_path)?;
        let mut w = std::io::BufWriter::new(file);
        for s in &filtered {
            serde_json::to_writer(&mut w, s)?;
            writeln!(w)?;
        }
        log::info!("已输出 {} 条信号到 {}", filtered.len(), output_path.display());
    } else {
        for s in &filtered {
            println!("{}", serde_json::to_string(s)?);
        }
    }

    log::info!("推理完成 ✅");
    Ok(())
}

// ─── Candle 原生推理 ─────────────────────────────────

fn run_candle_infer(args: &InferArgs, records: &[OhlcvRecord]) -> Result<Vec<SignalRecord>> {
    log::info!("使用 Candle 金字塔推理");

    let device = Device::metal_if_available(0).unwrap_or_else(|_| Device::Cpu);
    log::info!("设备: {:?}", device);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    varmap.load(&args.model)?;
    let model = MultiTfModel::new(&DEFAULT_STRIDES, &vb)?;
    let n_heads = head_count(&DEFAULT_STRIDES);

    let n = records.len();
    let mut signals = Vec::with_capacity(n - args.seq_len + 1);

    for end in (args.seq_len - 1)..n {
        let start = end + 1 - args.seq_len;
        let data = normalize_window(records, start, end, args.seq_len);
        let x = Tensor::from_slice(&data, (1, args.seq_len, 5), &device)?;

        let outputs = model.forward(&x)?;
        if outputs.len() < n_heads {
            log::warn!("模型输出不足: 预期 {} 实际 {}", n_heads, outputs.len());
            continue;
        }

        let (direction, confidence, is_bsp_prob) = extract_signal(&outputs, args.threshold)?;

        signals.push(SignalRecord {
            time: records[end].dt.clone(),
            direction,
            confidence,
            is_bsp_prob,
        });
    }

    log::info!("推理了 {} 个窗口", signals.len());
    Ok(signals)
}

// ─── ONNX Runtime 推理 ───────────────────────────────

fn run_ort_infer(args: &InferArgs, records: &[OhlcvRecord]) -> Result<Vec<SignalRecord>> {
    log::info!("使用 ONNX Runtime 推理");

    let mut session = ort::session::Session::builder()?
        .commit_from_file(&args.model)?;

    let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
    log::info!("ONNX 输出: {:?}", output_names);

    let idx_return = output_names.iter().position(|n| n.contains("return_logits"));
    let idx_bsp = output_names.iter().position(|n| n.contains("is_bsp"));

    let n = records.len();
    let mut signals = Vec::with_capacity(n - args.seq_len + 1);

    for end in (args.seq_len - 1)..n {
        let start = end + 1 - args.seq_len;
        let data = normalize_window(records, start, end, args.seq_len);

        let input_tensor = ort::value::Tensor::from_array((
            [1_usize, args.seq_len, 5],
            data,
        ))?;

        let ort_outputs = session.run(ort::inputs![input_tensor])?;

        let (direction, confidence, is_bsp_prob) = if let Some(ret_idx) = idx_return {
            match extract_ort_f32(&ort_outputs, ret_idx) {
                Ok(logits) if logits.len() >= 3 => {
                    let probs = softmax_1d(&logits[..3]);
                    let p_drop = probs[0];
                    let p_rise = probs[2];
                    let thr = args.threshold;

                    let (dir, conf) = if p_rise > p_drop && p_rise >= thr {
                        (1_i64, p_rise)
                    } else if p_drop > p_rise && p_drop >= thr {
                        (2_i64, p_drop)
                    } else {
                        (0_i64, p_drop.max(p_rise))
                    };

                    let bsp = if let Some(bsp_idx) = idx_bsp {
                        extract_ort_f32(&ort_outputs, bsp_idx)
                            .ok().and_then(|v| v.first().copied()).unwrap_or(0.0) as f64
                    } else {
                        0.0
                    };

                    (dir, conf, bsp)
                }
                _ => (0, 0.0, 0.0),
            }
        } else {
            (0, 0.0, 0.0)
        };

        signals.push(SignalRecord {
            time: records[end].dt.clone(),
            direction,
            confidence,
            is_bsp_prob,
        });
    }

    Ok(signals)
}

// ─── 辅助函数 ───────────────────────────────────────

fn extract_ort_f32(outputs: &ort::session::SessionOutputs, index: usize) -> Result<Vec<f32>> {
    let (_shape, data) = outputs[index].try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

fn softmax_1d(x: &[f32]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = x.iter().map(|v| ((v - max) as f64).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / s).collect()
}
