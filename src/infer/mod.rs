//! 金字塔模型导出与推理
//!
//! 支持两种推理后端：
//! - **Candle 原生**：加载 `.safetensors` 权重
//! - **ONNX Runtime**：加载 `.onnx` 文件
//!
//! 价格标准化（除以窗口最后一根 close）与训练时一致。

pub mod candle;
pub mod onnx;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::model::multi_tf::{MultiTfModel, head_count, DEFAULT_STRIDES};
use candle::run_candle_infer;
use onnx::run_ort_infer;

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
pub struct OhlcvRecord {
    pub dt: String,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub vol: f32,
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
pub fn normalize_window(records: &[OhlcvRecord], start: usize, end: usize, seq_len: usize) -> Vec<f32> {
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
