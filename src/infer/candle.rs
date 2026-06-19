//! Candle 原生推理（.safetensors）

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::model::multi_tf::{MultiTfModel, head_count, DEFAULT_STRIDES};
use super::{InferArgs, OhlcvRecord, SignalRecord, normalize_window};

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

/// Candle 原生推理
pub fn run_candle_infer(args: &InferArgs, records: &[OhlcvRecord]) -> Result<Vec<SignalRecord>> {
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
