//! ONNX Runtime 推理（.onnx）

use anyhow::Result;

use super::{InferArgs, OhlcvRecord, SignalRecord, normalize_window};

/// ONNX Runtime 推理
pub fn run_ort_infer(args: &InferArgs, records: &[OhlcvRecord]) -> Result<Vec<SignalRecord>> {
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
