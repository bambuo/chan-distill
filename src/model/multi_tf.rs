//! 时序金字塔模型 + 不确定性加权损失
//!
//! 每个输出头有一个可学习的 log_var 参数，训练过程中自动平衡各任务权重。

use candle_core::{DType, Module, Result, Tensor, Var};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, VarBuilder};

const D_MODEL: usize = 128;

/// 默认金字塔步长
pub const DEFAULT_STRIDES: [usize; 7] = [5, 3, 2, 2, 4, 6, 7];

#[allow(dead_code)]
fn channel_for(level: usize) -> usize {
    match level {
        0 => 5, 1 => 32, 2 => 48, 3 => 64,
        4 => 80, 5 => 96, 6 => 112, _ => D_MODEL,
    }
}

pub fn head_count(strides: &[usize]) -> usize { 8 + strides.len() * 4 }

/// 时序金字塔模型（含可学习 log_vars 用于不确定性加权）
#[allow(dead_code)]
pub struct MultiTfModel {
    convs: Vec<Vec<Conv1d>>,
    heads: Vec<Linear>,
    pub log_vars: Vec<Var>,  // 可学习的任务权重参数
    strides: Vec<usize>,
}

#[allow(dead_code)]
impl MultiTfModel {
    pub fn new(strides: &[usize], vb: &VarBuilder) -> Result<Self> {
        let mut convs = Vec::with_capacity(strides.len());
        let mut in_c = 5;

        for (li, &stride) in strides.iter().enumerate() {
            let (mid_c, out_c) = match li {
                0 => (32, 48),
                1 => (48, D_MODEL),
                _ => (D_MODEL, D_MODEL),
            };

            let k1 = if stride >= 5 { 7 } else if stride >= 3 { 5 } else { 3 };
            let c1 = conv1d(in_c, mid_c, k1,
                Conv1dConfig { padding: k1 / 2, stride, ..Default::default() },
                vb.pp(&format!("c{}a", li)))?;
            let c2 = conv1d(mid_c, out_c, 5,
                Conv1dConfig { padding: 2, ..Default::default() },
                vb.pp(&format!("c{}b", li)))?;
            convs.push(vec![c1, c2]);
            in_c = out_c;  // 下一层的输入 = 本层输出
        }

        let n_heads = head_count(strides);
        let mut heads = Vec::with_capacity(n_heads);
        heads.push(linear(D_MODEL, 3, vb.pp("h0"))?);
        let h1_cfg: [(usize, &str); 7] = [
            (1,"h1tp"),(1,"h1bt"),(3,"h1bd"),(1,"h1bs"),(1,"h1zs"),(1,"h1bp"),(3,"h1bpd")];
        for (d, n) in &h1_cfg { heads.push(linear(D_MODEL, *d, vb.pp(n))?); }
        for li in 0..strides.len() {
            let p = format!("h{}", 8 + li * 4);
            heads.push(linear(D_MODEL, 1, vb.pp(&format!("{}tp",p)))?);
            heads.push(linear(D_MODEL, 1, vb.pp(&format!("{}bt",p)))?);
            heads.push(linear(D_MODEL, 3, vb.pp(&format!("{}bd",p)))?);
            heads.push(linear(D_MODEL, 1, vb.pp(&format!("{}zs",p)))?);
        }

        // 可学习的 log_var 参数：不指定 init，vb.get 会随机初始化
        let mut log_vars = Vec::with_capacity(n_heads);
        for i in 0..n_heads {
            log_vars.push(Var::from_tensor(&vb.get(&[1], &format!("lv{}", i))?)?);
        }

        Ok(Self { convs, heads, log_vars, strides: strides.to_vec() })
    }

    /// 前向: (B, T, 5) → Vec<Tensor>[N]
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = x.permute((0, 2, 1))?;
        let mut tlen = h.dim(2)?;
        for (li, layer) in self.convs.iter().enumerate() {
            let _ = li;
            if (tlen as i64) < 7 { break; }
            h = layer[0].forward(&h)?.relu()?;
            h = layer[1].forward(&h)?.relu()?;
            tlen = h.dim(2)?;
            if tlen <= 1 { break; }
        }
        let h = h.mean(2)?;

        let n_heads = head_count(&self.strides);
        let mut out = Vec::with_capacity(n_heads);
        // heads 0..n_heads 正常输出
        for i in 0..n_heads.min(self.heads.len()) {
            out.push(self.heads[i].forward(&h)?);
        }
        // 剩余充零
        while out.len() < n_heads {
            out.push(Tensor::zeros((x.dim(0)?, 1), DType::F32, x.device())?);
        }
        Ok(out)
    }

    /// 返回 log_vars 的值（用于损失计算）
    pub fn log_var_values(&self) -> Result<Vec<f64>> {
        self.log_vars.iter().map(|v| {
            let val: f32 = v.as_tensor().squeeze(0)?.to_scalar()?;
            Ok(val as f64)
        }).collect()
    }
}

// ─── 不确定性加权损失 ─────────────────────────────

/// 不确定性加权损失
///
/// 每个头 i 的可学习 log_var 控制其损失权重:
///   L_total = Σ (L_i * exp(-log_var_i) + 0.5 * log_var_i)
///
/// 噪声大的任务 log_var 自然上升，权重自动降低。
pub fn loss_uncertainty(
    preds: &[Tensor],
    targets_f32: &[&Tensor],
    targets_i64: &[&Tensor],
    log_vars: &[f64],
) -> Result<Tensor> {
    let n = preds.len().min(log_vars.len()).min(targets_f32.len()).min(targets_i64.len());
    let mut total: Option<Tensor> = None;

    for i in 0..n {
        let (typ, var): (u8, f64) = match i {
            0 | 3 | 7 | 10 | 14 | 18 | 22 | 26 | 30 | 34 => (0, log_vars[i]),
            1 | 2 | 6 | 8 | 9 | 12 | 13 | 16 | 17 | 20 | 21 | 24 | 25 | 28 | 29 | 32 | 33 => (1, log_vars[i]),
            _ => (2, log_vars[i]),
        };

        let loss_base = match typ {
            0 => candle_nn::loss::cross_entropy(&preds[i], targets_i64[i])?,
            1 => candle_nn::loss::binary_cross_entropy_with_logit(&preds[i], targets_f32[i])?,
            2 => candle_nn::loss::mse(&preds[i], &targets_f32[i].reshape(((), 1))?)?,
            _ => continue,
        };

        // 不确定性加权: L_i * exp(-log_var) + 0.5 * log_var
        // 不确定性加权: L_i * exp(-log_var) + 0.5 * log_var
        let w = var.exp();
        let r = var * 0.5;
        let term = ((loss_base * w)? + r as f64)?;
        total = Some(match total { None => term, Some(t) => (t + term)? });
    }

    Ok(total.unwrap_or(Tensor::zeros((1,), DType::F32, &candle_core::Device::Cpu)?))
}
