//! CZSC 标注提取
//!
//! TfCzsc 管理器、分型检测、笔检测、中枢计数、收益率计算

use std::sync::Arc;

use czsc_core::analyze::CZSC;
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::bi::BI;
use czsc_core::objects::direction::Direction;
use czsc_core::objects::freq::Freq;
use czsc_core::objects::fx::FX;
use czsc_core::objects::mark::Mark;

use super::bsp::{BspState, find_zs_groups};

pub const MAX_BI_NUM: usize = 5000;

/// 8 个周期的定义: (名称, 聚合因子)
pub const TIMEFRAMES: &[(&str, usize)] = &[
    ("1m", 1),
    ("5m", 5),
    ("15m", 15),
    ("30m", 30),
    ("1h", 60),
    ("4h", 240),
    ("1d", 1440),
    ("1w", 10080),
];

/// 单个周期的结构字段（4 个通用字段）
#[derive(Debug, Clone, Copy)]
pub struct TfLabels {
    pub has_top: f32,
    pub has_bottom: f32,
    pub bi_dir: f32,
    pub zs: f32,
}

/// 单根 K 线的完整标注快照（8 周期）
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct ChanLabel {
    pub tf: [TfLabels; 8], // 8 个周期，每个 4 个字段
    // 1m 专有字段（索引 0）
    pub bi_strength: f32,
    pub is_bsp: f32,
    pub bsp_direction: f32,
    // 主任务
    pub next_return: f32,
    pub return_label: i64,
}

impl ChanLabel {
    /// 创建空标签（全部为 0）
    pub fn zero() -> Self {
        ChanLabel {
            tf: [TfLabels { has_top: 0.0, has_bottom: 0.0, bi_dir: 0.0, zs: 0.0 }; 8],
            bi_strength: 0.0,
            is_bsp: 0.0,
            bsp_direction: 0.0,
            next_return: 0.0,
            return_label: 1,
        }
    }
}

// ─── 多周期 CZSC 管理器 ─────────────────────────────

/// 管理一个周期的 CZSC 实例 + bar 聚合 + 标签跟踪
pub struct TfCzsc {
    name: &'static str,
    factor: usize,
    czsc: Option<CZSC>,
    buffer: Vec<RawBar>,
    bar_id: i32,
    prev_fx: usize,
    prev_bi: usize,
    // BI 方向连续化
    running_bi_dir: f32,
    running_bi_strength: f32,
}

impl TfCzsc {
    pub fn new(name: &'static str, factor: usize) -> Self {
        TfCzsc {
            name,
            factor,
            czsc: None,
            buffer: Vec::with_capacity(factor),
            bar_id: 0,
            prev_fx: 0,
            prev_bi: 0,
            running_bi_dir: 0.0,
            running_bi_strength: 0.0,
        }
    }

    /// 推入一根 1m K 线，如果达到聚合因子则返回 true
    pub fn push(&mut self, bar: &RawBar) -> bool {
        self.buffer.push(bar.clone());
        if self.buffer.len() >= self.factor {
            self.flush();
            true
        } else {
            false
        }
    }

    /// 聚合缓冲区的 bar 并更新 CZSC
    fn flush(&mut self) {
        let buf: Vec<RawBar> = self.buffer.drain(..).collect();
        let open = buf[0].open;
        let close = buf.last().unwrap().close;
        let high = buf.iter().map(|b| b.high).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(open);
        let low = buf.iter().map(|b| b.low).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(open);
        let vol: f64 = buf.iter().map(|b| b.vol).sum();
        let amount: f64 = buf.iter().map(|b| b.amount).sum();

        let dt = buf[0].dt;

        let agg_bar = czsc_core::objects::bar::RawBarBuilder::default()
            .symbol(Arc::from(self.name))
            .dt(dt)
            .freq(Freq::Tick)
            .id(self.bar_id)
            .open(open)
            .close(close)
            .high(high)
            .low(low)
            .vol(vol)
            .amount(amount)
            .build()
            .expect("聚合 bar 构造失败");
        self.bar_id += 1;

        if let Some(ref mut cz) = self.czsc {
            self.prev_fx = cz.get_fx_list().len();
            self.prev_bi = cz.bi_list.len();
            cz.update_bar(agg_bar);
        } else {
            self.prev_fx = 0;
            self.prev_bi = 0;
            self.czsc = Some(CZSC::new(vec![agg_bar], MAX_BI_NUM));
        }
    }

    /// 提取当前周期的 4 个标签
    pub fn extract(&mut self, _max_bi_num: usize) -> TfLabels {
        let cz = match self.czsc.as_ref() {
            Some(c) => c,
            None => return TfLabels { has_top: 0.0, has_bottom: 0.0, bi_dir: 0.0, zs: 0.0 },
        };

        let curr_fx = cz.get_fx_list().len();
        let curr_bi = cz.bi_list.len();
        let new_fx = curr_fx > self.prev_fx;
        let new_bi = curr_bi > self.prev_bi;
        self.prev_fx = curr_fx;
        self.prev_bi = curr_bi;

        let fx_list = cz.get_fx_list();
        let (top, bottom) = detect_fractals(&fx_list, new_fx);

        let (bi_dir_raw, bi_str) = detect_bi(&cz.bi_list, new_bi);
        // BI 方向连续化
        if new_bi && bi_dir_raw != 0.0 {
            self.running_bi_dir = bi_dir_raw;
            self.running_bi_strength = bi_str;
        }
        let bi_dir = if bi_dir_raw != 0.0 { bi_dir_raw }
                     else if self.running_bi_dir != 0.0 { self.running_bi_dir }
                     else { 0.0 };

        let zs = count_zs(&cz.bi_list);

        TfLabels { has_top: top, has_bottom: bottom, bi_dir, zs }
    }
}

// ─── 标注提取函数 ────────────────────────────────────

/// 从 CZSC 分析器快照中提取分型标记
pub fn detect_fractals(fx_list: &[FX], new_fx: bool) -> (f32, f32) {
    if !new_fx {
        return (0.0, 0.0);
    }
    if let Some(last_fx) = fx_list.last() {
        match last_fx.mark {
            Mark::G => (1.0, 0.0),
            Mark::D => (0.0, 1.0),
        }
    } else {
        (0.0, 0.0)
    }
}

/// 笔检测：仅在新笔形成时返回方向/强度
pub fn detect_bi(bi_list: &[BI], new_bi: bool) -> (f32, f32) {
    if !new_bi {
        return (0.0, 0.0);
    }
    if let Some(last_bi) = bi_list.last() {
        let direction = match last_bi.direction {
            Direction::Up => 1.0,
            Direction::Down => 0.0,
        };
        let strength = last_bi.get_change().abs() as f32;
        (direction, strength)
    } else {
        (0.0, 0.0)
    }
}

/// 中枢计数：缠论正确的分组计数
pub fn count_zs(bis: &[BI]) -> f32 {
    find_zs_groups(bis).len() as f32
}

/// 计算未来 N 根 K 线的收益率和分类标签
pub fn compute_future_return(bars: &[RawBar], idx: usize, lookahead: usize) -> (f32, i64) {
    let next_idx = idx + lookahead;
    if next_idx >= bars.len() {
        return (0.0, 1);
    }

    let current_close = bars[idx].close;
    let future_close = bars[next_idx].close;

    if current_close <= 0.0 {
        return (0.0, 1);
    }

    let ret = (future_close - current_close) / current_close;

    // 自适应阈值：使用 lookahead 窗口内收益率的标准差估算
    let threshold = estimate_volatility(bars, idx, lookahead);

    let label = if ret < -threshold {
        0 // 跌
    } else if ret > threshold {
        2 // 涨
    } else {
        1 // 平
    };

    (ret as f32, label)
}

/// 估计当前窗口的波动率，用于自适应分类阈值
fn estimate_volatility(bars: &[RawBar], idx: usize, window: usize) -> f64 {
    let start = idx.saturating_sub(window).max(0);
    let n = bars.len().min(start + window);

    if n - start < 5 {
        return 0.02; // 数据不足时默认 2%
    }

    let mut returns = Vec::with_capacity(n - start);
    for i in start + 1..n {
        if bars[i - 1].close > 0.0 {
            returns.push((bars[i].close - bars[i - 1].close) / bars[i - 1].close);
        }
    }

    if returns.len() < 5 {
        return 0.02;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    let std = variance.sqrt();

    // 阈值 = 0.5 倍标准差，至少 1%，最多 5%
    (std * 0.5).clamp(0.01, 0.05)
}

// ─── Freq 解析 ──────────────────────────────────────

use anyhow::{bail, Result};

const FREQ_MAP: &[(&str, Freq)] = &[
    ("Tick", Freq::Tick),
    ("1m", Freq::F1),
    ("5m", Freq::F5),
    ("15m", Freq::F15),
    ("30m", Freq::F30),
    ("60m", Freq::F60),
    ("1h", Freq::F60),
    ("120m", Freq::F120),
    ("2h", Freq::F120),
    ("240m", Freq::F240),
    ("4h", Freq::F240),
    ("360m", Freq::F360),
    ("6h", Freq::F360),
    ("1d", Freq::D),
    ("1w", Freq::W),
    ("1M", Freq::M),
    ("1S", Freq::S),
    ("1Y", Freq::Y),
];

/// 将常见频率字符串转换为 czsc_core 的 Freq 枚举
pub fn parse_freq(s: &str) -> Result<Freq> {
    let s = s.trim().to_lowercase();
    // 支持 "1分钟" 等中文格式
    let cn_map: &[(&str, &str)] = &[
        ("1分钟", "1m"),
        ("5分钟", "5m"),
        ("15分钟", "15m"),
        ("30分钟", "30m"),
        ("60分钟", "1h"),
        ("240分钟", "4h"),
        ("日线", "1d"),
        ("周线", "1w"),
        ("月线", "1M"),
    ];
    let s = if s.ends_with('线') || s.ends_with('钟') {
        if let Some((_, eng)) = cn_map.iter().find(|(cn, _)| *cn == s.as_str()) {
            eng.to_string()
        } else {
            s.to_lowercase()
        }
    } else {
        s.to_lowercase()
    };

    for (key, freq) in FREQ_MAP {
        if key.eq_ignore_ascii_case(&s) {
            return Ok(*freq);
        }
    }

    // 支持纯数字分钟: "120" → F120
    if let Ok(minutes) = s.parse::<i32>() {
        for (key, freq) in FREQ_MAP {
            if *key != "Tick" {
                if let Ok(n) = key[..key.len() - 1].parse::<i32>() {
                    if n == minutes {
                        return Ok(*freq);
                    }
                }
            }
        }
    }

    bail!(
        "不支持的时间周期: {s}，支持格式: 1m/5m/15m/30m/1h/4h/1d/1w/1M/日线/周线 等"
    )
}
