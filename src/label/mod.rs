//! 数据标注模块
//!
//! 基于 CZSC（缠中说禅技术分析）逐 K 线回放，
//! 提取分型、笔、中枢、买卖点等缠论结构作为 **辅助学习信号**。
//!
//! 注意：这些标签不是"标准答案"，而是**结构化先验知识**——
//! 它们引导模型关注缠论认为重要的价格结构模式。
//! 标签的近似精度足够，不需要与缠论严格一致。
//!
//! # 标注流程
//!
//! ```text
//! K 线 CSV (OHLCV)
//!   │
//!   ├── czsc_core::CZSC + update_bar() 逐 K 线回放
//!   ├── 追踪 fx_list/bi_list 长度变化 → 仅新形成时标记
//!   │
//!   ├── 每根 K 线快照：
//!   │   ├── 当前是否新形成顶分型 / 底分型
//!   │   ├── 当前是否新形成笔及方向/强度
//!   │   ├── 当前中枢数量（缠论正确分组计数）
//!   │   ├── 当前是不是关键转折点 (BSP) 及方向
//!   │   └── 未来 N 根 K 线的收益率（波动率自适应阈值）
//!   │
//!   └── 输出 Parquet 标注数据集
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
use clap::Args;
use czsc_core::analyze::CZSC;
use czsc_core::objects::bar::{RawBar, RawBarBuilder};
use czsc_core::objects::bi::BI;
use czsc_core::objects::direction::Direction;
use czsc_core::objects::freq::Freq;
use czsc_core::objects::fx::FX;
use czsc_core::objects::mark::Mark;

// ─── CLI ──────────────────────────────────────────────

/// 数据标注 CLI 参数
#[derive(Args, Debug)]
pub struct LabelArgs {
    /// 输入 CSV 文件路径（OHLCV 格式）
    #[arg(short, long, value_name = "FILE")]
    pub data: PathBuf,

    /// 输出 Parquet 路径
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// K 线周期（如 1h、15m、4h、1d）
    #[arg(short, long, default_value = "1h")]
    pub freq: String,

    /// 序列长度（默认 60 根 K 线）
    #[arg(long, default_value_t = 60)]
    pub seq_len: usize,

    /// CZSC 最大笔数（默认 50）
    #[arg(long, default_value_t = 50)]
    pub fx_threshold: i32,

    /// 未来收益率向前看多少根 K 线（默认 seq_len/5）
    #[arg(long)]
    pub lookahead: Option<usize>,

    /// 涨跌分类阈值（百分比，默认自动按波动率计算）
    #[arg(long)]
    pub ret_threshold: Option<f64>,

    /// CSV 无表头（列顺序: dt,open,high,low,close,vol）
    #[arg(long)]
    pub no_header: bool,
}

// ─── ChanLabel ─────────────────────────────────────────

/// 8 个周期的定义: (名称, 聚合因子)
pub(crate) const TIMEFRAMES: &[(&str, usize)] = &[
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
    pub tf: [TfLabels; 8],   // 8 个周期，每个 4 个字段
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
struct TfCzsc {
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
    fn new(name: &'static str, factor: usize) -> Self {
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
    fn push(&mut self, bar: &RawBar) -> bool {
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

        let dt = buf[0].dt;  // 使用第一根的时间戳

        let agg_bar = RawBarBuilder::default()
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
    fn extract(&mut self, _max_bi_num: usize) -> TfLabels {
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

const MAX_BI_NUM: usize = 5000;

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
fn parse_freq(s: &str) -> Result<Freq> {
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
        // 查找中文映射
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

// ─── CSV 读取 ─────────────────────────────────────────

/// 从 CSV 文件读取 K 线数据，含数据完整性校验
fn read_csv_to_raw_bars(path: &std::path::Path, freq: &Freq, no_header: bool) -> Result<Vec<RawBar>> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .has_headers(!no_header)
        .from_path(path)
        .with_context(|| format!("无法打开 CSV 文件: {}", path.display()))?;

    // 列索引：有表头时按列名查找，无表头时用固定位置
    // 固定位置格式: dt(0), open(1), high(2), low(3), close(4), vol(5)
    let (idx_dt, idx_open, idx_high, idx_low, idx_close, idx_vol, idx_amount) = if no_header {
        (0usize, 1, 2, 3, 4, 5, 5)
    } else {
        let headers = reader.headers()?.clone();
        let cols: Vec<String> = headers.iter().map(|h| h.to_lowercase()).collect();
        (col_index(&cols, "dt")?,
         col_index_in(&cols, &["open", "开盘"])?,
         col_index_in(&cols, &["high", "最高"])?,
         col_index_in(&cols, &["low", "最低"])?,
         col_index_in(&cols, &["close", "收盘"])?,
         col_index_in(&cols, &["vol", "volume", "成交量"])?,
         col_index_in(&cols, &["amount", "成交额"])?)
    };

    let mut bars = Vec::new();
    let mut last_dt: Option<DateTime<Utc>> = None;
    let mut skipped = 0usize;

    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_e) => { skipped += 1; continue; }
        };
        let dt_str = match record.get(idx_dt) {
            Some(s) => s.trim(),
            None => { skipped += 1; continue; }
        };
        let dt = match parse_datetime(dt_str) {
            Ok(d) => d,
            Err(_) => { skipped += 1; continue; }
        };

        // 校验时间递增
        if let Some(prev) = last_dt {
            if dt <= prev {
                skipped += 1;
                continue;
            }
        }
        last_dt = Some(dt);

        let open: f64 = match record.get(idx_open).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let high: f64 = match record.get(idx_high).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let low: f64 = match record.get(idx_low).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let close: f64 = match record.get(idx_close).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let vol: f64 = match record.get(idx_vol).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v >= 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let amount: f64 = match record.get(idx_amount).and_then(|v| v.trim().parse().ok()) {
            Some(v) if v >= 0.0 => v,
            _ => { skipped += 1; continue; }
        };

        // 数据完整性校验
        if let Err(e) = validate_ohlcv(open, high, low, close, vol, amount, &dt) {
            log::warn!("跳过无效行 {}: {}", dt_str, e);
            skipped += 1;
            continue;
        }

        let bar = RawBarBuilder::default()
            .symbol(Arc::from("CHAN"))
            .dt(dt)
            .freq(*freq)
            .id(bars.len() as i32)
            .open(open)
            .close(close)
            .high(high)
            .low(low)
            .vol(vol)
            .amount(amount)
            .build()
            .map_err(|e| anyhow::anyhow!("RawBarBuilder 失败: {e}"))?;

        bars.push(bar);
    }

    if bars.is_empty() {
        bail!("CSV 文件中没有数据");
    }

    log::info!("读取了 {} 根 K 线（跳过 {} 行脏数据）", bars.len(), skipped);
    Ok(bars)
}

/// OHLCV 数据完整性校验
fn validate_ohlcv(
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    vol: f64,
    amount: f64,
    dt: &DateTime<Utc>,
) -> Result<()> {
    if high < low {
        bail!(
            "{}: high ({}) < low ({})",
            dt.format("%Y-%m-%d %H:%M:%S"),
            high,
            low
        );
    }
    if high < open || high < close {
        bail!(
            "{}: high ({}) < open/close ({}/{})",
            dt.format("%Y-%m-%d %H:%M:%S"),
            high,
            open,
            close
        );
    }
    if low > open || low > close {
        bail!(
            "{}: low ({}) > open/close ({}/{})",
            dt.format("%Y-%m-%d %H:%M:%S"),
            low,
            open,
            close
        );
    }
    if vol < 0.0 || amount < 0.0 {
        bail!(
            "{}: 负值成交量 ({}) 或成交额 ({})",
            dt.format("%Y-%m-%d %H:%M:%S"),
            vol,
            amount
        );
    }
    if open <= 0.0 || close <= 0.0 || high <= 0.0 || low <= 0.0 {
        bail!(
            "{}: 非正价格 O={} H={} L={} C={}",
            dt.format("%Y-%m-%d %H:%M:%S"),
            open,
            high,
            low,
            close
        );
    }
    Ok(())
}

fn col_index(cols: &[String], name: &str) -> Result<usize> {
    cols.iter()
        .position(|c| c == name)
        .with_context(|| format!("CSV 缺少必需列: {name}，可用列: {}", cols.join(", ")))
}

fn col_index_in(cols: &[String], names: &[&str]) -> Result<usize> {
    for name in names {
        if let Some(pos) = cols.iter().position(|c| c == *name) {
            return Ok(pos);
        }
    }
    bail!(
        "CSV 缺少必需列 ({}), 可用列: {}",
        names.join("/"),
        cols.join(", ")
    )
}

/// 解析多种常见时间格式
fn parse_datetime(s: &str) -> Result<DateTime<Utc>> {
    let s = s.trim();

    // Unix 毫秒时间戳（13 位数字，最常见于交易所 API）
    if s.len() == 13 && s.bytes().all(|b| b.is_ascii_digit()) {
        let ms: i64 = s.parse()?;
        let secs = ms / 1000;
        let nsecs = ((ms % 1000) * 1_000_000) as u32;
        return DateTime::from_timestamp(secs, nsecs)
            .context("Unix ms 时间戳超出范围");
    }

    // Unix 秒时间戳（10 位数字）
    if s.len() == 10 && s.bytes().all(|b| b.is_ascii_digit()) {
        let ts: i64 = s.parse()?;
        return DateTime::from_timestamp(ts, 0)
            .context("Unix 秒时间戳超出范围");
    }

    // ISO 8601 with timezone: "2024-01-01T00:00:00Z" / "+08:00"
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }

    // ISO 8601 with fractional seconds + Z: "2024-01-01T00:00:00.000Z"
    if let Ok(dt) = DateTime::parse_from_rfc3339(&(s.to_owned() + "Z")) {
        return Ok(dt.with_timezone(&Utc));
    }

    // "2024-01-01T00:00:00" (无时区，视为UTC)
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
    }

    // "2024-01-01 00:00:00"
    for fmt in &[
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
        }
        // Try as date only (time = 00:00:00)
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, fmt) {
            if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
            }
        }
    }

    bail!("不支持的时间格式: {s}")
}

// ─── 标注提取 ─────────────────────────────────────────

/// 从 CZSC 分析器快照中提取训练标签
///
/// # 参数
/// - `new_fx`: 当前 K 线是否形成了新分型（由调用者追踪）
/// - `new_bi`: 当前 K 线是否形成了新笔（由调用者追踪）
/// 分型检测：仅在新分型形成时返回有效标记
fn detect_fractals(fx_list: &[FX], new_fx: bool) -> (f32, f32) {
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
fn detect_bi(bi_list: &[BI], new_bi: bool) -> (f32, f32) {
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
///
/// 缠论中，连续三笔有重叠区域（ZG ≥ ZD）即构成一个中枢。
/// 之后继续重叠的笔属于同一中枢的延伸。
/// 当出现不重叠的笔（突破中枢区间）时，中枢完成。
/// 突破笔之后的重叠三笔构成下一个中枢。
///
/// 例如：
///   B1 B2 B3 B4 B5 B6 B7 B8
///   [── ZS1 ──────]  B6 突破
///                      [ ZS2 ]
///   计数结果: 2 个中枢
fn count_zs(bis: &[BI]) -> f32 {
    if bis.len() < 3 {
        return 0.0;
    }

    let mut count = 0;
    let mut i = 0;

    while i < bis.len() {
        // 需要至少 3 笔来检查
        if i + 2 >= bis.len() {
            break;
        }

        // 取三笔的重叠区间
        let zg = bis[i]
            .get_high()
            .min(bis[i + 1].get_high())
            .min(bis[i + 2].get_high());
        let zd = bis[i]
            .get_low()
            .max(bis[i + 1].get_low())
            .max(bis[i + 2].get_low());

        if zg >= zd {
            // ✅ 中枢形成！扩展阶段：检查后续笔是否重叠
            // 缠论中，中枢的 ZG/ZD 由前三笔固定，后续笔重叠则属于延伸
            let mut j = i + 3;
            while j < bis.len() {
                let bi = &bis[j];
                // 检查是否重叠: bi 的区间 [low, high] 是否与 [zd, zg] 相交
                if bi.get_high() >= zd && bi.get_low() <= zg {
                    j += 1; // 继续延伸
                } else {
                    break; // 突破中枢，中枢完成
                }
            }
            count += 1;
            i = j; // 从突破笔开始寻找下一个中枢
        } else {
            i += 1;
        }
    }

    count as f32
}

/// 中枢分组信息
#[allow(dead_code)]
struct ZsGroup {
    zg: f64,        // 中枢上沿
    zd: f64,        // 中枢下沿
    start: usize,   // 起始 BI 索引
    end: usize,     // 结束 BI 索引（突破笔的前一笔）
}

/// 找到所有完成的中枢分组
fn find_zs_groups(bis: &[BI]) -> Vec<ZsGroup> {
    if bis.len() < 3 { return vec![]; }
    let mut groups = Vec::new();
    let mut i = 0;
    while i + 2 < bis.len() {
        let zg = bis[i].get_high().min(bis[i+1].get_high()).min(bis[i+2].get_high());
        let zd = bis[i].get_low().max(bis[i+1].get_low()).max(bis[i+2].get_low());
        if zg >= zd {
            let mut j = i + 3;
            while j < bis.len() {
                let b = &bis[j];
                if b.get_high() >= zd && b.get_low() <= zg {
                    j += 1;
                } else {
                    break;
                }
            }
            groups.push(ZsGroup { zg, zd, start: i, end: j - 1 });
            i = j; // 跳过已分组的中枢
        } else {
            i += 1;
        }
    }
    groups
}

/// BSP 状态机（跨 bar 追踪买卖点）
struct BspState {
    last_buy_idx: Option<usize>,          // 一买发生的 BI 索引
    last_sell_idx: Option<usize>,         // 一卖发生的 BI 索引
}

impl BspState {
    fn new() -> Self {
        BspState { last_buy_idx: None, last_sell_idx: None }
    }

    /// 全三类买卖点检测
    fn detect(&mut self, bis: &[BI], new_bi: bool) -> (f32, f32) {
        if !new_bi || bis.len() < 4 { return (0.0, 0.0); }
        let n = bis.len();
        let last = &bis[n - 1];
        let groups = find_zs_groups(&bis[..n - 1]);

        // 一买: 下跌段背驰 + 反转向上
        if self.detect_buy1(bis, last, n, &groups) { return (1.0, 1.0); }
        // 二买: 一买后的回调不破前低
        if self.detect_buy2(bis, last, n, &groups) { return (1.0, 1.0); }
        // 三买: 突破中枢后回拉不进中枢
        if self.detect_buy3(bis, last, &groups) { return (1.0, 1.0); }

        // 卖点对称
        if self.detect_sell1(bis, last, n, &groups) { return (1.0, 2.0); }
        if self.detect_sell2(bis, last, n) { return (1.0, 2.0); }
        if self.detect_sell3(bis, last, &groups) { return (1.0, 2.0); }

        (0.0, 0.0)
    }

    /// 一买: 下跌段背驰 → 向上笔确认反转
    ///
    /// 序列: ...→ Down_prev → Up_mid → Down_cur → Up_last(确认)
    ///         ↑—背驰比较这两个下跌段—↑
    fn detect_buy1(&mut self, bis: &[BI], last: &BI, n: usize, _groups: &[ZsGroup]) -> bool {
        if last.direction != Direction::Up { return false; }
        if n < 4 { return false; }  // 需要至少 4 笔: U-D-U-D
        // 倒数第二笔必须是 Down
        if bis[n - 2].direction != Direction::Down { return false; }

        // 找到 cur_down 之前的那个 Down BI
        // 跳过 last(n-1) 和 cur_down(n-2)，找下一个 Down
        let prev_down = bis.iter().rev().skip(2).find(|b| b.direction == Direction::Down);

        if let Some(prev_d) = prev_down {
            let cur_down = &bis[n - 2];
            // 底背驰: 新低 + 力度减弱
            let divergence = cur_down.get_low() < prev_d.get_low()
                && cur_down.get_change().abs() < prev_d.get_change().abs();

            if divergence {
                self.last_buy_idx = Some(n - 1);
                self.last_sell_idx = None;
                return true;
            }
        }
        false
    }

    /// 二买: 一买之后回拉不破前低
    fn detect_buy2(&self, bis: &[BI], last: &BI, n: usize, _groups: &[ZsGroup]) -> bool {
        if last.direction != Direction::Down { return false; }
        if let Some(buy_idx) = self.last_buy_idx {
            if n - 1 > buy_idx {
                let buy_low = bis[buy_idx].get_low();
                // 回调不破一买最低点
                if last.get_low() > buy_low {
                    return true;
                }
            }
        }
        false
    }

    /// 三买: 向上突破中枢后回拉不进中枢（回调低点 > ZG）
    fn detect_buy3(&self, _bis: &[BI], last: &BI, groups: &[ZsGroup]) -> bool {
        if last.direction != Direction::Down || groups.is_empty() { return false; }
        let last_zs = match groups.last() { Some(z) => z, None => return false };
        // 最新下跌笔没有回到中枢内
        last.get_low() > last_zs.zg
    }

    /// 一卖: 上涨段背驰 → 向下笔确认反转
    ///
    /// 序列: ...→ Up_prev → Down_mid → Up_cur → Down_last(确认)
    ///         ↑—背驰比较这两个上涨段—↑
    fn detect_sell1(&mut self, bis: &[BI], last: &BI, n: usize, _groups: &[ZsGroup]) -> bool {
        if last.direction != Direction::Down { return false; }
        if n < 4 { return false; }
        if bis[n - 2].direction != Direction::Up { return false; }

        // 找到 cur_up 之前的那个 Up BI
        let prev_up = bis.iter().rev().skip(2).find(|b| b.direction == Direction::Up);

        if let Some(prev_u) = prev_up {
            let cur_up = &bis[n - 2];
            // 顶背驰: 新高 + 力度减弱
            let divergence = cur_up.get_high() > prev_u.get_high()
                && cur_up.get_change().abs() < prev_u.get_change().abs();

            if divergence {
                self.last_sell_idx = Some(n - 1);
                self.last_buy_idx = None;
                return true;
            }
        }
        false
    }

    /// 二卖: 一卖之后反弹不破前高
    fn detect_sell2(&self, bis: &[BI], last: &BI, n: usize) -> bool {
        if last.direction != Direction::Up { return false; }
        if let Some(sell_idx) = self.last_sell_idx {
            if n - 1 > sell_idx {
                let sell_high = bis[sell_idx].get_high();
                if last.get_high() < sell_high {
                    return true;
                }
            }
        }
        false
    }

    /// 三卖: 向下突破中枢后反弹不进中枢（反弹高点 < ZD）
    fn detect_sell3(&self, _bis: &[BI], last: &BI, groups: &[ZsGroup]) -> bool {
        if last.direction != Direction::Up || groups.is_empty() { return false; }
        let last_zs = match groups.last() { Some(z) => z, None => return false };
        last.get_high() < last_zs.zd
    }
}

/// 计算未来 N 根 K 线的收益率和分类标签
///
/// 分类阈值基于收益率标准差的 0.5 倍自适应，或使用预设值。
fn compute_future_return(bars: &[RawBar], idx: usize, lookahead: usize) -> (f32, i64) {
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

// ─── Level-2 递归标注 ─────────────────────────────────

/// 将一根 Level-1 笔构建为 Level-2 "K 线"
///
/// 缠论递归：本级别一根笔 = 高级别一根 K 线
/// open = 笔起始分型价格, close = 笔结束分型价格
/// high = 笔最高价, low = 笔最低价
// ─── Parquet 输出 ─────────────────────────────────────

use polars::prelude::*;

/// 将 8 周期标注写入 Parquet（44 列）
fn write_parquet(bars: &[RawBar], labels: &[ChanLabel], output: &std::path::Path) -> Result<()> {
    let n = bars.len();
    assert_eq!(n, labels.len());

    // OHLCV 列
    let mut dt_col = Vec::with_capacity(n);
    let mut open_col = Vec::with_capacity(n);
    let mut close_col = Vec::with_capacity(n);
    let mut high_col = Vec::with_capacity(n);
    let mut low_col = Vec::with_capacity(n);
    let mut vol_col = Vec::with_capacity(n);
    let mut amount_col = Vec::with_capacity(n);

    // 时间段特定列: 8 × 4 = 32
    let mut tf_cols: Vec<Vec<f64>> = (0..32).map(|_| Vec::with_capacity(n)).collect();
    let mut bi_str_col = Vec::with_capacity(n);
    let mut is_bsp_col = Vec::with_capacity(n);
    let mut bsp_dir_col = Vec::with_capacity(n);
    let mut next_ret_col = Vec::with_capacity(n);
    let mut ret_label_col = Vec::with_capacity(n);

    for (bar, lbl) in bars.iter().zip(labels.iter()) {
        dt_col.push(bar.dt.timestamp_nanos_opt().unwrap_or(0));
        open_col.push(bar.open);
        close_col.push(bar.close);
        high_col.push(bar.high);
        low_col.push(bar.low);
        vol_col.push(bar.vol);
        amount_col.push(bar.amount);

        for (ti, tf) in lbl.tf.iter().enumerate() {
            let base = ti * 4;
            tf_cols[base].push(tf.has_top as f64);
            tf_cols[base + 1].push(tf.has_bottom as f64);
            tf_cols[base + 2].push(tf.bi_dir as f64);
            tf_cols[base + 3].push(tf.zs as f64);
        }

        bi_str_col.push(lbl.bi_strength as f64);
        is_bsp_col.push(lbl.is_bsp as f64);
        bsp_dir_col.push(lbl.bsp_direction as f64);
        next_ret_col.push(lbl.next_return as f64);
        ret_label_col.push(lbl.return_label);
    }

    // 构建 df!() 调用
    // 列格式: 7 OHLCV + 32 tf + 3 1m额外 + 2 主任务 = 44
    let tf_names = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"];
    let tf_fields = ["has_top", "has_bottom", "bi_dir", "zs"];

    // 使用 polars 的 Series 构建
    let mut series_list: Vec<polars::prelude::Series> = Vec::with_capacity(44);

    series_list.push(Series::new("dt".into(), &dt_col));
    series_list.push(Series::new("open".into(), &open_col));
    series_list.push(Series::new("close".into(), &close_col));
    series_list.push(Series::new("high".into(), &high_col));
    series_list.push(Series::new("low".into(), &low_col));
    series_list.push(Series::new("vol".into(), &vol_col));
    series_list.push(Series::new("amount".into(), &amount_col));

    for (ti, tf_name) in tf_names.iter().enumerate() {
        let base = ti * 4;
        series_list.push(Series::new(format!("{}_{}", tf_name, tf_fields[0]).into(), &tf_cols[base]));
        series_list.push(Series::new(format!("{}_{}", tf_name, tf_fields[1]).into(), &tf_cols[base + 1]));
        series_list.push(Series::new(format!("{}_{}", tf_name, tf_fields[2]).into(), &tf_cols[base + 2]));
        series_list.push(Series::new(format!("{}_{}", tf_name, tf_fields[3]).into(), &tf_cols[base + 3]));
    }

    series_list.push(Series::new("1m_bi_str".into(), &bi_str_col));
    series_list.push(Series::new("1m_is_bsp".into(), &is_bsp_col));
    series_list.push(Series::new("1m_bsp_dir".into(), &bsp_dir_col));
    series_list.push(Series::new("next_return".into(), &next_ret_col));
    series_list.push(Series::new("return_label".into(), &ret_label_col));

    let df = DataFrame::new(
        series_list.into_iter().map(|s| s.into()).collect()
    ).context("构建 DataFrame 失败")?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(output)
        .with_context(|| format!("无法创建输出文件: {}", output.display()))?;
    let mut df_mut = df;
    ParquetWriter::new(file).finish(&mut df_mut)?;

    log::info!("已写入 {} 条标注（44 列）到 {}", n, output.display());
    Ok(())
}

// ─── 主入口 ───────────────────────────────────────────

/// 运行数据标注（8 周期并行标注）
pub fn run(args: LabelArgs) -> Result<()> {
    log::info!("加载 K 线数据: {:?}", args.data);
    log::info!("输出路径: {:?}", args.output);
    log::info!("序列长度: {}", args.seq_len);

    let freq = parse_freq(&args.freq)?;
    let lookahead = args.lookahead.unwrap_or_else(|| std::cmp::max(args.seq_len / 5, 1));

    let bars = read_csv_to_raw_bars(&args.data, &freq, args.no_header)?;
    let n = bars.len();
    log::info!("共 {} 根 1m K 线, 向前看 {} 根", n, lookahead);

    // 初始化 8 个周期的 CZSC 管理器
    let mut tf_managers: Vec<TfCzsc> = TIMEFRAMES
        .iter()
        .map(|(name, factor)| TfCzsc::new(name, *factor))
        .collect();

    // 1m 周期特殊处理：直接更新（factor=1）
    // 1m CZSC 直接用原始 bars，不走 buffer 聚合
    let mut czsc_1m = CZSC::new(vec![bars[0].clone()], MAX_BI_NUM);
    let mut labels = Vec::with_capacity(n);

    // 第一根 bar
    labels.push(ChanLabel::zero());

    // 1m 笔方向状态（跨 bar 持久化，必须在循环外声明）
    let mut running_bi_1m_val: f32 = 0.0;
    let mut running_str_1m_val: f32 = 0.0;
    // 完整 BSP 检测状态机
    let mut bsp_state = BspState::new();

    // 后续 bars 增量处理
    for i in 1..n {
        // 1m CZSC 直接更新
        let before_fx = czsc_1m.get_fx_list().len();
        let before_bi = czsc_1m.bi_list.len();
        czsc_1m.update_bar(bars[i].clone());
        let new_fx_1m = czsc_1m.get_fx_list().len() > before_fx;
        let new_bi_1m = czsc_1m.bi_list.len() > before_bi;

        // 提取 1m BSP、强度等
        let fx_list = czsc_1m.get_fx_list();
        let (top_1m, bottom_1m) = detect_fractals(&fx_list, new_fx_1m);
        let (bi_dir_1m_raw, bi_str_1m) = detect_bi(&czsc_1m.bi_list, new_bi_1m);
        let zs_1m = count_zs(&czsc_1m.bi_list);
        let (is_bsp_val, bsp_dir_val) = bsp_state.detect(&czsc_1m.bi_list, new_bi_1m);
        let (next_ret, ret_lbl) = compute_future_return(&bars, i, lookahead);

        // 其他周期：插入 bar，达到聚合因子时自动更新 CZSC
        for tfm in tf_managers.iter_mut().skip(1) {
            tfm.push(&bars[i]);
        }

        // 提取全部 8 个周期的标签
        let mut tf_labels = [TfLabels { has_top: 0.0, has_bottom: 0.0, bi_dir: 0.0, zs: 0.0 }; 8];

        // 1m: 从直接更新的 czsc_1m 提取，BI 方向跨 bar 连续
        if new_bi_1m && bi_dir_1m_raw != 0.0 {
            running_bi_1m_val = bi_dir_1m_raw;
            running_str_1m_val = bi_str_1m;
        }
        let bi_dir_1m_cont = if bi_dir_1m_raw != 0.0 { bi_dir_1m_raw } else { running_bi_1m_val };

        tf_labels[0] = TfLabels { has_top: top_1m, has_bottom: bottom_1m, bi_dir: bi_dir_1m_cont, zs: zs_1m };

        // 5m-1w: 从 TfCzsc 提取
        for (ti, tfm) in tf_managers.iter_mut().enumerate().skip(1) {
            tf_labels[ti] = tfm.extract(50);
        }

        labels.push(ChanLabel {
            tf: tf_labels,
            bi_strength: running_str_1m_val,
            is_bsp: is_bsp_val,
            bsp_direction: bsp_dir_val,
            next_return: next_ret,
            return_label: ret_lbl,
        });
    }

    log::info!("标注完成，共 {} 条", labels.len());

    // 统计
    let n_bsp = labels.iter().filter(|l| l.is_bsp > 0.5).count();
    log::info!("BSP 总数: {}", n_bsp);

    write_parquet(&bars, &labels, &args.output)?;
    log::info!("标注完成 ✅");
    Ok(())
}
