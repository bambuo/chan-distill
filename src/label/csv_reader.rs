//! CSV 读取 + 时间解析 + OHLCV 校验

use std::path::Path;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::freq::Freq;
use std::sync::Arc;

/// 从 CSV 文件读取 K 线数据，含数据完整性校验
pub fn read_csv_to_raw_bars(path: &Path, freq: &Freq, no_header: bool) -> Result<Vec<RawBar>> {
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

        let bar = czsc_core::objects::bar::RawBarBuilder::default()
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
pub fn parse_datetime(s: &str) -> Result<DateTime<Utc>> {
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
