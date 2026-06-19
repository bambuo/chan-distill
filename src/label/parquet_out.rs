//! Parquet 输出（44 列）

use anyhow::{Context, Result};
use czsc_core::objects::bar::RawBar;
use polars::prelude::*;

use super::czsc_label::ChanLabel;

/// 将 8 周期标注写入 Parquet（44 列）
pub fn write_parquet(bars: &[RawBar], labels: &[ChanLabel], output: &std::path::Path) -> Result<()> {
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

    let tf_names = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"];
    let tf_fields = ["has_top", "has_bottom", "bi_dir", "zs"];

    let mut series_list: Vec<Series> = Vec::with_capacity(44);

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
