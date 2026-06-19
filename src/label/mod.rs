//! 数据标注模块
//!
//! 基于 CZSC（缠中说禅技术分析）逐 K 线回放，
//! 提取分型、笔、中枢、买卖点等缠论结构作为 **辅助学习信号**。
//!
//! 注意：这些标签不是"标准答案"，而是**结构化先验知识**——
//! 它们引导模型关注缠论认为重要的价格结构模式。
//! 标签的近似精度足够，不需要与缠论严格一致。

pub mod bsp;
pub mod csv_reader;
pub mod czsc_label;
pub mod parquet_out;

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use czsc_core::analyze::CZSC;

use bsp::BspState;
use czsc_label::{ChanLabel, TfCzsc, TfLabels, TIMEFRAMES, MAX_BI_NUM,
                 detect_fractals, detect_bi, count_zs, compute_future_return, parse_freq};
use csv_reader::read_csv_to_raw_bars;
use parquet_out::write_parquet;

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
