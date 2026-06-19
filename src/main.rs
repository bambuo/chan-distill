//! Chan-Distill RS（缠论辅助学习 · Rust 版）
//!
//! 用缠论标注作为结构化先验，引导 1D-CNN 学习价格模式，
//! 实现从 K 线到交易信号的端到端流水线：
//!   CZSC 标注 → Candle 训练 → ONNX / Candle 推理
//!
//! 这不是"蒸馏"（不追求模型替代缠论），
//! 而是"缠论辅助学习"——缠论标注提供结构先验，
//! 主任务（涨跌预测）提供密集训练信号。


mod label;
mod model;
mod train;
mod infer;

use clap::{Parser, Subcommand};

/// Chan-Distill RS：纯 Rust 端到端缠论信号模型
#[derive(Parser)]
#[command(name = "chan-distill-rs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 用 CZSC 给 K 线数据打缠论标签
    Label(label::LabelArgs),

    /// 训练多任务 1D-CNN 模型
    Train(train::TrainArgs),

    /// 导出 ONNX 模型
    Export(Box<infer::ExportArgs>),

    /// 用 ONNX 模型推理
    Infer(infer::InferArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Label(args) => label::run(args),
        Commands::Train(args) => train::run(args),
        Commands::Export(args) => infer::run_export(*args),
        Commands::Infer(args) => infer::run_infer(args),
    }
}
