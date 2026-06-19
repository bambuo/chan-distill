//! BSP 状态机 + 中枢分组
//!
//! 全三类买卖点检测：一买（背驰）、二买（回拉不破前低）、三买（突破中枢不回）

use czsc_core::objects::bi::BI;
use czsc_core::objects::direction::Direction;

/// 中枢分组信息
#[allow(dead_code)]
pub struct ZsGroup {
    pub zg: f64,      // 中枢上沿
    pub zd: f64,      // 中枢下沿
    pub start: usize, // 起始 BI 索引
    pub end: usize,   // 结束 BI 索引（突破笔的前一笔）
}

/// 找到所有完成的中枢分组
pub fn find_zs_groups(bis: &[BI]) -> Vec<ZsGroup> {
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
pub struct BspState {
    last_buy_idx: Option<usize>,  // 一买发生的 BI 索引
    last_sell_idx: Option<usize>, // 一卖发生的 BI 索引
}

impl BspState {
    pub fn new() -> Self {
        BspState { last_buy_idx: None, last_sell_idx: None }
    }

    /// 全三类买卖点检测
    pub fn detect(&mut self, bis: &[BI], new_bi: bool) -> (f32, f32) {
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
        if n < 4 { return false; }
        if bis[n - 2].direction != Direction::Down { return false; }

        let prev_down = bis.iter().rev().skip(2).find(|b| b.direction == Direction::Down);

        if let Some(prev_d) = prev_down {
            let cur_down = &bis[n - 2];
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

        let prev_up = bis.iter().rev().skip(2).find(|b| b.direction == Direction::Up);

        if let Some(prev_u) = prev_up {
            let cur_up = &bis[n - 2];
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
