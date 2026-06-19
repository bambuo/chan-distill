// fetch_1m.go — 从 data.binance.vision 下载 1m K 线数据
//
// 数据源:
//   完整月份 → https://data.binance.vision/data/spot/monthly/klines/...  (月度 zip)
//   当前月份 → https://data.binance.vision/data/spot/daily/klines/...   (每日 zip)
//
// 用法:
//     cd chan-distill && go run scripts/fetch_1m.go
//
// 输出: data/processed/{SYMBOL}_1m.csv
//   每行: timestamp,open,high,low,close,volume
//   按时间戳升序排列，跨月边界已去重

package main

import (
	"archive/zip"
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var symbols = []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}

const startYear = 2023
const startMonth = 1

// 内存中的一行 K 线，时间戳用 int64 方便排序去重
type klineRow struct {
	ts  int64
	row []string // [ts_str, open, high, low, close, volume]
}

func main() {
	now := time.Now().UTC()
	endYear := now.Year()
	endMonth := int(now.Month())

	// ── 生成完整月份列表（不含当月） ─────────────────
	var fullMonths []string
	for y := startYear; y <= endYear; y++ {
		mStart := 1
		if y == startYear {
			mStart = startMonth
		}
		mEnd := 12
		if y == endYear {
			mEnd = endMonth - 1 // 上月为止
		}
		if mStart > mEnd {
			continue
		}
		for m := mStart; m <= mEnd; m++ {
			fullMonths = append(fullMonths, fmt.Sprintf("%d-%02d", y, m))
		}
	}

	fmt.Printf("📅 完整月份: %d 个月  |  当前月份: %d-%02d (每日 zip)\n", len(fullMonths), endYear, endMonth)
	fmt.Printf("   数据源: data.binance.vision (月度 zip + 每日 zip)\n\n")

	processedDir := filepath.Join("data", "processed")
	if err := os.MkdirAll(processedDir, 0755); err != nil {
		log.Fatalf("创建目录失败: %v", err)
	}

	client := &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 20,
		},
	}

	var wg sync.WaitGroup
	for _, sym := range symbols {
		wg.Add(1)
		go func(symbol string) {
			defer wg.Done()
			downloadSymbol(client, symbol, fullMonths, endYear, endMonth, processedDir)
		}(sym)
	}
	wg.Wait()

	fmt.Println("\n✅ 全部完成")
}

func downloadSymbol(client *http.Client, symbol string, fullMonths []string, curYear, curMonth int, outDir string) {
	fmt.Printf("  🔄 %s 开始下载 ...\n", symbol)

	type monthData struct {
		month string
		rows  [][]string
	}

	// ── 第 1 步: 拉取完整月份的月度 zip ────────
	ch := make(chan monthData, len(fullMonths))
	var dlWg sync.WaitGroup
	sem := make(chan struct{}, 3) // 最多 3 并发

	for _, m := range fullMonths {
		dlWg.Add(1)
		go func(month string) {
			defer dlWg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			rows := downloadMonthZip(client, symbol, month)
			ch <- monthData{month, rows}
		}(m)
	}

	go func() {
		dlWg.Wait()
		close(ch)
	}()

	var allRows []klineRow
	totalMonthly := 0

	for md := range ch {
		for _, r := range md.rows {
			ts, err := strconv.ParseInt(r[0], 10, 64)
			if err != nil {
				continue
			}
			allRows = append(allRows, klineRow{ts, r})
		}
		totalMonthly += len(md.rows)
		fmt.Printf("     %s %s: %d 根 (月度累计 %d)\n", symbol, md.month, len(md.rows), totalMonthly)
	}

	// ── 第 2 步: 用每日 zip 补充当前月份 ──────
	fmt.Printf("     %s %d-%02d: 用每日 zip 补充当月数据 ...\n", symbol, curYear, curMonth)

	now := time.Now().UTC()
	var days []int
	for d := 1; d < now.Day(); d++ {
		days = append(days, d)
	}

	type dayResult struct {
		day  int
		rows [][]string
	}
	dayCh := make(chan dayResult, len(days))

	for _, d := range days {
		dlWg.Add(1)
		go func(day int) {
			defer dlWg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			rows := downloadDayZip(client, symbol, curYear, curMonth, day)
			dayCh <- dayResult{day, rows}
		}(d)
	}

	go func() {
		dlWg.Wait()
		close(dayCh)
	}()

	apiTotal := 0
	for dr := range dayCh {
		for _, r := range dr.rows {
			ts, err := strconv.ParseInt(r[0], 10, 64)
			if err != nil {
				continue
			}
			allRows = append(allRows, klineRow{ts, r})
		}
		apiTotal += len(dr.rows)
	}
	fmt.Printf("     %s %d-%02d: 每日 zip 获取 %d 根\n", symbol, curYear, curMonth, apiTotal)

	if len(allRows) == 0 {
		fmt.Printf("  ⚠️ %s: 没有数据\n", symbol)
		return
	}

	// ── 第 3 步: 排序 + 去重 ──────────────────
	sort.Slice(allRows, func(i, j int) bool {
		return allRows[i].ts < allRows[j].ts
	})

	unique := make([]klineRow, 0, len(allRows))
	var prevTs int64 = -1
	for _, kr := range allRows {
		if kr.ts != prevTs {
			unique = append(unique, kr)
			prevTs = kr.ts
		}
	}
	dedupCount := len(allRows) - len(unique)
	if dedupCount > 0 {
		fmt.Printf("     %s: 去重 %d 行\n", symbol, dedupCount)
	}
	allRows = unique

	// ── 第 4 步: 填充缺失的 1m 间隙（前值填充）──
	beforeFill := len(allRows)
	allRows = fillGaps(allRows)
	filledCount := len(allRows) - beforeFill
	if filledCount > 0 {
		fmt.Printf("     %s: 前值填充 %d 根缺失 K 线\n", symbol, filledCount)
	}

	// ── 第 5 步: 时间戳转可读格式 → 写 CSV ──
	outPath := filepath.Join(outDir, fmt.Sprintf("%s_1m.csv", symbol))
	f, err := os.Create(outPath)
	if err != nil {
		log.Printf("  ❌ %s 创建文件失败: %v", symbol, err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	for _, kr := range allRows {
		// 第 0 列从 Unix ms 转可读格式
		tsMs, _ := strconv.ParseInt(kr.row[0], 10, 64)
		if tsMs > 1_000_000_000_000_000 { // 微秒→毫秒 安全兜底
			tsMs /= 1000
		}
		readable := time.UnixMilli(tsMs).UTC().Format("2006-01-02 15:04:05")
		outRow := []string{readable, kr.row[1], kr.row[2], kr.row[3], kr.row[4], kr.row[5]}
		if err := w.Write(outRow); err != nil {
			log.Printf("  ❌ %s 写入失败: %v", symbol, err)
			return
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Printf("  ❌ %s CSV 刷新失败: %v", symbol, err)
		return
	}

	// 时序完整性检查（填充后应无间隔异常）
	checkContinuity(allRows, symbol)

	fmt.Printf("  ✅ %s 1m: %d 根 K 线 → %s\n", symbol, len(allRows), outPath)
}

// downloadMonthZip 从 data.binance.vision 下载单月的 zip 并解析
func downloadMonthZip(client *http.Client, symbol, month string) [][]string {
	url := fmt.Sprintf("https://data.binance.vision/data/spot/monthly/klines/%s/1m/%s-1m-%s.zip",
		symbol, symbol, month)

	resp, err := client.Get(url)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil
	}

	zipReader, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return nil
	}

	for _, f := range zipReader.File {
		if !strings.HasSuffix(f.Name, ".csv") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil
		}
		rows := parseBinanceCSV(rc)
		rc.Close()
		return rows
	}

	return nil
}

// downloadDayZip 从 data.binance.vision 下载单日的 zip
func downloadDayZip(client *http.Client, symbol string, year, month, day int) [][]string {
	url := fmt.Sprintf("https://data.binance.vision/data/spot/daily/klines/%s/1m/%s-1m-%d-%02d-%02d.zip",
		symbol, symbol, year, month, day)

	resp, err := client.Get(url)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// 当天还没生成 zip 是正常的
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil
	}

	zipReader, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return nil
	}

	for _, f := range zipReader.File {
		if !strings.HasSuffix(f.Name, ".csv") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil
		}
		rows := parseBinanceCSV(rc)
		rc.Close()
		return rows
	}

	return nil
}

// parseBinanceCSV 解析 Binance Data Portal CSV
// 0: open_time(ms), 1: open, 2: high, 3: low, 4: close, 5: volume, ...
func parseBinanceCSV(r io.Reader) [][]string {
	reader := csv.NewReader(r)
	rows := make([][]string, 0, 50000)

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(record) < 6 {
			continue
		}
		if record[0] == "open_time" {
			continue
		}

		rawTs, err := strconv.ParseInt(record[0], 10, 64)
		if err != nil {
			continue
		}

		// Binance Data Portal 在 2025 年初把时间戳从毫秒(13位)改成了微秒(16位)
		if rawTs > 1_000_000_000_000_000 { // > 1e15 = 微秒
			rawTs /= 1000
		}

		row := []string{
			strconv.FormatInt(rawTs, 10),
			record[1], record[2], record[3], record[4], record[5],
		}
		rows = append(rows, row)
	}
	return rows
}

// fillGaps 用前值填充缺失的 1m K 线（按 60s 间隔）
func fillGaps(rows []klineRow) []klineRow {
	if len(rows) < 2 {
		return rows
	}

	filled := make([]klineRow, 0, len(rows))
	filled = append(filled, rows[0])

	for i := 1; i < len(rows); i++ {
		prev := filled[len(filled)-1]
		curr := rows[i]

		// 逐分钟填充 gap
		for ts := prev.ts + 60000; ts < curr.ts; ts += 60000 {
			newRow := klineRow{
				ts: ts,
				row: []string{
					strconv.FormatInt(ts, 10),
					prev.row[1], prev.row[2], prev.row[3], prev.row[4], prev.row[5],
				},
			}
			filled = append(filled, newRow)
		}

		filled = append(filled, curr)
	}

	return filled
}

// checkContinuity 检查 1m K 线是否连续（每根间隔 60s）
func checkContinuity(rows []klineRow, symbol string) {
	if len(rows) < 2 {
		return
	}
	var gaps int
	var maxGap int64
	for i := 1; i < len(rows); i++ {
		diff := rows[i].ts - rows[i-1].ts
		if diff != 60000 {
			gaps++
			if diff > maxGap {
				maxGap = diff
			}
			if gaps <= 3 {
				t1 := time.UnixMilli(rows[i-1].ts).UTC().Format(time.RFC3339)
				t2 := time.UnixMilli(rows[i].ts).UTC().Format(time.RFC3339)
				fmt.Printf("    ⚠️ %s 时间间隔异常: %s → %s (间隔 %dms)\n", symbol, t1, t2, diff)
			}
		}
	}
	if gaps > 0 {
		fmt.Printf("    ⚠️ %s: 共 %d 处间隔异常, 最大间隔 %dms\n", symbol, gaps, maxGap)
	} else {
		fmt.Printf("    ✓ %s: 时序连续无间隔\n", symbol)
	}
}
