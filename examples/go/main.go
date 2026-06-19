// inference — ChanDistill 1m 实时推理示例（Go + ONNX Runtime）
//
// 用法:
//   cd examples/go/inference
//   go run main.go ../../BTCUSDT_1m.csv
//
// 每读一根 K 线推进一步，积累 10080 根后开始输出信号。

package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	SeqLen    = 10080   // 7 天的 1m 数据
	InputName = "input" // ONNX 输入名称
)

// Kline 单根 K 线
type Kline struct {
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// Engine 推理引擎
type Engine struct {
	session   *ort.AdvancedSession
	inputData []float32            // 输入 tensor 的底层数据（循环写入后再 copy 进 tensor）
	inputT    *ort.Tensor[float32] // 绑定到 session 的输入 tensor
	outputD   *ort.Tensor[int64]   // signal_direction
	outputC   *ort.Tensor[float32] // signal_confidence
	outputB   *ort.Tensor[float32] // is_bsp
	buffer    []float32            // 滚动窗口，长度 SeqLen × 5
	pos       int                  // 当前写入位置（循环 buffer）
	filled    bool                 // 是否已填满
}

func NewEngine(modelPath string) (*Engine, error) {
	// ONNX Runtime 初始化：优先读环境变量，否则探测常见 macOS 路径
	ortPath := os.Getenv("ORT_LIB_PATH")
	if ortPath == "" {
		for _, p := range []string{
			"/opt/homebrew/lib/libonnxruntime.dylib", // Apple Silicon Homebrew
			"/usr/local/lib/libonnxruntime.dylib",    // Intel Homebrew
			"onnxruntime.so",                         // Linux 默认
		} {
			if _, err := os.Stat(p); err == nil {
				ortPath = p
				break
			}
		}
	}
	if ortPath != "" {
		ort.SetSharedLibraryPath(ortPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("初始化 ONNX Runtime 失败: %v\n  请安装: brew install onnxruntime\n  或设置环境变量: export ORT_LIB_PATH=/path/to/libonnxruntime.dylib", err)
	}

	// 构建输入/输出 tensor（创建时即绑定到 session）
	inputShape := ort.NewShape(1, SeqLen, 5)
	inputData := make([]float32, SeqLen*5)
	inputT, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("创建输入 tensor 失败: %v", err)
	}

	outputD, err := ort.NewEmptyTensor[int64](ort.NewShape(1, 1))
	if err != nil {
		return nil, fmt.Errorf("创建输出 tensor 失败: %v", err)
	}
	outputC, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return nil, fmt.Errorf("创建输出 tensor 失败: %v", err)
	}
	outputB, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return nil, fmt.Errorf("创建输出 tensor 失败: %v", err)
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{InputName},
		[]string{"signal_direction", "signal_confidence", "is_bsp"},
		[]ort.Value{inputT},
		[]ort.Value{outputD, outputC, outputB},
		nil)
	if err != nil {
		return nil, fmt.Errorf("创建推理会话失败: %v", err)
	}

	buf := make([]float32, SeqLen*5)
	return &Engine{
		session:   session,
		inputData: inputData,
		inputT:    inputT,
		outputD:   outputD,
		outputC:   outputC,
		outputB:   outputB,
		buffer:    buf,
	}, nil
}

func (e *Engine) Close() {
	e.session.Destroy()
	e.inputT.Destroy()
	e.outputD.Destroy()
	e.outputC.Destroy()
	e.outputB.Destroy()
}

func (e *Engine) Update(k Kline) *Signal {
	// 写入循环 buffer（[open, high, low, close, volume] 交错存储）
	base := e.pos * 5
	e.buffer[base+0] = float32(k.Open)
	e.buffer[base+1] = float32(k.High)
	e.buffer[base+2] = float32(k.Low)
	e.buffer[base+3] = float32(k.Close)
	e.buffer[base+4] = float32(k.Volume)
	e.pos = (e.pos + 1) % SeqLen

	if !e.filled && e.pos == 0 {
		e.filled = true
	}
	if !e.filled {
		return nil // 还没攒够
	}

	// 构建连续 buffer（循环 buffer → 线性化）
	linear := make([]float32, SeqLen*5)
	copy(linear, e.buffer[e.pos*5:])                    // pos 到末尾
	copy(linear[(SeqLen-e.pos)*5:], e.buffer[:e.pos*5]) // 开头到 pos-1

	// 归一化
	basePrice := linear[3] // 第一根 close
	if basePrice <= 0 {
		basePrice = 1
	}
	var volSum float32
	for i := 0; i < SeqLen; i++ {
		off := i * 5
		linear[off+0] /= basePrice // open
		linear[off+1] /= basePrice // high
		linear[off+2] /= basePrice // low
		linear[off+3] /= basePrice // close
		volSum += linear[off+4]
	}
	volMean := volSum / SeqLen
	if volMean < 1e-10 {
		volMean = 1
	}
	for i := 0; i < SeqLen; i++ {
		linear[i*5+4] /= volMean // volume
	}

	// 将数据写入已绑定的输入 tensor
	copy(e.inputData, linear)

	// 推理
	if err := e.session.Run(); err != nil {
		return nil
	}

	dirData := e.outputD.GetData()
	confData := e.outputC.GetData()
	bspData := e.outputB.GetData()

	if len(dirData) == 0 || len(confData) == 0 || len(bspData) == 0 {
		return nil
	}

	d := dirData[0]
	c := confData[0]
	b := bspData[0]

	if b > 0.5 && d > 0 {
		label := "buy"
		if d == 2 {
			label = "sell"
		}
		return &Signal{
			Direction:  label,
			Confidence: float64(c),
			BspScore:   float64(b),
		}
	}
	return nil
}

// Signal 交易信号
type Signal struct {
	Direction  string  // "buy" / "sell"
	Confidence float64 // 综合置信度
	BspScore   float64 // BSP 分数
}

func parseKline(row []string) (Kline, error) {
	if len(row) < 6 {
		return Kline{}, fmt.Errorf("列数不足: %d", len(row))
	}
	ts, err := time.Parse("2006-01-02 15:04:05", row[0])
	if err != nil {
		return Kline{}, fmt.Errorf("解析时间戳失败: %v", err)
	}
	vals := make([]float64, 5)
	for i := 0; i < 5; i++ {
		v, err := strconv.ParseFloat(row[i+1], 64)
		if err != nil {
			return Kline{}, fmt.Errorf("解析第%d列失败: %v", i+1, err)
		}
		vals[i] = v
	}
	return Kline{
		Timestamp: ts,
		Open:      vals[0],
		High:      vals[1],
		Low:       vals[2],
		Close:     vals[3],
		Volume:    vals[4],
	}, nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("用法: go run main.go <1m_csv_path>")
		os.Exit(1)
	}
	csvPath := os.Args[1]

	// 找 ONNX 模型（跟 CSV 同目录或当前目录）
	modelPath := "chan_distill.onnx"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = "../chan_distill.onnx"
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("❌ 模型文件不存在: chan_distill.onnx\n")
		fmt.Println("   请先运行: python export_onnx.py")
		os.Exit(1)
	}

	engine, err := NewEngine(modelPath)
	if err != nil {
		fmt.Printf("❌ 加载模型失败: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()
	defer ort.DestroyEnvironment()

	fmt.Printf("📦 模型: %s\n", modelPath)
	fmt.Printf("📐 输入: (1, %d, 5) OHLCV\n", SeqLen)
	fmt.Println("🚀 开始推理...")
	fmt.Println()

	file, err := os.Open(csvPath)
	if err != nil {
		fmt.Printf("❌ 打开 CSV 失败: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lineNo := 0
	signalCount := 0

	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		lineNo++

		k, err := parseKline(record)
		if err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  第%d行解析失败: %v\n", lineNo, err)
			continue
		}

		signal := engine.Update(k)
		if signal != nil {
			signalCount++
			if signalCount <= 10 || signalCount%100 == 0 {
				fmt.Printf("  ⚡ %s  %s  置信度 %.4f  BSP %.4f\n",
					k.Timestamp.Format("2006-01-02 15:04"),
					signal.Direction, signal.Confidence, signal.BspScore)
			}
		}
	}

	elapsed := float64(lineNo) / 60 // 分钟
	fmt.Printf("\n✅ 完成: %d 根 K 线 (%.1f 小时), %d 个信号\n",
		lineNo, elapsed/60, signalCount)
}
