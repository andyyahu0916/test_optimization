# Poisson Solver Benchmark Tools

這個目錄包含兩個專門用來測試 Poisson 算法性能的 benchmark 工具。

## 📁 檔案說明

### 1. `benchmark_poisson.py` - 完整系統測試
使用真實的 OpenMM 系統來測試 Poisson solver 性能。

**特點：**
- 載入完整的 PDB 結構和力場
- 初始化真實的電極系統
- 測試完整的 `Poisson_solver_fixed_voltage()` 函數
- 更接近實際模擬情況

**使用方法：**
```bash
# 基本使用（使用預設參數）
python benchmark_poisson.py

# 自訂參數
python benchmark_poisson.py -c config_refactored.ini -n 10 -r 5 --warmup 2

# 參數說明：
#   -c, --config    : 配置檔路徑（預設: config_refactored.ini）
#   -n, --iterations: 每次測試的 Poisson iterations 數量（預設: 10）
#   -r, --repeats   : 重複測試次數（預設: 5）
#   --warmup        : 預熱次數，不計入統計（預設: 2）
```

**輸出內容：**
- 每個版本的詳細時間統計（平均、標準差、最小、最大）
- 版本間的速度比較和加速比
- 推算到完整模擬的時間節省
- 自動保存結果到 `benchmark_results_YYYYMMDD_HHMMSS.txt`

---

### 2. `benchmark_poisson_minimal.py` - 最小化測試
直接測試核心計算函數，不需要完整 OpenMM 系統。

**特點：**
- 使用模擬數據（隨機生成的電荷和力）
- 只測試核心的電荷計算邏輯
- 啟動快速，適合快速迭代測試
- 可以控制電極原子數量

**使用方法：**
```bash
# 基本使用
python benchmark_poisson_minimal.py

# 自訂參數
python benchmark_poisson_minimal.py --cathode 2000 --anode 2000 -n 5000 --warmup 200

# 參數說明：
#   --cathode    : 陰極原子數量（預設: 1000）
#   --anode      : 陽極原子數量（預設: 1000）
#   -n, --iterations: 測試迭代次數（預設: 1000）
#   --warmup     : 預熱次數（預設: 100）
```

**輸出內容：**
- 微秒級的精確時間測量
- 統計數據（平均、中位數、標準差等）
- 版本間比較和加速比
- 推算到 1 ns 和 10 ns 模擬的時間節省

---

## 🎯 使用建議

### 何時使用 `benchmark_poisson.py`？
- 想要測試完整系統的性能
- 需要了解實際模擬中的加速效果
- 已經有配置好的系統和 PDB 檔案

### 何時使用 `benchmark_poisson_minimal.py`？
- 快速測試 Cython 模組是否正常工作
- 開發和調試 Cython 優化時
- 想要測試不同規模的電極系統
- 不想等待完整系統初始化

---

## 📊 典型輸出範例

### benchmark_poisson.py 輸出：
```
============================================================
POISSON SOLVER BENCHMARK
============================================================

🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍 🐍
Setting up original version...
✓ original system initialized
  Cathode atoms: 1024
  Anode atoms: 1024

============================================================
Benchmarking Original Python Poisson Solver
============================================================
Run 1/5... 0.145230 s (14.523 ms/iter)
Run 2/5... 0.143891 s (14.389 ms/iter)
...

============================================================
PERFORMANCE COMPARISON
============================================================

Version         Total Time           Per Iteration
------------------------------------------------------------
Original            144.200 ms            14.420 ms
Cython               12.340 ms             1.234 ms
------------------------------------------------------------
Speedup              11.69x                11.69x

⚡ Time saved per 10 iterations: 131.860 ms (91.4%)
```

### benchmark_poisson_minimal.py 輸出：
```
============================================================
MINIMAL POISSON SOLVER BENCHMARK
============================================================

============================================================
Benchmarking: Original Python/NumPy
============================================================
Results:
  Mean:   234.56 ± 12.34 μs
  Median: 232.10 μs

============================================================
Benchmarking: Cython Optimized
============================================================
Results:
  Mean:   18.23 ± 1.45 μs
  Median: 18.01 μs

============================================================
COMPARISON
============================================================
Version              Mean Time
----------------------------------------
Original                234.56 μs
Cython                   18.23 μs
----------------------------------------
Speedup:                 12.87x
Time saved:             216.33 μs (92.2%)

============================================================
EXTRAPOLATION TO FULL SIMULATION
============================================================

For 1 ns simulation:
  Charge updates: 100,000
  Total Poisson solver calls: 400,000
  
Poisson solver time:
  Original: 1.6 minutes
  Cython:   0.1 minutes
  Saved:    1.4 minutes (92.2%)
```

---

## 🔧 故障排除

### Cython 模組未找到
如果看到 "Cython module not available" 錯誤：

1. 確認 Cython 模組已編譯：
```bash
cd lib
python setup.py build_ext --inplace
```

2. 檢查是否生成了 `.so` 或 `.pyd` 檔案：
```bash
ls lib/electrode_charge_cython*.so
```

### OpenMM 相關錯誤
如果 `benchmark_poisson.py` 出現 OpenMM 錯誤：
- 確認 OpenMM 已正確安裝
- 檢查 `config_refactored.ini` 配置是否正確
- 確認 PDB 檔案和力場檔案存在

---

## 📝 開發注意事項

### 添加新的測試指標
在 `benchmark_poisson.py` 或 `benchmark_poisson_minimal.py` 中：
1. 修改 `benchmark_*` 函數以收集更多統計數據
2. 更新 `print_statistics()` 函數來顯示新指標
3. 在 `compare_versions()` 中添加比較邏輯

### 測試不同的系統規模
修改電極原子數量來測試擴展性：
```bash
# 小系統
python benchmark_poisson_minimal.py --cathode 500 --anode 500

# 大系統
python benchmark_poisson_minimal.py --cathode 5000 --anode 5000
```

---

## 📚 相關檔案

- `MM_classes.py` - Original Python 版本
- `MM_classes_CYTHON.py` - Cython 優化版本
- `lib/electrode_charge_cython.pyx` - Cython 核心實現
- `config_refactored.ini` - 模擬配置檔

---

## 🎓 結果解讀

### 加速比 (Speedup)
- **< 2x**: 優化效果有限，可能需要檢查實現
- **2-5x**: 不錯的優化，值得使用
- **5-10x**: 很好的優化效果
- **> 10x**: 優秀的優化，Cython 發揮了很大作用

### 時間節省百分比
- 這個數字顯示你能節省多少計算時間
- 對於長時間模擬（幾天到幾週），即使 10-20% 的節省也很有價值
- > 50% 的節省意味著你可以在相同時間內跑兩倍長的模擬

---

## 💡 進階使用

### 自動化多次測試
```bash
# 測試不同規模
for size in 500 1000 2000 5000; do
    echo "Testing with $size atoms..."
    python benchmark_poisson_minimal.py --cathode $size --anode $size -n 1000
done
```

### 產生詳細報告
修改程式碼以輸出 CSV 或 JSON 格式，方便後續分析和畫圖。

---

**作者建議：**
建議先用 `benchmark_poisson_minimal.py` 快速驗證 Cython 優化是否工作，
然後用 `benchmark_poisson.py` 測試實際系統的性能提升。
