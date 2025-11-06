# 快速開始：Poisson Solver Benchmark

## 🚀 最快速的測試方法

```bash
cd /home/andy/test_optimization/openMM_constant_V_beta
./run_benchmark.sh
```

這個腳本會引導你運行兩種測試。

---

## 📊 方法 1: 最小化測試（推薦先用這個！）

**特點：** 快速、不需要 OpenMM 系統、純測試算法

```bash
python benchmark_poisson_minimal.py
```

**預期輸出：**
- 顯示 Original 和 Cython 版本的執行時間
- 計算加速比（預期 10-15x）
- 推算到完整模擬的時間節省

**自訂測試規模：**
```bash
# 測試更大的系統（2000 個電極原子）
python benchmark_poisson_minimal.py --cathode 2000 --anode 2000 -n 2000

# 快速測試（更少迭代）
python benchmark_poisson_minimal.py -n 500 --warmup 50
```

---

## 🔬 方法 2: 完整系統測試

**特點：** 使用真實 OpenMM 系統、更接近實際情況

```bash
python benchmark_poisson.py
```

**注意事項：**
- 需要配置檔案 `config_refactored.ini`
- 需要 PDB 檔案和力場檔案
- 執行時間較長（2-5 分鐘）
- 結果會自動保存到 `benchmark_results_*.txt`

**自訂參數：**
```bash
# 更多 iterations（更準確，但更慢）
python benchmark_poisson.py -n 20 -r 10

# 快速測試
python benchmark_poisson.py -n 5 -r 3 --warmup 1
```

---

## 📈 如何解讀結果

### 加速比 (Speedup)
```
Speedup: 12.50x
```
這表示 Cython 版本比 Original 快 12.5 倍！

### 時間節省
```
Time saved: 131.860 ms (91.4%)
```
每次 Poisson 求解節省 91.4% 的時間。

### 推算到完整模擬
```
For 1 ns simulation:
  Original: 15.6 minutes
  Cython:   1.2 minutes
  Saved:    14.4 minutes (92.2%)
```
1 ns 模擬能節省 14 分鐘的 Poisson 求解時間。

---

## 🔍 故障排除

### 問題 1: "Cython module not available"
**解決方法：**
```bash
cd lib
python setup.py build_ext --inplace
ls electrode_charge_cython*.so  # 確認編譯成功
```

### 問題 2: "Config file not found"
**解決方法：**
```bash
# 使用自訂配置檔
python benchmark_poisson.py -c /path/to/your/config.ini
```

### 問題 3: OpenMM 相關錯誤
**解決方法：**
- 確認 OpenMM 已安裝：`python -c "import simtk.openmm; print('OK')"`
- 使用最小化測試代替：`python benchmark_poisson_minimal.py`

---

## 💡 建議的測試流程

1. **第一步：驗證 Cython 工作**
   ```bash
   python benchmark_poisson_minimal.py -n 500
   ```
   看到加速比 > 5x 就是正常的。

2. **第二步：詳細性能測試**
   ```bash
   python benchmark_poisson_minimal.py -n 5000
   ```
   獲得更精確的統計數據。

3. **第三步：真實系統測試**（可選）
   ```bash
   python benchmark_poisson.py
   ```
   在實際 OpenMM 系統中驗證性能。

---

## 📊 輸出檔案

- `benchmark_results_YYYYMMDD_HHMMSS.txt` - 完整測試結果（由 benchmark_poisson.py 生成）
- 終端輸出包含所有統計數據和比較

---

## 🎯 典型結果範例

```
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

For 10 ns simulation:
  Original: 0.3 hours
  Cython:   0.0 hours
  Saved:    0.2 hours
```

---

## ❓ 常見問題

**Q: 需要多長時間？**
- 最小化測試：約 30 秒
- 完整系統測試：約 2-5 分鐘

**Q: 我應該期待多少加速？**
- 典型加速比：10-15x
- 如果 < 5x，可能有問題需要檢查

**Q: 可以測試自己的系統嗎？**
- 可以！修改 `config_refactored.ini` 指向你的 PDB 檔案
- 或使用 `benchmark_poisson_minimal.py` 並調整 `--cathode` 和 `--anode` 參數

**Q: 結果不穩定怎麼辦？**
- 增加重複次數：`-r 10`
- 增加 warmup 次數：`--warmup 5`
- 確保系統負載不高（關閉其他程式）

---

需要更多幫助？查看 `BENCHMARK_README.md` 獲取詳細文檔。
