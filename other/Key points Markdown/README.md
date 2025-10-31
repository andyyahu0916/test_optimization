# OpenMM Fixed-Voltage MD Simulation (優化版本)

**最終性能**: 3.76x 加速 (Original → CYTHON)  
**詳細文檔**: 請見 `OPTIMIZATION_SUMMARY.md`

---

## 📁 檔案結構說明

### 核心模擬檔案

```
run_openMM.py              # 主程式 - MD 模擬執行
config.ini                 # 模擬參數設定
sapt_exclusions.py         # SAPT-FF 排除規則
```

### 版本選擇 (選擇其中一個)

**lib/** 目錄包含三個版本:

#### 1. Original 版本 (baseline)
```
lib/MM_classes.py                    # 原始 MM 類
lib/Fixed_Voltage_routines.py       # 原始電壓計算
```
- **用途**: 參考基準,驗證正確性
- **性能**: 1.00x (baseline)
- **何時使用**: 需要確認數值正確性時

#### 2. OPTIMIZED 版本 (NumPy 優化)
```
lib/MM_classes_OPTIMIZED.py                # NumPy 向量化
lib/Fixed_Voltage_routines_OPTIMIZED.py   # 向量化電壓計算
```
- **用途**: 生產環境 (如果不想編譯 Cython)
- **性能**: 2.88x
- **優點**: 
  - ✅ 不需要編譯
  - ✅ 純 Python + NumPy
  - ✅ 易於修改和除錯

#### 3. CYTHON 版本 (最快!) ⭐
```
lib/MM_classes_CYTHON.py                  # Cython 加速版
lib/Fixed_Voltage_routines_CYTHON.py     # Cython 電壓計算
lib/electrode_charges_cython.pyx         # Cython 核心函數
lib/setup_cython.py                      # 編譯腳本
lib/electrode_charges_cython*.so         # 編譯後的共享庫
```
- **用途**: 生產環境 (推薦!)
- **性能**: **3.76x** 🚀
- **優點**:
  - ✅ 最快
  - ✅ 保持數值精度
  - ✅ 自動 fallback 到 OPTIMIZED (如果編譯失敗)

---

## 🚀 快速開始

### 1. 編譯 Cython 模組 (首次使用)

```bash
cd lib/
python setup_cython.py build_ext --inplace

# 驗證編譯成功
ls electrode_charges_cython*.so
```

### 2. 選擇版本並運行

編輯 `run_openMM.py` 的 import:

```python
# 選擇 CYTHON 版本 (推薦)
from lib.MM_classes_CYTHON import MM
from lib.Fixed_Voltage_routines_CYTHON import *

# 或選擇 OPTIMIZED 版本
# from lib.MM_classes_OPTIMIZED import MM
# from lib.Fixed_Voltage_routines_OPTIMIZED import *

# 或選擇 Original 版本
# from lib.MM_classes import MM
# from lib.Fixed_Voltage_routines import *
```

### 3. 執行模擬

```bash
python run_openMM.py
```

---

## 🧪 測試與驗證

### Benchmark (性能測試)

```bash
python bench.py
```

**輸出範例**:
```
Version              Time (s)        Speedup        
----------------------------------------------------------------------
Original             0.2840          1.00x (baseline)
Optimized (NumPy)    0.0986          2.88x
Cython               0.0756          3.76x          ⭐
----------------------------------------------------------------------
```

### Profiling (瓶頸分析)

```bash
python profile_bottleneck.py
```

**輸出**: 詳細的時間分解,找出性能瓶頸

---

## 📊 性能對比

| 操作 | Original | OPTIMIZED | CYTHON | 加速比 |
|------|----------|-----------|--------|--------|
| **Poisson solver** | 28.4 ms | 9.9 ms | **7.6 ms** | **3.76x** |
| Forces 提取 | 3.7 ms | 0.05 ms | 0.05 ms | 74x |
| 電荷計算 | 5 ms | 2 ms | 0.5 ms | 10x |
| 參數更新 | 8 ms | 6 ms | 6 ms | 1.3x |

---

## ⚠️ 重要注意事項

### Cython 編譯失敗?

如果看到:
```
⚠️  Cython module not found. Falling back to NumPy implementation.
```

**解決方法**:
1. 確認已安裝 Cython: `pip install cython`
2. 確認有 C 編譯器: `gcc --version`
3. 重新編譯: `cd lib && python setup_cython.py build_ext --inplace`

**Fallback 機制**: 即使編譯失敗,程式仍會自動使用 OPTIMIZED 版本 (2.88x 加速)

### 驗證數值正確性

```bash
python bench.py
```

檢查輸出:
```
RESULT CONSISTENCY CHECK
----------------------------------------------------------------------
Cython vs Orig    9.82e-13       3.78e-14       ✓ OK
```

誤差 < 1e-12 表示**完全正確** ✅

---

## 🔬 技術細節

詳見 **`OPTIMIZATION_SUMMARY.md`** 完整文檔,包含:

- ✅ 優化策略詳解
- ✅ 失敗的優化嘗試 (避免重複錯誤)
- ✅ Cython 實現細節
- ✅ 數據傳輸分析
- ✅ 未來優化方向

---

## 📞 問題排查

### 問題 1: 性能沒有提升

**檢查**:
1. 確認使用了正確的版本 (CYTHON 不是 Original)
2. 檢查 Cython 模組是否載入: 看啟動訊息 "✅ Cython module loaded"
3. 系統規模是否太小 (< 1000 原子優化不明顯)

### 問題 2: 結果不一致

**檢查**:
1. 運行 `bench.py` 驗證精度
2. 確認所有版本使用相同的 `config.ini`
3. 檢查隨機種子設定

### 問題 3: 編譯錯誤

```bash
# 查看詳細錯誤
cd lib
python setup_cython.py build_ext --inplace --verbose

# 生成 Cython 註解 HTML (黃色 = Python 交互多 = 慢)
cython -a electrode_charges_cython.pyx
# 開啟 electrode_charges_cython.html 查看
```

---

## 📚 相關檔案

- `OPTIMIZATION_SUMMARY.md` - **完整技術文檔** (必讀!)
- `config.ini` - 模擬參數設定
- `for_openmm.pdb` - 初始結構
- `ffdir/` - 力場參數

---

## 🎯 推薦工作流程

### 開發階段
```python
# 使用 OPTIMIZED 版本 (方便除錯)
from lib.MM_classes_OPTIMIZED import MM
```

### 生產運行
```python
# 使用 CYTHON 版本 (最快)
from lib.MM_classes_CYTHON import MM
```

### 驗證正確性
```python
# 比較所有版本
python bench.py
```

---

**版本**: v1.0 (2025-10-24)  
**狀態**: Production Ready ✅  
**建議**: 使用 CYTHON 版本獲得最佳性能
