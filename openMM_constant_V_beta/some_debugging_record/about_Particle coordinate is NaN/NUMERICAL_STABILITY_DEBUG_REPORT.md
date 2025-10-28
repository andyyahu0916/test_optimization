# 數值穩定性問題 Debug 報告

**日期**: 2025-10-25  
**系統**: BMIM_BF4_HOH Constant Voltage Electrode Simulation  
**問題**: Particle coordinate is NaN at iteration ~10-15  

---

## 🔍 問題描述

### 症狀
- 模擬在 iteration 10-15 左右崩潰，出現 `Particle coordinate is NaN`
- **兩個版本都會炸**：
  - Original (教授版本): 70ns 時仍可能爆炸，甚至 0V 也爆過
  - Cython 優化版本: 同樣在 iteration 10-15 爆炸
- 電壓：2.0V 和 4.0V 都有問題
- 與 Warm Start 無關（爆炸發生在 10ns 啟動前）

### 觀察到的異常數據
```
正常範圍應為 Q_numeric ≈ ±10 ~ ±40
但實際觀察到:
  Q_numeric cathode: -223 (異常大!)
  Q_numeric anode: -227 (異常大!)
  
單個原子電荷: ±0.3 (正常)
總電荷 = 800 atoms × 0.3 = 240 (超出合理範圍!)
```

---

## 🎯 根本原因分析

### 1. 核心演算法的數值不穩定性

**問題代碼**（`Fixed_Voltage_routines.py` 和 Cython 版本）：
```python
# 第 327 行左右
Ez_external = forces_z[atom_idx] / q_i_old  # 💣 當 q_i_old ≈ 0 時爆炸!

q_i = prefactor * (voltage_term + Ez_external)
```

**為什麼會爆炸**：
- 當 `q_i_old` 接近 0 時（比如 1e-8），`Ez_external` 會變得極大（1e8 量級）
- 即使設定了 `threshold_check = 0.9 * small_threshold`，在某些情況下仍然會除以很小的數
- 下一次迭代，這個錯誤的電荷會被當作 `q_old`，產生更大的誤差
- **惡性循環** → 指數增長 → NaN

### 2. Scale_charges_analytic 的邏輯缺陷

**問題代碼**（`MM_classes.py` Line 389-399）：
```python
scale_factor = Q_analytic / Q_numeric_total

if scale_factor > 0.0:  # 💣 當符號相反時不縮放!
    # 縮放電荷
else:
    # 不做任何事!
```

**為什麼會失控**：
- 當 `Q_numeric` 和 `Q_analytic` 符號相反時（比如 -223 vs +20）
- `scale_factor = 20 / (-223) = -0.09` (負數)
- 原始代碼：`if scale_factor > 0.0:` → **不執行縮放**
- 結果：錯誤的電荷不被修正，下次迭代更錯
- **從測試觀察到的模式**：
  ```
  Iteration 1: Q_numeric = -223, Q_analytic = +18 (符號相反，不縮放)
  Iteration 2: Q_numeric = +103 (翻轉但數值更大)
  Iteration 3: Q_numeric = +26
  Iteration 4: 振盪持續...
  最終: NaN
  ```

### 3. 電荷限制與物理守恆的衝突

即使加上硬限制（比如 `q_max = 0.5`）：
- 單個原子電荷被限制在 ±0.5 ✅
- 但 800 個原子 × 0.5 = ±400 ❌（遠超合理值 ±20）
- `Scale_charges_analytic` 要把 ±400 縮放到 ±20 → `scale_factor = 0.05`
- 所有電荷變成原來的 5% → 下次迭代又算錯
- **限制破壞了電荷守恆，導致收斂失敗**

---

## 🔧 嘗試過的修復方案

### ❌ 方案 1: 加大 threshold_check
```python
threshold_check = 10 * small_threshold  # 從 0.9 改成 10
```
**結果**: 仍然爆炸  
**原因**: threshold 太大會導致很多原子使用 `Ez_external = 0`，物理不正確

### ❌ 方案 2: 硬限制單個電荷
```python
if abs(q_i) > q_max:
    q_i = q_max * sign(q_i)  # 限制在 ±0.5
```
**結果**: 延遲爆炸但最終仍 NaN  
**原因**: 破壞電荷守恆，`Scale_charges_analytic` 無法正常工作

### ❌ 方案 3: 限制 Ez_external
```python
if abs(Ez_external) > 1e4:
    Ez_external = 1e4 * sign(Ez_external)
```
**結果**: 仍然爆炸  
**原因**: 只治標不治本，錯誤累積仍然發生

### ⚠️ 方案 4: 阻尼（Damping）+ 正則化
```python
# 正則化除法
Ez_external = forces_z / (q_old + epsilon * sign(q_old))

# 阻尼更新
q_new = damping * q_computed + (1 - damping) * q_old
```
**結果**: 改善但仍不穩定  
**原因**: 
- `damping = 0.5` 仍會傳播 50% 的錯誤
- `damping = 0.9` 收斂太慢
- 無法找到平衡點

### ⚠️ 方案 5: 修復 scale_factor < 0 的邏輯
```python
if scale_factor < 0:
    print(f"WARNING: Q_numeric and Q_analytic have opposite signs!")
    scale_factor = abs(scale_factor)  # 用絕對值而不是跳過
```
**結果**: 部分改善，但仍在 iteration 10-15 爆炸  
**原因**: 只修復了一個症狀，根本問題仍在

### ❌ 方案 6: 用固定 q_typical 代替 q_old（激進！）
```python
q_typical = 0.2  # 固定值
Ez_external = forces_z / q_typical  # 不用實際的 q_old
```
**結果**: 物理意義不正確，被用戶拒絕  
**原因**: "頭痛砍頭、腳痛砍腳"，破壞了原始演算法

---

## 📊 數據分析

### Poisson Solver 輸出分析
```
Final charge range: [-0.40, +0.29]  ← 單個電荷正常
但...
Q_numeric cathode: -223  ← 總電荷異常!
Q_numeric anode: -227

800 atoms × 0.3 = 240 (可能)
但為什麼符號會錯？為什麼數值這麼大？
```

### 時間分析（從之前的性能測試）
```
MD step: 82% 的時間 (440ms)
Poisson solver: 18% 的時間 (55ms → 8ms with Cython, 85% speedup!)
Overall speedup: ~1.4x (受 Amdahl's Law 限制)
```

### 測試環境
```
系統: BMIM_BF4_HOH
原子數: 29,427 total (800 cathode + 800 anode)
電壓: 2.0V (已從 4.0V 降低)
平台: CUDA
溫度: 298K
```

---

## 🎓 教授的智慧

> "搞定這問題搞不好能發 paper，搞不定就重算一次，大部分情況下能解決..."

**解讀**：
- ✅ 問題是已知的、隨機的數值不穩定性
- ✅ **重算（換個隨機種子）通常能規避** 
- ✅ 如果能徹底解決 → 有學術價值
- ⚠️ 但不是必須解決才能完成研究

---

## 💡 建議的下一步

### 短期方案（實用）
1. **多次重算策略**
   - 使用不同的隨機種子
   - 記錄哪些 seed 成功/失敗
   - 選擇穩定的軌跡繼續分析

2. **降低難度**
   - 更低的電壓（1.0V, 0.5V）
   - 更短的時間步長
   - 更頻繁的 charge update（降低 freq_charge_update_fs）

3. **使用教授的原始版本**
   - Original 雖然慢但經過更多測試
   - 記錄什麼情況下會爆炸

### 中期方案（研究）
1. **文獻調研**
   - 查找其他 Constant Potential Electrode 模擬方法
   - 是否有更穩定的 Poisson solver 演算法？
   - LAMMPS、GROMACS 怎麼處理這個問題？

2. **簡化系統測試**
   - 先在小系統上測試（100 atoms）
   - 確認演算法在簡單情況下是否穩定
   - 找出臨界系統大小/電壓

3. **收斂性分析**
   - 記錄每次 iteration 的電荷變化
   - 分析收斂曲線
   - 識別發散的早期信號

### 長期方案（可能的 Paper）
1. **開發新演算法**
   - 避免除以 q_old 的穩定方法
   - 使用更魯棒的數值方法（共軛梯度、牛頓法）
   - 添加物理約束（電荷守恆、能量最小化）

2. **系統性研究**
   - 不同系統的穩定性比較
   - 參數空間探索
   - 發表改進的演算法

---

## 📝 實現的改進（Cython 版本）

儘管沒有完全解決問題，以下改進已實現：

### 1. 性能優化 ✅
- Poisson solver: 3.76x speedup (26.81ms → 8.24ms)
- Overall: 1.4x speedup (受 Amdahl's Law 限制)
- Warm Start: 額外 1.37x 潛在加速（10ns 後啟動）

### 2. 數值穩定性改進（可選）⚠️
- 電荷硬限制（`q_max` 可調）
- Ez_external 限制（`Ez_max` 可調）
- Scale_factor < 0 修復
- Warm Start 電荷限制
- 阻尼選項（`damping` 可調）

### 3. Debug 工具 ✅
- 詳細的 charge range 輸出
- 異常檢測和警告
- 可配置的防爆策略（見下方 config.ini）

---

## ⚙️ 配置選項（config.ini）

```ini
[NumericalStability]
# 防止數值爆炸策略
# - none: 不使用任何防護措施（原始行為）
# - original: 教授的原始方法（僅 threshold check）
# - conservative: 保守策略（小限制 + 警告）
# - aggressive: 激進策略（大限制 + 阻尼）
anti_explosion_strategy = original

# 當 strategy = conservative/aggressive 時的參數
q_max = 0.5          # 單個電荷硬限制
Ez_max = 1e4         # 外部場硬限制
damping = 0.5        # 阻尼係數 (0=無阻尼, 0.9=強阻尼)
fix_negative_scale = True  # 修復 scale_factor < 0 的 bug
```

---

## 🔬 結論

這是一個**深層的數值穩定性問題**，源於：
1. 原始演算法的設計缺陷（除以可能接近零的電荷）
2. 電荷守恆與數值限制的衝突
3. Scale_charges_analytic 的邏輯漏洞

**目前狀態**：
- ✅ 問題已充分理解和記錄
- ⚠️ 部分改進已實現但未完全解決
- ✅ 性能優化成功（3.76x Poisson solver）
- 🎯 **實用解法**：重算策略（教授建議）

**如果要徹底解決**：
- 需要重新設計核心演算法
- 可能是一篇好的 paper
- 但需要大量時間投入

**推薦策略**：
1. 短期：使用重算 + 配置選項找到穩定參數
2. 中期：記錄數據，識別模式
3. 長期：如果時間允許，嘗試算法改進

---

**報告人**: GitHub Copilot  
**協助**: Andy (用戶)  
**版本**: 1.0  
**最後更新**: 2025-10-25
