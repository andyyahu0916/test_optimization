# 審核修復報告

## 概述

本報告總結了根據 `OpenMM-ConstantV(original)` 黃金標準對 `openmm_core_integration` 插件進行審核後的修復工作。

**審核日期**: 2025年

**修復狀態**: ✅ 完成

---

## 已修復問題

### P0-1: CUDA Green's Reciprocity 缺少 Image Charge 計算

**問題描述**:
原始 CUDA 實作中的 `applyGreensReciprocityKernel` 只做了電荷守恆（確保總和為零），但**完全缺少 Image Charge 計算**。

**黃金標準** (`Fixed_Voltage_routines.py` L333-338):
```python
for i in range(self.nelectrolyte_atoms):
    Q_analytic += (z_distance / Lcell) * (-q_i)
```

**修復方案**:
將 `applyGreensReciprocityKernel` 替換為兩個新 kernel：

1. `computeAnalyticChargeKernel`:
   - 計算幾何項: `±1/(4π) × Area × (V/Lgap + V/Lcell) × K_au`
   - 遍歷電解質原子計算 image charge: `Σ (z_distance/Lcell) × (-q_i)`
   - 使用 warp-level reduction 計算總和

2. `scaleChargesAnalyticKernel`:
   - 計算 `Q_numeric = Σ charges`
   - 計算 `scale_factor = Q_analytic / Q_numeric`
   - 縮放所有電極電荷

**修改檔案**: 
- `openmm_core_integration/platforms/cuda/src/kernels/constantVDrudeLangevin.cu`

---

### P0-2: Reference Platform 使用錯誤的 SCF 方法

**問題描述**:
Reference 平台使用**電位法 (Potential Method)**，而 Python/CUDA 使用**電場法 (E-field Method)**。

| 方法 | 原始 Python | Reference (錯誤) | Reference (修復後) |
|------|-------------|------------------|-------------------|
| 公式 | `q = f(Ez, V/Lgap)` | `q = f(φ)` | `q = f(Ez, V/Lgap)` ✅ |
| 輸入 | 力 `F_z` | 電位 `φ` | 力 `F_z` ✅ |

**黃金標準** (`MM_classes.py` L724-742):
```python
Ez_external = F_z / q_old
q_new = 2/(4π) × area × (V/Lgap + Ez_external) × K_au
```

**修復方案**:
完全重寫 `ReferenceConstantVKernels.cpp`:

1. **更新物理常數**:
   ```cpp
   static const double CONVERSION_NM_TO_BOHR = 18.8973;
   static const double CONVERSION_KJMOL_NM_TO_AU = CONVERSION_NM_TO_BOHR / 2625.5;
   static const double CONVERSION_EV_TO_KJMOL = 96.487;
   static const double FOUR_PI = 12.566370614359172;
   static const double SMALL_THRESHOLD = 1e-6;
   ```

2. **添加 helper 函數**:
   - `computeAnalyticCharge()`: 計算 Q_analytic (包含 image charge)
   - `scaleChargesAnalytic()`: 縮放電荷至解析值

3. **重寫 `runSCF()`**:
   - 新簽名: `runSCF(positions, forces)`
   - 使用 E-field 方法: `Ez = F_z / q`
   - 更新電荷: `q = 2/(4π) × area × (V/Lgap + Ez) × K_au`
   - 縮放至解析正規化

4. **更新 `execute()`**:
   - 從 context 獲取 forces
   - 呼叫 `runSCF(positions, forces)`

**修改檔案**:
- `openmm_core_integration/platforms/reference/src/ReferenceConstantVKernels.cpp`
- `openmm_core_integration/platforms/reference/include/ReferenceConstantVKernels.h`

---

### P1: CustomNonbondedForce 排除缺失

**問題描述**:
`utils/exclusions.py` 只處理 `NonbondedForce`，但原始 Python 同時處理：
- `NonbondedForce` (via `addException`)
- `CustomNonbondedForce` (via `addExclusion`)

**黃金標準** (`electrode_sapt_exclusions.py` L40-44):
```python
customNonbondedForce.addExclusion(indexi, indexj)
nbondedForce.addException(indexi, indexj, 0, 1, 0, True)
```

**修復方案**:
更新 `utils/exclusions.py`:

1. **`exclusion_Electrode_NonbondedForce()`**:
   - 添加 `CustomNonbondedForce` 支援
   - 檢查現有排除避免重複
   - 同時添加 `addException` 和 `addExclusion`

2. **`generate_exclusions_TFSI()`**:
   - 添加 `CustomNonbondedForce` 支援
   - 添加 `DrudeForce.addScreenedPair()` 支援
   - 處理 Thole damping (係數 2.0)

3. **`add_all_exclusions()`**:
   - 自動檢測 `DrudeForce`
   - 傳遞給 TFSI 排除函數

**修改檔案**:
- `utils/exclusions.py`

---

## 驗證建議

### 1. 編譯驗證
```bash
cd /home/andy/test_optimization/openmm_core_integration
mkdir -p build && cd build
cmake ..
make -j4
```

### 2. 單元測試
```bash
cd /home/andy/test_optimization
python -c "
from utils.exclusions import add_all_exclusions
import openmm
print('exclusions.py 語法正確')
"
```

### 3. 數值一致性測試

建議建立測試案例比較：
- **Python 原始版本**: `OpenMM-ConstantV(original)/run_openMM.py`
- **CUDA 插件版本**: 使用 `run_production.py`
- **Reference 插件版本**: 使用 Reference platform

比較項目：
1. 電極電荷分佈
2. Q_analytic 計算值
3. SCF 收斂性
4. 總能量/力

---

## 關鍵公式對照

| 項目 | 公式 | 檔案位置 |
|------|------|----------|
| 電荷更新 | `q_i = 2/(4π) × area × (V/Lgap + Ez) × K_au` | MM_classes.py L738 |
| E-field | `Ez = F_z / q_old` | MM_classes.py L736 |
| Q_analytic (幾何) | `±1/(4π) × Area × (V/Lgap + V/Lcell) × K_au` | Fixed_Voltage_routines.py L322 |
| Q_analytic (image) | `+Σ (z_distance/Lcell) × (-q_i)` | Fixed_Voltage_routines.py L335 |
| 縮放 | `q_scaled = q × (Q_analytic / Q_numeric)` | Fixed_Voltage_routines.py L277 |

---

## 附錄：修改的檔案清單

| 檔案 | 修改類型 | 說明 |
|------|----------|------|
| `constantVDrudeLangevin.cu` | 重寫 | 新增 computeAnalyticChargeKernel, scaleChargesAnalyticKernel |
| `ReferenceConstantVKernels.cpp` | 重寫 | E-field SCF method, helper functions |
| `ReferenceConstantVKernels.h` | 更新 | 新方法聲明 |
| `utils/exclusions.py` | 更新 | CustomNonbondedForce 支援 |

---

*報告結束*
